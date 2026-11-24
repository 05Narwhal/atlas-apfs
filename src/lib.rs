// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! APFS, read-only (little-endian).
//!
//! [`mount`] opens the container in a source: the superblock of the newest
//! valid checkpoint, the container object map and a shared node cache. Each
//! volume in it is mounted as a [`Volume`], which answers the
//! [`FileSystem`](atlas_fs_core::fs::FileSystem) calls (folders, content,
//! extended attributes, snapshots, decmpfs compressed files). A volume that
//! cannot be read is reported as [`VolumeState::Failed`] beside the others.
//!
//! Damaged input is an error, never a panic: every length, offset and count
//! read from the disk is checked before use. Encrypted volumes are not
//! read.
//!
//! ```no_run
//! use std::path::Path;
//! use std::sync::Arc;
//!
//! use atlas_fs_core::fs::{FileSystem, VolumeState};
//! use atlas_fs_core::io::file::FileSource;
//!
//! # fn main() -> Result<(), atlas_fs_core::error::Error> {
//! let source = Arc::new(FileSource::open(Path::new("container.img"))?);
//! for volume in atlas_apfs::mount(source)? {
//!     if let VolumeState::Ready(fs) = volume.state {
//!         println!("{}: {}", volume.name, fs.info().name);
//!     }
//! }
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]
// On-disk bytes are hostile. These lints make the usual ways of trusting
// them (unchecked indexing, arithmetic and casts, panicking shortcuts) fail
// the clippy gate. Unit tests and the test fixtures are exempt.
#![warn(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unimplemented,
    clippy::unreachable,
    clippy::unwrap_used
)]
#![cfg_attr(
    any(test, feature = "test-util"),
    allow(
        clippy::arithmetic_side_effects,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used
    )
)]

use std::sync::Arc;

use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};
use atlas_fs_core::fs::{ContainerVolume, VolumeRole, VolumeState};
use atlas_fs_core::io::block::BlockSource;

pub(crate) mod btree;
pub(crate) mod bytes;
pub(crate) mod container;
#[cfg(test)]
mod fuzz_seeds;
pub(crate) mod object;
pub(crate) mod omap;
pub(crate) mod records;
pub(crate) mod snapshot;
pub(crate) mod spaceman;
#[cfg(any(test, feature = "test-util"))]
#[doc(hidden)]
#[allow(
    dead_code,
    missing_docs,
    private_interfaces,
    clippy::new_without_default
)]
pub mod test_container;
pub mod volume;

pub use volume::Volume;

use container::Container;
use object::TYPE_FS;
use volume::ApfsSuperblock;

/// Opens the APFS container in `source` and every volume in it, in the
/// container's slot order.
///
/// Fails only when the container itself cannot be opened; a volume that
/// cannot be read is returned as [`VolumeState::Failed`] beside the others.
pub fn mount(source: Arc<dyn BlockSource>) -> Result<Vec<ContainerVolume>> {
    let container = Container::open(source)?;
    Ok(container
        .superblock
        .fs_oids
        .iter()
        .map(|&oid| open_slot(&container, oid))
        .collect())
}

/// The volume superblock named by the virtual oid `oid`, as of the
/// container's checkpoint.
fn read_superblock(container: &Container, oid: u64) -> Result<ApfsSuperblock> {
    let mapping = container
        .omap
        .lookup(&container.blocks, oid, container.xid())?
        .ok_or_else(|| {
            Error::Corrupt(Corruption::new(
                "container object map",
                Location::Object(oid),
                CorruptKind::Other("volume is not in the container's object map"),
            ))
        })?;
    let block = container.blocks.block(mapping.paddr)?;
    let header = object::verify(
        &block,
        mapping.paddr,
        "volume superblock",
        &[TYPE_FS],
        Some(oid),
    )?;
    ApfsSuperblock::parse(&block, mapping.paddr, header.xid)
}

fn open_slot(container: &Arc<Container>, oid: u64) -> ContainerVolume {
    let sb = match read_superblock(container, oid) {
        Ok(sb) => sb,
        Err(e) => return ContainerVolume::new("", None, VolumeState::Failed(e)),
    };
    let name = sb.name.clone();
    let role = role_of(sb.role);
    let state = match Volume::open(Arc::clone(container), sb, container.xid()) {
        Ok(volume) => VolumeState::Ready(Arc::new(volume)),
        Err(e) => VolumeState::Failed(e),
    };
    ContainerVolume::new(name, role, state)
}

/// The sidebar role of `apfs_role` (§7): user and data are both Data; a
/// role without a variant is `None`.
fn role_of(role: u16) -> Option<VolumeRole> {
    match role {
        0x1 => Some(VolumeRole::System),
        0x2 | 0x40 => Some(VolumeRole::Data),
        0x4 => Some(VolumeRole::Recovery),
        0x8 => Some(VolumeRole::Vm),
        0x10 => Some(VolumeRole::Preboot),
        _ => None,
    }
}

/// Entry points for the fuzz targets in `fuzz/`. Not an API: they exist so
/// the targets can reach parsers that are private to the engine.
#[doc(hidden)]
pub mod fuzzing {
    use std::sync::Arc;

    use super::btree::{Layout, Node, NodeSpec};
    use super::container::Container;
    use atlas_fs_core::io::block::MemSource;

    /// Parses `data` as a B-tree node of each layout and reads every entry
    /// of the ones that parse. Errors are fine; a panic is the bug.
    pub fn btree_node(data: &[u8]) {
        for layout in [Layout::Variable, Layout::Fixed { key: 16, value: 16 }] {
            let spec = NodeSpec {
                oid: None,
                headers: true,
                layout,
                subtype: 0,
                structure: "fuzzed node",
            };
            if let Ok(node) = Node::parse(data.to_vec().into_boxed_slice(), 0, &spec) {
                for i in 0..node.len() {
                    let _ = (node.key(i).len(), node.value(i).map(<[u8]>::len));
                }
            }
        }
    }

    /// Opens the container in `data` and resolves every volume through its
    /// object map. Errors are fine; a panic is the bug.
    pub fn container(data: &[u8]) {
        let Ok(container) = Container::open(Arc::new(MemSource::new(data))) else {
            return;
        };
        let _ = container.ephemeral(container.superblock.spaceman_oid);
        for &oid in &container.superblock.fs_oids {
            let _ = container
                .omap
                .lookup(&container.blocks, oid, container.xid());
        }
    }
}
