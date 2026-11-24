// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Snapshots (design §10): the snapshot metadata tree, and opening a
//! snapshot as a volume that resolves every virtual object at the
//! snapshot's transaction.

use std::sync::Arc;

use super::btree::{Layout, Tree, TreeSpec};
use super::bytes::{bytes, le_u16, le_u64};
use super::container::Container;
use super::object::{self, OBJ_NOHEADER, TYPE_FS, TYPE_SNAPMETATREE};
use super::records::{T_SNAP_METADATA, first_of, key_header, name_text};
use super::volume::ApfsSuperblock;
use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};
use atlas_fs_core::fs::{SnapshotInfo, Timestamp};

/// Most snapshots listed.
pub(crate) const MAX_SNAPSHOTS: usize = 65_536;

/// Size of a snapshot metadata value before its name.
const SNAP_FIXED_LEN: usize = 50;

/// One snapshot metadata record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SnapRecord {
    /// The snapshot's transaction: its id.
    pub xid: u64,
    /// Where its volume superblock is.
    pub superblock: u64,
    pub created: u64,
    pub name: String,
}

impl SnapRecord {
    fn parse(xid: u64, value: &[u8]) -> Option<Self> {
        let name_len = usize::from(le_u16(value, 48)?);
        Some(Self {
            xid,
            superblock: le_u64(value, 8)?,
            created: le_u64(value, 16)?,
            name: name_text(bytes(value, SNAP_FIXED_LEN, name_len)?),
        })
    }

    pub fn info(&self) -> SnapshotInfo {
        SnapshotInfo {
            id: self.xid,
            name: self.name.clone(),
            created: Some(Timestamp::from_unix_nanos(
                i64::try_from(self.created).unwrap_or(i64::MAX),
            )),
        }
    }
}

/// The snapshot metadata tree of `sb`, or `None` when the volume has none.
fn tree(container: &Container, sb: &ApfsSuperblock) -> Result<Option<Tree>> {
    if sb.snap_meta_tree_oid == 0 {
        return Ok(None);
    }
    Tree::open(
        &container.blocks,
        TreeSpec {
            root: sb.snap_meta_tree_oid,
            root_oid: sb.snap_meta_tree_oid,
            headers: sb.snap_meta_tree_type & OBJ_NOHEADER == 0,
            layout: Layout::Variable,
            subtype: TYPE_SNAPMETATREE,
            structure: "snapshot metadata tree",
            omap: None,
        },
    )
    .map(Some)
}

/// Every snapshot of the volume `sb`, oldest first, at most
/// [`MAX_SNAPSHOTS`].
pub(crate) fn list(container: &Container, sb: &ApfsSuperblock) -> Result<Vec<SnapRecord>> {
    let Some(tree) = tree(container, sb)? else {
        return Ok(Vec::new());
    };
    let mut cursor = tree.seek(
        Arc::clone(&container.blocks),
        &mut first_of(0, T_SNAP_METADATA),
    )?;
    let mut out = Vec::new();
    while let Some((key, value)) = cursor.next()? {
        let Some((xid, kind)) = key_header(key) else {
            continue;
        };
        // Keys order by id, then type: metadata records (id = xid) come
        // before the name records (type 11, id all ones), which are skipped.
        if kind != T_SNAP_METADATA {
            continue;
        }
        if out.len() >= MAX_SNAPSHOTS {
            break;
        }
        if let Some(record) = SnapRecord::parse(xid, value) {
            out.push(record);
        }
    }
    Ok(out)
}

/// The snapshot `xid`'s record.
pub(crate) fn find(container: &Container, sb: &ApfsSuperblock, xid: u64) -> Result<SnapRecord> {
    let not_found = || Error::NotFound { node: xid };
    let tree = tree(container, sb)?.ok_or_else(not_found)?;
    let mut cursor = tree.seek(
        Arc::clone(&container.blocks),
        &mut first_of(xid, T_SNAP_METADATA),
    )?;
    match cursor.next()? {
        Some((key, value)) if key_header(key) == Some((xid, T_SNAP_METADATA)) => {
            SnapRecord::parse(xid, value).ok_or_else(|| {
                Error::Corrupt(Corruption::new(
                    "snapshot metadata tree",
                    Location::Object(xid),
                    CorruptKind::Other("snapshot record is too short"),
                ))
            })
        }
        _ => Err(not_found()),
    }
}

/// The volume superblock a snapshot was taken with.
pub(crate) fn superblock(container: &Container, record: &SnapRecord) -> Result<ApfsSuperblock> {
    let block = container.blocks.block(record.superblock)?;
    let header = object::verify(
        &block,
        record.superblock,
        "snapshot volume superblock",
        &[TYPE_FS],
        None,
    )?;
    ApfsSuperblock::parse(&block, record.superblock, header.xid)
}
