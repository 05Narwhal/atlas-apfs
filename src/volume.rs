// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! An APFS volume (design §3.6, §5): its superblock, the feature policy, and
//! the `FileSystem` implementation over its file-system tree.

use std::cmp::Ordering;
use std::ops::ControlFlow;
use std::sync::Arc;

use super::btree::{Cursor, Layout, Tree, TreeSpec};
use super::bytes::{array, bytes, le_u16, le_u32, le_u64, len64};
use super::container::{Blocks, Container};
use super::object::{OBJ_NOHEADER, TYPE_FEXT_TREE, TYPE_FSTREE};
use super::omap::{OMAP_VAL_ENCRYPTED, OMAP_VAL_NOHEADER, Omap};
use super::records::{
    DT_DIR, DT_LNK, DT_REG, DirRec, FEXT_PREALLOCATED, FileExtent, INODE_HAS_UNCOMPRESSED_SIZE,
    Inode, T_DIR_REC, T_FILE_EXTENT, T_INODE, T_XATTR, UF_HIDDEN, XattrData, drec_name,
    finder_info, first_of, key_header, name_text, xattr_data, xattr_name,
};
use super::snapshot;
use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};
use atlas_fs_core::fs::{
    AllocationMap, DirBatch, DirEntry, DirProgress, DirStream, EntryFlags, Extent, FileSystem,
    FinderCodes, Metadata, NodeId, NodeKind, Owner, ScanEntry, ScanSummary, SnapshotInfo, Stream,
    Timestamp, VolumeIdentity, VolumeInfo, XattrInfo,
};
use atlas_fs_core::io::block::{BlockSource, MemSource, check_range};
use atlas_hfsplus::catalog::entry_flags;
use atlas_hfsplus::compression::{self as decmpfs, DecmpfsHeader, MAX_DECMPFS_XATTR};
use atlas_hfsplus::unicode::decompose;

/// `APSB`, read as a little-endian `u32`.
const APSB_MAGIC: u32 = u32::from_le_bytes(*b"APSB");

/// Inode numbers.
pub(crate) const ROOT_DIR_ID: u64 = 2;
const FIRST_USER_ID: u64 = 16;

/// Volume incompatible features (§3.6).
const INCOMPAT_CASE_INSENSITIVE: u64 = 0x1;
const INCOMPAT_DATALESS_SNAPS: u64 = 0x2;
const INCOMPAT_ENC_ROLLED: u64 = 0x4;
const INCOMPAT_NORMALIZATION_INSENSITIVE: u64 = 0x8;
const INCOMPAT_INCOMPLETE_RESTORE: u64 = 0x10;
const INCOMPAT_SEALED: u64 = 0x20;
const INCOMPAT_PFK: u64 = 0x40;
const INCOMPAT_EXTENT_PREALLOC: u64 = 0x80;
const INCOMPAT_SECONDARY_FSROOT: u64 = 0x100;

/// `apfs_fs_flags`: the volume is not encrypted.
const FS_UNENCRYPTED: u64 = 0x1;

/// Inode internal flag: the file is sparse; missing extents read as zeros.
const INODE_IS_SPARSE: u64 = 0x200;

/// The three attributes the format keeps for itself.
const RESOURCE_FORK_XATTR: &[u8] = b"com.apple.ResourceFork";
const DECMPFS_XATTR: &[u8] = b"com.apple.decmpfs";
const SYMLINK_XATTR: &[u8] = b"com.apple.fs.symlink";
/// Counted as an attribute, and read for the Finder flags and codes.
const FINDER_INFO_XATTR: &[u8] = b"com.apple.FinderInfo";

/// Longest symbolic link target read.
pub(crate) const MAX_LINK_TARGET: usize = 4096;
/// Most extents of one stream.
pub(crate) const MAX_EXTENTS: usize = 4_000_000;

/// The mode's file type bits.
const S_IFMT: u16 = 0o170_000;
const S_IFDIR: u16 = 0o040_000;
const S_IFREG: u16 = 0o100_000;
const S_IFLNK: u16 = 0o120_000;

fn corrupt(structure: &'static str, location: Location, kind: CorruptKind) -> Error {
    Error::Corrupt(Corruption::new(structure, location, kind))
}

fn bad_record(id: u64, what: &'static str) -> Error {
    corrupt(
        "file-system tree",
        Location::Object(id),
        CorruptKind::Other(what),
    )
}

/// A volume superblock's fields this reader uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ApfsSuperblock {
    /// Where it was read from.
    pub paddr: u64,
    /// The header's transaction.
    pub xid: u64,
    pub incompatible: u64,
    pub root_tree_type: u32,
    pub snap_meta_tree_type: u32,
    pub omap_oid: u64,
    pub root_tree_oid: u64,
    pub snap_meta_tree_oid: u64,
    pub uuid: [u8; 16],
    pub last_modified: u64,
    pub fs_flags: u64,
    pub name: String,
    pub role: u16,
    pub fext_tree_oid: u64,
    pub fext_tree_type: u32,
}

impl ApfsSuperblock {
    /// Parses a verified volume superblock read from `paddr`.
    pub fn parse(block: &[u8], paddr: u64, xid: u64) -> Result<Self> {
        let short = || {
            corrupt(
                "volume superblock",
                Location::Block(paddr),
                CorruptKind::Truncated {
                    needed: 1044,
                    available: len64(block.len()),
                },
            )
        };
        let magic = le_u32(block, 32).ok_or_else(short)?;
        if magic != APSB_MAGIC {
            return Err(corrupt(
                "volume superblock",
                Location::Block(paddr),
                CorruptKind::BadSignature {
                    expected: u64::from(APSB_MAGIC),
                    found: u64::from(magic),
                },
            ));
        }
        Ok(Self {
            paddr,
            xid,
            incompatible: le_u64(block, 56).ok_or_else(short)?,
            root_tree_type: le_u32(block, 116).ok_or_else(short)?,
            snap_meta_tree_type: le_u32(block, 124).ok_or_else(short)?,
            omap_oid: le_u64(block, 128).ok_or_else(short)?,
            root_tree_oid: le_u64(block, 136).ok_or_else(short)?,
            snap_meta_tree_oid: le_u64(block, 152).ok_or_else(short)?,
            uuid: array(block, 240).ok_or_else(short)?,
            last_modified: le_u64(block, 256).ok_or_else(short)?,
            fs_flags: le_u64(block, 264).ok_or_else(short)?,
            name: name_text(bytes(block, 704, 256).ok_or_else(short)?),
            role: le_u16(block, 964).ok_or_else(short)?,
            fext_tree_oid: le_u64(block, 1032).ok_or_else(short)?,
            fext_tree_type: le_u32(block, 1040).ok_or_else(short)?,
        })
    }

    /// True when the volume is encrypted.
    pub fn encrypted(&self) -> bool {
        self.fs_flags & FS_UNENCRYPTED == 0
    }
}

/// The feature policy of §3.6: mounts, or says why not.
pub(crate) fn check_features(sb: &ApfsSuperblock) -> Result<()> {
    let known = INCOMPAT_CASE_INSENSITIVE
        | INCOMPAT_DATALESS_SNAPS
        | INCOMPAT_ENC_ROLLED
        | INCOMPAT_NORMALIZATION_INSENSITIVE
        | INCOMPAT_INCOMPLETE_RESTORE
        | INCOMPAT_SEALED
        | INCOMPAT_PFK
        | INCOMPAT_EXTENT_PREALLOC
        | INCOMPAT_SECONDARY_FSROOT;
    let unknown = sb.incompatible & !known;
    if unknown != 0 {
        return Err(Error::unsupported(format!(
            "this APFS volume uses a feature newer than this reader (incompatible feature {unknown:#x})"
        )));
    }
    if sb.incompatible & INCOMPAT_ENC_ROLLED != 0 {
        return Err(Error::unsupported(
            "this volume is part-way through changing its encryption",
        ));
    }
    if sb.incompatible & INCOMPAT_PFK != 0 && sb.encrypted() {
        return Err(Error::unsupported(
            "this volume can only be unlocked on the Mac it came from",
        ));
    }
    Ok(())
}

/// Everything a volume's operations share; behind an `Arc` so folder
/// streams own it.
pub(crate) struct Core {
    pub container: Arc<Container>,
    pub blocks: Arc<Blocks>,
    pub sb: ApfsSuperblock,
    pub omap: Arc<Omap>,
    fs_tree: Tree,
    /// A sealed volume's file extents (§5.3).
    fext_tree: Option<Tree>,
    /// Directory record keys carry a name hash.
    hashed: bool,
    case_insensitive: bool,
}

/// A mounted APFS volume.
pub struct Volume {
    core: Arc<Core>,
    info: VolumeInfo,
    identity: VolumeIdentity,
}

// `FileSystem` requires it; this names the failure here if a field breaks it.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Volume>();
};

impl Volume {
    /// Opens the volume `sb` of `container`, resolving its objects as of
    /// transaction `xid`.
    pub(crate) fn open(container: Arc<Container>, sb: ApfsSuperblock, xid: u64) -> Result<Self> {
        let name = sb.name.clone();
        Self::open_at(container, sb, xid, None, name)
    }

    /// [`Self::open`] with the object map given (a snapshot shares the live
    /// volume's) and the name to show.
    fn open_at(
        container: Arc<Container>,
        sb: ApfsSuperblock,
        xid: u64,
        omap: Option<Arc<Omap>>,
        name: String,
    ) -> Result<Self> {
        check_features(&sb)?;
        if sb.encrypted() {
            return Err(Error::unsupported("encrypted volume"));
        }
        let blocks = Arc::clone(&container.blocks);
        let omap = match omap {
            Some(omap) => omap,
            None => Arc::new(Omap::open(&blocks, sb.omap_oid)?),
        };
        let fext_tree = if sb.incompatible & INCOMPAT_SEALED != 0 && sb.fext_tree_oid != 0 {
            Some(Tree::open(
                &blocks,
                TreeSpec {
                    root: sb.fext_tree_oid,
                    root_oid: sb.fext_tree_oid,
                    headers: sb.fext_tree_type & OBJ_NOHEADER == 0,
                    layout: Layout::Fixed { key: 16, value: 16 },
                    subtype: TYPE_FEXT_TREE,
                    structure: "file-extent tree",
                    omap: None,
                },
            )?)
        } else {
            None
        };
        let root = omap
            .lookup(&blocks, sb.root_tree_oid, xid)?
            .ok_or_else(|| {
                corrupt(
                    "file-system tree",
                    Location::Object(sb.root_tree_oid),
                    CorruptKind::Other("root is not in the volume's object map"),
                )
            })?;
        if root.flags & OMAP_VAL_ENCRYPTED != 0 {
            return Err(Error::unsupported("encrypted volume"));
        }
        let headers = sb.root_tree_type & OBJ_NOHEADER == 0 && root.flags & OMAP_VAL_NOHEADER == 0;
        let fs_tree = Tree::open(
            &blocks,
            TreeSpec {
                root: root.paddr,
                root_oid: sb.root_tree_oid,
                headers,
                layout: Layout::Variable,
                subtype: TYPE_FSTREE,
                structure: "file-system tree",
                omap: Some((Arc::clone(&omap), xid)),
            },
        )?;
        let insensitive =
            sb.incompatible & (INCOMPAT_CASE_INSENSITIVE | INCOMPAT_NORMALIZATION_INSENSITIVE);

        let mut info = VolumeInfo::new(name);
        info.format = Some("APFS");
        info.total_bytes = Some(container.byte_len());
        // The space manager is read once here; a damaged one only loses this.
        info.free_bytes = super::spaceman::read(&container).ok().map(|s| {
            s.free_count
                .saturating_mul(u64::from(container.blocks.block_size))
        });
        info.resource_forks = true;

        let mut change = [0u8; 16];
        change[..8].copy_from_slice(&xid.to_le_bytes());
        change[8..].copy_from_slice(&sb.last_modified.to_le_bytes());
        let identity = VolumeIdentity {
            id: sb.uuid,
            change,
        };
        let core = Core {
            hashed: insensitive != 0,
            case_insensitive: sb.incompatible & INCOMPAT_CASE_INSENSITIVE != 0,
            container,
            blocks,
            sb,
            omap,
            fs_tree,
            fext_tree,
        };
        Ok(Self {
            core: Arc::new(core),
            info,
            identity,
        })
    }
}

/// An attribute value, owned.
#[derive(Debug, Clone, PartialEq, Eq)]
enum XValue {
    Embedded(Vec<u8>),
    Stream { id: u64, size: u64 },
}

impl XValue {
    fn from(data: XattrData<'_>) -> Self {
        match data {
            XattrData::Embedded(d) => Self::Embedded(d.to_vec()),
            XattrData::Stream { id, size } => Self::Stream { id, size },
        }
    }

    fn size(&self) -> u64 {
        match self {
            Self::Embedded(d) => len64(d.len()),
            Self::Stream { size, .. } => *size,
        }
    }
}

/// The attributes an object visit collects before the inode is known.
#[derive(Default)]
struct Attrs {
    resource: Option<XValue>,
    decmpfs: Option<XValue>,
    symlink: Option<String>,
    finder_info: Option<[u8; 32]>,
    attributes: u32,
}

/// What one object visit learns about an inode (§5.1).
struct Facts {
    id: u64,
    inode: Inode,
    resource: Option<XValue>,
    decmpfs: Option<XValue>,
    symlink: Option<String>,
    finder_info: Option<[u8; 32]>,
    attributes: u32,
}

fn kind_of_mode(mode: u16) -> NodeKind {
    match mode & S_IFMT {
        S_IFDIR => NodeKind::Directory,
        S_IFREG => NodeKind::File,
        S_IFLNK => NodeKind::Symlink,
        _ => NodeKind::Other,
    }
}

fn kind_of_drec(kind: u16) -> NodeKind {
    match kind {
        DT_DIR => NodeKind::Directory,
        DT_REG => NodeKind::File,
        DT_LNK => NodeKind::Symlink,
        _ => NodeKind::Other,
    }
}

fn timestamp(ns: u64) -> Timestamp {
    Timestamp::from_unix_nanos(i64::try_from(ns).unwrap_or(i64::MAX))
}

fn file_id32(id: u64) -> u32 {
    u32::try_from(id).unwrap_or(u32::MAX)
}

impl Facts {
    fn kind(&self) -> NodeKind {
        kind_of_mode(self.inode.mode)
    }

    fn compressed(&self) -> bool {
        self.kind() == NodeKind::File && self.inode.is_compressed()
    }

    fn size(&self) -> Option<u64> {
        match self.kind() {
            NodeKind::Directory => None,
            _ if self.compressed() => {
                if self.inode.internal_flags & INODE_HAS_UNCOMPRESSED_SIZE != 0 {
                    return Some(self.inode.uncompressed_size);
                }
                match &self.decmpfs {
                    Some(XValue::Embedded(d)) => DecmpfsHeader::parse(d, file_id32(self.id))
                        .ok()
                        .map(|h| h.uncompressed_size),
                    _ => None,
                }
            }
            NodeKind::Symlink => Some(self.inode.dstream.map_or_else(
                || self.symlink.as_ref().map_or(0, |t| len64(t.len())),
                |d| d.size,
            )),
            _ => Some(self.inode.dstream.map_or(0, |d| d.size)),
        }
    }

    fn resource_size(&self) -> Option<u64> {
        if self.kind() != NodeKind::File {
            return None;
        }
        if self.compressed() {
            return Some(0);
        }
        Some(self.resource.as_ref().map_or(0, XValue::size))
    }

    fn flags(&self, name_starts_with_dot: bool) -> EntryFlags {
        let finder_flags = self
            .finder_info
            .map_or(0, |fi| u16::from_be_bytes([fi[8], fi[9]]));
        let mut flags = entry_flags(name_starts_with_dot, finder_flags);
        if self.inode.bsd_flags & UF_HIDDEN != 0 {
            flags = flags.with(EntryFlags::HIDDEN);
        }
        if self.compressed() {
            flags = flags.with(EntryFlags::COMPRESSED);
        }
        flags
    }

    fn finder_codes(&self) -> Option<FinderCodes> {
        if self.kind() != NodeKind::File {
            return None;
        }
        let fi = self.finder_info?;
        let file_type: [u8; 4] = array(&fi, 0)?;
        let creator: [u8; 4] = array(&fi, 4)?;
        (file_type != [0; 4] || creator != [0; 4]).then_some(FinderCodes { file_type, creator })
    }
}

/// What [`visit`] calls per record: (type, key, value).
type RecordVisitor<'a> = dyn FnMut(u8, &[u8], &[u8]) -> Result<ControlFlow<()>> + 'a;

/// Calls `f` with every record of object `id` from type `from` on, in key
/// order, until `f` breaks or the object's records end.
fn visit(cursor: &mut Cursor, id: u64, from: u8, f: &mut RecordVisitor<'_>) -> Result<()> {
    cursor.seek(&mut first_of(id, from))?;
    while let Some((key, value)) = cursor.next()? {
        let (kid, kind) = key_header(key).ok_or_else(|| bad_record(id, "key is too short"))?;
        if kid != id || f(kind, key, value)?.is_break() {
            break;
        }
    }
    Ok(())
}

impl Core {
    fn cursor(&self, id: u64, kind: u8) -> Result<Cursor> {
        self.fs_tree
            .seek(Arc::clone(&self.blocks), &mut first_of(id, kind))
    }

    /// The inode `id` with its attributes, through `cursor` (created on
    /// first use). `None` when there is no inode record.
    fn facts_with(&self, cursor: &mut Option<Cursor>, id: u64) -> Result<Option<Facts>> {
        let cursor = match cursor {
            Some(c) => c,
            None => cursor.insert(self.cursor(id, T_INODE)?),
        };
        let mut inode = None;
        let mut facts = Attrs::default();
        visit(cursor, id, T_INODE, &mut |kind, key, value| {
            match kind {
                T_INODE => {
                    inode = Some(
                        Inode::parse(value)
                            .ok_or_else(|| bad_record(id, "inode record is too short"))?,
                    );
                }
                T_XATTR => {
                    let (Some(name), Some(data)) = (xattr_name(key), xattr_data(value)) else {
                        return Ok(ControlFlow::Continue(()));
                    };
                    match name {
                        RESOURCE_FORK_XATTR => facts.resource = Some(XValue::from(data)),
                        DECMPFS_XATTR => facts.decmpfs = Some(XValue::from(data)),
                        SYMLINK_XATTR => {
                            if let XattrData::Embedded(target) = data
                                && target.len() <= MAX_LINK_TARGET
                            {
                                facts.symlink = Some(name_text(target));
                            }
                        }
                        _ => {
                            if name == FINDER_INFO_XATTR
                                && let XattrData::Embedded(fi) = data
                            {
                                facts.finder_info = finder_info(fi);
                            }
                            facts.attributes = facts.attributes.saturating_add(1);
                        }
                    }
                }
                _ => return Ok(ControlFlow::Break(())),
            }
            Ok(ControlFlow::Continue(()))
        })?;
        Ok(inode.map(|inode| Facts {
            id,
            inode,
            resource: facts.resource,
            decmpfs: facts.decmpfs,
            symlink: facts.symlink,
            finder_info: facts.finder_info,
            attributes: facts.attributes,
        }))
    }

    fn facts(&self, id: u64) -> Result<Facts> {
        self.facts_with(&mut None, id)?
            .ok_or(Error::NotFound { node: id })
    }

    /// The runs of the stream whose extents are keyed by `id`: from the
    /// sealed file-extent tree when the volume has one and it holds the
    /// stream, else from the file-system tree (§5.3).
    fn runs(&self, id: u64) -> Result<Vec<Run>> {
        let mut runs = Vec::new();
        let mut end = 0u64;
        if let Some(fext) = &self.fext_tree {
            let mut cursor = fext.seek(Arc::clone(&self.blocks), &mut |key: &[u8]| {
                let stored = (le_u64(key, 0), le_u64(key, 8));
                match stored {
                    (Some(kid), Some(logical)) => Ok((kid, logical).cmp(&(id, 0))),
                    _ => Err(bad_record(id, "file extent key is too short")),
                }
            })?;
            while let Some((key, value)) = cursor.next()? {
                let (kid, e) = FileExtent::parse_sealed(key, value)
                    .ok_or_else(|| bad_record(id, "file extent record is too short"))?;
                if kid != id {
                    break;
                }
                self.push_run(&mut runs, &mut end, id, e)?;
            }
            if !runs.is_empty() {
                return Ok(runs);
            }
        }
        let mut cursor = self.cursor(id, T_FILE_EXTENT)?;
        while let Some((key, value)) = cursor.next()? {
            let (kid, kind) = key_header(key).ok_or_else(|| bad_record(id, "key is too short"))?;
            if kid != id || kind != T_FILE_EXTENT {
                break;
            }
            let e = FileExtent::parse(key, value)
                .ok_or_else(|| bad_record(id, "file extent record is too short"))?;
            self.push_run(&mut runs, &mut end, id, e)?;
        }
        Ok(runs)
    }

    /// Checks one extent of stream `id` against the ones before it (ending
    /// at `end`) and the container, then appends it.
    fn push_run(&self, runs: &mut Vec<Run>, end: &mut u64, id: u64, e: FileExtent) -> Result<()> {
        let bs = u64::from(self.blocks.block_size);
        if e.len == 0 || e.len.checked_rem(bs) != Some(0) || e.logical < *end {
            return Err(bad_record(id, "file extent length or order is invalid"));
        }
        let zeros = e.paddr == 0 || e.flags & FEXT_PREALLOCATED != 0;
        if !zeros {
            let inside = e
                .paddr
                .checked_add(e.len.checked_div(bs).unwrap_or(0))
                .is_some_and(|last| last <= self.blocks.block_count);
            if !inside {
                return Err(corrupt(
                    "file-system tree",
                    Location::Object(id),
                    CorruptKind::OutOfRange {
                        field: "file extent block",
                        value: e.paddr,
                        limit_name: "container blocks",
                        limit: self.blocks.block_count,
                    },
                ));
            }
        }
        *end = e
            .logical
            .checked_add(e.len)
            .ok_or_else(|| bad_record(id, "file extent end overflows"))?;
        if runs.len() >= MAX_EXTENTS {
            return Err(Error::unsupported(format!(
                "a stream of more than {MAX_EXTENTS} extents"
            )));
        }
        runs.push(Run {
            logical: e.logical,
            len: e.len,
            paddr: e.paddr,
            zeros,
        });
        Ok(())
    }

    /// A readable source for an attribute value.
    fn open_xvalue(&self, value: &XValue) -> Result<Arc<dyn BlockSource>> {
        match value {
            XValue::Embedded(d) => Ok(Arc::new(MemSource::new(d.clone()))),
            XValue::Stream { id, size } => Ok(Arc::new(ExtentSource {
                blocks: Arc::clone(&self.blocks),
                runs: self.runs(*id)?,
                len: *size,
                sparse: false,
                id: *id,
            })),
        }
    }

    /// An attribute value read whole, failing when it is longer than `max`.
    fn read_xvalue(&self, value: &XValue, max: usize) -> Result<Vec<u8>> {
        let size = value.size();
        if size > len64(max) {
            return Err(Error::unsupported(format!(
                "an attribute of {size} bytes (the limit is {max})"
            )));
        }
        match value {
            XValue::Embedded(d) => Ok(d.clone()),
            XValue::Stream { .. } => {
                let source = self.open_xvalue(value)?;
                let mut out = vec![0u8; usize::try_from(size).unwrap_or(0)];
                source.read_at(0, &mut out)?;
                Ok(out)
            }
        }
    }

    /// The name of a directory record key, without the NUL.
    fn record_name(&self, key: &[u8], dir: u64) -> Result<String> {
        drec_name(key, self.hashed)
            .map(name_text)
            .ok_or_else(|| bad_record(dir, "directory record key is too short"))
    }

    /// Checks `dir` is a folder.
    fn require_folder(&self, dir: u64) -> Result<()> {
        match self.facts(dir)?.kind() {
            NodeKind::Directory => Ok(()),
            _ => Err(Error::NotADirectory { node: dir }),
        }
    }
}

/// One run of a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Run {
    logical: u64,
    len: u64,
    paddr: u64,
    /// A hole or preallocated space: reads as zeros.
    zeros: bool,
}

/// A stream's content through its file extents (§5.3).
pub(crate) struct ExtentSource {
    blocks: Arc<Blocks>,
    runs: Vec<Run>,
    len: u64,
    /// Missing extents read as zeros (a sparse file).
    sparse: bool,
    id: u64,
}

impl ExtentSource {
    fn read_into(&self, offset: u64, buf: &mut [u8], bulk: bool) -> Result<()> {
        check_range(offset, buf.len(), self.len)?;
        let bs = u64::from(self.blocks.block_size);
        let source = self.blocks.source();
        let mut pos = offset;
        let mut done = 0usize;
        // Each pass fills at least one byte: every branch below has n >= 1.
        while done < buf.len() {
            let left = buf.len().saturating_sub(done);
            let after = self.runs.partition_point(|r| r.logical <= pos);
            let current = after.checked_sub(1).and_then(|i| self.runs.get(i));
            let (n, from) = match current {
                Some(r) if pos < r.logical.saturating_add(r.len) => {
                    let n = usize::try_from(r.logical.saturating_add(r.len).saturating_sub(pos))
                        .map_or(left, |room| room.min(left));
                    let at = (!r.zeros).then(|| {
                        r.paddr
                            .checked_mul(bs)
                            .and_then(|b| b.checked_add(pos.saturating_sub(r.logical)))
                    });
                    (n, at)
                }
                _ => match self.runs.get(after) {
                    Some(next) => {
                        let gap = usize::try_from(next.logical.saturating_sub(pos))
                            .map_or(left, |room| room.min(left));
                        (gap, None)
                    }
                    None if self.sparse => (left, None),
                    None => {
                        return Err(corrupt(
                            "file content",
                            Location::Object(self.id),
                            CorruptKind::Other("content ends before the stream's size"),
                        ));
                    }
                },
            };
            let n = n.max(1).min(left);
            let chunk = buf
                .get_mut(done..done.saturating_add(n))
                .ok_or_else(|| bad_record(self.id, "read past the buffer"))?;
            match from {
                None => chunk.fill(0),
                Some(None) => return Err(bad_record(self.id, "file extent address overflows")),
                Some(Some(byte)) if bulk => source.read_bulk(byte, chunk)?,
                Some(Some(byte)) => source.read_at(byte, chunk)?,
            }
            done = done.saturating_add(n);
            pos = pos.saturating_add(len64(n));
        }
        Ok(())
    }

    /// The runs as byte extents of the container, holes left out, adjacent
    /// runs merged.
    fn extents(&self) -> Vec<Extent> {
        let bs = u64::from(self.blocks.block_size);
        let mut out: Vec<Extent> = Vec::new();
        for r in self.runs.iter().filter(|r| !r.zeros) {
            let Some(offset) = r.paddr.checked_mul(bs) else {
                continue;
            };
            if let Some(last) = out.last_mut()
                && last.offset.checked_add(last.len) == Some(offset)
            {
                last.len = last.len.saturating_add(r.len);
                continue;
            }
            out.push(Extent { offset, len: r.len });
        }
        out
    }
}

impl BlockSource for ExtentSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn sector_size(&self) -> u32 {
        512
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.read_into(offset, buf, false)
    }

    fn read_bulk(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.read_into(offset, buf, true)
    }
}

/// A folder listing: one cursor walks the directory records, a second
/// visits each child's inode (decision 5: children are usually neighbours).
struct ApfsDir {
    core: Arc<Core>,
    dir: u64,
    entries: Cursor,
    inodes: Option<Cursor>,
    done: bool,
}

impl ApfsDir {
    fn step(&mut self, out: &mut DirBatch, max: usize) -> Result<DirProgress> {
        let mut added = 0usize;
        while added < max {
            let Some((key, value)) = self.entries.next()? else {
                return Ok(DirProgress::Done);
            };
            let (id, kind) =
                key_header(key).ok_or_else(|| bad_record(self.dir, "key is too short"))?;
            if id != self.dir || kind != T_DIR_REC {
                return Ok(DirProgress::Done);
            }
            let name = self.core.record_name(key, self.dir)?;
            let rec = DirRec::parse(value)
                .ok_or_else(|| bad_record(self.dir, "directory record is too short"))?;
            // A child whose inode cannot be read is still listed.
            let facts = self
                .core
                .facts_with(&mut self.inodes, rec.file_id)
                .ok()
                .flatten();
            let dot = name.starts_with('.');
            let entry = match &facts {
                Some(f) => DirEntry {
                    node: NodeId(rec.file_id),
                    name: &name,
                    kind: f.kind(),
                    size: f.size(),
                    resource_size: f.resource_size(),
                    modified: Some(timestamp(f.inode.modified)),
                    flags: f.flags(dot),
                },
                None => DirEntry {
                    node: NodeId(rec.file_id),
                    name: &name,
                    kind: kind_of_drec(rec.kind),
                    size: None,
                    resource_size: None,
                    modified: None,
                    flags: if dot {
                        EntryFlags::HIDDEN
                    } else {
                        EntryFlags::EMPTY
                    },
                },
            };
            out.push(entry);
            added = added.saturating_add(1);
        }
        Ok(DirProgress::More)
    }
}

impl DirStream for ApfsDir {
    fn next_batch(&mut self, out: &mut DirBatch, max: usize) -> Result<DirProgress> {
        if self.done {
            return Ok(DirProgress::Done);
        }
        let result = self.step(out, max);
        if !matches!(result, Ok(DirProgress::More)) {
            self.done = true;
        }
        result
    }
}

/// `name` as compared on an insensitive volume: decomposed, and lowercased
/// when the volume ignores case.
fn folded(name: &str, case_insensitive: bool) -> Vec<u16> {
    let mut out = Vec::with_capacity(name.len());
    if case_insensitive {
        decompose(&name.to_lowercase(), &mut out);
    } else {
        decompose(name, &mut out);
    }
    out
}

impl FileSystem for Volume {
    fn info(&self) -> VolumeInfo {
        self.info.clone()
    }

    fn root(&self) -> NodeId {
        NodeId(ROOT_DIR_ID)
    }

    fn metadata(&self, node: NodeId) -> Result<Metadata> {
        let f = self.core.facts(node.0)?;
        let kind = f.kind();
        Ok(Metadata {
            node,
            kind,
            size: f.size(),
            resource_size: f.resource_size(),
            created: Some(timestamp(f.inode.created)),
            modified: Some(timestamp(f.inode.modified)),
            accessed: Some(timestamp(f.inode.accessed)),
            flags: f.flags(f.inode.name.as_deref().is_some_and(|n| n.starts_with('.'))),
            link_target: (kind == NodeKind::Symlink)
                .then(|| f.symlink.clone())
                .flatten(),
            finder_codes: f.finder_codes(),
            owner: Some(Owner {
                user: f.inode.owner,
                group: f.inode.group,
                mode: f.inode.mode,
            }),
            attributes: Some(f.attributes),
            finder_info: f.finder_info,
        })
    }

    fn lookup(&self, dir: NodeId, name: &str) -> Result<Option<NodeId>> {
        let mut cursor = self.core.cursor(dir.0, T_DIR_REC)?;
        let insensitive = self.core.hashed;
        let want = insensitive.then(|| folded(name, self.core.case_insensitive));
        let mut fallback = None;
        while let Some((key, value)) = cursor.next()? {
            let (id, kind) =
                key_header(key).ok_or_else(|| bad_record(dir.0, "key is too short"))?;
            if id != dir.0 || kind != T_DIR_REC {
                break;
            }
            let stored = self.core.record_name(key, dir.0)?;
            let Some(rec) = DirRec::parse(value) else {
                continue;
            };
            if stored == name {
                return Ok(Some(NodeId(rec.file_id)));
            }
            if fallback.is_none()
                && let Some(want) = &want
                && folded(&stored, self.core.case_insensitive) == *want
            {
                fallback = Some(NodeId(rec.file_id));
            }
        }
        Ok(fallback)
    }

    fn read_dir(&self, dir: NodeId) -> Result<Box<dyn DirStream>> {
        self.core.require_folder(dir.0)?;
        Ok(Box::new(ApfsDir {
            entries: self.core.cursor(dir.0, T_DIR_REC)?,
            core: Arc::clone(&self.core),
            dir: dir.0,
            inodes: None,
            done: false,
        }))
    }

    fn extents(&self, node: NodeId, stream: Stream) -> Result<Vec<Extent>> {
        let f = self.core.facts(node.0)?;
        if f.kind() == NodeKind::Directory {
            return Err(Error::IsADirectory { node: node.0 });
        }
        let id = match stream {
            Stream::Data if f.compressed() => match &f.resource {
                Some(XValue::Stream { id, .. }) => *id,
                _ => return Ok(Vec::new()),
            },
            Stream::Data if f.kind() == NodeKind::File => f.inode.private_id,
            Stream::Resource if !f.compressed() => match &f.resource {
                Some(XValue::Stream { id, .. }) => *id,
                _ => return Ok(Vec::new()),
            },
            _ => return Ok(Vec::new()),
        };
        let source = ExtentSource {
            blocks: Arc::clone(&self.core.blocks),
            runs: self.core.runs(id)?,
            len: 0,
            sparse: false,
            id,
        };
        Ok(source.extents())
    }

    fn open(&self, node: NodeId, stream: Stream) -> Result<Arc<dyn BlockSource>> {
        let f = self.core.facts(node.0)?;
        match (f.kind(), stream) {
            (NodeKind::Directory, _) => Err(Error::IsADirectory { node: node.0 }),
            (NodeKind::Symlink, Stream::Data) => Ok(Arc::new(MemSource::new(
                f.symlink.clone().unwrap_or_default().into_bytes(),
            ))),
            (_, Stream::Data) if f.compressed() => {
                let xattr = match &f.decmpfs {
                    Some(value) => self.core.read_xvalue(value, MAX_DECMPFS_XATTR)?,
                    None => {
                        return Err(bad_record(
                            node.0,
                            "compressed file has no decmpfs attribute",
                        ));
                    }
                };
                let core = &self.core;
                let resource = f.resource.clone();
                decmpfs::open(&xattr, file_id32(node.0), &mut || match &resource {
                    Some(value) => core.open_xvalue(value),
                    None => Err(Error::NoSuchStream { node: node.0 }),
                })
            }
            (_, Stream::Data) => Ok(Arc::new(ExtentSource {
                blocks: Arc::clone(&self.core.blocks),
                runs: self.core.runs(f.inode.private_id)?,
                len: f.inode.dstream.map_or(0, |d| d.size),
                sparse: f.inode.internal_flags & INODE_IS_SPARSE != 0,
                id: f.inode.private_id,
            })),
            (_, Stream::Resource) => match &f.resource {
                Some(value) if !f.compressed() && value.size() > 0 => self.core.open_xvalue(value),
                _ => Err(Error::NoSuchStream { node: node.0 }),
            },
        }
    }

    fn xattrs(&self, node: NodeId) -> Result<Vec<XattrInfo>> {
        let mut out = Vec::new();
        let mut cursor = self.core.cursor(node.0, T_XATTR)?;
        visit(&mut cursor, node.0, T_XATTR, &mut |kind, key, value| {
            if kind != T_XATTR {
                return Ok(ControlFlow::Break(()));
            }
            if let (Some(name), Some(data)) = (xattr_name(key), xattr_data(value))
                && !matches!(name, RESOURCE_FORK_XATTR | DECMPFS_XATTR | SYMLINK_XATTR)
            {
                out.push(XattrInfo {
                    name: name_text(name),
                    size: data.size(),
                });
            }
            Ok(ControlFlow::Continue(()))
        })?;
        Ok(out)
    }

    fn read_xattr(&self, node: NodeId, name: &str, max: usize) -> Result<Option<Vec<u8>>> {
        let want = name.as_bytes();
        if matches!(want, RESOURCE_FORK_XATTR | DECMPFS_XATTR | SYMLINK_XATTR) {
            return Ok(None);
        }
        let mut found = None;
        let mut cursor = self.core.cursor(node.0, T_XATTR)?;
        visit(&mut cursor, node.0, T_XATTR, &mut |kind, key, value| {
            if kind != T_XATTR {
                return Ok(ControlFlow::Break(()));
            }
            if xattr_name(key) == Some(want) {
                found = xattr_data(value).map(XValue::from);
                return Ok(ControlFlow::Break(()));
            }
            Ok(ControlFlow::Continue(()))
        })?;
        found
            .map(|value| self.core.read_xvalue(&value, max))
            .transpose()
    }

    fn allocation_map(&self, buckets: usize) -> Result<AllocationMap> {
        super::spaceman::allocation_map(&self.core.container, buckets)
    }

    fn scan(&self, visit: &mut dyn FnMut(ScanEntry<'_>) -> ControlFlow<()>) -> Result<ScanSummary> {
        scan(&self.core, visit)
    }

    fn snapshots(&self) -> Result<Vec<SnapshotInfo>> {
        Ok(snapshot::list(&self.core.container, &self.core.sb)?
            .iter()
            .map(snapshot::SnapRecord::info)
            .collect())
    }

    /// The volume at snapshot `id` (its xid): the superblock stored with the
    /// snapshot, read through this volume's object map at that transaction.
    fn open_snapshot(&self, id: u64) -> Result<Arc<dyn FileSystem>> {
        let container = &self.core.container;
        let record = snapshot::find(container, &self.core.sb, id)?;
        let sb = snapshot::superblock(container, &record)?;
        if sb.incompatible & INCOMPAT_DATALESS_SNAPS != 0 {
            return Err(Error::unsupported("a snapshot that holds no data"));
        }
        let name = format!("{} \u{2014} {}", self.info.name, record.name);
        Ok(Arc::new(Self::open_at(
            Arc::clone(container),
            sb,
            record.xid,
            Some(Arc::clone(&self.core.omap)),
            name,
        )?))
    }

    fn identity(&self) -> Option<VolumeIdentity> {
        Some(self.identity)
    }
}

/// One in-order pass over the whole file-system tree (§5.2): each inode
/// with an id of 16 or more and a name is reported once its attributes have
/// been seen. A damaged node ends the pass; it is counted, not an error.
fn scan(
    core: &Core,
    visit: &mut dyn FnMut(ScanEntry<'_>) -> ControlFlow<()>,
) -> Result<ScanSummary> {
    let mut summary = ScanSummary::default();
    let mut cursor = core
        .fs_tree
        .seek(Arc::clone(&core.blocks), &mut |_| Ok(Ordering::Greater))?;
    let mut pending: Option<Facts> = None;

    let mut emit = |facts: Facts, summary: &mut ScanSummary| -> ControlFlow<()> {
        let Some(name) = facts.inode.name.as_deref() else {
            summary.skipped = summary.skipped.saturating_add(1);
            return ControlFlow::Continue(());
        };
        summary.entries = summary.entries.saturating_add(1);
        visit(ScanEntry {
            node: NodeId(facts.id),
            parent: NodeId(facts.inode.parent),
            name,
            kind: facts.kind(),
            size: facts.size(),
            resource_size: facts.resource_size(),
            modified: Some(timestamp(facts.inode.modified)),
            flags: facts.flags(name.starts_with('.')),
            finder_type: facts
                .finder_codes()
                .map(|c| c.file_type)
                .filter(|t| *t != [0; 4]),
        })
    };

    loop {
        let record = match cursor.next() {
            Ok(Some((key, value))) => key_header(key).map(|(id, kind)| {
                let xattr = (kind == T_XATTR)
                    .then(|| xattr_name(key).zip(xattr_data(value)))
                    .flatten()
                    .map(|(n, d)| (n.to_vec(), XValue::from(d)));
                let inode = (kind == T_INODE).then(|| Inode::parse(value)).flatten();
                (id, kind, xattr, inode)
            }),
            Ok(None) => break,
            Err(_) => {
                summary.skipped = summary.skipped.saturating_add(1);
                break;
            }
        };
        let Some((id, kind, xattr, inode)) = record else {
            summary.skipped = summary.skipped.saturating_add(1);
            continue;
        };
        if pending.as_ref().is_some_and(|p| p.id != id)
            && let Some(done) = pending.take()
            && emit(done, &mut summary).is_break()
        {
            return Ok(summary);
        }
        match (kind, inode) {
            (T_INODE, Some(inode)) if id >= FIRST_USER_ID => {
                pending = Some(Facts {
                    id,
                    inode,
                    resource: None,
                    decmpfs: None,
                    symlink: None,
                    finder_info: None,
                    attributes: 0,
                });
            }
            (T_INODE, None) => summary.skipped = summary.skipped.saturating_add(1),
            (T_XATTR, _) => {
                if let (Some(p), Some((name, data))) = (pending.as_mut(), xattr) {
                    match name.as_slice() {
                        RESOURCE_FORK_XATTR => p.resource = Some(data),
                        DECMPFS_XATTR => p.decmpfs = Some(data),
                        FINDER_INFO_XATTR => {
                            if let XValue::Embedded(fi) = &data {
                                p.finder_info = finder_info(fi);
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    if let Some(done) = pending.take() {
        let _ = emit(done, &mut summary);
    }
    Ok(summary)
}

#[cfg(test)]
mod tests;
