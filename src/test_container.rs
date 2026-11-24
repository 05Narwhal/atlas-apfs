// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Builds APFS containers in memory, with correct checksums, for tests and
//! fuzz seeds (design §12). Test code: it panics on misuse.

use std::sync::Arc;

use super::btree::{INFO_LEN, Layout};
use super::container::Blocks;
use super::object::{
    OBJ_EPHEMERAL, OBJ_NOHEADER, OBJ_PHYSICAL, TYPE_BTREE, TYPE_BTREE_NODE, TYPE_CHECKPOINT_MAP,
    TYPE_FEXT_TREE, TYPE_FS, TYPE_FSTREE, TYPE_NX_SUPERBLOCK, TYPE_OMAP, TYPE_OMAP_TREE,
    TYPE_SNAPMETATREE, TYPE_SPACEMAN, TYPE_SPACEMAN_CIB, seal,
};
use atlas_fs_core::io::block::MemSource;

/// Block size of every built container.
pub const BS: usize = 4096;
/// The container superblock's oid.
const NX_OID: u64 = 1;
/// Ephemeral oid of the space manager.
pub const SPACEMAN_OID: u64 = 0x400;
/// First virtual oid handed out.
const FIRST_VIRTUAL_OID: u64 = 1026;
/// Transaction every virtual mapping is made at.
pub const MAP_XID: u64 = 1;
/// The container UUID.
pub const NX_UUID: [u8; 16] = *b"disk-atlas-apfs!";

/// Blocks of an image under construction.
#[derive(Default)]
pub struct Image {
    blocks: Vec<Vec<u8>>,
}

impl Image {
    /// The next free block, zeroed.
    pub fn alloc(&mut self) -> u64 {
        self.blocks.push(vec![0u8; BS]);
        (self.blocks.len() - 1) as u64
    }

    /// Replaces block `paddr`, which must be allocated.
    pub fn put(&mut self, paddr: u64, block: Vec<u8>) {
        assert_eq!(block.len(), BS);
        self.blocks[paddr as usize] = block;
    }

    /// The image as bytes.
    pub fn bytes(&self) -> Vec<u8> {
        self.blocks.concat()
    }
}

fn put_u16(b: &mut [u8], at: usize, v: u16) {
    b[at..at + 2].copy_from_slice(&v.to_le_bytes());
}
fn put_u32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}
fn put_u64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

/// Writes an object header (checksum left for [`seal`]).
pub fn header(b: &mut [u8], oid: u64, xid: u64, type_and_flags: u32, subtype: u16) {
    put_u64(b, 8, oid);
    put_u64(b, 16, xid);
    put_u32(b, 24, type_and_flags);
    put_u32(b, 28, u32::from(subtype));
}

/// Key and value pairs.
pub type Records = Vec<(Vec<u8>, Vec<u8>)>;

/// How [`write_tree`] lays out a tree.
#[derive(Clone, Copy)]
pub struct NodeOpts {
    pub layout: Layout,
    /// Most records per node, so small inputs still give deep trees.
    pub per_node: usize,
    pub subtype: u16,
    /// Children are block addresses (else virtual oids).
    pub physical: bool,
    /// Nodes carry object headers.
    pub headers: bool,
    pub xid: u64,
}

impl NodeOpts {
    /// An object map tree.
    pub fn omap(per_node: usize) -> Self {
        Self {
            layout: Layout::Fixed { key: 16, value: 16 },
            per_node,
            subtype: TYPE_OMAP_TREE,
            physical: true,
            headers: true,
            xid: MAP_XID,
        }
    }

    /// A file-system tree (virtual, variable).
    pub fn fs_tree(per_node: usize) -> Self {
        Self {
            layout: Layout::Variable,
            per_node,
            subtype: TYPE_FSTREE,
            physical: false,
            headers: true,
            xid: MAP_XID,
        }
    }
}

/// A written tree: its root and, for a virtual tree, the oid of every node
/// with where it was put.
pub struct WrittenTree {
    pub root_paddr: u64,
    pub root_oid: u64,
    pub mappings: Vec<(u64, u64)>,
}

/// Writes a tree of `records` (sorted by the caller) into `img`. Virtual
/// node oids come from `next_oid`.
pub fn write_tree(
    img: &mut Image,
    records: &[(Vec<u8>, Vec<u8>)],
    opts: NodeOpts,
    next_oid: &mut u64,
) -> WrittenTree {
    assert!(opts.per_node >= 2);
    let mut mappings = Vec::new();
    let longest_key = records.iter().map(|r| r.0.len()).max().unwrap_or(0) as u32;
    let longest_value = records.iter().map(|r| r.1.len()).max().unwrap_or(0) as u32;

    // Level 0: (first key, records) per leaf.
    let mut level: Vec<(Vec<u8>, Records)> = if records.is_empty() {
        vec![(Vec::new(), Vec::new())]
    } else {
        pack(records.to_vec(), opts)
    };
    let mut depth: u16 = 0;
    let mut written = 0u64;
    loop {
        let is_root = level.len() == 1;
        let mut ids = Vec::new();
        for (first, entries) in &level {
            written += 1;
            let paddr = img.alloc();
            let oid = if opts.physical {
                paddr
            } else {
                let oid = *next_oid;
                *next_oid += 1;
                mappings.push((oid, paddr));
                oid
            };
            let info = is_root.then_some(RootInfo {
                physical: opts.physical,
                key_count: records.len() as u64,
                node_count: written,
                longest_key,
                longest_value,
            });
            let block = node_block(oid, depth, entries, opts, info);
            img.put(paddr, block);
            ids.push((first.clone(), oid, paddr));
        }
        if is_root {
            let (_, oid, paddr) = ids.pop().unwrap();
            return WrittenTree {
                root_paddr: paddr,
                root_oid: oid,
                mappings,
            };
        }
        level = pack(
            ids.iter()
                .map(|(k, oid, _)| (k.clone(), oid.to_le_bytes().to_vec()))
                .collect(),
            opts,
        );
        depth += 1;
    }
}

/// Groups entries into nodes of at most `per_node` entries that fit a block
/// (with room for the tree info, so any node could be the root).
fn pack(entries: Records, opts: NodeOpts) -> Vec<(Vec<u8>, Records)> {
    let entry_size = if matches!(opts.layout, Layout::Fixed { .. }) {
        4
    } else {
        8
    };
    let room = BS - 56 - INFO_LEN;
    let mut groups = Vec::new();
    let mut current: Records = Vec::new();
    let mut used = 0;
    for e in entries {
        let need = e.0.len() + e.1.len() + entry_size;
        if !current.is_empty() && (current.len() == opts.per_node || used + need > room) {
            groups.push((current[0].0.clone(), std::mem::take(&mut current)));
            used = 0;
        }
        used += need;
        current.push(e);
    }
    if !current.is_empty() {
        groups.push((current[0].0.clone(), current));
    }
    groups
}

struct RootInfo {
    physical: bool,
    key_count: u64,
    node_count: u64,
    longest_key: u32,
    longest_value: u32,
}

fn node_block(
    oid: u64,
    level: u16,
    entries: &[(Vec<u8>, Vec<u8>)],
    opts: NodeOpts,
    info: Option<RootInfo>,
) -> Vec<u8> {
    let mut b = vec![0u8; BS];
    let root = info.is_some();
    let fixed = matches!(opts.layout, Layout::Fixed { .. });
    let entry_size = if fixed { 4 } else { 8 };
    let mut flags = 0u16;
    if root {
        flags |= 1;
    }
    if level == 0 {
        flags |= 2;
    }
    if fixed {
        flags |= 4;
    }
    put_u16(&mut b, 32, flags);
    put_u16(&mut b, 34, level);
    put_u32(&mut b, 36, entries.len() as u32);
    let toc_len = entries.len() * entry_size;
    put_u16(&mut b, 40, 0);
    put_u16(&mut b, 42, toc_len as u16);
    let key_start = 56 + toc_len;
    let val_end = if root { BS - INFO_LEN } else { BS };
    let mut key_off = 0usize;
    let mut val_off = 0usize;
    for (i, (k, v)) in entries.iter().enumerate() {
        let at = 56 + i * entry_size;
        let ks = key_start + key_off;
        b[ks..ks + k.len()].copy_from_slice(k);
        val_off += v.len();
        let vs = val_end - val_off;
        assert!(ks + k.len() <= vs, "node overflows: lower per_node");
        b[vs..vs + v.len()].copy_from_slice(v);
        if fixed {
            put_u16(&mut b, at, key_off as u16);
            put_u16(&mut b, at + 2, val_off as u16);
        } else {
            put_u16(&mut b, at, key_off as u16);
            put_u16(&mut b, at + 2, k.len() as u16);
            put_u16(&mut b, at + 4, val_off as u16);
            put_u16(&mut b, at + 6, v.len() as u16);
        }
        key_off += k.len();
    }
    if let Some(info) = info {
        let at = BS - INFO_LEN;
        let (ks, vs) = match opts.layout {
            Layout::Fixed { key, value } => (u32::from(key), u32::from(value)),
            Layout::Variable => (0, 0),
        };
        let mut tree_flags = 0u32;
        if info.physical {
            tree_flags |= 0x10;
        }
        if !opts.headers {
            tree_flags |= 0x100;
        }
        put_u32(&mut b, at, tree_flags);
        put_u32(&mut b, at + 4, BS as u32);
        put_u32(&mut b, at + 8, ks);
        put_u32(&mut b, at + 12, vs);
        put_u32(&mut b, at + 16, info.longest_key);
        put_u32(&mut b, at + 20, info.longest_value);
        put_u64(&mut b, at + 24, info.key_count);
        put_u64(&mut b, at + 32, info.node_count);
    }
    if opts.headers {
        let kind = if root { TYPE_BTREE } else { TYPE_BTREE_NODE };
        let storage = if opts.physical { OBJ_PHYSICAL } else { 0 };
        header(
            &mut b,
            oid,
            opts.xid,
            u32::from(kind) | storage,
            opts.subtype,
        );
        seal(&mut b);
    } else {
        put_u16(&mut b, 32, flags | 0x10);
    }
    b
}

/// An object map key.
pub fn omap_key(oid: u64, xid: u64) -> Vec<u8> {
    let mut k = oid.to_le_bytes().to_vec();
    k.extend_from_slice(&xid.to_le_bytes());
    k
}

/// An object map value.
pub fn omap_value(flags: u32, paddr: u64) -> Vec<u8> {
    let mut v = flags.to_le_bytes().to_vec();
    v.extend_from_slice(&(BS as u32).to_le_bytes());
    v.extend_from_slice(&paddr.to_le_bytes());
    v
}

/// Writes an object map holding `entries` (oid, xid, flags, paddr), in any
/// order; returns the header's address.
pub fn write_omap(img: &mut Image, entries: &[(u64, u64, u32, u64)], per_node: usize) -> u64 {
    let mut sorted = entries.to_vec();
    sorted.sort_by_key(|e| (e.0, e.1));
    let records: Vec<_> = sorted
        .iter()
        .map(|&(oid, xid, flags, paddr)| (omap_key(oid, xid), omap_value(flags, paddr)))
        .collect();
    let paddr = img.alloc();
    let tree = write_tree(img, &records, NodeOpts::omap(per_node), &mut 0);
    let mut b = vec![0u8; BS];
    header(
        &mut b,
        paddr,
        MAP_XID,
        OBJ_PHYSICAL | u32::from(TYPE_OMAP),
        0,
    );
    put_u32(&mut b, 40, OBJ_PHYSICAL | u32::from(TYPE_BTREE));
    put_u32(&mut b, 44, OBJ_PHYSICAL | u32::from(TYPE_BTREE));
    put_u64(&mut b, 48, tree.root_paddr);
    seal(&mut b);
    img.put(paddr, b);
    paddr
}

/// What an item of a built volume is.
#[derive(Clone)]
pub enum ItemKind {
    Folder,
    /// A regular file with this content, stored in `fragments` runs.
    File(Vec<u8>),
    Symlink(String),
    /// A decmpfs file: the whole `com.apple.decmpfs` value and, for the
    /// resource fork kinds, the fork.
    Compressed {
        decmpfs: Vec<u8>,
        resource: Option<Vec<u8>>,
    },
}

/// One item of a built volume.
#[derive(Clone)]
pub struct FsItem {
    pub id: u64,
    pub parent: u64,
    pub name: String,
    pub kind: ItemKind,
    pub bsd_flags: u32,
    pub internal_flags: u64,
    pub uncompressed_size: u64,
    /// Runs a file's content is split into (at least 1).
    pub fragments: usize,
    /// (name, value, stored as a stream).
    pub xattrs: Vec<(String, Vec<u8>, bool)>,
    /// Size recorded in the data stream when it differs from the content
    /// (a sparse or truncated file).
    pub size_override: Option<u64>,
}

/// The files and folders of a built volume. The root (2) exists already.
#[derive(Clone)]
pub struct FsSpec {
    pub items: Vec<FsItem>,
    next_id: u64,
}

/// Fixed dates of every built inode: created, modified, changed, accessed.
pub const TIMES: [u64; 4] = [
    1_600_000_000_000_000_000,
    1_650_000_000_000_000_000,
    1_660_000_000_000_000_000,
    1_670_000_000_000_000_000,
];

impl FsSpec {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            next_id: 16,
        }
    }

    fn add(&mut self, parent: u64, name: &str, kind: ItemKind) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.items.push(FsItem {
            id,
            parent,
            name: name.to_owned(),
            kind,
            bsd_flags: 0,
            internal_flags: 0,
            uncompressed_size: 0,
            fragments: 1,
            xattrs: Vec::new(),
            size_override: None,
        });
        id
    }

    pub fn folder(&mut self, parent: u64, name: &str) -> u64 {
        self.add(parent, name, ItemKind::Folder)
    }

    pub fn file(&mut self, parent: u64, name: &str, content: &[u8]) -> u64 {
        self.add(parent, name, ItemKind::File(content.to_vec()))
    }

    pub fn symlink(&mut self, parent: u64, name: &str, target: &str) -> u64 {
        self.add(parent, name, ItemKind::Symlink(target.to_owned()))
    }

    pub fn compressed(
        &mut self,
        parent: u64,
        name: &str,
        decmpfs: Vec<u8>,
        resource: Option<Vec<u8>>,
    ) -> u64 {
        let id = self.add(parent, name, ItemKind::Compressed { decmpfs, resource });
        self.item(id).bsd_flags |= super::records::UF_COMPRESSED;
        id
    }

    pub fn xattr(&mut self, id: u64, name: &str, value: &[u8], stream: bool) {
        self.item(id)
            .xattrs
            .push((name.to_owned(), value.to_vec(), stream));
    }

    pub fn item(&mut self, id: u64) -> &mut FsItem {
        self.items.iter_mut().find(|i| i.id == id).expect("item")
    }

    /// The records of the volume, sorted, with content written into `img`.
    pub fn records(&self, img: &mut Image, hashed: bool) -> Records {
        use super::records::encode::*;
        use super::records::{DT_DIR, DT_LNK, DT_REG};
        let mut records: Records = Vec::new();
        let mut next_stream_id = 0x0100_0000u64;
        let children = |id: u64| self.items.iter().filter(|i| i.parent == id).count() as u32;
        records.push((
            inode_key(2),
            inode_value(&InodeSpec {
                parent: 1,
                private_id: 2,
                mode: 0o040755,
                bsd_flags: 0,
                internal_flags: 0,
                uncompressed_size: 0,
                name: Some("root".into()),
                size: None,
                times: TIMES,
                nchildren: children(2),
            }),
        ));
        for item in &self.items {
            let (mode, dt, size) = match &item.kind {
                ItemKind::Folder => (0o040755, DT_DIR, None),
                ItemKind::File(c) => (
                    0o100644,
                    DT_REG,
                    Some(item.size_override.unwrap_or(c.len() as u64)),
                ),
                ItemKind::Symlink(t) => (0o120755, DT_LNK, Some(t.len() as u64)),
                ItemKind::Compressed { .. } => (0o100644, DT_REG, None),
            };
            records.push((
                inode_key(item.id),
                inode_value(&InodeSpec {
                    parent: item.parent,
                    private_id: item.id,
                    mode,
                    bsd_flags: item.bsd_flags,
                    internal_flags: item.internal_flags,
                    uncompressed_size: item.uncompressed_size,
                    name: Some(item.name.clone()),
                    size,
                    times: TIMES,
                    nchildren: if dt == DT_DIR { children(item.id) } else { 1 },
                }),
            ));
            records.push((
                drec_key(item.parent, &item.name, hashed),
                drec_value(item.id, dt),
            ));
            let mut xattrs = item.xattrs.clone();
            match &item.kind {
                ItemKind::File(content) => {
                    records.extend(write_content(img, item.id, content, item.fragments));
                }
                ItemKind::Symlink(target) => {
                    let mut t = target.as_bytes().to_vec();
                    t.push(0);
                    xattrs.push(("com.apple.fs.symlink".into(), t, false));
                }
                ItemKind::Compressed { decmpfs, resource } => {
                    xattrs.push(("com.apple.decmpfs".into(), decmpfs.clone(), false));
                    if let Some(fork) = resource {
                        xattrs.push(("com.apple.ResourceFork".into(), fork.clone(), true));
                    }
                }
                ItemKind::Folder => {}
            }
            for (name, value, stream) in xattrs {
                if stream {
                    let sid = next_stream_id;
                    next_stream_id += 1;
                    records.extend(write_content(img, sid, &value, 1));
                    records.push((
                        xattr_key(item.id, &name),
                        xattr_stream(sid, value.len() as u64),
                    ));
                } else {
                    records.push((xattr_key(item.id, &name), xattr_embedded(&value)));
                }
            }
        }
        records.sort_by_key(|(k, _)| order(k, hashed));
        records
    }
}

/// Writes `content` into new blocks in `fragments` runs (a spare block
/// between runs keeps them apart); returns the file extent records.
fn write_content(img: &mut Image, id: u64, content: &[u8], fragments: usize) -> Records {
    use super::records::encode::{extent_key, extent_value};
    let blocks = content.len().div_ceil(BS);
    if blocks == 0 {
        return Vec::new();
    }
    let per = blocks.div_ceil(fragments.max(1));
    let mut out = Vec::new();
    for (i, piece) in content.chunks(per * BS).enumerate() {
        if i > 0 {
            img.alloc();
        }
        let n = piece.len().div_ceil(BS);
        let first = img.alloc();
        for _ in 1..n {
            img.alloc();
        }
        for (j, chunk) in piece.chunks(BS).enumerate() {
            let mut b = vec![0u8; BS];
            b[..chunk.len()].copy_from_slice(chunk);
            img.put(first + j as u64, b);
        }
        out.push((
            extent_key(id, (i * per * BS) as u64),
            extent_value((n * BS) as u64, 0, first),
        ));
    }
    out
}

/// The order of file-system keys (§3.7, as verified): object id, type, then
/// the hash bits and name, the name, or the logical address.
fn order(key: &[u8], hashed: bool) -> (u64, u8, Vec<u8>) {
    use super::records::{T_DIR_REC, T_FILE_EXTENT, T_XATTR, key_header};
    let (id, kind) = key_header(key).expect("key");
    let rest = match kind {
        T_DIR_REC if hashed => {
            let v = u32::from_le_bytes(key[8..12].try_into().unwrap());
            let mut r = (v & !0x3FF).to_be_bytes().to_vec();
            r.extend_from_slice(&key[12..]);
            r
        }
        T_DIR_REC | T_XATTR => key[10..].to_vec(),
        T_FILE_EXTENT => u64::from_le_bytes(key[8..16].try_into().unwrap())
            .to_be_bytes()
            .to_vec(),
        _ => key.get(8..).unwrap_or(&[]).to_vec(),
    };
    (id, kind, rest)
}

/// Flips one byte in the middle of block `paddr`.
pub fn corrupt_block(image: &mut [u8], paddr: u64, block_size: usize) {
    let at = paddr as usize * block_size + block_size / 2;
    image[at] ^= 0x5A;
}

/// A built tree or object map, for the B-tree and object map tests.
pub struct Built {
    pub image: Vec<u8>,
    pub blocks: Arc<Blocks>,
    pub root: u64,
}

impl Built {
    /// Block access over `image`.
    pub fn blocks_over(image: Vec<u8>) -> Arc<Blocks> {
        let count = (image.len() / BS) as u64;
        Arc::new(Blocks::new(
            Arc::new(MemSource::new(image)),
            BS as u32,
            count,
        ))
    }

    /// Every byte `blocks` reads.
    pub fn image_of(blocks: &Blocks) -> Vec<u8> {
        let mut out = vec![0u8; blocks.source().len() as usize];
        blocks.source().read_at(0, &mut out).unwrap();
        out
    }
}

/// A physical tree of `records` alone in an image (block 0 left empty).
pub fn build_tree_image(records: &[(Vec<u8>, Vec<u8>)], opts: NodeOpts) -> Built {
    let mut img = Image::default();
    img.alloc();
    let tree = write_tree(&mut img, records, opts, &mut 0);
    let image = img.bytes();
    Built {
        blocks: Built::blocks_over(image.clone()),
        image,
        root: tree.root_paddr,
    }
}

/// An object map alone in an image: its blocks and header address.
pub fn omap_image(entries: &[(u64, u64, u32, u64)], per_node: usize) -> (Arc<Blocks>, u64) {
    let mut img = Image::default();
    img.alloc();
    let paddr = write_omap(&mut img, entries, per_node.max(2));
    (Built::blocks_over(img.bytes()), paddr)
}

/// One volume of a built container.
#[derive(Clone)]
pub struct VolumeSpec {
    pub name: String,
    /// `apfs_role`.
    pub role: u16,
    /// `apfs_incompatible_features`.
    pub incompatible: u64,
    /// `apfs_fs_flags` (0x1 unencrypted).
    pub fs_flags: u64,
    /// The files and folders.
    pub fs: FsSpec,
    /// Raw sorted records used instead of `fs`, for damage tests.
    pub records: Option<Records>,
    /// Sealed: the file-system tree has no headers, extents are in a
    /// file-extent tree.
    pub sealed: bool,
    /// Older states of the volume.
    pub snapshots: Vec<SnapSpec>,
    /// Records per file-system tree node.
    pub per_node: usize,
    pub uuid: [u8; 16],
}

impl VolumeSpec {
    /// An unencrypted, case-sensitive volume with an empty root folder.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            role: 0,
            incompatible: 0,
            fs_flags: 1,
            fs: FsSpec::new(),
            records: None,
            sealed: false,
            snapshots: Vec::new(),
            per_node: 16,
            uuid: [0x11; 16],
        }
    }

    /// Hashed directory keys: case- or normalization-insensitive.
    pub fn hashed(&self) -> bool {
        self.incompatible & 0x9 != 0
    }
}

/// A built container and where its parts are.
pub struct BuiltContainer {
    pub image: Vec<u8>,
    /// The newest checkpoint's transaction.
    pub xid: u64,
    /// Where each volume superblock is.
    pub volume_paddrs: Vec<u64>,
    pub spaceman_oid: u64,
    pub spaceman_paddr: u64,
    /// The ring superblock and map of each requested checkpoint, in order.
    pub checkpoint_superblocks: Vec<u64>,
    pub checkpoint_maps: Vec<u64>,
}

/// Blocks per space manager chunk in built containers (real ones use
/// 32,768); small, so a small image has several chunks.
const BLOCKS_PER_CHUNK: u64 = 8;

/// Builds a container.
pub struct ContainerBuilder {
    checkpoints: Vec<u64>,
    block_zero_xid: Option<u64>,
    volumes: Vec<VolumeSpec>,
    free_blocks: u64,
}

impl ContainerBuilder {
    /// One checkpoint at xid 10 and one empty volume.
    pub fn new() -> Self {
        Self {
            checkpoints: vec![10],
            block_zero_xid: None,
            volumes: vec![VolumeSpec::new("Atlas")],
            free_blocks: 0,
        }
    }

    /// Blank, free blocks at the end of the container.
    pub fn free_blocks(mut self, n: u64) -> Self {
        self.free_blocks = n;
        self
    }

    /// Checkpoints at these transactions, in ring order.
    pub fn checkpoints(mut self, xids: &[u64]) -> Self {
        self.checkpoints = xids.to_vec();
        self
    }

    /// Block 0 as a copy of a checkpoint at this transaction. When no ring
    /// checkpoint has it, its map is in the ring and its superblock is not
    /// (the ring wrapped over it).
    pub fn block_zero_xid(mut self, xid: u64) -> Self {
        self.block_zero_xid = Some(xid);
        self
    }

    /// Replaces the volumes.
    pub fn volumes(mut self, volumes: Vec<VolumeSpec>) -> Self {
        self.volumes = volumes;
        self
    }

    pub fn build(self) -> BuiltContainer {
        let mut img = Image::default();
        img.alloc(); // block 0
        let zero_xid = self
            .block_zero_xid
            .unwrap_or(*self.checkpoints.last().expect("a checkpoint"));
        let stale = !self.checkpoints.contains(&zero_xid);
        let slots = self.checkpoints.len() + usize::from(stale);
        let desc_blocks = (2 * slots).max(8) as u32;
        let desc_base = img.alloc();
        for _ in 1..desc_blocks {
            img.alloc();
        }
        let data_blocks = 4u32;
        let data_base = img.alloc();
        for _ in 1..data_blocks {
            img.alloc();
        }
        let spaceman_paddr = data_base;

        // Volumes, then the container object map that finds them.
        let mut next_oid = FIRST_VIRTUAL_OID;
        let mut fs_oids = Vec::new();
        let mut volume_paddrs = Vec::new();
        let mut nx_map = Vec::new();
        for (index, spec) in self.volumes.iter().enumerate() {
            let oid = next_oid;
            next_oid += 1;
            let paddr = write_volume(&mut img, spec, index as u32, oid, &mut next_oid);
            fs_oids.push(oid);
            volume_paddrs.push(paddr);
            nx_map.push((oid, MAP_XID, 0, paddr));
        }
        let omap = write_omap(&mut img, &nx_map, 16);

        // The space manager: one chunk-info block over chunks of
        // BLOCKS_PER_CHUNK; every block written so far is used, the blank
        // tail is free.
        let cib = img.alloc();
        let used = img.blocks.len() as u64;
        for _ in 0..self.free_blocks {
            img.alloc();
        }
        let block_count = img.blocks.len() as u64;
        let chunks = block_count.div_ceil(BLOCKS_PER_CHUNK);
        let mut b = vec![0u8; BS];
        header(
            &mut b,
            cib,
            MAP_XID,
            OBJ_PHYSICAL | u32::from(TYPE_SPACEMAN_CIB),
            0,
        );
        put_u32(&mut b, 36, chunks as u32);
        for c in 0..chunks {
            let first = c * BLOCKS_PER_CHUNK;
            let n = BLOCKS_PER_CHUNK.min(block_count - first);
            let free = (first + n).saturating_sub(used.max(first));
            let at = 40 + 32 * c as usize;
            put_u64(&mut b, at, MAP_XID);
            put_u64(&mut b, at + 8, first);
            put_u32(&mut b, at + 16, n as u32);
            put_u32(&mut b, at + 20, free as u32);
        }
        seal(&mut b);
        img.put(cib, b);
        let mut sm = vec![0u8; BS];
        header(
            &mut sm,
            SPACEMAN_OID,
            MAP_XID,
            OBJ_EPHEMERAL | u32::from(TYPE_SPACEMAN),
            0,
        );
        put_u32(&mut sm, 32, BS as u32);
        put_u32(&mut sm, 36, BLOCKS_PER_CHUNK as u32);
        put_u32(&mut sm, 40, 126);
        put_u64(&mut sm, 48, block_count);
        put_u64(&mut sm, 56, chunks);
        put_u32(&mut sm, 64, 1);
        put_u64(&mut sm, 72, self.free_blocks);
        put_u32(&mut sm, 80, 2048);
        put_u64(&mut sm, 2048, cib);
        seal(&mut sm);
        img.put(spaceman_paddr, sm);

        // The ring: checkpoint k at slots 2k (map) and 2k+1 (superblock);
        // a stale block 0's map last, with no superblock.
        let nx = |xid: u64, index: u32| {
            let mut b = vec![0u8; BS];
            header(
                &mut b,
                NX_OID,
                xid,
                OBJ_EPHEMERAL | u32::from(TYPE_NX_SUPERBLOCK),
                0,
            );
            b[32..36].copy_from_slice(b"NXSB");
            put_u32(&mut b, 36, BS as u32);
            put_u64(&mut b, 40, block_count);
            put_u64(&mut b, 64, 2);
            b[72..88].copy_from_slice(&NX_UUID);
            put_u64(&mut b, 88, next_oid);
            put_u64(&mut b, 96, xid + 1);
            put_u32(&mut b, 104, desc_blocks);
            put_u32(&mut b, 108, data_blocks);
            put_u64(&mut b, 112, desc_base);
            put_u64(&mut b, 120, data_base);
            put_u32(&mut b, 136, index);
            put_u32(&mut b, 140, 2);
            put_u32(&mut b, 148, 1);
            put_u64(&mut b, 152, SPACEMAN_OID);
            put_u64(&mut b, 160, omap);
            put_u32(&mut b, 180, 100);
            for (i, oid) in fs_oids.iter().enumerate() {
                put_u64(&mut b, 184 + 8 * i, *oid);
            }
            seal(&mut b);
            b
        };
        let map = |xid: u64, paddr: u64| {
            let mut b = vec![0u8; BS];
            header(
                &mut b,
                paddr,
                xid,
                OBJ_PHYSICAL | u32::from(TYPE_CHECKPOINT_MAP),
                0,
            );
            put_u32(&mut b, 32, 1); // last
            put_u32(&mut b, 36, 1);
            put_u32(&mut b, 40, OBJ_EPHEMERAL | u32::from(TYPE_SPACEMAN));
            put_u32(&mut b, 48, BS as u32);
            put_u64(&mut b, 64, SPACEMAN_OID);
            put_u64(&mut b, 72, spaceman_paddr);
            seal(&mut b);
            b
        };
        let mut checkpoint_superblocks = Vec::new();
        let mut checkpoint_maps = Vec::new();
        let mut zero_index = 0;
        for (k, &xid) in self.checkpoints.iter().enumerate() {
            let index = 2 * k as u32;
            let map_paddr = desc_base + u64::from(index);
            let sb_paddr = map_paddr + 1;
            img.put(map_paddr, map(xid, map_paddr));
            img.put(sb_paddr, nx(xid, index));
            checkpoint_maps.push(map_paddr);
            checkpoint_superblocks.push(sb_paddr);
            if xid == zero_xid {
                zero_index = index;
            }
        }
        if stale {
            let index = 2 * self.checkpoints.len() as u32;
            let map_paddr = desc_base + u64::from(index);
            img.put(map_paddr, map(zero_xid, map_paddr));
            zero_index = index;
        }
        img.put(0, nx(zero_xid, zero_index));

        BuiltContainer {
            image: img.bytes(),
            xid: self.checkpoints.iter().copied().max().unwrap_or(zero_xid),
            volume_paddrs,
            spaceman_oid: SPACEMAN_OID,
            spaceman_paddr,
            checkpoint_superblocks,
            checkpoint_maps,
        }
    }
}

/// Writes one volume: its file-system tree, its object map and its
/// superblock. Returns the superblock's address.
fn write_volume(
    img: &mut Image,
    spec: &VolumeSpec,
    index: u32,
    oid: u64,
    next_oid: &mut u64,
) -> u64 {
    let hashed = spec.hashed();
    let sealed = spec.sealed;
    // The live tree is mapped after every snapshot.
    let live_xid = spec
        .snapshots
        .iter()
        .map(|s| s.xid)
        .max()
        .map_or(MAP_XID, |x| x + 1);
    let fs_opts = |xid: u64| NodeOpts {
        headers: !sealed,
        xid,
        ..NodeOpts::fs_tree(spec.per_node)
    };
    let map_flags = if sealed { OMAP_NOHEADER } else { 0 };
    let mut entries = Vec::new();
    let first_oid = *next_oid;
    let mut last_oid = first_oid;

    // Snapshots: older trees from the same first oid, mapped at their own
    // transactions, so the oids the live tree reuses resolve by xid.
    let mut snaps = Vec::new();
    for snap in &spec.snapshots {
        let mut counter = first_oid;
        let (records, _) = split_extents(snap.fs.records(img, hashed), false);
        let tree = write_tree(img, &records, fs_opts(snap.xid), &mut counter);
        last_oid = last_oid.max(counter);
        entries.extend(
            tree.mappings
                .iter()
                .map(|&(o, p)| (o, snap.xid, map_flags, p)),
        );
        snaps.push((snap, tree.root_oid, img.alloc()));
    }
    let mut counter = first_oid;
    let records = match &spec.records {
        Some(r) => r.clone(),
        None => spec.fs.records(img, hashed),
    };
    let (records, fext) = split_extents(records, sealed);
    let tree = write_tree(img, &records, fs_opts(live_xid), &mut counter);
    last_oid = last_oid.max(counter);
    *next_oid = last_oid;
    entries.extend(
        tree.mappings
            .iter()
            .map(|&(o, p)| (o, live_xid, map_flags, p)),
    );
    let omap = write_omap(img, &entries, 16);

    // The snapshot metadata tree: one record per snapshot and one name
    // record per snapshot (type 11, skipped by the reader).
    let mut meta: Records = Vec::new();
    for (snap, _, sb_paddr) in &snaps {
        let mut v = vec![0u8; 50];
        v[8..16].copy_from_slice(&sb_paddr.to_le_bytes());
        v[16..24].copy_from_slice(&(TIMES[0] + snap.xid).to_le_bytes());
        let mut name = snap.name.as_bytes().to_vec();
        name.push(0);
        v[48..50].copy_from_slice(&(name.len() as u16).to_le_bytes());
        v.extend_from_slice(&name);
        meta.push((super::records::encode::key(snap.xid, 1), v));
        let mut k = super::records::encode::key(super::records::OBJ_ID_MASK, 11);
        k.extend_from_slice(&(name.len() as u16).to_le_bytes());
        k.extend_from_slice(&name);
        meta.push((k, snap.xid.to_le_bytes().to_vec()));
    }
    meta.sort_by_key(|(k, _)| order(k, false));
    let meta_tree = write_tree(
        img,
        &meta,
        NodeOpts {
            layout: Layout::Variable,
            per_node: 8,
            subtype: TYPE_SNAPMETATREE,
            physical: true,
            headers: true,
            xid: live_xid,
        },
        &mut 0,
    );
    let fext_root = (sealed && !fext.is_empty()).then(|| {
        write_tree(
            img,
            &fext,
            NodeOpts {
                layout: Layout::Fixed { key: 16, value: 16 },
                per_node: 6,
                subtype: TYPE_FEXT_TREE,
                physical: true,
                headers: false,
                xid: live_xid,
            },
            &mut 0,
        )
        .root_paddr
    });

    let superblock = |xid: u64, root_oid: u64| {
        let mut b = vec![0u8; BS];
        header(&mut b, oid, xid, u32::from(TYPE_FS), 0);
        b[32..36].copy_from_slice(b"APSB");
        put_u32(&mut b, 36, index);
        let mut incompatible = spec.incompatible;
        if sealed {
            incompatible |= 0x20;
        }
        put_u64(&mut b, 56, incompatible);
        let noheader = if sealed { OBJ_NOHEADER } else { 0 };
        put_u32(&mut b, 116, u32::from(TYPE_BTREE) | noheader);
        put_u32(&mut b, 120, OBJ_PHYSICAL | u32::from(TYPE_BTREE));
        put_u32(&mut b, 124, OBJ_PHYSICAL | u32::from(TYPE_BTREE));
        put_u64(&mut b, 128, omap);
        put_u64(&mut b, 136, root_oid);
        put_u64(&mut b, 152, meta_tree.root_paddr);
        b[240..256].copy_from_slice(&spec.uuid);
        put_u64(&mut b, 256, 1_700_000_000_000_000_000 + xid);
        put_u64(&mut b, 264, spec.fs_flags);
        let name = spec.name.as_bytes();
        b[704..704 + name.len()].copy_from_slice(name);
        put_u16(&mut b, 964, spec.role);
        if let Some(root) = fext_root {
            put_u64(&mut b, 1032, root);
            put_u32(
                &mut b,
                1040,
                OBJ_PHYSICAL | OBJ_NOHEADER | u32::from(TYPE_BTREE),
            );
        }
        seal(&mut b);
        b
    };
    for (snap, root_oid, sb_paddr) in &snaps {
        img.put(*sb_paddr, superblock(snap.xid, *root_oid));
    }
    let paddr = img.alloc();
    img.put(paddr, superblock(live_xid, tree.root_oid));
    paddr
}

/// Object map value flag: the node has no header.
const OMAP_NOHEADER: u32 = 0x8;

/// For a sealed volume, moves the file extent records out of `records` into
/// sealed file-extent records (key: id, logical; value: length and flags,
/// block), sorted. Otherwise leaves them.
fn split_extents(records: Records, sealed: bool) -> (Records, Records) {
    use super::records::{T_FILE_EXTENT, key_header};
    if !sealed {
        return (records, Vec::new());
    }
    let (extents, rest): (Records, Records) = records
        .into_iter()
        .partition(|(k, _)| key_header(k).map(|h| h.1) == Some(T_FILE_EXTENT));
    let mut fext: Records = extents
        .into_iter()
        .map(|(k, v)| {
            let (id, _) = key_header(&k).unwrap();
            let mut key = id.to_le_bytes().to_vec();
            key.extend_from_slice(&k[8..16]);
            (key, v[..16].to_vec())
        })
        .collect();
    fext.sort_by_key(|(k, _)| {
        (
            u64::from_le_bytes(k[..8].try_into().unwrap()),
            u64::from_le_bytes(k[8..16].try_into().unwrap()),
        )
    });
    (rest, fext)
}

/// A snapshot of a built volume: its contents at transaction `xid`.
#[derive(Clone)]
pub struct SnapSpec {
    pub name: String,
    pub xid: u64,
    pub fs: FsSpec,
}

/// A volume with one of each kind of item: folders, a fragmented file with
/// attributes and a resource fork, a symlink, files compressed in the
/// attribute and in the resource fork. For seeds and end-to-end tests.
pub fn standard_volume(name: &str, incompatible: u64) -> VolumeSpec {
    use atlas_hfsplus::compression::fixtures as cmp;
    let mut fs = FsSpec::new();
    let docs = fs.folder(2, "Documents");
    let content: Vec<u8> = (0..20_000u32).map(|i| (i * 7 % 253) as u8).collect();
    let big = fs.file(docs, "big.bin", &content);
    fs.item(big).fragments = 2;
    fs.xattr(big, "user.small", b"hello", false);
    fs.xattr(big, "user.large", &[0xAB; 5000], true);
    fs.xattr(big, "com.apple.ResourceFork", &[0xCD; 3000], true);
    fs.symlink(2, "link", "Documents/big.bin");
    let plain = cmp::content(6000);
    let mut inline = cmp::header(3, plain.len() as u64);
    inline.extend_from_slice(&cmp::zlib(&plain));
    fs.compressed(docs, "inline.txt", inline, None);
    let chunks: Vec<Vec<u8>> = cmp::pieces(&content)
        .into_iter()
        .map(cmp::lzvn_literals)
        .collect();
    fs.compressed(
        docs,
        "forked.bin",
        cmp::header(8, content.len() as u64),
        Some(cmp::offsets_fork(&chunks)),
    );
    let sub = fs.folder(docs, "Nested");
    for i in 0..12 {
        fs.file(
            sub,
            &format!("note {i}.txt"),
            format!("note {i}").as_bytes(),
        );
    }
    fs.file(2, ".hidden", b"x");
    let mut spec = VolumeSpec::new(name);
    spec.incompatible = incompatible;
    spec.fs = fs;
    spec.per_node = 6;
    spec
}

/// The standard volume sealed, with one snapshot of an older state.
pub fn sealed_with_snapshot() -> VolumeSpec {
    let mut spec = standard_volume("System", 0x8);
    spec.sealed = true;
    spec.role = 0x1;
    let mut then = FsSpec::new();
    let f = then.file(2, "old.txt", &[7u8; 9000]);
    then.item(f).fragments = 2;
    then.folder(2, "Old folder");
    spec.snapshots = vec![SnapSpec {
        name: "before update".into(),
        xid: 3,
        fs: then,
    }];
    spec
}
