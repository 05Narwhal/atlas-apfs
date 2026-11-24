// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! APFS B-trees (design §3.5): the validated node view, the node cache, and
//! `Tree` with a `Cursor` that seeks and iterates without sibling links.
//!
//! Every node is checked whole by [`Node::parse`] before any record is handed
//! out, so the accessors below index only ranges already proved to lie inside
//! the block.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use super::bytes::{le_u16, le_u32, le_u64, len64};
use super::container::Blocks;
use super::object::{self, HEADER_LEN, ObjHeader, TYPE_BTREE, TYPE_BTREE_NODE};
use super::omap::Omap;
use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};

/// Deepest tree accepted (`btn_level` of a root).
pub(crate) const MAX_LEVEL: u16 = 16;
/// Nodes the cache holds at 4 KiB blocks; fewer at larger blocks.
pub(crate) const NODE_CACHE: usize = 4096;
/// Bytes the node cache may hold, whatever the block size.
const NODE_CACHE_BYTES: usize = 64 << 20;
/// Shards of the node cache.
const SHARDS: usize = 16;

/// Node flag: the root.
const BTNODE_ROOT: u16 = 0x1;
/// Node flag: a leaf.
const BTNODE_LEAF: u16 = 0x2;
/// Node flag: fixed-size keys and values.
const BTNODE_FIXED: u16 = 0x4;
/// Tree info flag: child ids are block addresses.
const BTREE_PHYSICAL: u32 = 0x10;
/// Tree info flag: children are ephemeral objects.
const BTREE_EPHEMERAL: u32 = 0x8;

/// Start of the table of contents, after the node header.
const NODE_HEADER_END: usize = 56;
/// Size of the tree info at the end of a root node.
pub(crate) const INFO_LEN: usize = 40;
/// A fixed tree's value offset meaning "no value".
const GHOST: u16 = 0xFFFF;
/// Size of a child id at the start of an index value.
const CHILD_ID_LEN: u32 = 8;

/// How a tree's keys and values are sized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Layout {
    /// Keys and values of their own lengths, in 8-byte table entries.
    Variable,
    /// Every key `key` bytes and every leaf value `value` bytes, in 4-byte
    /// table entries. Index values are 8-byte child ids.
    Fixed {
        /// Key size.
        key: u16,
        /// Leaf value size.
        value: u16,
    },
}

/// The tree info stored at the end of a root node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TreeInfo {
    /// `bt_flags`.
    pub flags: u32,
    /// `bt_node_size`.
    pub node_size: u32,
    /// `bt_key_size` (fixed trees).
    pub key_size: u32,
    /// `bt_val_size` (fixed trees).
    pub value_size: u32,
    /// `bt_key_count`.
    pub key_count: u64,
    /// `bt_node_count`.
    pub node_count: u64,
}

impl TreeInfo {
    fn parse(b: &[u8]) -> Option<Self> {
        Some(Self {
            flags: le_u32(b, 0)?,
            node_size: le_u32(b, 4)?,
            key_size: le_u32(b, 8)?,
            value_size: le_u32(b, 12)?,
            key_count: le_u64(b, 24)?,
            node_count: le_u64(b, 32)?,
        })
    }
}

/// What a caller expects of a node it is about to read.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NodeSpec {
    /// The object id the header must carry (the virtual oid, or the paddr of
    /// a physical node). `None` for a no-header node.
    pub oid: Option<u64>,
    /// False for a node stored without an object header (§3.5).
    pub headers: bool,
    /// The tree's layout; the node's fixed flag must agree.
    pub layout: Layout,
    /// The `o_subtype` the tree's nodes carry.
    pub subtype: u16,
    /// What the tree is, for errors.
    pub structure: &'static str,
}

/// One validated table entry: absolute byte ranges inside the node.
#[derive(Debug, Clone, Copy)]
struct Entry {
    key: u32,
    key_len: u32,
    value: u32,
    value_len: u32,
    ghost: bool,
}

/// A verified, parsed B-tree node.
#[derive(Debug)]
pub(crate) struct Node {
    /// Where it was read from.
    pub paddr: u64,
    /// The header's object id (0 for a no-header node).
    pub oid: u64,
    /// `btn_flags`.
    pub flags: u16,
    /// `btn_level`: 0 for a leaf.
    pub level: u16,
    /// Present on a root node.
    pub info: Option<TreeInfo>,
    layout: Layout,
    headers: bool,
    data: Box<[u8]>,
    entries: Vec<Entry>,
}

fn node_corrupt(structure: &'static str, paddr: u64, kind: CorruptKind) -> Error {
    Error::Corrupt(Corruption::new(structure, Location::Block(paddr), kind))
}

fn out_of_range(
    field: &'static str,
    value: u64,
    limit_name: &'static str,
    limit: u64,
) -> CorruptKind {
    CorruptKind::OutOfRange {
        field,
        value,
        limit_name,
        limit,
    }
}

impl Node {
    /// Parses and validates the block read from `paddr`. The object header
    /// (or its absence) must already have been checked against `spec`.
    pub fn parse(data: Box<[u8]>, paddr: u64, spec: &NodeSpec) -> Result<Self> {
        let corrupt = |kind| node_corrupt(spec.structure, paddr, kind);
        let short = || {
            corrupt(CorruptKind::Truncated {
                needed: len64(NODE_HEADER_END),
                available: len64(data.len()),
            })
        };
        let len = data.len();
        let flags = le_u16(&data, 32).ok_or_else(short)?;
        let level = le_u16(&data, 34).ok_or_else(short)?;
        let nkeys = le_u32(&data, 36).ok_or_else(short)?;
        let table_off = usize::from(le_u16(&data, 40).ok_or_else(short)?);
        let table_len = usize::from(le_u16(&data, 42).ok_or_else(short)?);

        if level > MAX_LEVEL {
            return Err(corrupt(out_of_range(
                "node level",
                u64::from(level),
                "MAX_LEVEL",
                u64::from(MAX_LEVEL),
            )));
        }
        let leaf = flags & BTNODE_LEAF != 0;
        if leaf != (level == 0) {
            return Err(corrupt(CorruptKind::InvalidValue {
                field: "node flags against level",
                value: u64::from(flags),
            }));
        }
        let fixed = flags & BTNODE_FIXED != 0;
        if fixed != matches!(spec.layout, Layout::Fixed { .. }) {
            return Err(corrupt(CorruptKind::InvalidValue {
                field: "node fixed-size flag",
                value: u64::from(flags),
            }));
        }
        let root = flags & BTNODE_ROOT != 0;
        let val_end = if root {
            len.checked_sub(INFO_LEN).ok_or_else(short)?
        } else {
            len
        };
        let info = if root {
            Some(
                data.get(val_end..)
                    .and_then(TreeInfo::parse)
                    .ok_or_else(short)?,
            )
        } else {
            None
        };

        let entry_size: usize = if fixed { 4 } else { 8 };
        let toc_start = NODE_HEADER_END.checked_add(table_off).ok_or_else(short)?;
        let key_start = toc_start.checked_add(table_len).ok_or_else(short)?;
        if key_start > val_end {
            return Err(corrupt(out_of_range(
                "table of contents end",
                len64(key_start),
                "value area end",
                len64(val_end),
            )));
        }
        let max_keys = len
            .saturating_sub(NODE_HEADER_END)
            .checked_div(entry_size)
            .unwrap_or(0);
        let count = usize::try_from(nkeys).unwrap_or(usize::MAX);
        if count > max_keys || count.saturating_mul(entry_size) > table_len {
            return Err(corrupt(out_of_range(
                "key count",
                u64::from(nkeys),
                "table capacity",
                len64(table_len.checked_div(entry_size).unwrap_or(0)),
            )));
        }
        if !leaf && count == 0 {
            return Err(corrupt(CorruptKind::InvalidValue {
                field: "key count of an index node",
                value: 0,
            }));
        }

        let mut entries = Vec::with_capacity(count);
        for i in 0..count {
            // i < count <= table_len / entry_size, so this stays in the table.
            let at = toc_start.saturating_add(i.saturating_mul(entry_size));
            let (key_off, key_len, value_off, value_len) = match spec.layout {
                Layout::Fixed { key, value } => {
                    let k = le_u16(&data, at).ok_or_else(short)?;
                    let v = le_u16(&data, at.saturating_add(2)).ok_or_else(short)?;
                    let vlen = if leaf { u32::from(value) } else { CHILD_ID_LEN };
                    (k, u32::from(key), v, vlen)
                }
                Layout::Variable => (
                    le_u16(&data, at).ok_or_else(short)?,
                    u32::from(le_u16(&data, at.saturating_add(2)).ok_or_else(short)?),
                    le_u16(&data, at.saturating_add(4)).ok_or_else(short)?,
                    u32::from(le_u16(&data, at.saturating_add(6)).ok_or_else(short)?),
                ),
            };
            let key = key_start.saturating_add(usize::from(key_off));
            let key_end = key.saturating_add(usize::try_from(key_len).unwrap_or(usize::MAX));
            if key_end > val_end {
                return Err(corrupt(out_of_range(
                    "key end",
                    len64(key_end),
                    "value area end",
                    len64(val_end),
                )));
            }
            let ghost = fixed && value_off == GHOST;
            if ghost && !leaf {
                return Err(corrupt(CorruptKind::InvalidValue {
                    field: "index entry without a value",
                    value: u64::from(value_off),
                }));
            }
            let (value, value_len) = if ghost {
                (0, 0)
            } else {
                let start = val_end
                    .checked_sub(usize::from(value_off))
                    .filter(|&s| s >= key_start)
                    .ok_or_else(|| {
                        corrupt(out_of_range(
                            "value offset",
                            u64::from(value_off),
                            "value area",
                            len64(val_end.saturating_sub(key_start)),
                        ))
                    })?;
                let end = start.saturating_add(usize::try_from(value_len).unwrap_or(usize::MAX));
                if end > val_end {
                    return Err(corrupt(out_of_range(
                        "value end",
                        len64(end),
                        "value area end",
                        len64(val_end),
                    )));
                }
                if !leaf && value_len < CHILD_ID_LEN {
                    return Err(corrupt(CorruptKind::InvalidValue {
                        field: "index value length",
                        value: u64::from(value_len),
                    }));
                }
                (start, value_len)
            };
            // Every offset is below the block length, at most 64 KiB.
            entries.push(Entry {
                key: u32::try_from(key).unwrap_or(u32::MAX),
                key_len,
                value: u32::try_from(value).unwrap_or(u32::MAX),
                value_len,
                ghost,
            });
        }

        let oid = ObjHeader::read(&data).map_or(0, |h| h.oid);
        Ok(Self {
            paddr,
            oid,
            flags,
            level,
            info,
            layout: spec.layout,
            headers: spec.headers,
            data,
            entries,
        })
    }

    /// Number of table entries, ghosts included.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True for a leaf.
    pub fn is_leaf(&self) -> bool {
        self.level == 0
    }

    fn range(&self, start: u32, len: u32) -> &[u8] {
        let start = usize::try_from(start).unwrap_or(usize::MAX);
        let len = usize::try_from(len).unwrap_or(0);
        self.data
            .get(start..start.saturating_add(len))
            .unwrap_or(&[])
    }

    /// The key of entry `i`. Empty past the end (never happens for an index
    /// below [`Self::len`]).
    pub fn key(&self, i: usize) -> &[u8] {
        self.entries
            .get(i)
            .map_or(&[], |e| self.range(e.key, e.key_len))
    }

    /// The value of entry `i`; `None` for a ghost or past the end.
    pub fn value(&self, i: usize) -> Option<&[u8]> {
        let e = self.entries.get(i)?;
        (!e.ghost).then(|| self.range(e.value, e.value_len))
    }

    /// The child id stored in index entry `i`.
    fn child(&self, i: usize) -> Option<u64> {
        le_u64(self.value(i)?, 0)
    }

    /// Index of the first entry whose key is not less than the target.
    /// `cmp(key)` orders a key against the target.
    fn lower_bound(&self, cmp: &mut dyn FnMut(&[u8]) -> Result<Ordering>) -> Result<usize> {
        self.partition(|key, c| Ok(c(key)? == Ordering::Less), cmp)
    }

    /// Index of the first entry whose key is greater than the target.
    fn upper_bound(&self, cmp: &mut dyn FnMut(&[u8]) -> Result<Ordering>) -> Result<usize> {
        self.partition(|key, c| Ok(c(key)? != Ordering::Greater), cmp)
    }

    /// Binary search for the first entry where `before` is false. Terminates
    /// on any input; on unsorted keys the answer is merely wrong.
    fn partition(
        &self,
        before: impl Fn(&[u8], &mut dyn FnMut(&[u8]) -> Result<Ordering>) -> Result<bool>,
        cmp: &mut dyn FnMut(&[u8]) -> Result<Ordering>,
    ) -> Result<usize> {
        let mut lo = 0usize;
        let mut hi = self.len();
        while lo < hi {
            // lo < hi <= len, so mid < hi and the sum cannot overflow usize
            // for a node of at most 64 KiB.
            let mid = lo.saturating_add((hi.saturating_sub(lo)) / 2);
            if before(self.key(mid), cmp)? {
                lo = mid.saturating_add(1);
            } else {
                hi = mid;
            }
        }
        Ok(lo)
    }
}

/// Nodes already verified and parsed, keyed by (address space, paddr): the
/// space is 0 for blocks as stored and a volume's own number for blocks it
/// decrypted, so a decrypted node is never served to another volume.
pub(crate) struct NodeCache {
    shards: Box<[Mutex<Shard>]>,
    per_shard: usize,
}

struct Slot {
    key: (u16, u64),
    node: Arc<Node>,
    referenced: bool,
}

#[derive(Default)]
struct Shard {
    map: HashMap<(u16, u64), usize>,
    slots: Vec<Slot>,
    hand: usize,
}

impl NodeCache {
    /// A cache sized for blocks of `block_size` bytes: [`NODE_CACHE`] nodes,
    /// fewer when that would exceed 64 MiB.
    pub fn new(block_size: u32) -> Self {
        let size = usize::try_from(block_size).unwrap_or(usize::MAX).max(1);
        let total = NODE_CACHE
            .min(NODE_CACHE_BYTES.checked_div(size).unwrap_or(0))
            .max(SHARDS);
        let shards = (0..SHARDS).map(|_| Mutex::new(Shard::default())).collect();
        Self {
            shards,
            per_shard: total / SHARDS,
        }
    }

    fn shard(&self, key: (u16, u64)) -> Option<&Mutex<Shard>> {
        // Fibonacci hashing spreads neighbouring addresses over the shards;
        // the top four bits index the 16 shards.
        let mixed = key
            .1
            .wrapping_add(u64::from(key.0))
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.shards.get(usize::try_from(mixed >> 60).unwrap_or(0))
    }

    pub fn get(&self, key: (u16, u64)) -> Option<Arc<Node>> {
        let mut shard = self
            .shard(key)?
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let at = *shard.map.get(&key)?;
        let slot = shard.slots.get_mut(at)?;
        slot.referenced = true;
        Some(Arc::clone(&slot.node))
    }

    pub fn insert(&self, key: (u16, u64), node: Arc<Node>) {
        let Some(shard) = self.shard(key) else {
            return;
        };
        let mut guard = shard.lock().unwrap_or_else(PoisonError::into_inner);
        let shard = &mut *guard;
        if let Some(&at) = shard.map.get(&key)
            && let Some(slot) = shard.slots.get_mut(at)
        {
            slot.node = node;
            slot.referenced = true;
            return;
        }
        if shard.slots.len() < self.per_shard {
            shard.map.insert(key, shard.slots.len());
            shard.slots.push(Slot {
                key,
                node,
                referenced: false,
            });
            return;
        }
        // CLOCK: at most two sweeps, since the first clears every bit.
        let n = shard.slots.len();
        for _ in 0..n.saturating_mul(2) {
            let at = shard.hand.checked_rem(n).unwrap_or(0);
            shard.hand = at.saturating_add(1);
            let Some(slot) = shard.slots.get_mut(at) else {
                return;
            };
            if slot.referenced {
                slot.referenced = false;
                continue;
            }
            let old = slot.key;
            *slot = Slot {
                key,
                node,
                referenced: false,
            };
            shard.map.remove(&old);
            shard.map.insert(key, at);
            return;
        }
    }
}

/// Reads and verifies one node, through the cache.
pub(crate) fn read_node(blocks: &Blocks, paddr: u64, spec: &NodeSpec) -> Result<Arc<Node>> {
    let key = (0u16, paddr);
    if let Some(node) = blocks.nodes.get(key) {
        check_cached(&node, spec)?;
        return Ok(node);
    }
    let data = blocks.block(paddr)?;
    if spec.headers {
        let header = object::verify(
            &data,
            paddr,
            spec.structure,
            &[TYPE_BTREE, TYPE_BTREE_NODE],
            spec.oid,
        )?;
        if header.subtype != u32::from(spec.subtype) {
            return Err(node_corrupt(
                spec.structure,
                paddr,
                CorruptKind::InvalidValue {
                    field: "node subtype",
                    value: u64::from(header.subtype),
                },
            ));
        }
    } else if data
        .get(..HEADER_LEN)
        .is_none_or(|h| h.iter().any(|&b| b != 0))
    {
        // A node of a no-header tree that has a header is damage.
        return Err(node_corrupt(
            spec.structure,
            paddr,
            CorruptKind::InvalidValue {
                field: "header of a no-header node",
                value: 0,
            },
        ));
    }
    let node = Arc::new(Node::parse(data.into_boxed_slice(), paddr, spec)?);
    blocks.nodes.insert(key, Arc::clone(&node));
    Ok(node)
}

/// A cached node was parsed for some tree; it must suit this one too, so a
/// block shared by two trees on a damaged disk cannot cross between them.
fn check_cached(node: &Node, spec: &NodeSpec) -> Result<()> {
    let oid_ok = match spec.oid {
        Some(oid) => spec.headers && node.oid == oid,
        None => !spec.headers,
    };
    if node.layout != spec.layout || node.headers != spec.headers || !oid_ok {
        return Err(node_corrupt(
            spec.structure,
            node.paddr,
            CorruptKind::Other("block is shared with another tree"),
        ));
    }
    Ok(())
}

/// How a tree's index entries name their children.
#[derive(Clone)]
pub(crate) enum Children {
    /// Child ids are block addresses.
    Physical,
    /// Child ids are virtual; resolve them in `omap` as of `xid`.
    Virtual {
        /// The object map that resolves them.
        omap: Arc<Omap>,
        /// The transaction to resolve at.
        xid: u64,
    },
}

/// One B-tree of a container, ready to search.
#[derive(Clone)]
pub(crate) struct Tree {
    root: u64,
    root_oid: Option<u64>,
    children: Children,
    headers: bool,
    layout: Layout,
    subtype: u16,
    structure: &'static str,
    /// Most node loads one iteration may make.
    node_limit: u64,
}

/// What [`Tree::open`] needs to know of a tree.
pub(crate) struct TreeSpec {
    /// Block address of the root.
    pub root: u64,
    /// The id the root's header carries (its virtual oid, or `root` for a
    /// physical tree). Ignored without headers.
    pub root_oid: u64,
    /// False for a tree stored without object headers.
    pub headers: bool,
    /// The tree's layout. A fixed tree's info must give exactly these sizes.
    pub layout: Layout,
    /// The `o_subtype` of its nodes.
    pub subtype: u16,
    /// What it is, for errors.
    pub structure: &'static str,
    /// The object map for virtual children, with the transaction to read
    /// at. `None` when the tree must be physical.
    pub omap: Option<(Arc<Omap>, u64)>,
}

impl Tree {
    /// Reads the root and checks it describes the tree `spec` expects.
    pub fn open(blocks: &Blocks, spec: TreeSpec) -> Result<Self> {
        let node_spec = NodeSpec {
            oid: spec.headers.then_some(spec.root_oid),
            headers: spec.headers,
            layout: spec.layout,
            subtype: spec.subtype,
            structure: spec.structure,
        };
        let root = read_node(blocks, spec.root, &node_spec)?;
        let corrupt = |kind| node_corrupt(spec.structure, spec.root, kind);
        let info = match (root.flags & BTNODE_ROOT != 0, root.info) {
            (true, Some(info)) => info,
            _ => {
                return Err(corrupt(CorruptKind::InvalidValue {
                    field: "root node flags",
                    value: u64::from(root.flags),
                }));
            }
        };
        if info.node_size != blocks.block_size {
            return Err(corrupt(CorruptKind::InvalidValue {
                field: "tree node size",
                value: u64::from(info.node_size),
            }));
        }
        if let Layout::Fixed { key, value } = spec.layout
            && (info.key_size != u32::from(key) || info.value_size != u32::from(value))
        {
            return Err(corrupt(CorruptKind::InvalidValue {
                field: "tree key or value size",
                value: (u64::from(info.key_size) << 32) | u64::from(info.value_size),
            }));
        }
        let children = if info.flags & BTREE_EPHEMERAL != 0 {
            return Err(Error::unsupported(
                "an APFS tree of ephemeral nodes outside the space manager",
            ));
        } else if info.flags & BTREE_PHYSICAL != 0 {
            Children::Physical
        } else {
            match spec.omap {
                Some((omap, xid)) => Children::Virtual { omap, xid },
                None => {
                    return Err(corrupt(CorruptKind::InvalidValue {
                        field: "tree flags (virtual tree with no object map)",
                        value: u64::from(info.flags),
                    }));
                }
            }
        };
        Ok(Self {
            root: spec.root,
            root_oid: spec.headers.then_some(spec.root_oid),
            children,
            headers: spec.headers,
            layout: spec.layout,
            subtype: spec.subtype,
            structure: spec.structure,
            node_limit: info.node_count.min(blocks.block_count).saturating_add(1),
        })
    }

    fn root_node(&self, blocks: &Blocks) -> Result<Arc<Node>> {
        read_node(blocks, self.root, &self.spec(self.root_oid, self.headers))
    }

    fn spec(&self, oid: Option<u64>, headers: bool) -> NodeSpec {
        NodeSpec {
            oid: if headers { oid } else { None },
            headers,
            layout: self.layout,
            subtype: self.subtype,
            structure: self.structure,
        }
    }

    /// The child `i` of index node `parent`, checked to be one level below.
    fn child(&self, blocks: &Blocks, parent: &Node, i: usize) -> Result<Arc<Node>> {
        let corrupt = |kind| node_corrupt(self.structure, parent.paddr, kind);
        let id = parent
            .child(i)
            .ok_or_else(|| corrupt(CorruptKind::Other("index entry without a child id")))?;
        let (paddr, oid, headers) = match &self.children {
            Children::Physical => (id, Some(id), self.headers),
            Children::Virtual { omap, xid } => {
                let mapping = omap.lookup(blocks, id, *xid)?.ok_or_else(|| {
                    Error::Corrupt(Corruption::new(
                        self.structure,
                        Location::Object(id),
                        CorruptKind::Other("child node is not in the object map"),
                    ))
                })?;
                if mapping.flags & super::omap::OMAP_VAL_ENCRYPTED != 0 {
                    return Err(Error::unsupported("encrypted volume"));
                }
                let headers = self.headers && mapping.flags & super::omap::OMAP_VAL_NOHEADER == 0;
                (mapping.paddr, Some(id), headers)
            }
        };
        let node = read_node(blocks, paddr, &self.spec(oid, headers))?;
        if node.flags & BTNODE_ROOT != 0 || Some(node.level) != parent.level.checked_sub(1) {
            return Err(node_corrupt(
                self.structure,
                paddr,
                CorruptKind::InvalidValue {
                    field: "child node level",
                    value: u64::from(node.level),
                },
            ));
        }
        Ok(node)
    }

    /// A cursor on the first record whose key is not less than the target.
    /// `cmp(key)` orders a stored key against the target.
    pub fn seek(
        &self,
        container: Arc<Blocks>,
        cmp: &mut dyn FnMut(&[u8]) -> Result<Ordering>,
    ) -> Result<Cursor> {
        let mut cursor = Cursor {
            blocks: container,
            tree: self.clone(),
            path: Vec::new(),
            loads: 0,
        };
        cursor.seek(cmp)?;
        Ok(cursor)
    }

    /// The last record whose key is not greater than the target, as the
    /// leaf that holds it and its index.
    pub fn seek_le(
        &self,
        blocks: &Blocks,
        cmp: &mut dyn FnMut(&[u8]) -> Result<Ordering>,
    ) -> Result<Option<(Arc<Node>, usize)>> {
        let mut node = self.root_node(blocks)?;
        // Each pass descends one level; levels strictly decrease from at
        // most MAX_LEVEL, so this ends.
        loop {
            let upper = node.upper_bound(cmp)?;
            let Some(at) = upper.checked_sub(1) else {
                return Ok(None);
            };
            if node.is_leaf() {
                return Ok(node.value(at).is_some().then_some((node, at)));
            }
            node = self.child(blocks, &node, at)?;
        }
    }
}

/// A position in a tree. Holds the path from the root; advancing climbs to
/// the first ancestor with a next child and descends leftmost.
pub(crate) struct Cursor {
    blocks: Arc<Blocks>,
    tree: Tree,
    /// From the root down; the last is a leaf and its index is the next
    /// record [`Self::next`] returns.
    path: Vec<(Arc<Node>, usize)>,
    /// Nodes loaded by `next` since the last seek.
    loads: u64,
}

impl Cursor {
    /// Repositions on the first record whose key is not less than the
    /// target, reusing the current leaf when the target lies inside it.
    pub fn seek(&mut self, cmp: &mut dyn FnMut(&[u8]) -> Result<Ordering>) -> Result<()> {
        self.loads = 0;
        if let Some((leaf, index)) = self.path.last_mut()
            && leaf.len() > 0
        {
            let first = cmp(leaf.key(0))?;
            let last = cmp(leaf.key(leaf.len().saturating_sub(1)))?;
            if first == Ordering::Less && last != Ordering::Less {
                *index = leaf.lower_bound(cmp)?;
                return Ok(());
            }
        }
        self.path.clear();
        let mut node = self.tree.root_node(&self.blocks)?;
        // Bounded: levels strictly decrease (checked in `child`).
        loop {
            if node.is_leaf() {
                let at = node.lower_bound(cmp)?;
                self.path.push((node, at));
                return Ok(());
            }
            // The last child whose key is less than the target: records
            // equal to it may start there.
            let at = node.lower_bound(cmp)?.saturating_sub(1);
            let child = self.tree.child(&self.blocks, &node, at)?;
            self.path.push((node, at));
            node = child;
        }
    }

    /// The current record's key and value, then advances. `None` at the end.
    /// Ghost entries are skipped.
    pub fn next(&mut self) -> Result<Option<(&[u8], &[u8])>> {
        loop {
            let Some((leaf, index)) = self.path.last_mut() else {
                return Ok(None);
            };
            if *index < leaf.len() {
                let at = *index;
                *index = at.saturating_add(1);
                if leaf.value(at).is_none() {
                    continue;
                }
                break;
            }
            if !self.advance_leaf()? {
                self.path.clear();
                return Ok(None);
            }
        }
        let Some((leaf, index)) = self.path.last() else {
            return Ok(None);
        };
        let at = index.saturating_sub(1);
        Ok(Some((leaf.key(at), leaf.value(at).unwrap_or(&[]))))
    }

    /// Moves to the start of the next leaf. False at the end of the tree.
    fn advance_leaf(&mut self) -> Result<bool> {
        self.path.pop();
        // Climb to the first ancestor with a next child.
        loop {
            let Some((node, index)) = self.path.last_mut() else {
                return Ok(false);
            };
            let next = index.saturating_add(1);
            if next < node.len() {
                *index = next;
                break;
            }
            self.path.pop();
        }
        // Descend leftmost. Bounded by the levels.
        loop {
            let Some((parent, index)) = self.path.last() else {
                return Ok(false);
            };
            if parent.is_leaf() {
                return Ok(true);
            }
            self.loads = self.loads.saturating_add(1);
            if self.loads > self.tree.node_limit {
                return Err(node_corrupt(
                    self.tree.structure,
                    parent.paddr,
                    CorruptKind::LimitExceeded {
                        what: "nodes visited in one pass over a tree",
                        limit: self.tree.node_limit,
                    },
                ));
            }
            let (parent, index) = (Arc::clone(parent), *index);
            let child = self.tree.child(&self.blocks, &parent, index)?;
            self.path.push((child, 0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_container::{Built, NodeOpts, build_tree_image};

    const SPEC: NodeSpec = NodeSpec {
        oid: None,
        headers: true,
        layout: Layout::Fixed { key: 16, value: 16 },
        subtype: 0xB,
        structure: "test tree",
    };

    fn key(oid: u64, xid: u64) -> Vec<u8> {
        let mut k = oid.to_le_bytes().to_vec();
        k.extend_from_slice(&xid.to_le_bytes());
        k
    }

    fn cmp_to(target: (u64, u64)) -> impl FnMut(&[u8]) -> Result<Ordering> {
        move |k: &[u8]| {
            let a = (le_u64(k, 0).unwrap(), le_u64(k, 8).unwrap());
            Ok(a.cmp(&target))
        }
    }

    /// A physical fixed tree of `n` records (oid i*2, xid 1) over several
    /// levels, with `per_node` records per node.
    fn tree(n: u64, per_node: usize) -> (Built, Tree) {
        let records: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
            .map(|i| (key(i * 2, 1), key(i * 2 + 1000, 0)))
            .collect();
        let built = build_tree_image(&records, NodeOpts::omap(per_node));
        let tree = Tree::open(
            &built.blocks,
            TreeSpec {
                root: built.root,
                root_oid: built.root,
                headers: true,
                layout: SPEC.layout,
                subtype: 0xB,
                structure: "test tree",
                omap: None,
            },
        )
        .unwrap();
        (built, tree)
    }

    #[test]
    fn iterates_every_record_in_order_across_levels() {
        let (built, tree) = tree(500, 7);
        assert!(tree.root_node(&built.blocks).unwrap().level >= 2);
        let mut cursor = tree
            .seek(Arc::clone(&built.blocks), &mut cmp_to((0, 0)))
            .unwrap();
        let mut seen = Vec::new();
        while let Some((k, v)) = cursor.next().unwrap() {
            seen.push((le_u64(k, 0).unwrap(), le_u64(v, 0).unwrap()));
        }
        let want: Vec<_> = (0..500).map(|i| (i * 2, i * 2 + 1000)).collect();
        assert_eq!(seen, want);
    }

    #[test]
    fn seek_and_seek_le_find_neighbours() {
        let (built, tree) = tree(300, 5);
        // An absent key between two present ones.
        let mut cursor = tree
            .seek(Arc::clone(&built.blocks), &mut cmp_to((101, 0)))
            .unwrap();
        assert_eq!(le_u64(cursor.next().unwrap().unwrap().0, 0), Some(102));
        // Reuse within the leaf, then a far jump.
        cursor.seek(&mut cmp_to((104, 0))).unwrap();
        assert_eq!(le_u64(cursor.next().unwrap().unwrap().0, 0), Some(104));
        cursor.seek(&mut cmp_to((500, 1))).unwrap();
        assert_eq!(le_u64(cursor.next().unwrap().unwrap().0, 0), Some(500));
        cursor.seek(&mut cmp_to((10_000, 0))).unwrap();
        assert!(cursor.next().unwrap().is_none());

        let (leaf, at) = tree
            .seek_le(&built.blocks, &mut cmp_to((101, 9)))
            .unwrap()
            .unwrap();
        assert_eq!(le_u64(leaf.key(at), 0), Some(100));
        assert!(
            tree.seek_le(&built.blocks, &mut cmp_to((0, 0)))
                .unwrap()
                .is_none()
        );
        let (leaf, at) = tree
            .seek_le(&built.blocks, &mut cmp_to((u64::MAX, 0)))
            .unwrap()
            .unwrap();
        assert_eq!(le_u64(leaf.key(at), 0), Some(598));
    }

    #[test]
    fn parse_rejects_malformed_nodes() {
        let (built, tree) = tree(3, 8);
        let node = tree.root_node(&built.blocks).unwrap();
        let good = node.data.to_vec();
        let parse = |b: &[u8]| Node::parse(b.to_vec().into_boxed_slice(), 1, &SPEC);
        assert!(parse(&good).is_ok());
        // Truncated below the node header.
        assert!(parse(&good[..40]).is_err());
        // Key count past the table.
        let mut bad = good.clone();
        bad[36..40].copy_from_slice(&10_000u32.to_le_bytes());
        assert!(parse(&bad).is_err());
        // A value offset pointing before the key area.
        let mut bad = good.clone();
        let toc = 56 + usize::from(u16::from_le_bytes([bad[40], bad[41]]));
        bad[toc + 2..toc + 4].copy_from_slice(&0xFFF0u16.to_le_bytes());
        assert!(parse(&bad).is_err());
        // Level past MAX_LEVEL, leaf flag disagreeing with the level.
        let mut bad = good.clone();
        bad[34..36].copy_from_slice(&17u16.to_le_bytes());
        assert!(parse(&bad).is_err());
        let mut bad = good.clone();
        bad[34..36].copy_from_slice(&1u16.to_le_bytes());
        assert!(parse(&bad).is_err());
        // Variable layout expected, fixed node found.
        let variable = NodeSpec {
            layout: Layout::Variable,
            ..SPEC
        };
        assert!(Node::parse(good.clone().into_boxed_slice(), 1, &variable).is_err());
        // Every truncation fails or parses; none panics.
        for len in 0..good.len() {
            let _ = parse(&good[..len]);
        }
    }

    #[test]
    fn a_self_referencing_index_is_rejected() {
        // Two levels; point the root's second child at the root itself.
        let (built, _tree) = tree(40, 5);
        let mut image = built.image.clone();
        let root = usize::try_from(built.root).unwrap() * 4096;
        let node = Node::parse(
            image[root..root + 4096].to_vec().into_boxed_slice(),
            built.root,
            &SPEC,
        )
        .unwrap();
        let e = node.entries[1];
        let v = usize::try_from(e.value).unwrap();
        image[root + v..root + v + 8].copy_from_slice(&built.root.to_le_bytes());
        crate::object::seal(&mut image[root..root + 4096]);
        let blocks = Built::blocks_over(image);
        let tree = Tree::open(
            &blocks,
            TreeSpec {
                root: built.root,
                root_oid: built.root,
                headers: true,
                layout: SPEC.layout,
                subtype: 0xB,
                structure: "test tree",
                omap: None,
            },
        )
        .unwrap();
        let mut cursor = tree.seek(Arc::clone(&blocks), &mut cmp_to((0, 0))).unwrap();
        let mut result = Ok(());
        for _ in 0..1000 {
            match cursor.next() {
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        assert!(result.is_err(), "the loop must be reported");
    }
}
