// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Object maps (design §3.4): virtual object id and transaction to block
//! address, with a resolution cache.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};

use super::btree::{Layout, Tree, TreeSpec};
use super::bytes::{le_u32, le_u64, len64};
use super::container::Blocks;
use super::object::{self, OBJ_NOHEADER, TYPE_OMAP, TYPE_OMAP_TREE};
use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};

/// Mapping flag: the object was deleted at this transaction.
pub(crate) const OMAP_VAL_DELETED: u32 = 0x1;
/// Mapping flag: the object is encrypted.
pub(crate) const OMAP_VAL_ENCRYPTED: u32 = 0x4;
/// Mapping flag: the object is stored without a header.
pub(crate) const OMAP_VAL_NOHEADER: u32 = 0x8;

/// Resolutions cached per object map.
pub(crate) const OMAP_CACHE: usize = 65_536;
const SHARDS: usize = 16;

/// Object map keys and values are both 16 bytes.
const LAYOUT: Layout = Layout::Fixed { key: 16, value: 16 };

/// Where a virtual object is stored at some transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Mapping {
    /// Its block address.
    pub paddr: u64,
    /// `ov_flags`.
    pub flags: u32,
    /// `ov_size`: bytes the object occupies.
    pub size: u32,
    /// The transaction of the mapping found.
    pub xid: u64,
}

/// An object map.
pub(crate) struct Omap {
    /// Where its header is.
    pub paddr: u64,
    tree: Tree,
    cache: Box<[Mutex<ResolutionShard>]>,
}

/// Cached answers of one shard, by (oid, transaction asked for).
type ResolutionShard = HashMap<(u64, u64), Option<Mapping>>;

fn corrupt(paddr: u64, kind: CorruptKind) -> Error {
    Error::Corrupt(Corruption::new("object map", Location::Block(paddr), kind))
}

impl Omap {
    /// Reads the object map whose header is at `paddr` (a physical object).
    pub fn open(blocks: &Blocks, paddr: u64) -> Result<Self> {
        let block = blocks.block(paddr)?;
        object::verify(&block, paddr, "object map", &[TYPE_OMAP], Some(paddr))?;
        let short = || {
            corrupt(
                paddr,
                CorruptKind::Truncated {
                    needed: 72,
                    available: len64(block.len()),
                },
            )
        };
        let tree_type = le_u32(&block, 40).ok_or_else(short)?;
        let tree_oid = le_u64(&block, 48).ok_or_else(short)?;
        if u32::from(object::TYPE_BTREE) != tree_type & object::TYPE_MASK {
            return Err(corrupt(
                paddr,
                CorruptKind::InvalidValue {
                    field: "object map tree type",
                    value: u64::from(tree_type),
                },
            ));
        }
        let tree = Tree::open(
            blocks,
            TreeSpec {
                root: tree_oid,
                root_oid: tree_oid,
                headers: tree_type & OBJ_NOHEADER == 0,
                layout: LAYOUT,
                subtype: TYPE_OMAP_TREE,
                structure: "object map tree",
                omap: None,
            },
        )?;
        let cache = (0..SHARDS).map(|_| Mutex::new(HashMap::new())).collect();
        Ok(Self { paddr, tree, cache })
    }

    /// Where `oid` is stored as of transaction `max_xid`: the mapping with
    /// the highest transaction not above it. `None` when there is none or
    /// it was deleted.
    pub fn lookup(&self, blocks: &Blocks, oid: u64, max_xid: u64) -> Result<Option<Mapping>> {
        let key = (oid, max_xid);
        let index = oid.checked_rem(len64(SHARDS)).unwrap_or(0);
        let Some(shard) = self.cache.get(usize::try_from(index).unwrap_or(0)) else {
            return self.lookup_uncached(blocks, oid, max_xid);
        };
        if let Some(hit) = shard
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&key)
        {
            return Ok(*hit);
        }
        let found = self.lookup_uncached(blocks, oid, max_xid)?;
        let mut map = shard.lock().unwrap_or_else(PoisonError::into_inner);
        if map.len() >= OMAP_CACHE / SHARDS {
            map.clear();
        }
        map.insert(key, found);
        Ok(found)
    }

    fn lookup_uncached(&self, blocks: &Blocks, oid: u64, max_xid: u64) -> Result<Option<Mapping>> {
        let target = (oid, max_xid);
        let mut cmp = |k: &[u8]| -> Result<Ordering> {
            let stored = parse_key(k).ok_or_else(|| short_record(self.paddr))?;
            Ok(stored.cmp(&target))
        };
        let Some((leaf, at)) = self.tree.seek_le(blocks, &mut cmp)? else {
            return Ok(None);
        };
        let (found_oid, xid) = parse_key(leaf.key(at)).ok_or_else(|| short_record(self.paddr))?;
        if found_oid != oid {
            return Ok(None);
        }
        let value = leaf.value(at).unwrap_or(&[]);
        let (flags, size, paddr) =
            (|| Some((le_u32(value, 0)?, le_u32(value, 4)?, le_u64(value, 8)?)))()
                .ok_or_else(|| short_record(self.paddr))?;
        if flags & OMAP_VAL_DELETED != 0 {
            return Ok(None);
        }
        if paddr >= blocks.block_count {
            return Err(Error::Corrupt(Corruption::new(
                "object map",
                Location::Object(oid),
                CorruptKind::OutOfRange {
                    field: "mapped address",
                    value: paddr,
                    limit_name: "container blocks",
                    limit: blocks.block_count,
                },
            )));
        }
        Ok(Some(Mapping {
            paddr,
            flags,
            size,
            xid,
        }))
    }
}

fn parse_key(k: &[u8]) -> Option<(u64, u64)> {
    Some((le_u64(k, 0)?, le_u64(k, 8)?))
}

fn short_record(paddr: u64) -> Error {
    corrupt(paddr, CorruptKind::Other("object map record is too short"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_container::{Built, omap_image};

    #[test]
    fn lookup_takes_the_newest_mapping_not_after_the_transaction() {
        // oid 1026 at xids 3 and 8; oid 1030 deleted at xid 9.
        let entries = [
            (1026u64, 3u64, 0u32, 1u64),
            (1026, 8, 0, 2),
            (1030, 4, 0, 3),
            (1030, 9, OMAP_VAL_DELETED, 0),
        ];
        let (blocks, paddr) = omap_image(&entries, 2);
        let omap = Omap::open(&blocks, paddr).unwrap();
        let at = |oid, xid| omap.lookup(&blocks, oid, xid).unwrap().map(|m| m.paddr);
        assert_eq!(at(1026, 2), None);
        assert_eq!(at(1026, 3), Some(1));
        assert_eq!(at(1026, 7), Some(1));
        assert_eq!(at(1026, 100), Some(2));
        assert_eq!(at(1030, 8), Some(3));
        assert_eq!(at(1030, 9), None);
        assert_eq!(at(1027, 100), None);
        // Cached answers agree.
        assert_eq!(at(1026, 100), Some(2));
    }

    #[test]
    fn a_mapping_outside_the_container_is_corrupt() {
        let (blocks, paddr) = omap_image(&[(1026, 1, 0, 1 << 40)], 4);
        let omap = Omap::open(&blocks, paddr).unwrap();
        assert!(omap.lookup(&blocks, 1026, 5).is_err());
        // And a damaged header does not open.
        let mut image = Built::image_of(&blocks);
        let at = usize::try_from(paddr).unwrap() * 4096 + 100;
        image[at] ^= 1;
        assert!(Omap::open(&Built::blocks_over(image), paddr).is_err());
    }
}
