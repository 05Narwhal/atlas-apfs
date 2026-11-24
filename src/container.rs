// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! The APFS container (design §3.2–§3.3, §4): block reads, the container
//! superblock, checkpoint selection, ephemeral objects and the container's
//! object map.

use std::sync::Arc;

use super::btree::NodeCache;
use super::bytes::{array, le_u32, le_u64, len64};
use super::object::{self, ObjHeader, TYPE_CHECKPOINT_MAP, TYPE_NX_SUPERBLOCK};
use super::omap::Omap;
use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};
use atlas_fs_core::io::block::BlockSource;

/// `NXSB`, read as a little-endian `u32`.
pub(crate) const NX_MAGIC: u32 = u32::from_le_bytes(*b"NXSB");
/// Smallest and largest block sizes accepted.
pub(crate) const MIN_BLOCK_SIZE: u32 = 4096;
pub(crate) const MAX_BLOCK_SIZE: u32 = 65536;
/// Most blocks in the checkpoint descriptor area.
pub(crate) const MAX_XP_DESC: u32 = 65_536;
/// Most checkpoint mappings kept.
pub(crate) const MAX_EPHEMERAL: usize = 65_536;
/// Most volumes a container holds.
pub(crate) const MAX_FILE_SYSTEMS: u32 = 100;
/// The descriptor area is read in one piece below this size.
const BULK_READ_LIMIT: u64 = 16 << 20;

/// Incompatible feature: pre-release version 1.
const NX_INCOMPAT_VERSION1: u64 = 0x1;
/// Incompatible feature: version 2, required.
const NX_INCOMPAT_VERSION2: u64 = 0x2;
/// Incompatible feature: a Fusion container.
const NX_INCOMPAT_FUSION: u64 = 0x100;
/// High bit of `nx_xp_desc_blocks`: the area is not contiguous.
const XP_NONCONTIGUOUS: u32 = 0x8000_0000;

const STRUCTURE: &str = "container superblock";

fn corrupt_at(structure: &'static str, paddr: u64, kind: CorruptKind) -> Error {
    Error::Corrupt(Corruption::new(structure, Location::Block(paddr), kind))
}

/// Block access to a container: range-checked reads and the node cache.
pub(crate) struct Blocks {
    source: Arc<dyn BlockSource>,
    /// Bytes per block.
    pub block_size: u32,
    /// Blocks readable: the superblock's count, clamped to the source.
    pub block_count: u64,
    pub(crate) nodes: NodeCache,
}

impl Blocks {
    /// Blocks of `block_size` over `source`, at most `claimed` of them.
    pub fn new(source: Arc<dyn BlockSource>, block_size: u32, claimed: u64) -> Self {
        let in_source = source.len().checked_div(u64::from(block_size)).unwrap_or(0);
        Self {
            block_count: claimed.min(in_source),
            nodes: NodeCache::new(block_size),
            source,
            block_size,
        }
    }

    /// The source the container was opened on.
    pub fn source(&self) -> &Arc<dyn BlockSource> {
        &self.source
    }

    /// Fills `buf` from block `paddr` on. `buf` may span several blocks; all
    /// of them must lie inside the container.
    pub fn read(&self, paddr: u64, buf: &mut [u8]) -> Result<()> {
        let size = u64::from(self.block_size);
        let blocks = len64(buf.len()).div_ceil(size.max(1));
        let inside = paddr
            .checked_add(blocks)
            .is_some_and(|end| end <= self.block_count);
        let offset = paddr.checked_mul(size).filter(|_| inside).ok_or_else(|| {
            corrupt_at(
                "container",
                paddr,
                CorruptKind::OutOfRange {
                    field: "block address",
                    value: paddr,
                    limit_name: "container blocks",
                    limit: self.block_count,
                },
            )
        })?;
        self.source.read_at(offset, buf)
    }

    /// One whole block.
    pub fn block(&self, paddr: u64) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; usize::try_from(self.block_size).unwrap_or(0)];
        self.read(paddr, &mut buf)?;
        Ok(buf)
    }
}

/// The container superblock's fields this reader uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NxSuperblock {
    /// The header's transaction.
    pub xid: u64,
    pub block_size: u32,
    pub block_count: u64,
    pub incompatible_features: u64,
    pub uuid: [u8; 16],
    /// Blocks in the descriptor area, without the high bit.
    pub xp_desc_blocks: u32,
    pub xp_desc_contiguous: bool,
    pub xp_desc_base: u64,
    pub xp_desc_index: u32,
    pub xp_desc_len: u32,
    pub spaceman_oid: u64,
    pub omap_oid: u64,
    /// The volumes' virtual oids, in slot order, zero slots left out.
    pub fs_oids: Vec<u64>,
    /// `nx_keylocker`: address and block count.
    pub keylocker: (u64, u64),
}

impl NxSuperblock {
    /// Parses a verified container superblock read from `paddr`.
    pub fn parse(block: &[u8], paddr: u64) -> Result<Self> {
        let short = || {
            corrupt_at(
                STRUCTURE,
                paddr,
                CorruptKind::Truncated {
                    needed: 1312,
                    available: len64(block.len()),
                },
            )
        };
        let header = ObjHeader::read(block).ok_or_else(short)?;
        let magic = le_u32(block, 32).ok_or_else(short)?;
        if magic != NX_MAGIC {
            return Err(corrupt_at(
                STRUCTURE,
                paddr,
                CorruptKind::BadSignature {
                    expected: u64::from(NX_MAGIC),
                    found: u64::from(magic),
                },
            ));
        }
        let block_size = le_u32(block, 36).ok_or_else(short)?;
        check_block_size(block_size, paddr)?;
        let max_fs = le_u32(block, 180).ok_or_else(short)?;
        if max_fs > MAX_FILE_SYSTEMS {
            return Err(corrupt_at(
                STRUCTURE,
                paddr,
                CorruptKind::OutOfRange {
                    field: "maximum volumes",
                    value: u64::from(max_fs),
                    limit_name: "MAX_FILE_SYSTEMS",
                    limit: u64::from(MAX_FILE_SYSTEMS),
                },
            ));
        }
        let mut fs_oids = Vec::new();
        for slot in 0..usize::try_from(max_fs).unwrap_or(0) {
            // slot < 100, so the offset is at most 184 + 792.
            let oid =
                le_u64(block, 184usize.saturating_add(slot.saturating_mul(8))).ok_or_else(short)?;
            if oid != 0 {
                fs_oids.push(oid);
            }
        }
        let desc_blocks = le_u32(block, 104).ok_or_else(short)?;
        Ok(Self {
            xid: header.xid,
            block_size,
            block_count: le_u64(block, 40).ok_or_else(short)?,
            incompatible_features: le_u64(block, 64).ok_or_else(short)?,
            uuid: array(block, 72).ok_or_else(short)?,
            xp_desc_blocks: desc_blocks & !XP_NONCONTIGUOUS,
            xp_desc_contiguous: desc_blocks & XP_NONCONTIGUOUS == 0,
            xp_desc_base: le_u64(block, 112).ok_or_else(short)?,
            xp_desc_index: le_u32(block, 136).ok_or_else(short)?,
            xp_desc_len: le_u32(block, 140).ok_or_else(short)?,
            spaceman_oid: le_u64(block, 152).ok_or_else(short)?,
            omap_oid: le_u64(block, 160).ok_or_else(short)?,
            fs_oids,
            keylocker: (
                le_u64(block, 1296).ok_or_else(short)?,
                le_u64(block, 1304).ok_or_else(short)?,
            ),
        })
    }
}

fn check_block_size(block_size: u32, paddr: u64) -> Result<()> {
    if !block_size.is_power_of_two() || !(MIN_BLOCK_SIZE..=MAX_BLOCK_SIZE).contains(&block_size) {
        return Err(corrupt_at(
            STRUCTURE,
            paddr,
            CorruptKind::InvalidValue {
                field: "block size",
                value: u64::from(block_size),
            },
        ));
    }
    Ok(())
}

/// One entry of a checkpoint map: where an ephemeral object is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CheckpointMapping {
    /// The object's type and storage flags.
    pub type_and_flags: u32,
    /// Bytes the object occupies.
    pub size: u32,
    /// The volume it belongs to (0 for the container).
    pub fs_oid: u64,
    /// The ephemeral oid.
    pub oid: u64,
    /// Where it is stored.
    pub paddr: u64,
}

/// An opened container.
pub(crate) struct Container {
    /// Block access, shared with every tree and cursor.
    pub blocks: Arc<Blocks>,
    /// The superblock of the checkpoint in use.
    pub superblock: NxSuperblock,
    /// The container's object map.
    pub omap: Arc<Omap>,
    ephemeral: Vec<CheckpointMapping>,
}

impl Container {
    /// Opens the container in `source`: reads block 0, then chooses the
    /// newest checkpoint whose maps and object map verify (§3.3).
    pub fn open(source: Arc<dyn BlockSource>) -> Result<Arc<Self>> {
        let block0 = read_block_zero(&*source)?;
        object::verify(&block0, 0, STRUCTURE, &[TYPE_NX_SUPERBLOCK], None)?;
        let sb0 = NxSuperblock::parse(&block0, 0)?;
        let blocks = Arc::new(Blocks::new(source, sb0.block_size, sb0.block_count));
        let desc = descriptor_area(&blocks, &sb0)?;

        let mut candidates = find_candidates(&blocks, &sb0, desc)?;
        // Newest first; ties keep ring order.
        candidates.sort_by_key(|&(xid, _)| std::cmp::Reverse(xid));

        let mut last_error = None;
        for &(_, paddr) in &candidates {
            let sb = match blocks
                .block(paddr)
                .and_then(|b| NxSuperblock::parse(&b, paddr))
            {
                Ok(sb) => sb,
                Err(e) => {
                    last_error = Some(e);
                    continue;
                }
            };
            match accept(&blocks, sb, desc) {
                Ok(container) => return Ok(Arc::new(container)),
                Err(e) => last_error = Some(e),
            }
        }
        // Block 0 under the same test.
        match accept(&blocks, sb0, desc) {
            Ok(container) => Ok(Arc::new(container)),
            Err(e) => Err(last_error.unwrap_or(e)),
        }
    }

    /// Where the ephemeral object `oid` is, from the checkpoint maps.
    pub fn ephemeral(&self, oid: u64) -> Option<&CheckpointMapping> {
        self.ephemeral.iter().find(|m| m.oid == oid)
    }

    /// The transaction of the checkpoint in use.
    pub fn xid(&self) -> u64 {
        self.superblock.xid
    }

    /// Bytes in the container: its blocks as readable.
    pub fn byte_len(&self) -> u64 {
        self.blocks
            .block_count
            .saturating_mul(u64::from(self.blocks.block_size))
    }
}

/// Reads block 0 at its own block size.
fn read_block_zero(source: &dyn BlockSource) -> Result<Vec<u8>> {
    let mut head = vec![0u8; MIN_BLOCK_SIZE as usize];
    if source.len() < u64::from(MIN_BLOCK_SIZE) {
        return Err(corrupt_at(
            STRUCTURE,
            0,
            CorruptKind::Truncated {
                needed: u64::from(MIN_BLOCK_SIZE),
                available: source.len(),
            },
        ));
    }
    source.read_at(0, &mut head)?;
    let block_size = le_u32(&head, 36).unwrap_or(0);
    check_block_size(block_size, 0)?;
    if block_size == MIN_BLOCK_SIZE {
        return Ok(head);
    }
    let mut block = vec![0u8; usize::try_from(block_size).unwrap_or(0)];
    source.read_at(0, &mut block)?;
    Ok(block)
}

/// The descriptor area as (first block, block count), checked.
fn descriptor_area(blocks: &Blocks, sb: &NxSuperblock) -> Result<(u64, u32)> {
    if !sb.xp_desc_contiguous {
        return Err(Error::unsupported(
            "an APFS container whose checkpoint area is not contiguous",
        ));
    }
    let n = sb.xp_desc_blocks;
    if n == 0 || n > MAX_XP_DESC {
        return Err(corrupt_at(
            STRUCTURE,
            0,
            CorruptKind::OutOfRange {
                field: "checkpoint descriptor blocks",
                value: u64::from(n),
                limit_name: "MAX_XP_DESC",
                limit: u64::from(MAX_XP_DESC),
            },
        ));
    }
    let end = sb.xp_desc_base.checked_add(u64::from(n));
    if end.is_none_or(|end| end > blocks.block_count) {
        return Err(corrupt_at(
            STRUCTURE,
            0,
            CorruptKind::OutOfRange {
                field: "checkpoint descriptor area",
                value: sb.xp_desc_base,
                limit_name: "container blocks",
                limit: blocks.block_count,
            },
        ));
    }
    Ok((sb.xp_desc_base, n))
}

/// Every block of the descriptor area that verifies as a superblock of
/// this container, as (xid, paddr).
fn find_candidates(
    blocks: &Blocks,
    sb0: &NxSuperblock,
    desc: (u64, u32),
) -> Result<Vec<(u64, u64)>> {
    let (base, n) = desc;
    let size = usize::try_from(blocks.block_size).unwrap_or(0);
    let total = u64::from(n).saturating_mul(u64::from(blocks.block_size));
    let mut candidates = Vec::new();
    let mut consider = |block: &[u8], paddr: u64| {
        let is_sb = ObjHeader::read(block).is_some_and(|h| h.kind() == TYPE_NX_SUPERBLOCK);
        if !is_sb || object::verify(block, paddr, STRUCTURE, &[TYPE_NX_SUPERBLOCK], None).is_err() {
            return;
        }
        if let Ok(sb) = NxSuperblock::parse(block, paddr)
            && sb.uuid == sb0.uuid
            && sb.block_size == sb0.block_size
        {
            candidates.push((sb.xid, paddr));
        }
    };
    if total <= BULK_READ_LIMIT {
        let mut area = vec![0u8; usize::try_from(total).unwrap_or(0)];
        blocks.read(base, &mut area)?;
        for (i, block) in area.chunks_exact(size.max(1)).enumerate() {
            consider(block, base.saturating_add(len64(i)));
        }
    } else {
        let mut block = vec![0u8; size];
        for i in 0..u64::from(n) {
            let paddr = base.saturating_add(i);
            // A block that cannot be read is not a candidate.
            if blocks.read(paddr, &mut block).is_ok() {
                consider(&block, paddr);
            }
        }
    }
    Ok(candidates)
}

/// Checks the checkpoint `sb` closes: every map block verifies, belongs to
/// its transaction and fits; the container object map opens.
fn accept(blocks: &Arc<Blocks>, sb: NxSuperblock, desc: (u64, u32)) -> Result<Container> {
    let (base, n) = desc;
    if sb.incompatible_features & NX_INCOMPAT_FUSION != 0 {
        return Err(Error::unsupported("an APFS Fusion container"));
    }
    if sb.incompatible_features & NX_INCOMPAT_VERSION1 != 0 {
        return Err(Error::unsupported(
            "a pre-release (version 1) APFS container",
        ));
    }
    if sb.incompatible_features != NX_INCOMPAT_VERSION2 {
        return Err(Error::unsupported(format!(
            "this APFS container uses a feature newer than this reader (incompatible features {:#x})",
            sb.incompatible_features
        )));
    }
    if sb.block_size != blocks.block_size {
        return Err(corrupt_at(
            STRUCTURE,
            0,
            CorruptKind::InvalidValue {
                field: "block size",
                value: u64::from(sb.block_size),
            },
        ));
    }
    if sb.xp_desc_len == 0 || sb.xp_desc_len > n || sb.xp_desc_index >= n {
        return Err(corrupt_at(
            STRUCTURE,
            0,
            CorruptKind::OutOfRange {
                field: "checkpoint length or index",
                value: u64::from(sb.xp_desc_len),
                limit_name: "checkpoint descriptor blocks",
                limit: u64::from(n),
            },
        ));
    }
    let per_block = usize::try_from(blocks.block_size)
        .unwrap_or(0)
        .saturating_sub(40)
        / 40;
    let mut ephemeral = Vec::new();
    // All but the last block of the checkpoint are maps. At most n - 1.
    for k in 0..sb.xp_desc_len.saturating_sub(1) {
        let ring = u64::from(sb.xp_desc_index)
            .saturating_add(u64::from(k))
            .checked_rem(u64::from(n))
            .unwrap_or(0);
        let paddr = base.saturating_add(ring);
        let block = blocks.block(paddr)?;
        let header = object::verify(
            &block,
            paddr,
            "checkpoint map",
            &[TYPE_CHECKPOINT_MAP],
            Some(paddr),
        )?;
        if header.xid != sb.xid {
            return Err(corrupt_at(
                "checkpoint map",
                paddr,
                CorruptKind::InvalidValue {
                    field: "transaction",
                    value: header.xid,
                },
            ));
        }
        let count = le_u32(&block, 36).unwrap_or(u32::MAX);
        let count = usize::try_from(count).unwrap_or(usize::MAX);
        if count > per_block || ephemeral.len().saturating_add(count) > MAX_EPHEMERAL {
            return Err(corrupt_at(
                "checkpoint map",
                paddr,
                CorruptKind::OutOfRange {
                    field: "mapping count",
                    value: len64(count),
                    limit_name: "entries per block",
                    limit: len64(per_block),
                },
            ));
        }
        for i in 0..count {
            // i < per_block, so the entry lies inside the block.
            let at = 40usize.saturating_add(i.saturating_mul(40));
            let entry = (|| {
                Some(CheckpointMapping {
                    type_and_flags: le_u32(&block, at)?,
                    size: le_u32(&block, at.saturating_add(8))?,
                    fs_oid: le_u64(&block, at.saturating_add(16))?,
                    oid: le_u64(&block, at.saturating_add(24))?,
                    paddr: le_u64(&block, at.saturating_add(32))?,
                })
            })();
            if let Some(entry) = entry {
                ephemeral.push(entry);
            }
        }
    }
    let omap = Arc::new(Omap::open(blocks, sb.omap_oid)?);
    Ok(Container {
        blocks: Arc::clone(blocks),
        superblock: sb,
        omap,
        ephemeral,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_container::{ContainerBuilder, corrupt_block};
    use atlas_fs_core::io::block::MemSource;

    fn open(image: Vec<u8>) -> Result<Arc<Container>> {
        Container::open(Arc::new(MemSource::new(image)))
    }

    #[test]
    fn opens_a_built_container() {
        let built = ContainerBuilder::new().build();
        let c = open(built.image).unwrap();
        assert_eq!(c.blocks.block_size, 4096);
        assert_eq!(c.xid(), built.xid);
        assert_eq!(c.superblock.fs_oids.len(), 1);
        let vol = c
            .omap
            .lookup(&c.blocks, c.superblock.fs_oids[0], c.xid())
            .unwrap()
            .unwrap();
        assert_eq!(vol.paddr, built.volume_paddrs[0]);
        assert!(c.ephemeral(built.spaceman_oid).is_some());
    }

    #[test]
    fn chooses_the_newest_checkpoint_that_verifies() {
        // Block 0 at xid 5, the ring holding xid 9 (valid) and xid 12 (bad
        // checksum): 9 is chosen.
        let built = ContainerBuilder::new()
            .checkpoints(&[9, 12])
            .block_zero_xid(5)
            .build();
        let mut image = built.image;
        corrupt_block(&mut image, built.checkpoint_superblocks[1], 4096);
        let c = open(image).unwrap();
        assert_eq!(c.xid(), 9);
    }

    #[test]
    fn falls_back_to_block_zero_and_fails_without_any() {
        let built = ContainerBuilder::new()
            .checkpoints(&[9])
            .block_zero_xid(5)
            .build();
        let mut image = built.image.clone();
        corrupt_block(&mut image, built.checkpoint_superblocks[0], 4096);
        assert_eq!(open(image.clone()).unwrap().xid(), 5);
        corrupt_block(&mut image, 0, 4096);
        assert!(open(image).is_err());
    }

    #[test]
    fn a_damaged_checkpoint_map_rejects_its_checkpoint() {
        let built = ContainerBuilder::new()
            .checkpoints(&[9, 12])
            .block_zero_xid(5)
            .build();
        let mut image = built.image;
        corrupt_block(&mut image, built.checkpoint_maps[1], 4096);
        assert_eq!(open(image).unwrap().xid(), 9);
    }

    #[test]
    fn rejects_hostile_superblocks() {
        let built = ContainerBuilder::new().build();
        let set = |at: usize, bytes: &[u8]| {
            let mut image = built.image.clone();
            // Block 0 and every ring copy, resealed, so only the field is wrong.
            for paddr in std::iter::once(0).chain(built.checkpoint_superblocks.iter().copied()) {
                let base = usize::try_from(paddr).unwrap() * 4096;
                image[base + at..base + at + bytes.len()].copy_from_slice(bytes);
                crate::object::seal(&mut image[base..base + 4096]);
            }
            open(image)
        };
        // Block size not a power of two, too small, too large.
        assert!(set(36, &5000u32.to_le_bytes()).is_err());
        assert!(set(36, &512u32.to_le_bytes()).is_err());
        assert!(set(36, &(1u32 << 20).to_le_bytes()).is_err());
        // Descriptor area past the end, zero blocks, non-contiguous.
        assert!(set(112, &u64::MAX.to_le_bytes()).is_err());
        assert!(set(104, &0u32.to_le_bytes()).is_err());
        assert!(matches!(
            set(104, &(XP_NONCONTIGUOUS | 8).to_le_bytes()),
            Err(Error::Unsupported(_))
        ));
        // Fusion, version 1, unknown incompatible bit.
        assert!(matches!(
            set(64, &0x102u64.to_le_bytes()),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            set(64, &0x1u64.to_le_bytes()),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            set(64, &0x6u64.to_le_bytes()),
            Err(Error::Unsupported(_))
        ));
        // More than 100 volumes.
        assert!(set(180, &101u32.to_le_bytes()).is_err());
        // Truncated images.
        for len in [0usize, 100, 4095, 4096, 8192] {
            assert!(open(built.image[..len].to_vec()).is_err(), "{len}");
        }
    }
}
