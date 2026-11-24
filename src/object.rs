// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! The object header every APFS metadata block starts with, its type
//! constants, and the Fletcher-64 checksum (design §3.1).

use super::bytes::{le_u32, le_u64, len64};
use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};

/// Size of the object header.
pub(crate) const HEADER_LEN: usize = 32;

/// Low 16 bits of `o_type`: the object's type.
pub(crate) const TYPE_MASK: u32 = 0x0000_FFFF;
/// Storage flag: the object lives in the checkpoint data area.
#[cfg_attr(not(test), allow(dead_code, reason = "named for the test builder"))]
pub(crate) const OBJ_EPHEMERAL: u32 = 0x8000_0000;
/// Storage flag: the object id is the block address.
#[cfg_attr(not(test), allow(dead_code, reason = "named for the test builder"))]
pub(crate) const OBJ_PHYSICAL: u32 = 0x4000_0000;
/// Storage flag: the object is stored without a header (§3.5).
pub(crate) const OBJ_NOHEADER: u32 = 0x2000_0000;

/// Container superblock.
pub(crate) const TYPE_NX_SUPERBLOCK: u16 = 0x1;
/// B-tree root node.
pub(crate) const TYPE_BTREE: u16 = 0x2;
/// B-tree non-root node.
pub(crate) const TYPE_BTREE_NODE: u16 = 0x3;
/// Space manager.
pub(crate) const TYPE_SPACEMAN: u16 = 0x5;
/// Space manager chunk-info block.
pub(crate) const TYPE_SPACEMAN_CIB: u16 = 0x7;
/// Space manager chunk-info address block.
pub(crate) const TYPE_SPACEMAN_CAB: u16 = 0x6;
/// Object map.
pub(crate) const TYPE_OMAP: u16 = 0xB;
/// Checkpoint map.
pub(crate) const TYPE_CHECKPOINT_MAP: u16 = 0xC;
/// Volume superblock.
pub(crate) const TYPE_FS: u16 = 0xD;
/// File-system tree (a B-tree subtype).
pub(crate) const TYPE_FSTREE: u16 = 0xE;
/// Snapshot metadata tree (a B-tree subtype).
pub(crate) const TYPE_SNAPMETATREE: u16 = 0x10;
/// Object map tree (a B-tree subtype).
pub(crate) const TYPE_OMAP_TREE: u16 = 0xB;
/// Sealed-volume file-extent tree (a B-tree subtype).
pub(crate) const TYPE_FEXT_TREE: u16 = 0x1F;

/// `2^32 - 1`, the Fletcher-64 modulus.
const MOD: u64 = 0xFFFF_FFFF;
/// Words summed between reductions; see the proof in [`fletcher64`].
const REDUCE_EVERY: usize = 1024;

/// The Fletcher-64 checksum APFS stores in `o_cksum`, over `block[8..]` read
/// as little-endian `u32` words. Trailing bytes that do not fill a word are
/// ignored (a block is always a multiple of 4096 bytes).
///
/// A valid checksum is never 0: each half is `MOD - (x mod MOD)`, which lies
/// in `1 ..= MOD`.
pub(crate) fn fletcher64(block: &[u8]) -> u64 {
    let data = block.get(8..).unwrap_or(&[]);
    let (words, _rest) = data.as_chunks::<4>();
    let mut sum1: u64 = 0;
    let mut sum2: u64 = 0;
    // Entering a run both sums are below 2^32. After k <= 1024 words, each
    // below 2^32: sum1 < 2^32 + 1024 * 2^32 < 2^43 and
    // sum2 < 2^32 + 1024 * 2^43 < 2^54. Neither wraps, so the wrapping adds
    // below are plain adds.
    for run in words.chunks(REDUCE_EVERY) {
        for word in run {
            sum1 = sum1.wrapping_add(u64::from(u32::from_le_bytes(*word)));
            sum2 = sum2.wrapping_add(sum1);
        }
        sum1 %= MOD;
        sum2 %= MOD;
    }
    // Both below MOD, so the sums below stay under 2^33.
    let c1 = MOD.wrapping_sub(sum1.wrapping_add(sum2) % MOD);
    let c2 = MOD.wrapping_sub(sum1.wrapping_add(c1) % MOD);
    (c2 << 32) | c1
}

/// The fields of an object header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ObjHeader {
    /// `o_cksum`.
    pub cksum: u64,
    /// `o_oid`.
    pub oid: u64,
    /// `o_xid`: the transaction that wrote the object.
    pub xid: u64,
    /// `o_type`, type and storage flags together.
    pub type_and_flags: u32,
    /// `o_subtype`.
    pub subtype: u32,
}

impl ObjHeader {
    /// Reads the header at the start of `block`, without checking anything.
    pub fn read(block: &[u8]) -> Option<Self> {
        Some(Self {
            cksum: le_u64(block, 0)?,
            oid: le_u64(block, 8)?,
            xid: le_u64(block, 16)?,
            type_and_flags: le_u32(block, 24)?,
            subtype: le_u32(block, 28)?,
        })
    }

    /// The object's type, without the storage flags.
    pub fn kind(&self) -> u16 {
        // Masked to 16 bits: always fits.
        u16::try_from(self.type_and_flags & TYPE_MASK).unwrap_or(u16::MAX)
    }
}

fn corrupt(structure: &'static str, paddr: u64, kind: CorruptKind) -> Error {
    Error::Corrupt(Corruption::new(structure, Location::Block(paddr), kind))
}

/// Checks the block read from `paddr`: its checksum, that its type is one of
/// `types`, and, when the caller knows it, its object id. `structure` names
/// what was being read in the error.
///
/// Never called for a block stored without a header (§3.5).
pub(crate) fn verify(
    block: &[u8],
    paddr: u64,
    structure: &'static str,
    types: &[u16],
    oid: Option<u64>,
) -> Result<ObjHeader> {
    let header = ObjHeader::read(block).ok_or_else(|| {
        corrupt(
            structure,
            paddr,
            CorruptKind::Truncated {
                needed: len64(HEADER_LEN),
                available: len64(block.len()),
            },
        )
    })?;
    let computed = fletcher64(block);
    if header.cksum == 0 || header.cksum != computed {
        return Err(corrupt(
            structure,
            paddr,
            CorruptKind::BadSignature {
                expected: computed,
                found: header.cksum,
            },
        ));
    }
    if !types.contains(&header.kind()) {
        return Err(corrupt(
            structure,
            paddr,
            CorruptKind::InvalidValue {
                field: "object type",
                value: u64::from(header.type_and_flags),
            },
        ));
    }
    if let Some(want) = oid
        && header.oid != want
    {
        return Err(corrupt(
            structure,
            paddr,
            CorruptKind::InvalidValue {
                field: "object id",
                value: header.oid,
            },
        ));
    }
    Ok(header)
}

/// Writes the checksum of `block` into its first 8 bytes. For the test
/// builder and fuzz seeds.
#[cfg(any(test, feature = "test-util"))]
pub(crate) fn seal(block: &mut [u8]) {
    let sum = fletcher64(block);
    block[..8].copy_from_slice(&sum.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object(kind: u16, oid: u64) -> Vec<u8> {
        let mut block = vec![0u8; 4096];
        block[8..16].copy_from_slice(&oid.to_le_bytes());
        block[16..24].copy_from_slice(&7u64.to_le_bytes());
        block[24..28].copy_from_slice(&(u32::from(kind) | OBJ_PHYSICAL).to_le_bytes());
        for (i, b) in block.iter_mut().enumerate().skip(32) {
            *b = (i * 31 % 251) as u8;
        }
        seal(&mut block);
        block
    }

    #[test]
    fn fletcher_of_known_words() {
        // Hand-computed: words 1, 2 after the checksum field (o_oid = 1 + 2<<32).
        let mut block = [0u8; 16];
        block[8..12].copy_from_slice(&1u32.to_le_bytes());
        block[12..16].copy_from_slice(&2u32.to_le_bytes());
        // sum1 = 3, sum2 = 1 + 3 = 4; c1 = MOD - 7; c2 = MOD - ((3 + c1) mod MOD) = 4.
        assert_eq!(fletcher64(&block), (4u64 << 32) | (MOD - 7));
        // An all-zero block sums to MOD in both halves, never to 0.
        assert_eq!(fletcher64(&[0u8; 4096]), u64::MAX);
    }

    #[test]
    fn reduction_matches_per_word_reduction() {
        // A block of maximal words exercises the deferred reduction.
        let mut block = vec![0xFFu8; 65536];
        block[..8].fill(0);
        let mut s1: u64 = 0;
        let mut s2: u64 = 0;
        for w in block[8..].as_chunks::<4>().0 {
            s1 = (s1 + u64::from(u32::from_le_bytes(*w))) % MOD;
            s2 = (s2 + s1) % MOD;
        }
        let c1 = MOD - ((s1 + s2) % MOD);
        let c2 = MOD - ((s1 + c1) % MOD);
        assert_eq!(fletcher64(&block), (c2 << 32) | c1);
    }

    #[test]
    fn verify_accepts_a_sealed_block_and_rejects_damage() {
        let block = object(TYPE_OMAP, 42);
        let header = verify(&block, 42, "object map", &[TYPE_OMAP], Some(42)).unwrap();
        assert_eq!((header.oid, header.xid, header.kind()), (42, 7, TYPE_OMAP));

        // Every single-bit flip after the checksum field is caught.
        for byte in [8usize, 31, 32, 2000, 4095] {
            for bit in [0u8, 3, 7] {
                let mut bad = block.clone();
                bad[byte] ^= 1 << bit;
                assert!(verify(&bad, 42, "x", &[TYPE_OMAP], None).is_err());
            }
        }
        // Wrong type, wrong oid, zero checksum, truncated.
        assert!(verify(&block, 42, "x", &[TYPE_FS], None).is_err());
        assert!(verify(&block, 42, "x", &[TYPE_OMAP], Some(43)).is_err());
        let mut zero = block.clone();
        zero[..8].fill(0);
        assert!(verify(&zero, 42, "x", &[TYPE_OMAP], None).is_err());
        assert!(verify(&block[..20], 42, "x", &[TYPE_OMAP], None).is_err());
    }
}
