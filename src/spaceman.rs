// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! The space manager (design §6): free space, and the disk strip's density
//! from each chunk's free count rather than from the bitmaps.

use super::bytes::{le_u32, le_u64, len64};
use super::container::Container;
use super::object::{self, TYPE_SPACEMAN, TYPE_SPACEMAN_CAB, TYPE_SPACEMAN_CIB};
use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};
use atlas_fs_core::fs::{AllocationMap, MAX_ALLOCATION_BUCKETS};

/// Most chunk-info blocks walked.
pub(crate) const MAX_CIB: u64 = 1_048_576;
/// Largest space manager object read.
const MAX_SPACEMAN_BYTES: u64 = 16 << 20;

const STRUCTURE: &str = "space manager";

fn corrupt(location: Location, kind: CorruptKind) -> Error {
    Error::Corrupt(Corruption::new(STRUCTURE, location, kind))
}

fn limit(field: &'static str, value: u64, limit_name: &'static str, limit: u64) -> Error {
    corrupt(
        Location::Unspecified,
        CorruptKind::OutOfRange {
            field,
            value,
            limit_name,
            limit,
        },
    )
}

/// The main device's record of the space manager.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Spaceman {
    pub blocks_per_chunk: u32,
    pub block_count: u64,
    pub chunk_count: u64,
    pub cib_count: u32,
    pub cab_count: u32,
    /// Free blocks on the main device.
    pub free_count: u64,
    /// The chunk-info (or address) block addresses.
    addresses: Vec<u64>,
}

/// Reads and verifies the container's space manager.
pub(crate) fn read(container: &Container) -> Result<Spaceman> {
    let oid = container.superblock.spaceman_oid;
    let mapping = container.ephemeral(oid).ok_or_else(|| {
        corrupt(
            Location::Object(oid),
            CorruptKind::Other("not in the checkpoint maps"),
        )
    })?;
    let bs = u64::from(container.blocks.block_size);
    let size = u64::from(mapping.size);
    if size < bs || size > MAX_SPACEMAN_BYTES || size.checked_rem(bs) != Some(0) {
        return Err(limit(
            "object size",
            size,
            "MAX_SPACEMAN_BYTES",
            MAX_SPACEMAN_BYTES,
        ));
    }
    let mut raw = vec![0u8; usize::try_from(size).unwrap_or(0)];
    container.blocks.read(mapping.paddr, &mut raw)?;
    object::verify(&raw, mapping.paddr, STRUCTURE, &[TYPE_SPACEMAN], Some(oid))?;
    let short = || {
        corrupt(
            Location::Block(mapping.paddr),
            CorruptKind::Truncated {
                needed: 88,
                available: len64(raw.len()),
            },
        )
    };
    let block_size = le_u32(&raw, 32).ok_or_else(short)?;
    if block_size != container.blocks.block_size {
        return Err(corrupt(
            Location::Block(mapping.paddr),
            CorruptKind::InvalidValue {
                field: "block size",
                value: u64::from(block_size),
            },
        ));
    }
    let blocks_per_chunk = le_u32(&raw, 36).ok_or_else(short)?;
    let block_count = le_u64(&raw, 48).ok_or_else(short)?;
    let chunk_count = le_u64(&raw, 56).ok_or_else(short)?;
    let cib_count = le_u32(&raw, 64).ok_or_else(short)?;
    let cab_count = le_u32(&raw, 68).ok_or_else(short)?;
    let free_count = le_u64(&raw, 72).ok_or_else(short)?;
    let addr_offset = le_u32(&raw, 80).ok_or_else(short)?;

    if blocks_per_chunk == 0 {
        return Err(limit("blocks per chunk", 0, "at least", 1));
    }
    let max_chunks = block_count
        .checked_div(u64::from(blocks_per_chunk))
        .unwrap_or(0)
        .saturating_add(1);
    if chunk_count > max_chunks {
        return Err(limit(
            "chunk count",
            chunk_count,
            "blocks / blocks per chunk + 1",
            max_chunks,
        ));
    }
    if free_count > block_count {
        return Err(limit("free blocks", free_count, "blocks", block_count));
    }
    if u64::from(cib_count) > MAX_CIB {
        return Err(limit(
            "chunk-info blocks",
            u64::from(cib_count),
            "MAX_CIB",
            MAX_CIB,
        ));
    }
    let count = if cab_count == 0 { cib_count } else { cab_count };
    let mut addresses = Vec::with_capacity(usize::try_from(count).unwrap_or(0).min(4096));
    for i in 0..usize::try_from(count).unwrap_or(0) {
        let at = usize::try_from(addr_offset)
            .unwrap_or(usize::MAX)
            .saturating_add(i.saturating_mul(8));
        addresses.push(le_u64(&raw, at).ok_or_else(short)?);
    }
    Ok(Spaceman {
        blocks_per_chunk,
        block_count,
        chunk_count,
        cib_count,
        cab_count,
        free_count,
        addresses,
    })
}

/// One chunk: its first block, its size and its free blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Chunk {
    first: u64,
    blocks: u32,
    free: u32,
}

/// Every chunk-info block's address, through the address blocks if any.
fn cib_addresses(container: &Container, sm: &Spaceman) -> Result<Vec<u64>> {
    if sm.cab_count == 0 {
        return Ok(sm.addresses.clone());
    }
    let mut out = Vec::new();
    for &paddr in &sm.addresses {
        let block = container.blocks.block(paddr)?;
        object::verify(&block, paddr, STRUCTURE, &[TYPE_SPACEMAN_CAB], None)?;
        let count = u64::from(le_u32(&block, 36).unwrap_or(0));
        let per_block = len64(block.len().saturating_sub(40) / 8);
        if count > per_block || len64(out.len()).saturating_add(count) > u64::from(sm.cib_count) {
            return Err(limit(
                "chunk-info blocks in an address block",
                count,
                "room",
                per_block,
            ));
        }
        for i in 0..usize::try_from(count).unwrap_or(0) {
            if let Some(a) = le_u64(&block, 40usize.saturating_add(i.saturating_mul(8))) {
                out.push(a);
            }
        }
    }
    Ok(out)
}

fn chunks(container: &Container, sm: &Spaceman) -> Result<Vec<Chunk>> {
    let mut out = Vec::new();
    for paddr in cib_addresses(container, sm)? {
        let block = container.blocks.block(paddr)?;
        object::verify(&block, paddr, STRUCTURE, &[TYPE_SPACEMAN_CIB], None)?;
        let count = u64::from(le_u32(&block, 36).unwrap_or(0));
        let per_block = len64(block.len().saturating_sub(40) / 32);
        if count > per_block || len64(out.len()).saturating_add(count) > sm.chunk_count {
            return Err(limit(
                "chunks in a chunk-info block",
                count,
                "room",
                per_block,
            ));
        }
        for i in 0..usize::try_from(count).unwrap_or(0) {
            let at = 40usize.saturating_add(i.saturating_mul(32));
            let (Some(first), Some(blocks), Some(free)) = (
                le_u64(&block, at.saturating_add(8)),
                le_u32(&block, at.saturating_add(16)),
                le_u32(&block, at.saturating_add(20)),
            ) else {
                continue;
            };
            out.push(Chunk {
                first,
                blocks,
                free,
            });
        }
    }
    Ok(out)
}

/// How full the container is in up to `buckets` equal slices (§6). Every
/// volume of a container reports this same map.
pub(crate) fn allocation_map(container: &Container, buckets: usize) -> Result<AllocationMap> {
    let sm = read(container)?;
    let total = container.blocks.block_count;
    if total == 0 {
        return Err(limit("blocks", 0, "at least", 1));
    }
    let n = buckets
        .clamp(1, MAX_ALLOCATION_BUCKETS)
        .min(usize::try_from(total).unwrap_or(usize::MAX));
    let n64 = len64(n);
    // i <= 4096 and total < 2^64 / 4096 for any real container; saturating
    // keeps a hostile count from wrapping.
    let bound = |i: u64| -> u64 {
        u64::try_from(
            u128::from(i)
                .saturating_mul(u128::from(total))
                .checked_div(u128::from(n64))
                .unwrap_or(0),
        )
        .unwrap_or(total)
    };
    // Used blocks per bucket, scaled: a chunk spreads its used count evenly.
    let mut used = vec![0u128; n];
    for chunk in chunks(container, &sm)? {
        let blocks = u64::from(chunk.blocks);
        if blocks == 0 {
            continue;
        }
        // A free count past the chunk's size is damage: treat it as full.
        let used_blocks = if chunk.free <= chunk.blocks {
            u128::from(chunk.blocks.saturating_sub(chunk.free))
        } else {
            u128::from(blocks)
        };
        let start = chunk.first.min(total);
        let end = chunk.first.saturating_add(blocks).min(total);
        if start >= end || used_blocks == 0 {
            continue;
        }
        // The buckets the chunk overlaps, from the one holding `start`.
        // Bounded: at most n passes.
        let first = u128::from(start)
            .saturating_mul(u128::from(n64))
            .checked_div(u128::from(total))
            .unwrap_or(0);
        let mut i = u64::try_from(first).unwrap_or(n64);
        while i < n64 {
            let (lo, hi) = (bound(i), bound(i.saturating_add(1)));
            if lo >= end {
                break;
            }
            let overlap = u128::from(hi.min(end).saturating_sub(lo.max(start)));
            if let Some(slot) = used.get_mut(usize::try_from(i).unwrap_or(usize::MAX)) {
                let share = used_blocks
                    .saturating_mul(overlap)
                    .checked_div(u128::from(blocks))
                    .unwrap_or(0);
                // Any used space in the overlap shows: at least one block.
                *slot = slot.saturating_add(share.max(u128::from(overlap > 0)));
            }
            i = i.saturating_add(1);
        }
    }
    let density = used
        .iter()
        .enumerate()
        .map(|(i, &u)| {
            let size = bound(len64(i).saturating_add(1))
                .saturating_sub(bound(len64(i)))
                .max(1);
            let v = u
                .min(u128::from(size))
                .saturating_mul(255)
                .div_ceil(u128::from(size));
            u8::try_from(v).unwrap_or(u8::MAX)
        })
        .collect();
    Ok(AllocationMap::new(
        0,
        total.saturating_mul(u64::from(container.blocks.block_size)),
        density,
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::test_container::{BS, ContainerBuilder, corrupt_block};
    use atlas_fs_core::io::block::MemSource;

    fn open(image: Vec<u8>) -> Arc<Container> {
        Container::open(Arc::new(MemSource::new(image))).unwrap()
    }

    #[test]
    fn free_space_and_density_follow_the_chunks() {
        // 256 blank blocks at the end: free.
        let built = ContainerBuilder::new().free_blocks(256).build();
        let total = (built.image.len() / BS) as u64;
        let c = open(built.image.clone());
        let sm = read(&c).unwrap();
        assert_eq!(sm.free_count, 256);
        assert!(sm.chunk_count > 1);
        let map = allocation_map(&c, total as usize).unwrap();
        assert_eq!(map.len, total * BS as u64);
        // One bucket per block: used blocks are 255, the free tail is 0.
        let used = (total - 256) as usize;
        assert!(map.density[..used - 8].iter().all(|&d| d == 255));
        assert!(map.density[used + 8..].iter().all(|&d| d == 0));
        // Two buckets: every used block is in the first half.
        let two = allocation_map(&c, 2).unwrap().density;
        let half = total / 2;
        let want = ((total - 256) * 255).div_ceil(half) as u8;
        assert_eq!(two, vec![want, 0]);
    }

    #[test]
    fn damaged_space_manager_fails_alone() {
        let built = ContainerBuilder::new().free_blocks(16).build();
        let mut image = built.image;
        corrupt_block(&mut image, built.spaceman_paddr, BS);
        let c = open(image);
        assert!(read(&c).is_err());
        assert!(allocation_map(&c, 64).is_err());
    }
}
