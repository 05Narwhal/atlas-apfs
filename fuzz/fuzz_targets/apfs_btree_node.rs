// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Fuzzes the APFS B-tree node parser on one block, in both layouts.
//! Errors are fine; a panic is the bug.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    atlas_apfs::fuzzing::btree_node(data);
});
