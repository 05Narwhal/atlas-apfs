// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::ControlFlow;
use std::sync::Arc;

use super::*;
use crate::mount;
use crate::test_container::{
    BuiltContainer, ContainerBuilder, FsSpec, TIMES, VolumeSpec, corrupt_block,
};
use atlas_fs_core::fs::{VolumeRole, VolumeState};
use atlas_fs_core::io::block::MemSource;
use atlas_hfsplus::compression::fixtures as cmp;

fn build(volumes: Vec<VolumeSpec>) -> BuiltContainer {
    ContainerBuilder::new().volumes(volumes).build()
}

fn ready(image: Vec<u8>) -> Vec<Result<Arc<dyn FileSystem>>> {
    mount(Arc::new(MemSource::new(image)))
        .unwrap()
        .into_iter()
        .map(|v| match v.state {
            VolumeState::Ready(fs) => Ok(fs),
            VolumeState::Failed(e) => Err(e),
            _ => panic!("no locked volumes here"),
        })
        .collect()
}

fn one(spec: VolumeSpec) -> Arc<dyn FileSystem> {
    ready(build(vec![spec]).image).pop().unwrap().unwrap()
}

fn list(fs: &dyn FileSystem, dir: u64) -> Vec<(String, u64, NodeKind, Option<u64>)> {
    let mut stream = fs.read_dir(NodeId(dir)).unwrap();
    let mut out = Vec::new();
    let mut batch = DirBatch::new();
    loop {
        batch.clear();
        let progress = stream.next_batch(&mut batch, 3).unwrap();
        out.extend(
            batch
                .iter()
                .map(|e| (e.name.to_owned(), e.node.0, e.kind, e.size)),
        );
        if progress == DirProgress::Done {
            return out;
        }
    }
}

fn read_all(source: &dyn BlockSource) -> Vec<u8> {
    let mut out = vec![0u8; usize::try_from(source.len()).unwrap()];
    // Odd-sized reads cross block and run boundaries.
    for (i, chunk) in out.chunks_mut(5000).enumerate() {
        source.read_at(i as u64 * 5000, chunk).unwrap();
    }
    out
}

/// A volume with one of everything; returns it with the ids.
struct Rich {
    spec: VolumeSpec,
    docs: u64,
    big: u64,
    link: u64,
    inline: u64,
    forked: u64,
    content: Vec<u8>,
    plain: Vec<u8>,
}

fn rich(incompatible: u64) -> Rich {
    let mut fs = FsSpec::new();
    let docs = fs.folder(2, "Documents");
    let content: Vec<u8> = (0..50_000u32).map(|i| (i * 7 % 253) as u8).collect();
    let big = fs.file(docs, "big.bin", &content);
    fs.item(big).fragments = 3;
    let mut finder = [0u8; 32];
    finder[..8].copy_from_slice(b"TEXTttxt");
    finder[8..10].copy_from_slice(&0x0008u16.to_be_bytes()); // label 4
    fs.xattr(big, "com.apple.FinderInfo", &finder, false);
    fs.xattr(big, "user.small", b"hello", false);
    fs.xattr(big, "user.large", &vec![0xAB; 9000], true);
    fs.xattr(big, "com.apple.ResourceFork", &vec![0xCD; 5000], true);
    let link = fs.symlink(2, "link", "Documents/big.bin");
    let plain = cmp::content(8_000);
    // Type 3: zlib in the attribute.
    let mut inline_x = cmp::header(3, plain.len() as u64);
    inline_x.extend_from_slice(&cmp::zlib(&plain));
    let inline = fs.compressed(docs, "inline.txt", inline_x, None);
    // Type 8: LZVN chunks in the resource fork; size from the inode.
    let chunks: Vec<Vec<u8>> = cmp::pieces(&content)
        .into_iter()
        .map(cmp::lzvn_literals)
        .collect();
    let forked = fs.compressed(
        docs,
        "forked.bin",
        cmp::header(8, content.len() as u64),
        Some(cmp::offsets_fork(&chunks)),
    );
    fs.item(forked).internal_flags = INODE_HAS_UNCOMPRESSED_SIZE;
    fs.item(forked).uncompressed_size = content.len() as u64;
    fs.file(2, ".hidden", b"x");
    let mut spec = VolumeSpec::new("Macintosh HD");
    spec.incompatible = incompatible;
    spec.role = 0x40;
    spec.fs = fs;
    Rich {
        spec,
        docs,
        big,
        link,
        inline,
        forked,
        content,
        plain,
    }
}

#[test]
fn lists_describes_and_reads_a_volume() {
    let r = rich(0);
    let fs = one(r.spec.clone());
    let info = fs.info();
    assert_eq!(
        (info.name.as_str(), info.format),
        ("Macintosh HD", Some("APFS"))
    );
    assert!(info.resource_forks);

    let root = list(&*fs, 2);
    let names: BTreeSet<_> = root.iter().map(|e| e.0.as_str()).collect();
    assert_eq!(names, BTreeSet::from(["Documents", "link", ".hidden"]));
    let docs = list(&*fs, r.docs);
    let by_name: BTreeMap<_, _> = docs.iter().map(|e| (e.0.as_str(), e)).collect();
    assert_eq!(by_name["big.bin"].3, Some(50_000));
    assert_eq!(
        by_name["inline.txt"].3,
        Some(8_000),
        "size from the decmpfs header"
    );
    assert_eq!(by_name["forked.bin"].3, Some(50_000), "size from the inode");

    // Metadata.
    let m = fs.metadata(NodeId(r.big)).unwrap();
    assert_eq!(m.kind, NodeKind::File);
    assert_eq!(m.size, Some(50_000));
    assert_eq!(m.resource_size, Some(5000));
    assert_eq!(
        m.modified,
        Some(Timestamp::from_unix_nanos(TIMES[1] as i64))
    );
    assert_eq!(m.finder_codes.unwrap().file_type, *b"TEXT");
    assert_eq!(m.attributes, Some(3), "FinderInfo, user.small, user.large");
    assert_eq!(m.flags.label(), 4, "label from the Finder flags");
    let m = fs.metadata(NodeId(r.link)).unwrap();
    assert_eq!(m.kind, NodeKind::Symlink);
    assert_eq!(m.link_target.as_deref(), Some("Documents/big.bin"));
    let m = fs.metadata(NodeId(r.forked)).unwrap();
    assert!(m.flags.contains(EntryFlags::COMPRESSED));
    assert_eq!(m.resource_size, Some(0));
    let hidden = root.iter().find(|e| e.0 == ".hidden").unwrap().1;
    assert!(
        fs.metadata(NodeId(hidden))
            .unwrap()
            .flags
            .contains(EntryFlags::HIDDEN)
    );
    assert!(matches!(
        fs.metadata(NodeId(9999)),
        Err(Error::NotFound { node: 9999 })
    ));

    // Content: fragmented, compressed both ways, the fork, the link.
    assert_eq!(
        read_all(&*fs.open(NodeId(r.big), Stream::Data).unwrap()),
        r.content
    );
    assert_eq!(
        read_all(&*fs.open(NodeId(r.inline), Stream::Data).unwrap()),
        r.plain
    );
    assert_eq!(
        read_all(&*fs.open(NodeId(r.forked), Stream::Data).unwrap()),
        r.content
    );
    assert_eq!(
        read_all(&*fs.open(NodeId(r.big), Stream::Resource).unwrap()),
        vec![0xCD; 5000]
    );
    assert_eq!(
        read_all(&*fs.open(NodeId(r.link), Stream::Data).unwrap()),
        b"Documents/big.bin"
    );
    assert!(matches!(
        fs.open(NodeId(r.forked), Stream::Resource),
        Err(Error::NoSuchStream { .. })
    ));
    assert!(matches!(
        fs.open(NodeId(r.docs), Stream::Data),
        Err(Error::IsADirectory { .. })
    ));
    let source = fs.open(NodeId(r.big), Stream::Data).unwrap();
    assert!(
        source.read_at(49_999, &mut [0u8; 2]).is_err(),
        "past the end"
    );

    // Extents: 13 blocks in three runs that do not touch.
    let ext = fs.extents(NodeId(r.big), Stream::Data).unwrap();
    assert_eq!(ext.len(), 3);
    assert_eq!(ext.iter().map(|e| e.len).sum::<u64>(), 13 * 4096);

    // Attributes.
    let names: BTreeSet<_> = fs
        .xattrs(NodeId(r.big))
        .unwrap()
        .into_iter()
        .map(|x| (x.name, x.size))
        .collect();
    assert_eq!(
        names,
        BTreeSet::from([
            ("com.apple.FinderInfo".to_owned(), 32),
            ("user.large".to_owned(), 9000),
            ("user.small".to_owned(), 5)
        ])
    );
    assert_eq!(
        fs.read_xattr(NodeId(r.big), "user.small", 100).unwrap(),
        Some(b"hello".to_vec())
    );
    assert_eq!(
        fs.read_xattr(NodeId(r.big), "user.large", 9000).unwrap(),
        Some(vec![0xAB; 9000])
    );
    assert!(matches!(
        fs.read_xattr(NodeId(r.big), "user.large", 8999),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(
        fs.read_xattr(NodeId(r.big), "com.apple.ResourceFork", 1 << 20)
            .unwrap(),
        None
    );
    assert_eq!(fs.read_xattr(NodeId(r.big), "nope", 10).unwrap(), None);

    // Lookup is exact on a sensitive volume.
    assert_eq!(
        fs.lookup(NodeId(2), "Documents").unwrap(),
        Some(NodeId(r.docs))
    );
    assert_eq!(fs.lookup(NodeId(2), "documents").unwrap(), None);
    assert!(matches!(
        fs.read_dir(NodeId(r.big)),
        Err(Error::NotADirectory { .. })
    ));
}

#[test]
fn insensitive_volumes_use_hashed_keys_and_fold_names() {
    let mut fs = FsSpec::new();
    let cafe = fs.file(2, "Caf\u{e9}", b"1");
    let mut spec = VolumeSpec::new("ci");
    spec.incompatible = 0x1 | 0x8;
    spec.fs = fs;
    let fs = one(spec);
    assert_eq!(list(&*fs, 2)[0].0, "Caf\u{e9}");
    assert_eq!(
        fs.lookup(NodeId(2), "Caf\u{e9}").unwrap(),
        Some(NodeId(cafe))
    );
    assert_eq!(
        fs.lookup(NodeId(2), "CAFE\u{301}").unwrap(),
        Some(NodeId(cafe))
    );
    assert_eq!(fs.lookup(NodeId(2), "Cafe").unwrap(), None);
}

#[test]
fn deep_trees_list_fully_and_scan_matches_a_walk() {
    let mut fs = FsSpec::new();
    let mut folders = vec![2u64];
    for i in 0..30 {
        let parent = folders[i % folders.len()];
        folders.push(fs.folder(parent, &format!("folder {i}")));
    }
    for i in 0..300 {
        let parent = folders[i % folders.len()];
        fs.file(parent, &format!("file {i:03}"), &[i as u8; 10]);
    }
    let mut spec = VolumeSpec::new("deep");
    spec.fs = fs;
    spec.per_node = 4;
    let vol = one(spec);

    let mut walked = BTreeSet::new();
    let mut queue = vec![2u64];
    while let Some(dir) = queue.pop() {
        for (name, id, kind, _) in list(&*vol, dir) {
            if kind == NodeKind::Directory {
                queue.push(id);
            }
            walked.insert((id, dir, name));
        }
    }
    assert_eq!(walked.len(), 330);
    let mut scanned = BTreeSet::new();
    let summary = vol
        .scan(&mut |e| {
            scanned.insert((e.node.0, e.parent.0, e.name.to_owned()));
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(scanned, walked);
    assert_eq!((summary.entries, summary.skipped), (330, 0));
}

#[test]
fn a_damaged_volume_fails_alone_and_policy_names_unsupported_features() {
    let good = rich(0x8).spec;
    let mut newer = VolumeSpec::new("newer");
    newer.incompatible = 0x8000;
    let mut encrypted = VolumeSpec::new("locked");
    encrypted.fs_flags = 0;
    let mut rolling = VolumeSpec::new("rolling");
    rolling.incompatible = 0x4;
    let built = build(vec![good, newer, encrypted, rolling]);
    let vols = ready(built.image.clone());
    assert!(vols[0].is_ok());
    for v in &vols[1..] {
        assert!(
            matches!(v, Err(Error::Unsupported(_))),
            "{:?}",
            v.as_ref().err()
        );
    }

    // Damage the first volume's superblock: it fails, the others are unchanged.
    let mut image = built.image;
    corrupt_block(&mut image, built.volume_paddrs[0], 4096);
    let vols = ready(image);
    assert!(matches!(vols[0], Err(Error::Corrupt(_))));
    assert!(matches!(vols[1], Err(Error::Unsupported(_))));
}

#[test]
fn damage_anywhere_in_a_volume_never_panics() {
    let r = rich(0x8);
    let built = build(vec![r.spec]);
    let blocks = built.image.len() / 4096;
    for paddr in 0..blocks as u64 {
        let mut image = built.image.clone();
        corrupt_block(&mut image, paddr, 4096);
        let Ok(vols) = mount(Arc::new(MemSource::new(image))) else {
            continue;
        };
        for v in vols {
            if let VolumeState::Ready(fs) = v.state {
                let _ = fs.scan(&mut |_| ControlFlow::Continue(()));
                if let Ok(mut s) = fs.read_dir(fs.root()) {
                    let mut b = DirBatch::new();
                    let _ = s.next_batch(&mut b, 100);
                    for e in b.iter() {
                        let _ = fs.metadata(e.node);
                        if let Ok(src) = fs.open(e.node, Stream::Data) {
                            let mut buf =
                                vec![0u8; usize::try_from(src.len().min(70_000)).unwrap()];
                            let _ = src.read_at(0, &mut buf);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn truncated_content_is_corrupt_unless_the_file_is_sparse() {
    let mut fs = FsSpec::new();
    let short = fs.file(2, "short", &[1u8; 100]);
    fs.item(short).size_override = Some(10_000);
    let sparse = fs.file(2, "sparse", &[1u8; 100]);
    fs.item(sparse).size_override = Some(10_000);
    fs.item(sparse).internal_flags = INODE_IS_SPARSE;
    let mut spec = VolumeSpec::new("v");
    spec.fs = fs;
    let vol = one(spec);
    let mut buf = vec![0u8; 10_000];
    let src = vol.open(NodeId(short), Stream::Data).unwrap();
    src.read_at(0, &mut buf[..4096]).unwrap();
    assert!(matches!(src.read_at(0, &mut buf), Err(Error::Corrupt(_))));
    let src = vol.open(NodeId(sparse), Stream::Data).unwrap();
    src.read_at(0, &mut buf).unwrap();
    assert_eq!(&buf[..100], &[1u8; 100]);
    assert!(buf[4096..].iter().all(|&b| b == 0));
}

#[test]
fn a_container_names_and_mounts_its_volumes() {
    let r = rich(0);
    let image = build(vec![r.spec, VolumeSpec::new("Second")]).image;
    let vols = mount(Arc::new(MemSource::new(image))).unwrap();
    assert_eq!(vols.len(), 2);
    assert_eq!(vols[0].name, "Macintosh HD");
    assert_eq!(vols[0].role, Some(VolumeRole::Data));
    assert_eq!(vols[1].name, "Second");
    assert_eq!(vols[1].role, None);
}

#[test]
fn the_standard_seed_volume_mounts_and_reads() {
    use crate::test_container::standard_volume;
    for incompatible in [0, 0x9] {
        let fs = one(standard_volume("Atlas", incompatible));
        let mut count = 0;
        fs.scan(&mut |e| {
            if e.kind == NodeKind::File
                && let Ok(src) = fs.open(e.node, Stream::Data)
            {
                read_all(&*src);
            }
            count += 1;
            ControlFlow::Continue(())
        })
        .unwrap();
        assert_eq!(count, 19);
    }
}

/// Every (path, content) of a volume, by walking it.
fn contents(fs: &dyn FileSystem) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    let mut queue = vec![(2u64, String::new())];
    while let Some((dir, path)) = queue.pop() {
        for (name, id, kind, _) in list(fs, dir) {
            let path = format!("{path}/{name}");
            match kind {
                NodeKind::Directory => queue.push((id, path)),
                _ => {
                    let data = read_all(&*fs.open(NodeId(id), Stream::Data).unwrap());
                    out.insert(path, data);
                }
            }
        }
    }
    out
}

#[test]
fn sealed_volumes_read_extents_from_their_own_tree() {
    use crate::test_container::standard_volume;
    let plain = one(standard_volume("Plain", 0x8));
    let mut spec = standard_volume("Sealed", 0x8);
    spec.sealed = true;
    let built = build(vec![spec]);
    // The file-system tree really is stored without headers.
    let image = &built.image;
    let vols = ready(image.clone());
    let sealed = vols[0].as_ref().unwrap();
    assert_eq!(contents(&**sealed), contents(&*plain));
    let headerless = image
        .chunks(4096)
        .filter(|b| {
            b[..32].iter().all(|&x| x == 0) && u16::from_le_bytes([b[32], b[33]]) & 0x10 != 0
        })
        .count();
    assert!(
        headerless >= 2,
        "no-header nodes in the image: {headerless}"
    );
}

#[test]
fn snapshots_list_and_open_at_their_transaction() {
    use crate::test_container::SnapSpec;
    // Then: "kept" with old content and "gone". Now: "kept" changed, "new".
    let mut then = FsSpec::new();
    then.file(2, "kept", b"old content");
    then.file(2, "gone", b"deleted since");
    let mut now = FsSpec::new();
    now.file(2, "kept", b"new content, longer");
    now.file(2, "new", b"added since");
    let mut spec = VolumeSpec::new("Data");
    spec.fs = now;
    spec.snapshots = vec![SnapSpec {
        name: "com.apple.TimeMachine.2026-11-01".into(),
        xid: 3,
        fs: then,
    }];
    let live = one(spec);
    let snaps = live.snapshots().unwrap();
    assert_eq!(snaps.len(), 1);
    assert_eq!(snaps[0].id, 3);
    assert_eq!(snaps[0].name, "com.apple.TimeMachine.2026-11-01");
    assert!(snaps[0].created.is_some());

    let old = live.open_snapshot(3).unwrap();
    assert_eq!(
        old.info().name,
        "Data \u{2014} com.apple.TimeMachine.2026-11-01"
    );
    assert_ne!(
        old.identity().unwrap().change,
        live.identity().unwrap().change
    );
    let want_old = BTreeMap::from([
        ("/gone".to_owned(), b"deleted since".to_vec()),
        ("/kept".to_owned(), b"old content".to_vec()),
    ]);
    let want_now = BTreeMap::from([
        ("/kept".to_owned(), b"new content, longer".to_vec()),
        ("/new".to_owned(), b"added since".to_vec()),
    ]);
    assert_eq!(contents(&*old), want_old);
    assert_eq!(contents(&*live), want_now);
    assert!(matches!(live.open_snapshot(4), Err(Error::NotFound { .. })));
    // A volume without snapshots lists none.
    assert!(one(VolumeSpec::new("v")).snapshots().unwrap().is_empty());
}

#[test]
fn damage_to_a_sealed_volume_or_its_snapshot_never_panics() {
    use crate::test_container::sealed_with_snapshot;
    let built = build(vec![sealed_with_snapshot()]);
    let vols = ready(built.image.clone());
    let fs = vols[0].as_ref().unwrap();
    assert_eq!(fs.snapshots().unwrap().len(), 1);
    assert_eq!(contents(&*fs.open_snapshot(3).unwrap()).len(), 1);
    for paddr in 0..(built.image.len() / 4096) as u64 {
        let mut image = built.image.clone();
        corrupt_block(&mut image, paddr, 4096);
        let Ok(vols) = mount(Arc::new(MemSource::new(image))) else {
            continue;
        };
        for v in vols {
            let VolumeState::Ready(fs) = v.state else {
                continue;
            };
            let mut views = vec![Arc::clone(&fs)];
            if let Ok(snap) = fs.open_snapshot(3) {
                views.push(snap);
            }
            for view in views {
                let _ = view.scan(&mut |e| {
                    if let Ok(src) = view.open(e.node, Stream::Data) {
                        let mut buf = vec![0u8; usize::try_from(src.len().min(40_000)).unwrap()];
                        let _ = src.read_at(0, &mut buf);
                    }
                    ControlFlow::Continue(())
                });
            }
        }
    }
}
