# atlas-apfs

A read-only APFS reader. It implements `atlas_fs_core::fs::FileSystem`, so each volume of a container lists
folders, looks up names, reads file content and extended attributes, and scans the whole file-system tree through
the same trait as the other `atlas-*` readers.

- Container superblock from the newest valid checkpoint, container and volume object maps, space manager.
- Case-sensitive and case-insensitive volumes, sealed (system) volumes, sparse and cloned extents.
- Snapshots, opened as read-only views.
- decmpfs compressed files, through `atlas-hfsplus`.
- Encrypted volumes are not read. A volume that cannot be read is reported as failed beside the others.
- No `unsafe`. Every length, offset and count read from the disk is checked; damage is an error, not a panic.

```rust,no_run
use std::sync::Arc;
use atlas_fs_core::fs::{FileSystem, VolumeState};
use atlas_fs_core::io::block::BlockSource;
use atlas_fs_core::io::file::FileSource;

let source: Arc<dyn BlockSource> = Arc::new(FileSource::open("container.img".as_ref())?);
for volume in atlas_apfs::mount(source)? {
    if let VolumeState::Ready(fs) = volume.state {
        let mut dir = fs.read_dir(fs.root())?;
    }
}
# Ok::<(), atlas_fs_core::error::Error>(())
```

The `test-util` feature exposes the test container builder; it is for tests of crates that use this one and is
not part of the stable API.

## Licence

Licensed under either of Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE)) or MIT license
([LICENSE-MIT](LICENSE-MIT)) at your option.
