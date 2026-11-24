// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! File-system tree records (design §3.7): key headers, the comparators the
//! reader seeks with, and the value parsers. Every parser returns `None` on
//! a record too short for what it claims; callers turn that into an error
//! or skip the record.

use std::cmp::Ordering;

use super::bytes::{array, bytes, le_u16, le_u32, le_u64};
use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};

/// Low 60 bits of a key header: the object id.
pub(crate) const OBJ_ID_MASK: u64 = 0x0FFF_FFFF_FFFF_FFFF;
/// The record type is the top 4 bits.
const TYPE_SHIFT: u32 = 60;

/// Record types.
pub(crate) const T_SNAP_METADATA: u8 = 1;
pub(crate) const T_INODE: u8 = 3;
pub(crate) const T_XATTR: u8 = 4;
pub(crate) const T_FILE_EXTENT: u8 = 8;
pub(crate) const T_DIR_REC: u8 = 9;

/// Inode internal flag: the uncompressed size field is valid.
pub(crate) const INODE_HAS_UNCOMPRESSED_SIZE: u64 = 0x0004_0000;
/// BSD flag: compressed (decmpfs).
pub(crate) const UF_COMPRESSED: u32 = 0x20;
/// BSD flag: hidden.
pub(crate) const UF_HIDDEN: u32 = 0x8000;

/// Xattr flag: the data is a stream descriptor.
pub(crate) const XATTR_DATA_STREAM: u16 = 0x1;
/// Xattr flag: the data is embedded in the record.
pub(crate) const XATTR_DATA_EMBEDDED: u16 = 0x2;

/// Extended field types of an inode.
const INO_EXT_TYPE_NAME: u8 = 4;
const INO_EXT_TYPE_DSTREAM: u8 = 8;
/// Most extended fields parsed.
pub(crate) const MAX_XFIELDS: usize = 64;

/// File extent flag (top byte of the length): preallocated, reads as zeros.
pub(crate) const FEXT_PREALLOCATED: u8 = 0x02;
/// Low 56 bits of a file extent's length field.
pub(crate) const FEXT_LEN_MASK: u64 = 0x00FF_FFFF_FFFF_FFFF;

/// Low 10 bits of a hashed directory key's `name_len_and_hash`.
const DREC_LEN_MASK: u32 = 0x3FF;

/// The (object id, record type) at the start of every key.
pub(crate) fn key_header(key: &[u8]) -> Option<(u64, u8)> {
    let raw = le_u64(key, 0)?;
    // The shift leaves 4 bits.
    Some((
        raw & OBJ_ID_MASK,
        u8::try_from(raw >> TYPE_SHIFT).unwrap_or(0),
    ))
}

fn short_key() -> Error {
    Error::Corrupt(Corruption::new(
        "file-system tree",
        Location::Unspecified,
        CorruptKind::Other("key is too short"),
    ))
}

/// Orders a key against "the first record of type `kind` of object `id`":
/// keys of that object and type compare greater, so a seek lands on the
/// first of them.
pub(crate) fn first_of(id: u64, kind: u8) -> impl FnMut(&[u8]) -> Result<Ordering> {
    move |key: &[u8]| {
        let (kid, ktype) = key_header(key).ok_or_else(short_key)?;
        Ok((kid, ktype).cmp(&(id, kind)).then(Ordering::Greater))
    }
}

/// Text of a stored name: the bytes before the first NUL, as UTF-8 with
/// invalid sequences replaced.
pub(crate) fn name_text(stored: &[u8]) -> String {
    let end = stored.iter().position(|&b| b == 0).unwrap_or(stored.len());
    String::from_utf8_lossy(stored.get(..end).unwrap_or(&[])).into_owned()
}

/// A directory record key's name bytes (with the NUL, if stored).
pub(crate) fn drec_name(key: &[u8], hashed: bool) -> Option<&[u8]> {
    if hashed {
        let len = le_u32(key, 8)? & DREC_LEN_MASK;
        bytes(key, 12, usize::try_from(len).ok()?)
    } else {
        let len = le_u16(key, 8)?;
        bytes(key, 10, usize::from(len))
    }
}

/// A directory record's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DirRec {
    /// The inode the entry names.
    pub file_id: u64,
    /// When it was added to the folder.
    pub date_added: u64,
    /// Low 4 bits: a `DT_*` type.
    pub kind: u16,
}

impl DirRec {
    pub fn parse(value: &[u8]) -> Option<Self> {
        Some(Self {
            file_id: le_u64(value, 0)?,
            date_added: le_u64(value, 8)?,
            kind: le_u16(value, 16)? & 0xF,
        })
    }
}

/// Directory record types (`DT_*`).
pub(crate) const DT_DIR: u16 = 4;
pub(crate) const DT_REG: u16 = 8;
pub(crate) const DT_LNK: u16 = 10;

/// The data stream extended field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct DStream {
    /// Bytes of content.
    pub size: u64,
    /// Bytes allocated.
    pub alloc_size: u64,
}

impl DStream {
    pub fn parse(b: &[u8]) -> Option<Self> {
        Some(Self {
            size: le_u64(b, 0)?,
            alloc_size: le_u64(b, 8)?,
        })
    }
}

/// Size of an inode value before its extended fields.
pub(crate) const INODE_FIXED_LEN: usize = 92;

/// An inode value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Inode {
    pub parent: u64,
    /// The id its file extents are keyed by.
    pub private_id: u64,
    pub created: u64,
    pub modified: u64,
    pub changed: u64,
    pub accessed: u64,
    pub internal_flags: u64,
    pub bsd_flags: u32,
    pub owner: u32,
    pub group: u32,
    pub mode: u16,
    pub uncompressed_size: u64,
    /// Extended field 4, without the NUL.
    pub name: Option<String>,
    /// Extended field 8.
    pub dstream: Option<DStream>,
}

impl Inode {
    pub fn parse(value: &[u8]) -> Option<Self> {
        let mut inode = Self {
            parent: le_u64(value, 0)?,
            private_id: le_u64(value, 8)?,
            created: le_u64(value, 16)?,
            modified: le_u64(value, 24)?,
            changed: le_u64(value, 32)?,
            accessed: le_u64(value, 40)?,
            internal_flags: le_u64(value, 48)?,
            bsd_flags: le_u32(value, 68)?,
            owner: le_u32(value, 72)?,
            group: le_u32(value, 76)?,
            mode: le_u16(value, 80)?,
            uncompressed_size: le_u64(value, 84)?,
            name: None,
            dstream: None,
        };
        for (kind, data) in xfields(value.get(INODE_FIXED_LEN..).unwrap_or(&[])) {
            match kind {
                INO_EXT_TYPE_NAME => inode.name = Some(name_text(data)),
                INO_EXT_TYPE_DSTREAM => inode.dstream = DStream::parse(data),
                _ => {}
            }
        }
        Some(inode)
    }

    /// True when the content is stored by decmpfs.
    pub fn is_compressed(&self) -> bool {
        self.bsd_flags & UF_COMPRESSED != 0
    }
}

/// The extended fields of a blob: (type, data) pairs. At most
/// [`MAX_XFIELDS`]; a field that runs past the end stops the parse and the
/// fields before it are kept.
pub(crate) fn xfields(blob: &[u8]) -> Vec<(u8, &[u8])> {
    let mut out = Vec::new();
    let (Some(count), Some(_used)) = (le_u16(blob, 0), le_u16(blob, 2)) else {
        return out;
    };
    let count = usize::from(count).min(MAX_XFIELDS);
    // Descriptors at 4, data after them.
    let mut data_at = 4usize.saturating_add(count.saturating_mul(4));
    for i in 0..count {
        let at = 4usize.saturating_add(i.saturating_mul(4));
        let (Some(kind), Some(size)) = (blob.get(at).copied(), le_u16(blob, at.saturating_add(2)))
        else {
            break;
        };
        let size = usize::from(size);
        let Some(data) = bytes(blob, data_at, size) else {
            break;
        };
        out.push((kind, data));
        // Each field is padded to 8 bytes.
        data_at = data_at.saturating_add(size.next_multiple_of(8));
    }
    out
}

/// An extended attribute's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum XattrData<'a> {
    /// Stored in the record.
    Embedded(&'a [u8]),
    /// Stored in blocks keyed by `id`, `size` bytes long.
    Stream {
        /// The object id its file extents are keyed by.
        id: u64,
        /// Bytes of content.
        size: u64,
    },
}

impl XattrData<'_> {
    /// The value's size in bytes.
    pub fn size(&self) -> u64 {
        match self {
            Self::Embedded(data) => u64::try_from(data.len()).unwrap_or(u64::MAX),
            Self::Stream { size, .. } => *size,
        }
    }
}

/// An xattr key's name, without the NUL.
pub(crate) fn xattr_name(key: &[u8]) -> Option<&[u8]> {
    let len = usize::from(le_u16(key, 8)?);
    let stored = bytes(key, 10, len)?;
    let end = stored.iter().position(|&b| b == 0).unwrap_or(stored.len());
    stored.get(..end)
}

/// Parses an xattr value. `None` when it is too short or its flags name
/// neither storage.
pub(crate) fn xattr_data(value: &[u8]) -> Option<XattrData<'_>> {
    let flags = le_u16(value, 0)?;
    let len = usize::from(le_u16(value, 2)?);
    let data = bytes(value, 4, len)?;
    if flags & XATTR_DATA_STREAM != 0 {
        Some(XattrData::Stream {
            id: le_u64(data, 0)?,
            size: DStream::parse(data.get(8..)?)?.size,
        })
    } else if flags & XATTR_DATA_EMBEDDED != 0 {
        Some(XattrData::Embedded(data))
    } else {
        None
    }
}

/// A file extent record: key's logical address and the value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FileExtent {
    /// Byte offset in the stream.
    pub logical: u64,
    /// Bytes covered.
    pub len: u64,
    /// The top 8 bits of the length field.
    pub flags: u8,
    /// First block; 0 is a hole.
    pub paddr: u64,
    /// Tweak base for encrypted data.
    pub crypto_id: u64,
}

impl FileExtent {
    /// From a file-system tree record (type 8).
    pub fn parse(key: &[u8], value: &[u8]) -> Option<Self> {
        let len_and_flags = le_u64(value, 0)?;
        Some(Self {
            logical: le_u64(key, 8)?,
            len: len_and_flags & FEXT_LEN_MASK,
            flags: u8::try_from(len_and_flags >> 56).unwrap_or(0),
            paddr: le_u64(value, 8)?,
            crypto_id: le_u64(value, 16)?,
        })
    }

    /// From a sealed volume's file-extent tree record: key (private id,
    /// logical address), value (length and flags, physical block).
    pub fn parse_sealed(key: &[u8], value: &[u8]) -> Option<(u64, Self)> {
        let len_and_flags = le_u64(value, 0)?;
        Some((
            le_u64(key, 0)?,
            Self {
                logical: le_u64(key, 8)?,
                len: len_and_flags & FEXT_LEN_MASK,
                flags: u8::try_from(len_and_flags >> 56).unwrap_or(0),
                paddr: le_u64(value, 8)?,
                crypto_id: 0,
            },
        ))
    }
}

/// Finder information from 32 bytes, or `None` when all zero.
pub(crate) fn finder_info(data: &[u8]) -> Option<[u8; 32]> {
    array::<32>(data, 0).filter(|b| b.iter().any(|&x| x != 0))
}

#[cfg(any(test, feature = "test-util"))]
pub(crate) mod encode {
    //! Encoders for the test builder: the inverse of the parsers above.

    use super::*;

    pub fn key(id: u64, kind: u8) -> Vec<u8> {
        ((u64::from(kind) << TYPE_SHIFT) | id)
            .to_le_bytes()
            .to_vec()
    }

    pub fn inode_key(id: u64) -> Vec<u8> {
        key(id, T_INODE)
    }

    /// A directory record key. Hashed keys carry a made-up hash: the reader
    /// never computes one (decision 4); only the order within a folder
    /// depends on it, and the builder sorts by the bytes it writes.
    pub fn drec_key(parent: u64, name: &str, hashed: bool) -> Vec<u8> {
        let mut k = key(parent, T_DIR_REC);
        let mut stored = name.as_bytes().to_vec();
        stored.push(0);
        if hashed {
            let hash = name
                .bytes()
                .fold(0x2u32, |h, b| h.wrapping_mul(31).wrapping_add(u32::from(b)))
                & 0x3F_FFFF;
            k.extend_from_slice(&((hash << 10) | stored.len() as u32).to_le_bytes());
        } else {
            k.extend_from_slice(&(stored.len() as u16).to_le_bytes());
        }
        k.extend_from_slice(&stored);
        k
    }

    pub fn drec_value(file_id: u64, kind: u16) -> Vec<u8> {
        let mut v = file_id.to_le_bytes().to_vec();
        v.extend_from_slice(&1_600_000_000_000_000_000u64.to_le_bytes());
        v.extend_from_slice(&kind.to_le_bytes());
        v
    }

    /// Extended fields from (type, data) pairs.
    pub fn xfields(fields: &[(u8, Vec<u8>)]) -> Vec<u8> {
        let mut out = (fields.len() as u16).to_le_bytes().to_vec();
        let used: usize = fields.iter().map(|f| f.1.len().next_multiple_of(8)).sum();
        out.extend_from_slice(&(used as u16).to_le_bytes());
        for (kind, data) in fields {
            out.extend_from_slice(&[*kind, 0]);
            out.extend_from_slice(&(data.len() as u16).to_le_bytes());
        }
        for (_, data) in fields {
            out.extend_from_slice(data);
            out.resize(out.len() + (data.len().next_multiple_of(8) - data.len()), 0);
        }
        out
    }

    /// Everything an inode value holds that the tests vary.
    #[derive(Clone)]
    pub struct InodeSpec {
        pub parent: u64,
        pub private_id: u64,
        pub mode: u16,
        pub bsd_flags: u32,
        pub internal_flags: u64,
        pub uncompressed_size: u64,
        pub name: Option<String>,
        pub size: Option<u64>,
        pub times: [u64; 4],
        pub nchildren: u32,
    }

    pub fn inode_value(spec: &InodeSpec) -> Vec<u8> {
        let mut v = vec![0u8; INODE_FIXED_LEN];
        v[0..8].copy_from_slice(&spec.parent.to_le_bytes());
        v[8..16].copy_from_slice(&spec.private_id.to_le_bytes());
        for (i, t) in spec.times.iter().enumerate() {
            v[16 + 8 * i..24 + 8 * i].copy_from_slice(&t.to_le_bytes());
        }
        v[48..56].copy_from_slice(&spec.internal_flags.to_le_bytes());
        v[56..60].copy_from_slice(&spec.nchildren.to_le_bytes());
        v[68..72].copy_from_slice(&spec.bsd_flags.to_le_bytes());
        v[72..76].copy_from_slice(&501u32.to_le_bytes());
        v[76..80].copy_from_slice(&20u32.to_le_bytes());
        v[80..82].copy_from_slice(&spec.mode.to_le_bytes());
        v[84..92].copy_from_slice(&spec.uncompressed_size.to_le_bytes());
        let mut fields = Vec::new();
        if let Some(name) = &spec.name {
            let mut n = name.as_bytes().to_vec();
            n.push(0);
            fields.push((INO_EXT_TYPE_NAME, n));
        }
        if let Some(size) = spec.size {
            let mut d = size.to_le_bytes().to_vec();
            d.extend_from_slice(&size.next_multiple_of(4096).to_le_bytes());
            d.extend_from_slice(&[0u8; 24]);
            fields.push((INO_EXT_TYPE_DSTREAM, d));
        }
        if !fields.is_empty() {
            v.extend_from_slice(&xfields(&fields));
        }
        v
    }

    pub fn xattr_key(id: u64, name: &str) -> Vec<u8> {
        let mut k = key(id, T_XATTR);
        let mut stored = name.as_bytes().to_vec();
        stored.push(0);
        k.extend_from_slice(&(stored.len() as u16).to_le_bytes());
        k.extend_from_slice(&stored);
        k
    }

    pub fn xattr_embedded(data: &[u8]) -> Vec<u8> {
        let mut v = XATTR_DATA_EMBEDDED.to_le_bytes().to_vec();
        v.extend_from_slice(&(data.len() as u16).to_le_bytes());
        v.extend_from_slice(data);
        v
    }

    pub fn xattr_stream(id: u64, size: u64) -> Vec<u8> {
        let mut v = XATTR_DATA_STREAM.to_le_bytes().to_vec();
        v.extend_from_slice(&48u16.to_le_bytes());
        v.extend_from_slice(&id.to_le_bytes());
        v.extend_from_slice(&size.to_le_bytes());
        v.extend_from_slice(&size.next_multiple_of(4096).to_le_bytes());
        v.extend_from_slice(&[0u8; 24]);
        v
    }

    pub fn extent_key(id: u64, logical: u64) -> Vec<u8> {
        let mut k = key(id, T_FILE_EXTENT);
        k.extend_from_slice(&logical.to_le_bytes());
        k
    }

    pub fn extent_value(len: u64, flags: u8, paddr: u64) -> Vec<u8> {
        let mut v = ((u64::from(flags) << 56) | len).to_le_bytes().to_vec();
        v.extend_from_slice(&paddr.to_le_bytes());
        v.extend_from_slice(&paddr.to_le_bytes());
        v
    }
}

#[cfg(test)]
mod tests {
    use super::encode::*;
    use super::*;

    fn spec() -> InodeSpec {
        InodeSpec {
            parent: 2,
            private_id: 20,
            mode: 0o100644,
            bsd_flags: UF_HIDDEN,
            internal_flags: 0,
            uncompressed_size: 0,
            name: Some("caf\u{e9}.txt".into()),
            size: Some(12_345),
            times: [1, 2, 3, 4],
            nchildren: 1,
        }
    }

    #[test]
    fn inode_round_trip_and_truncation() {
        let value = inode_value(&spec());
        let inode = Inode::parse(&value).unwrap();
        assert_eq!(inode.parent, 2);
        assert_eq!(inode.private_id, 20);
        assert_eq!(inode.mode, 0o100644);
        assert_eq!((inode.created, inode.modified, inode.accessed), (1, 2, 4));
        assert_eq!(inode.name.as_deref(), Some("caf\u{e9}.txt"));
        assert_eq!(inode.dstream.map(|d| d.size), Some(12_345));
        // Shorter than the fixed part: None. Cut inside the extended fields:
        // the fields before the cut survive, never a panic.
        assert!(Inode::parse(&value[..INODE_FIXED_LEN - 1]).is_none());
        for len in INODE_FIXED_LEN..value.len() {
            let cut = Inode::parse(&value[..len]).unwrap();
            assert!(cut.dstream.is_none() || len == value.len());
        }
    }

    #[test]
    fn xfields_are_bounded() {
        // A count of 65535 with no room: nothing parsed, no panic.
        let mut blob = vec![0xFF, 0xFF, 0, 0];
        blob.extend_from_slice(&[4, 0, 0xFF, 0xFF]);
        assert!(super::xfields(&blob).is_empty());
        // At most MAX_XFIELDS are read.
        let many: Vec<(u8, Vec<u8>)> = (0..100).map(|i| (i as u8, vec![i as u8])).collect();
        assert_eq!(super::xfields(&encode::xfields(&many)).len(), MAX_XFIELDS);
    }

    #[test]
    fn keys_order_and_parse() {
        let k = drec_key(2, "Hello", true);
        assert_eq!(key_header(&k), Some((2, T_DIR_REC)));
        assert_eq!(drec_name(&k, true), Some(&b"Hello\0"[..]));
        let k = drec_key(2, "Hello", false);
        assert_eq!(drec_name(&k, false), Some(&b"Hello\0"[..]));
        assert_eq!(drec_name(&k[..12], false), None);
        // first_of puts every key of (2, 9) after the target.
        let mut cmp = first_of(2, T_DIR_REC);
        assert_eq!(cmp(&k).unwrap(), Ordering::Greater);
        assert_eq!(cmp(&inode_key(2)).unwrap(), Ordering::Less);
        assert_eq!(cmp(&inode_key(3)).unwrap(), Ordering::Greater);
        assert!(cmp(&[1, 2, 3]).is_err());
    }

    #[test]
    fn xattr_values() {
        let v = xattr_embedded(b"abc");
        assert_eq!(xattr_data(&v), Some(XattrData::Embedded(b"abc")));
        let v = xattr_stream(77, 100_000);
        assert_eq!(
            xattr_data(&v),
            Some(XattrData::Stream {
                id: 77,
                size: 100_000
            })
        );
        assert_eq!(xattr_data(&v[..20]), None);
        let mut neither = xattr_embedded(b"abc");
        neither[0] = 0;
        assert_eq!(xattr_data(&neither), None);
        let k = xattr_key(9, "com.apple.decmpfs");
        assert_eq!(xattr_name(&k), Some(&b"com.apple.decmpfs"[..]));
        assert_eq!(xattr_name(&k[..11]), None);
    }
}
