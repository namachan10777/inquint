//! Interned string symbols.
//!
//! Names appear everywhere in the IR and in runtime values (record fields,
//! variant labels, string literals). Interning them makes equality and
//! hashing a `u32` comparison, and `Ord` remains string order so that any
//! container ordered by name enumerates in the same order as the previous
//! `Arc<str>` representation (Display, ITF traces, `pick` order).
//!
//! The table is a process-global concurrent interner: interning (rare —
//! essentially only while loading the IR) takes one mutex; `as_str` reads
//! are lock-free through an append-only [`Slab`]; `Ord` reads a rank
//! snapshot published through an `AtomicPtr` (rebuilt lazily after new
//! interns, then immutable — old snapshots are leaked). Interned strings
//! are leaked, so `as_str` hands out `&'static str`.

use crate::slab::Slab;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::cmp::Ordering;
use std::fmt;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicPtr, AtomicU32, Ordering as AO};
use std::sync::Mutex;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Symbol(NonZeroU32);

struct SymbolStore {
    map: Mutex<FxHashMap<&'static str, Symbol>>,
    strs: Slab<&'static str>,
    len: AtomicU32,
    /// String-order rank snapshot (leaked; null until first build). Covers
    /// exactly `ranks_len` symbols — reads that need more rebuild it.
    ranks: AtomicPtr<u32>,
    ranks_len: AtomicU32,
}

static STORE: SymbolStore = SymbolStore {
    map: Mutex::new(FxHashMap::with_hasher(rustc_hash::FxBuildHasher)),
    strs: Slab::new(),
    len: AtomicU32::new(0),
    ranks: AtomicPtr::new(std::ptr::null_mut()),
    ranks_len: AtomicU32::new(0),
};

impl SymbolStore {
    /// Rank snapshot covering at least `need` symbols, rebuilding once
    /// under the intern lock if the current one is stale.
    fn ranks_for(&self, need: u32) -> *const u32 {
        loop {
            if self.ranks_len.load(AO::Acquire) >= need {
                return self.ranks.load(AO::Acquire);
            }
            let _guard = self.map.lock().unwrap();
            if self.ranks_len.load(AO::Acquire) >= need {
                continue; // rebuilt while we waited for the lock
            }
            let len = self.len.load(AO::Acquire);
            let mut order: Vec<u32> = (0..len).collect();
            order.sort_unstable_by_key(|&i| *self.strs.get(i));
            let mut ranks = vec![0u32; len as usize];
            for (rank, &i) in order.iter().enumerate() {
                ranks[i as usize] = rank as u32;
            }
            let leaked: &'static mut [u32] = Box::leak(ranks.into_boxed_slice());
            self.ranks.store(leaked.as_mut_ptr(), AO::Release);
            self.ranks_len.store(len, AO::Release);
        }
    }
}

impl Symbol {
    pub fn intern(s: &str) -> Symbol {
        let mut map = STORE.map.lock().unwrap();
        if let Some(&sym) = map.get(s) {
            return sym;
        }
        let leaked: &'static str = Box::leak(s.into());
        let idx = STORE.len.load(AO::Relaxed);
        let sym = Symbol(NonZeroU32::new(idx + 1).unwrap());
        // Safety: idx is claimed under the lock; the release store of
        // `len` (and the lock release) publish the slot before anyone can
        // learn this symbol.
        unsafe { STORE.strs.write(idx, leaked) };
        STORE.len.store(idx + 1, AO::Release);
        map.insert(leaked, sym);
        sym
    }

    pub fn as_str(self) -> &'static str {
        STORE.strs.get(self.0.get() - 1)
    }
}

impl std::ops::Deref for Symbol {
    type Target = str;
    fn deref(&self) -> &'static str {
        self.as_str()
    }
}

impl AsRef<str> for Symbol {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl From<&str> for Symbol {
    fn from(s: &str) -> Self {
        Symbol::intern(s)
    }
}

impl From<String> for Symbol {
    fn from(s: String) -> Self {
        Symbol::intern(&s)
    }
}

impl PartialOrd for Symbol {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// String order, not id order: keeps name-keyed containers enumerating in
/// the same order as the old `Arc<str>` representation. Compares via the
/// rank snapshot (no string traversal, no lock on the hot path).
impl Ord for Symbol {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.0 == other.0 {
            return Ordering::Equal;
        }
        let need = self.0.get().max(other.0.get());
        let ranks = STORE.ranks_for(need);
        // Safety: the snapshot covers `need` symbols and is immutable.
        let (ra, rb) = unsafe {
            (
                *ranks.add(self.0.get() as usize - 1),
                *ranks.add(other.0.get() as usize - 1),
            )
        };
        ra.cmp(&rb)
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

impl<'de> Deserialize<'de> for Symbol {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V;
        impl serde::de::Visitor<'_> for V {
            type Value = Symbol;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a string")
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Symbol, E> {
                Ok(Symbol::intern(v))
            }
        }
        deserializer.deserialize_str(V)
    }
}

impl Serialize for Symbol {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_dedups() {
        let a = Symbol::intern("foo");
        let b = Symbol::intern("foo");
        let c = Symbol::intern("bar");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.as_str(), "foo");
    }

    #[test]
    fn ord_is_string_order() {
        let z = Symbol::intern("zzz");
        let a = Symbol::intern("aaa");
        assert!(a < z);
        assert_eq!(a.cmp(&a), Ordering::Equal);
        // interning after a rank rebuild still orders correctly
        let m = Symbol::intern("mmm");
        assert!(a < m && m < z);
    }

    #[test]
    fn deserialize_interns() {
        let s: Symbol = serde_json::from_str("\"hello\"").unwrap();
        assert_eq!(s, Symbol::intern("hello"));
    }
}
