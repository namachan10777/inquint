//! Interned string symbols.
//!
//! Names appear everywhere in the IR and in runtime values (record fields,
//! variant labels, string literals). Interning them makes equality and
//! hashing a `u32` comparison, and `Ord` remains string order so that any
//! container ordered by name enumerates in the same order as the previous
//! `Arc<str>` representation (Display, ITF traces, `pick` order).
//!
//! The table is thread-local: the checker is single-threaded, and `cargo
//! test` gets per-thread isolation for free. Interned strings are leaked, so
//! `as_str` hands out `&'static str`.

use rustc_hash::FxHashMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::fmt;
use std::num::NonZeroU32;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Symbol(NonZeroU32);

struct SymbolTable {
    map: FxHashMap<&'static str, Symbol>,
    strs: Vec<&'static str>,
    /// String-order rank per symbol index; rebuilt lazily after interning
    /// (the symbol set is essentially frozen once the IR is loaded, so the
    /// rebuild is amortized to nothing and `Ord` becomes two integer loads).
    ranks: Vec<u32>,
    ranks_dirty: bool,
}

impl SymbolTable {
    fn rebuild_ranks(&mut self) {
        let mut order: Vec<u32> = (0..self.strs.len() as u32).collect();
        order.sort_unstable_by_key(|&i| self.strs[i as usize]);
        self.ranks.resize(self.strs.len(), 0);
        for (rank, &i) in order.iter().enumerate() {
            self.ranks[i as usize] = rank as u32;
        }
        self.ranks_dirty = false;
    }
}

thread_local! {
    static SYMBOLS: RefCell<SymbolTable> = RefCell::new(SymbolTable {
        map: FxHashMap::default(),
        strs: Vec::new(),
        ranks: Vec::new(),
        ranks_dirty: false,
    });
}

impl Symbol {
    pub fn intern(s: &str) -> Symbol {
        SYMBOLS.with_borrow_mut(|t| {
            if let Some(&sym) = t.map.get(s) {
                return sym;
            }
            let leaked: &'static str = Box::leak(s.into());
            let sym = Symbol(NonZeroU32::new(t.strs.len() as u32 + 1).unwrap());
            t.strs.push(leaked);
            t.map.insert(leaked, sym);
            t.ranks_dirty = true;
            sym
        })
    }

    pub fn as_str(self) -> &'static str {
        SYMBOLS.with_borrow(|t| t.strs[self.0.get() as usize - 1])
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
/// cached ranks (no string traversal).
impl Ord for Symbol {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.0 == other.0 {
            return Ordering::Equal;
        }
        SYMBOLS.with_borrow_mut(|t| {
            if t.ranks_dirty {
                t.rebuild_ranks();
            }
            t.ranks[self.0.get() as usize - 1].cmp(&t.ranks[other.0.get() as usize - 1])
        })
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
    }

    #[test]
    fn deserialize_interns() {
        let s: Symbol = serde_json::from_str("\"hello\"").unwrap();
        assert_eq!(s, Symbol::intern("hello"));
    }
}
