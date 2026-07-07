//! Runtime values, hash-consed.
//!
//! Every value is interned into a thread-local [`ValueStore`] and handled
//! as a `Copy` 32-bit id ([`Value`]). Structural equality of normalized
//! values is therefore a single `u32` compare, cloning is free, and hashing
//! a state is a flat hash over ids — no recursive walks.
//!
//! Canonical form (the correctness core):
//! - containers store child ids in a canonical order: `Set`/`Map` sorted by
//!   [`value_cmp`] (the structural order, identical to the pre-hash-consing
//!   `Ord`), `Record` fields in `Symbol` (string) order via an interned
//!   [`RecordShape`];
//! - children are interned before parents, so by induction structurally
//!   equal values always intern to the same id;
//! - enumeration order (`pick`, folds, `Display`, ITF) is the structural
//!   order, unchanged from the previous representation.
//!
//! Symbolic ("lazy") set forms (`Interval`, `CrossProduct`, `PowerSet`,
//! `MapSet`, infinite sets) are interned like everything else but never
//! enter containers or state variables: container constructors normalize
//! their elements. [`value_cmp`] panics on them, as before.
//!
//! Lambdas are compile-time constants registered in a side table; their id
//! identity is the registration, they are never compared or stored in
//! containers.
//!
//! Interned nodes are leaked (`Box::leak`): the store lives for the whole
//! process, so accessors hand out `&'static` data.

use crate::error::{overflow, unsupported, QuintError};
use quint_ast::{QuintName, Symbol};
use quint_ast::slab::Slab;
use rustc_hash::FxHashMap;
use std::borrow::Cow;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::fmt;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};
use std::sync::{LazyLock, Mutex};

/// Enumeration guard: symbolic sets larger than this refuse to materialize.
/// TODO(v2): make configurable via CLI (--max-enum-size).
pub const MAX_ENUM: u64 = 1 << 20;

pub type EvalResult = Result<Value, QuintError>;

/// An interned value: a 32-bit id into the thread-local [`ValueStore`].
///
/// `PartialEq`/`Eq`/`Hash` are id-based: exact structural equality for
/// normalized values (hash-consing guarantees equal ⇔ same id). For
/// possibly-symbolic operands use [`value_eq`]. `Ord` is id order — an
/// arbitrary total order suitable for dedup containers only; user-visible
/// ordering must go through [`value_cmp`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Value(NonZeroU32);

/// The interned node: children are ids, so derived `Eq`/`Hash` are shallow
/// (and, by induction, structural).
#[derive(PartialEq, Eq, Hash, Debug)]
pub enum ValueData {
    Bool(bool),
    Int(i64),
    Str(Symbol),
    /// Sorted by [`value_cmp`], no duplicates.
    Set(Box<[Value]>),
    Tuple(Box<[Value]>),
    List(Box<[Value]>),
    /// Values in shape-field order (shapes are interned, field names sorted).
    Record(&'static RecordShape, Box<[Value]>),
    /// Sorted by key ([`value_cmp`]), unique keys.
    Map(Box<[(Value, Value)]>),
    Variant(Symbol, Value),
    /// Index into the lambda registry.
    Lambda(u32),
    // Symbolic set forms (never stored in containers/states):
    Interval(i64, i64),
    CrossProduct(Box<[Value]>),
    PowerSet(Value),
    MapSet(Value, Value),
    InfiniteInt,
    InfiniteNat,
}

/// An interned record shape: the field names, sorted. Interning makes
/// pointer equality complete, so `Eq` is `ptr::eq`.
#[derive(Debug)]
pub struct RecordShape {
    pub fields: Box<[Symbol]>,
}

impl PartialEq for RecordShape {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}
impl Eq for RecordShape {}
impl std::hash::Hash for RecordShape {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.fields.hash(state);
    }
}

/// A compiled lambda: the body's VM function id plus its parameter slots
/// (bound in the per-worker parameter bank). Created once per `Lambda`
/// IR node at lowering time; contains no shared mutable state, so the
/// store stays Send + Sync.
pub struct LambdaVal {
    pub slots: Box<[u32]>,
    pub fnid: crate::vm::FnId,
}

// ---------------------------------------------------------------------------
// The store
// ---------------------------------------------------------------------------

/// One interned value: its node plus the order-preserving 8-byte digest.
/// Kept together so a `value_cmp` operand costs one cache line.
#[derive(Clone, Copy)]
struct Entry {
    /// If two values' keys differ (and neither is [`NO_ORDER_KEY`]), their
    /// key order equals [`value_cmp`]. Precomputed at intern time so sorts
    /// and binary searches rarely need the deep structural compare.
    key: u64,
    data: &'static ValueData,
}

/// The process-global concurrent value store.
///
/// Reads (`data()`, order keys, comparisons) are lock-free through the
/// append-only [`Slab`]. Interning goes through a per-thread cache first
/// (same cost as the old thread-local store on a hit); a cache miss takes
/// one shard mutex. Truly new values additionally claim an id from the
/// global counter — the only cross-thread write traffic on the hot path.
struct ValueStore {
    /// Intern tables sharded by hash: stores only ids; hashing/comparison
    /// read `entries`. Lookups take borrowed slices (see `intern_seq`
    /// etc.), so an intern *hit* — the common case — allocates nothing.
    shards: [Mutex<hashbrown::HashTable<Value>>; N_SHARDS],
    entries: Slab<Entry>,
    len: AtomicU32,
    shapes: Mutex<FxHashMap<&'static [Symbol], &'static RecordShape>>,
    lambdas: Slab<&'static LambdaVal>,
    lambdas_len: Mutex<u32>,
}

const N_SHARDS: usize = 512;

fn shard_of(hash: u64) -> usize {
    (hash >> 54) as usize & (N_SHARDS - 1)
}

/// Sentinel key for lambdas and symbolic set forms: comparing those is a
/// bug that must keep panicking in the deep path, so they never win the
/// key short-circuit. Unreachable for real keys (their top byte is a
/// discriminant rank ≤ 15).
const NO_ORDER_KEY: u64 = u64::MAX;

/// The first (up to) 7 bytes of a string, big-endian in bits 0..56: byte
/// prefix order equals `str` order (UTF-8 comparison is bytewise).
fn str_prefix7(s: &str) -> u64 {
    let mut k = 0u64;
    for (i, b) in s.bytes().take(7).enumerate() {
        k |= (b as u64) << (48 - 8 * i);
    }
    k
}

/// Container kinds interned from borrowed slices.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SeqKind {
    Set,
    Tuple,
    List,
    CrossProduct,
}

// --- canonical hashing -----------------------------------------------------
// One hash function per logical value, shared by the slice-based lookups
// and the stored `ValueData` (so both sides of the table agree).

fn hash_seq(kind: SeqKind, elems: &[Value]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    let tag: u8 = match kind {
        SeqKind::Set => 3,
        SeqKind::Tuple => 4,
        SeqKind::List => 5,
        SeqKind::CrossProduct => 11,
    };
    tag.hash(&mut h);
    elems.hash(&mut h);
    h.finish()
}

fn hash_map_entries(entries: &[(Value, Value)]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    7u8.hash(&mut h);
    entries.hash(&mut h);
    h.finish()
}

fn hash_record(shape: &'static RecordShape, values: &[Value]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    6u8.hash(&mut h);
    // shapes are interned: the pointer identifies the field list
    (shape as *const RecordShape as usize).hash(&mut h);
    values.hash(&mut h);
    h.finish()
}

fn hash_data(d: &ValueData) -> u64 {
    use std::hash::{Hash, Hasher};
    match d {
        ValueData::Set(s) => return hash_seq(SeqKind::Set, s),
        ValueData::Tuple(s) => return hash_seq(SeqKind::Tuple, s),
        ValueData::List(s) => return hash_seq(SeqKind::List, s),
        ValueData::CrossProduct(s) => return hash_seq(SeqKind::CrossProduct, s),
        ValueData::Map(m) => return hash_map_entries(m),
        ValueData::Record(shape, values) => return hash_record(shape, values),
        _ => {}
    }
    let mut h = rustc_hash::FxHasher::default();
    discriminant_rank(d).hash(&mut h);
    match d {
        ValueData::Bool(b) => b.hash(&mut h),
        ValueData::Int(n) => n.hash(&mut h),
        ValueData::Str(s) => s.hash(&mut h),
        ValueData::Variant(label, v) => {
            label.hash(&mut h);
            v.hash(&mut h);
        }
        ValueData::Lambda(i) => i.hash(&mut h),
        ValueData::Interval(a, b) => {
            a.hash(&mut h);
            b.hash(&mut h);
        }
        ValueData::PowerSet(v) => v.hash(&mut h),
        ValueData::MapSet(a, b) => {
            a.hash(&mut h);
            b.hash(&mut h);
        }
        ValueData::InfiniteInt | ValueData::InfiniteNat => {}
        _ => unreachable!("containers handled above"),
    }
    h.finish()
}

// Reserved ids, in ValueStore::new interning order.
const FALSE_ID: u32 = 1;
const TRUE_ID: u32 = 2;
const SMALL_INT_MIN: i64 = -1024;
const SMALL_INT_MAX: i64 = 1023;
const INT_BASE: u32 = 3; // id of SMALL_INT_MIN

/// Ids of symbolic set forms carry this tag bit, so `is_symbolic` (and the
/// `normalize` fast path, hit on every assignment) needs no store access.
pub(crate) const SYMBOLIC_BIT: u32 = 1 << 31;

impl ValueStore {
    fn new() -> Self {
        let store = ValueStore {
            shards: [const { Mutex::new(hashbrown::HashTable::new()) }; N_SHARDS],
            entries: Slab::new(),
            len: AtomicU32::new(0),
            shapes: Mutex::new(FxHashMap::default()),
            lambdas: Slab::new(),
            lambdas_len: Mutex::new(0),
        };
        let f = store.intern(ValueData::Bool(false));
        let t = store.intern(ValueData::Bool(true));
        debug_assert_eq!((f.0.get(), t.0.get()), (FALSE_ID, TRUE_ID));
        for n in SMALL_INT_MIN..=SMALL_INT_MAX {
            let id = store.intern(ValueData::Int(n));
            debug_assert_eq!(id.0.get(), INT_BASE + (n - SMALL_INT_MIN) as u32);
        }
        store
    }

    /// The entry of an interned value (lock-free).
    #[inline]
    fn entry(&self, v: Value) -> &Entry {
        self.entries.get(v.index() as u32)
    }

    /// Order key of an interned child value.
    fn key_of(&self, v: Value) -> u64 {
        self.entry(v).key
    }

    /// Compute the order-preserving digest for a node whose children are
    /// already interned. Soundness: the digest is `[rank byte | 7-byte
    /// prefix of the first comparand's own key stream]`, and `value_cmp`
    /// orders same-rank values by exactly that first comparand — so a
    /// digest difference always decides the comparison the same way the
    /// deep compare would; digest equality is "undecided" and falls back.
    fn compute_order_key(&self, data: &ValueData) -> u64 {
        let rank = (discriminant_rank(data) as u64) << 56;
        let first_child = |c: Option<&Value>| -> u64 {
            // top 7 bytes of the child's key, shifted below the rank byte
            rank | c.map_or(0, |&e| self.key_of(e) >> 8)
        };
        match data {
            ValueData::Bool(b) => rank | ((*b as u64) << 48),
            ValueData::Int(n) => {
                // Exact (order-preserving, injective) for |n| < 2^54;
                // saturated outside — equal keys there fall back to the
                // deep compare, which stays correct.
                const LO: i64 = -(1 << 54);
                const HI: i64 = (1 << 54) - 1;
                rank | ((*n).clamp(LO, HI) - LO) as u64
            }
            ValueData::Str(s) => rank | str_prefix7(s.as_str()),
            ValueData::Tuple(es) => {
                // Pairs of small ints get an exact lexicographic digest
                // (28 bits each): tuple keys like (i, j) then never fall
                // back to the deep compare in map/set probes.
                const LO2: i64 = -(1 << 27);
                const HI2: i64 = (1 << 27) - 1;
                if let [a, b] = &es[..] {
                    if let (ValueData::Int(x), ValueData::Int(y)) =
                        (self.entry(*a).data, self.entry(*b).data)
                    {
                        if (LO2..=HI2).contains(x) && (LO2..=HI2).contains(y) {
                            return rank
                                | (((x - LO2) as u64) << 28)
                                | ((y - LO2) as u64);
                        }
                    }
                }
                first_child(es.first())
            }
            ValueData::Set(es) | ValueData::List(es) => first_child(es.first()),
            ValueData::Record(shape, _) => {
                rank | shape.fields.first().map_or(0, |f| str_prefix7(f.as_str()))
            }
            ValueData::Map(m) => first_child(m.first().map(|(k, _)| k)),
            ValueData::Variant(label, _) => rank | str_prefix7(label.as_str()),
            ValueData::Lambda(_) => NO_ORDER_KEY,
            _ => {
                debug_assert!(data.is_symbolic());
                NO_ORDER_KEY
            }
        }
    }

    /// The intern core: thread-local cache → hash shard → fresh insert.
    ///
    /// Publication protocol: a fresh entry is written into the slab while
    /// holding the shard lock, before the id is inserted into the shard
    /// map — every path an id can travel to another thread (shard map
    /// lookup, containment in a later entry, frontier handoff) is
    /// separated from the slab write by at least one release/acquire edge.
    fn lookup_or_insert(
        &self,
        hash: u64,
        eq: impl Fn(&ValueData) -> bool,
        make: impl FnOnce() -> ValueData,
    ) -> Value {
        let cached = CACHE.with_borrow(|c| {
            c.find(hash, |&v| eq(self.entry(v).data)).copied()
        });
        if let Some(id) = cached {
            return id;
        }
        let id = {
            let mut map = self.shards[shard_of(hash)].lock().unwrap();
            match map.find(hash, |&v| eq(self.entry(v).data)) {
                Some(&id) => id,
                None => {
                    let data = make();
                    let symbolic = data.is_symbolic();
                    let key = self.compute_order_key(&data);
                    let leaked: &'static ValueData = Box::leak(Box::new(data));
                    let index = self.len.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                    assert!(index < SYMBOLIC_BIT, "value store overflow");
                    // Safety: `index` was claimed uniquely above; the slot
                    // is published by the shard unlock below.
                    unsafe { self.entries.write(index - 1, Entry { key, data: leaked }) };
                    let bits = if symbolic { index | SYMBOLIC_BIT } else { index };
                    let id = Value(NonZeroU32::new(bits).unwrap());
                    map.insert_unique(hash, id, |&v| hash_data(self.entry(v).data));
                    id
                }
            }
        };
        CACHE.with_borrow_mut(|c| {
            c.insert_unique(hash, id, |&v| hash_data(self.entry(v).data));
        });
        id
    }

    fn intern(&self, data: ValueData) -> Value {
        let hash = hash_data(&data);
        // eq reads the not-yet-moved data; make takes it — it rides in a
        // RefCell so both closures can capture it by shared reference.
        let data = RefCell::new(Some(data));
        self.lookup_or_insert(
            hash,
            |cand| data.borrow().as_ref() == Some(cand),
            || data.borrow_mut().take().unwrap(),
        )
    }

    /// Intern a sequence container from a borrowed slice: nothing is
    /// allocated on a hit.
    fn intern_seq(&self, kind: SeqKind, elems: &[Value]) -> Value {
        let hash = hash_seq(kind, elems);
        self.lookup_or_insert(
            hash,
            |cand| match (kind, cand) {
                (SeqKind::Set, ValueData::Set(s)) => &**s == elems,
                (SeqKind::Tuple, ValueData::Tuple(s)) => &**s == elems,
                (SeqKind::List, ValueData::List(s)) => &**s == elems,
                (SeqKind::CrossProduct, ValueData::CrossProduct(s)) => &**s == elems,
                _ => false,
            },
            || match kind {
                SeqKind::Set => ValueData::Set(elems.into()),
                SeqKind::Tuple => ValueData::Tuple(elems.into()),
                SeqKind::List => ValueData::List(elems.into()),
                SeqKind::CrossProduct => ValueData::CrossProduct(elems.into()),
            },
        )
    }

    fn intern_map(&self, pairs: &[(Value, Value)]) -> Value {
        let hash = hash_map_entries(pairs);
        self.lookup_or_insert(
            hash,
            |cand| match cand {
                ValueData::Map(m) => &**m == pairs,
                _ => false,
            },
            || ValueData::Map(pairs.into()),
        )
    }

    fn intern_record(&self, shape: &'static RecordShape, values: &[Value]) -> Value {
        let hash = hash_record(shape, values);
        self.lookup_or_insert(
            hash,
            |cand| match cand {
                ValueData::Record(s, vs) => std::ptr::eq(*s, shape) && &**vs == values,
                _ => false,
            },
            || ValueData::Record(shape, values.into()),
        )
    }

    fn shape(&self, fields: &[Symbol]) -> &'static RecordShape {
        let mut shapes = self.shapes.lock().unwrap();
        if let Some(&s) = shapes.get(fields) {
            return s;
        }
        let shape: &'static RecordShape = Box::leak(Box::new(RecordShape {
            fields: fields.into(),
        }));
        shapes.insert(&shape.fields, shape);
        shape
    }
}

static GLOBAL: LazyLock<ValueStore> = LazyLock::new(ValueStore::new);

thread_local! {
    /// Per-thread intern cache in front of [`GLOBAL`]: hot values hit here
    /// at the same cost as the old thread-local store.
    static CACHE: RefCell<hashbrown::HashTable<Value>> =
        const { RefCell::new(hashbrown::HashTable::new()) };
}

fn intern(data: ValueData) -> Value {
    GLOBAL.intern(data)
}

fn intern_seq(kind: SeqKind, elems: &[Value]) -> Value {
    GLOBAL.intern_seq(kind, elems)
}

// ---------------------------------------------------------------------------
// Small stack buffers: constructor operands and pick indices live on the
// stack for the common small arities, so an intern hit allocates nothing.
// ---------------------------------------------------------------------------

/// Inline-first buffer of values (spills to a `Vec` beyond 16).
pub enum ValueBuf {
    Inline([Value; 16], usize),
    Spill(Vec<Value>),
}

impl ValueBuf {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        ValueBuf::Inline([Value::bool(false); 16], 0)
    }

    pub fn from_slice(s: &[Value]) -> Self {
        let mut buf = ValueBuf::new();
        for &v in s {
            buf.push(v);
        }
        buf
    }

    pub fn push(&mut self, v: Value) {
        match self {
            ValueBuf::Inline(buf, len) => {
                if *len < buf.len() {
                    buf[*len] = v;
                    *len += 1;
                } else {
                    let mut vec = buf.to_vec();
                    vec.push(v);
                    *self = ValueBuf::Spill(vec);
                }
            }
            ValueBuf::Spill(vec) => vec.push(v),
        }
    }

    pub fn as_slice(&self) -> &[Value] {
        match self {
            ValueBuf::Inline(buf, len) => &buf[..*len],
            ValueBuf::Spill(vec) => vec,
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [Value] {
        match self {
            ValueBuf::Inline(buf, len) => &mut buf[..*len],
            ValueBuf::Spill(vec) => vec,
        }
    }

    fn truncate(&mut self, n: usize) {
        match self {
            ValueBuf::Inline(_, len) => *len = n.min(*len),
            ValueBuf::Spill(vec) => vec.truncate(n),
        }
    }

    /// Sort by the structural order and drop duplicates (equal ⇔ same id).
    fn sort_dedup(&mut self) {
        sort_values_structural(self.as_mut_slice());
        let s = self.as_mut_slice();
        let mut w = 0;
        for r in 0..s.len() {
            if w == 0 || s[r] != s[w - 1] {
                s[w] = s[r];
                w += 1;
            }
        }
        self.truncate(w);
    }
}

/// [`value_cmp`] with the store already borrowed: lets a whole binary
/// search or sort run under a single store access.
fn value_cmp_in(s: &ValueStore, a: Value, b: Value) -> Ordering {
    if a == b {
        return Ordering::Equal;
    }
    let (ea, eb) = (*s.entry(a), *s.entry(b));
    if ea.key != eb.key && ea.key != NO_ORDER_KEY && eb.key != NO_ORDER_KEY {
        return ea.key.cmp(&eb.key);
    }
    finish_cmp(value_cmp_deep(ea.data, eb.data))
}

// ---------------------------------------------------------------------------
// Map update memo
// ---------------------------------------------------------------------------

/// Per-thread direct-mapped memo for `Map.set`/`Map.put`: map updates
/// dominate the transition relation in message-passing specs, and the
/// same (map, key, value) update recurs across huge numbers of states.
/// A hit skips the pair-vector rebuild, the full-map hash and the intern
/// (including its shard lock) entirely. Lossy (collisions overwrite) and
/// bounded (1 MiB per thread).
struct UpdateEntry {
    map: u32,
    key: u32,
    val: u32,
    /// 0 = empty slot (value ids are NonZeroU32).
    res: u32,
}

const UPDATE_CACHE_BITS: u32 = 16;

thread_local! {
    static UPDATE_CACHE: RefCell<Box<[UpdateEntry]>> = RefCell::new(
        (0..1usize << UPDATE_CACHE_BITS)
            .map(|_| UpdateEntry { map: 0, key: 0, val: 0, res: 0 })
            .collect(),
    );
}

fn update_slot(map: Value, key: Value, val: Value, must_exist: bool) -> usize {
    // fx-style mix of the three ids + the op tag
    let h = (map.to_bits() as u64)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add((key.to_bits() as u64) << 1)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add((val.to_bits() as u64) << 1 | must_exist as u64)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15);
    (h >> (64 - UPDATE_CACHE_BITS)) as usize
}

/// `Map.put` / `Map.set` with memoization. `key` and `val` must be
/// normalized. `must_exist` selects `set` semantics (error on a missing
/// key — errors are not cached).
pub fn map_update_cached(
    map: Value,
    key: Value,
    val: Value,
    must_exist: bool,
) -> EvalResult {
    let slot = update_slot(map, key, val, must_exist);
    let hit = UPDATE_CACHE.with_borrow(|c| {
        let e = &c[slot];
        if e.res != 0 && e.map == map.to_bits() && e.key == key.to_bits() && e.val == val.to_bits()
        {
            Some(Value::from_bits(e.res))
        } else {
            None
        }
    });
    if let Some(res) = hit {
        return Ok(res);
    }
    let entries = map.as_map();
    let res = match map_find_by_id(entries, key) {
        Some(i) => {
            if entries[i].1 == val {
                map // no-op update: the result is the map itself
            } else {
                let mut out = entries.to_vec();
                out[i].1 = val;
                Value::map_sorted(out)
            }
        }
        None if must_exist => {
            return Err(QuintError::new(
                "QNT507",
                "Called 'set' with a non-existing key",
            ));
        }
        None => {
            let mut out = entries.to_vec();
            let i = map_find(&out, key).unwrap_err();
            out.insert(i, (key, val));
            Value::map_sorted(out)
        }
    };
    UPDATE_CACHE.with_borrow_mut(|c| {
        c[slot] = UpdateEntry {
            map: map.to_bits(),
            key: key.to_bits(),
            val: val.to_bits(),
            res: res.to_bits(),
        };
    });
    Ok(res)
}

/// Position of a (normalized) key in sorted map entries, by id: keys are
/// normalized, so structural equality is id equality — a small map scans
/// with plain `u32` compares (zero store accesses); larger maps binary
/// search. Only for *lookup*; insertion position needs [`map_find`].
pub fn map_find_by_id(entries: &[(Value, Value)], key: Value) -> Option<usize> {
    if entries.len() <= 32 {
        entries.iter().position(|&(k, _)| k == key)
    } else {
        map_find(entries, key).ok()
    }
}

/// Binary search over sorted map entries, one store access for the whole
/// search (probes are mostly `u64` key compares).
pub fn map_find(entries: &[(Value, Value)], key: Value) -> Result<usize, usize> {
    let s = &*GLOBAL;
    entries.binary_search_by(|(k, _)| value_cmp_in(s, *k, key))
}

/// Binary search over a sorted set slice, one store access for the whole
/// search.
pub fn set_find(elems: &[Value], needle: Value) -> Result<usize, usize> {
    let s = &*GLOBAL;
    elems.binary_search_by(|&e| value_cmp_in(s, e, needle))
}

/// Sort values by [`value_cmp`], fetching every element's order key in a
/// single store access up front: the sort itself is then mostly `u64`
/// compares (deep fallback only on key collisions). Elements must be
/// normalized (containers never hold lambdas/symbolic forms).
pub(crate) fn sort_values_structural(vs: &mut [Value]) {
    let n = vs.len();
    if n <= 1 {
        return;
    }
    if n == 2 {
        // no batching machinery for the trivial case
        if value_cmp(vs[0], vs[1]) == Ordering::Greater {
            vs.swap(0, 1);
        }
        return;
    }
    fn sort_pairs(pairs: &mut [(u64, Value)]) {
        pairs.sort_unstable_by(|(ka, a), (kb, b)| ka.cmp(kb).then_with(|| value_cmp(*a, *b)));
    }
    if n <= 16 {
        let mut pairs = [(0u64, Value::bool(false)); 16];
        let s = &*GLOBAL;
        for (i, &v) in vs.iter().enumerate() {
            pairs[i] = (s.entry(v).key, v);
        }
        sort_pairs(&mut pairs[..n]);
        for (dst, (_, v)) in vs.iter_mut().zip(&pairs[..n]) {
            *dst = *v;
        }
    } else {
        let mut pairs: Vec<(u64, Value)> = Vec::with_capacity(n);
        let s = &*GLOBAL;
        pairs.extend(vs.iter().map(|&v| (s.entry(v).key, v)));
        sort_pairs(&mut pairs);
        for (dst, (_, v)) in vs.iter_mut().zip(&pairs) {
            *dst = *v;
        }
    }
}

/// Inline-first buffer of `u64` (pick bounds / choice indices; spills
/// beyond 8 — only deep `setOfMaps` nesting needs that).
pub enum U64Buf {
    Inline([u64; 8], usize),
    Spill(Vec<u64>),
}

impl U64Buf {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        U64Buf::Inline([0; 8], 0)
    }

    pub fn push(&mut self, v: u64) {
        match self {
            U64Buf::Inline(buf, len) => {
                if *len < buf.len() {
                    buf[*len] = v;
                    *len += 1;
                } else {
                    let mut vec = buf.to_vec();
                    vec.push(v);
                    *self = U64Buf::Spill(vec);
                }
            }
            U64Buf::Spill(vec) => vec.push(v),
        }
    }

    pub fn zeros(n: usize) -> Self {
        if n <= 8 {
            U64Buf::Inline([0; 8], n)
        } else {
            U64Buf::Spill(vec![0; n])
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u64] {
        match self {
            U64Buf::Inline(buf, len) => &mut buf[..*len],
            U64Buf::Spill(vec) => vec,
        }
    }
}

impl std::ops::Deref for U64Buf {
    type Target = [u64];
    fn deref(&self) -> &[u64] {
        match self {
            U64Buf::Inline(buf, len) => &buf[..*len],
            U64Buf::Spill(vec) => vec,
        }
    }
}

thread_local! {
    /// Shape lookups happen on every record construction; the global
    /// shapes table is behind a mutex, so cache the (very few) shapes
    /// per thread.
    static SHAPE_CACHE: RefCell<FxHashMap<&'static [Symbol], &'static RecordShape>> =
        RefCell::new(FxHashMap::default());
}

fn intern_shape(fields: &[Symbol]) -> &'static RecordShape {
    let cached = SHAPE_CACHE.with_borrow(|c| c.get(fields).copied());
    if let Some(s) = cached {
        return s;
    }
    let shape = GLOBAL.shape(fields);
    SHAPE_CACHE.with_borrow_mut(|c| c.insert(&shape.fields, shape));
    shape
}

/// Number of interned values (diagnostics).
pub fn store_len() -> usize {
    GLOBAL.len.load(AtomicOrdering::Relaxed) as usize
}

impl Value {
    #[inline]
    fn index(self) -> usize {
        (self.0.get() & !SYMBOLIC_BIT) as usize - 1
    }

    /// The interned node. `&'static`: the store is append-only and leaked.
    #[inline]
    pub fn data(self) -> &'static ValueData {
        GLOBAL.entry(self).data
    }

    /// Raw id bits, for embedding in bytecode immediates.
    #[inline]
    pub fn to_bits(self) -> u32 {
        self.0.get()
    }

    /// Inverse of [`Value::to_bits`]. The bits must come from `to_bits`.
    #[inline]
    pub fn from_bits(bits: u32) -> Self {
        Value(NonZeroU32::new(bits).expect("invalid value bits"))
    }

    pub fn as_lambda(self) -> &'static LambdaVal {
        // single store access for data + registry
        match GLOBAL.entry(self).data {
            ValueData::Lambda(i) => GLOBAL.lambdas.get(*i),
            v => panic!("expected lambda, got {v:?}"),
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.data(), f)
    }
}

// ---------------------------------------------------------------------------
// Structural order
// ---------------------------------------------------------------------------

fn discriminant_rank(v: &ValueData) -> u8 {
    match v {
        ValueData::Bool(_) => 0,
        ValueData::Int(_) => 1,
        ValueData::Str(_) => 2,
        ValueData::Set(_) => 3,
        ValueData::Tuple(_) => 4,
        ValueData::List(_) => 5,
        ValueData::Record(_, _) => 6,
        ValueData::Map(_) => 7,
        ValueData::Variant(_, _) => 8,
        ValueData::Lambda(_) => 9,
        ValueData::Interval(_, _) => 10,
        ValueData::CrossProduct(_) => 11,
        ValueData::PowerSet(_) => 12,
        ValueData::MapSet(_, _) => 13,
        ValueData::InfiniteInt => 14,
        ValueData::InfiniteNat => 15,
    }
}

impl ValueData {
    pub fn is_symbolic(&self) -> bool {
        matches!(
            self,
            ValueData::Interval(_, _)
                | ValueData::CrossProduct(_)
                | ValueData::PowerSet(_)
                | ValueData::MapSet(_, _)
                | ValueData::InfiniteInt
                | ValueData::InfiniteNat
        )
    }
}

/// Guard for early returns out of [`value_cmp`]: distinct ids must never
/// compare Equal (hash-consing invariant).
#[inline]
fn finish_cmp(ord: Ordering) -> Ordering {
    debug_assert!(
        ord != Ordering::Equal,
        "distinct ids compared equal: hash-consing invariant broken"
    );
    ord
}

/// Total structural order over *normalized* values — the same order as the
/// pre-hash-consing `Ord`, so all user-visible enumeration is unchanged.
/// Symbolic forms and lambdas must never be ordered (they never enter
/// containers); doing so is a bug. Hash-consing gives `Equal ⇔ same id`, so
/// comparison never recurses into equal subtrees, and precomputed order
/// keys decide most non-equal comparisons without touching the structure.
pub fn value_cmp(a: Value, b: Value) -> Ordering {
    if a == b {
        return Ordering::Equal;
    }
    // One store access for both nodes' data + order keys. Distinct valid
    // keys decide the comparison outright (see `compute_order_key` for the
    // soundness argument); equal keys fall back to the deep compare.
    let (da, ka, db, kb) = {
        let s = &*GLOBAL;
        let (ea, eb) = (s.entry(a), s.entry(b));
        (ea.data, ea.key, eb.data, eb.key)
    };
    if ka != kb && ka != NO_ORDER_KEY && kb != NO_ORDER_KEY {
        let ord = ka.cmp(&kb);
        debug_assert_eq!(
            ord,
            value_cmp_deep(da, db),
            "order key inconsistent with structural order: {da:?} vs {db:?}"
        );
        return ord;
    }
    finish_cmp(value_cmp_deep(da, db))
}

fn value_cmp_deep(da: &ValueData, db: &ValueData) -> Ordering {
    match (da, db) {
        (ValueData::Bool(x), ValueData::Bool(y)) => x.cmp(y),
        (ValueData::Int(x), ValueData::Int(y)) => x.cmp(y),
        (ValueData::Str(x), ValueData::Str(y)) => x.cmp(y),
        (ValueData::Set(x), ValueData::Set(y)) => cmp_slices(x, y),
        (ValueData::Tuple(x), ValueData::Tuple(y)) => cmp_slices(x, y),
        (ValueData::List(x), ValueData::List(y)) => cmp_slices(x, y),
        (ValueData::Record(sa, va), ValueData::Record(sb, vb)) => {
            // Same interned shape (the common case): every key pair compares
            // Equal, so the interleaved order reduces to the value slices.
            if std::ptr::eq(*sa, *sb) {
                return cmp_slices(va, vb);
            }
            // Field-interleaved comparison, matching BTreeMap<QuintName, Value> order.
            let mut it_a = sa.fields.iter().zip(va.iter());
            let mut it_b = sb.fields.iter().zip(vb.iter());
            loop {
                match (it_a.next(), it_b.next()) {
                    (None, None) => break Ordering::Equal,
                    (None, Some(_)) => break Ordering::Less,
                    (Some(_), None) => break Ordering::Greater,
                    (Some((ka, va)), Some((kb, vb))) => {
                        let o = ka.cmp(kb).then_with(|| value_cmp(*va, *vb));
                        if o != Ordering::Equal {
                            break o;
                        }
                    }
                }
            }
        }
        (ValueData::Map(x), ValueData::Map(y)) => {
            let mut it_a = x.iter();
            let mut it_b = y.iter();
            loop {
                match (it_a.next(), it_b.next()) {
                    (None, None) => break Ordering::Equal,
                    (None, Some(_)) => break Ordering::Less,
                    (Some(_), None) => break Ordering::Greater,
                    (Some((ka, va)), Some((kb, vb))) => {
                        let o = value_cmp(*ka, *kb).then_with(|| value_cmp(*va, *vb));
                        if o != Ordering::Equal {
                            break o;
                        }
                    }
                }
            }
        }
        (ValueData::Variant(lx, vx), ValueData::Variant(ly, vy)) => {
            lx.cmp(ly).then_with(|| value_cmp(*vx, *vy))
        }
        (ValueData::Lambda(_), _) | (_, ValueData::Lambda(_)) => {
            panic!("cannot compare lambdas")
        }
        _ if da.is_symbolic() || db.is_symbolic() => {
            panic!("cannot order symbolic set forms; normalize first (bug)")
        }
        _ => discriminant_rank(da).cmp(&discriminant_rank(db)),
    }
}

fn cmp_slices(a: &[Value], b: &[Value]) -> Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        let o = value_cmp(*x, *y);
        if o != Ordering::Equal {
            return o;
        }
    }
    a.len().cmp(&b.len())
}

// ---------------------------------------------------------------------------
// Constructors. Container constructors normalize their (potentially
// symbolic) elements so that everything reachable from a container is
// canonical.
// ---------------------------------------------------------------------------

impl Value {
    #[inline]
    pub fn int(n: i64) -> Self {
        if (SMALL_INT_MIN..=SMALL_INT_MAX).contains(&n) {
            Value(NonZeroU32::new(INT_BASE + (n - SMALL_INT_MIN) as u32).unwrap())
        } else {
            intern(ValueData::Int(n))
        }
    }

    #[inline]
    pub fn bool(b: bool) -> Self {
        Value(NonZeroU32::new(if b { TRUE_ID } else { FALSE_ID }).unwrap())
    }

    pub fn str(s: QuintName) -> Self {
        intern(ValueData::Str(s))
    }

    pub fn set(elems: impl IntoIterator<Item = Value>) -> Result<Self, QuintError> {
        let mut buf = ValueBuf::new();
        for v in elems {
            buf.push(v.normalize()?);
        }
        buf.sort_dedup();
        Ok(intern_seq(SeqKind::Set, buf.as_slice()))
    }

    /// Set from already-normalized elements (sorts and dedups).
    pub fn set_of_normalized(mut elems: Vec<Value>) -> Self {
        sort_values_structural(&mut elems);
        elems.dedup();
        intern_seq(SeqKind::Set, &elems)
    }

    /// Set from elements already sorted by [`value_cmp`] with no duplicates.
    pub fn set_sorted(elems: Vec<Value>) -> Self {
        debug_assert!(elems.windows(2).all(|w| value_cmp(w[0], w[1]) == Ordering::Less));
        intern_seq(SeqKind::Set, &elems)
    }

    pub fn tuple(elems: impl IntoIterator<Item = Value>) -> Result<Self, QuintError> {
        let mut buf = ValueBuf::new();
        for v in elems {
            buf.push(v.normalize()?);
        }
        Ok(intern_seq(SeqKind::Tuple, buf.as_slice()))
    }

    pub fn list(elems: impl IntoIterator<Item = Value>) -> Result<Self, QuintError> {
        let mut buf = ValueBuf::new();
        for v in elems {
            buf.push(v.normalize()?);
        }
        Ok(intern_seq(SeqKind::List, buf.as_slice()))
    }

    pub fn list_of_normalized(vs: Vec<Value>) -> Self {
        intern_seq(SeqKind::List, &vs)
    }

    pub fn record(
        fields: impl IntoIterator<Item = (QuintName, Value)>,
    ) -> Result<Self, QuintError> {
        let mut fs = fields
            .into_iter()
            .map(|(k, v)| Ok((k, v.normalize()?)))
            .collect::<Result<Vec<_>, QuintError>>()?;
        // Field order is Symbol (string) order; on duplicates the last
        // entry wins, matching BTreeMap insert semantics.
        fs.sort_by_key(|a| a.0);
        fs.dedup_by(|later, earlier| {
            if later.0 == earlier.0 {
                earlier.1 = later.1;
                true
            } else {
                false
            }
        });
        let shape = intern_shape(&fs.iter().map(|(k, _)| *k).collect::<Vec<_>>());
        let mut values = ValueBuf::new();
        for (_, v) in fs {
            values.push(v);
        }
        Ok(Self::record_shaped(shape, values.as_slice()))
    }

    /// Record with a known shape and values in shape order (all normalized).
    pub fn record_shaped(shape: &'static RecordShape, values: &[Value]) -> Self {
        debug_assert_eq!(shape.fields.len(), values.len());
        GLOBAL.intern_record(shape, values)
    }

    pub fn map(entries: impl IntoIterator<Item = (Value, Value)>) -> Result<Self, QuintError> {
        let entries = entries
            .into_iter()
            .map(|(k, v)| Ok((k.normalize()?, v.normalize()?)))
            .collect::<Result<Vec<_>, QuintError>>()?;
        Ok(Self::map_of_normalized(entries))
    }

    /// Map from already-normalized pairs (sorts by key; last duplicate wins,
    /// matching BTreeMap insert semantics).
    pub fn map_of_normalized(mut entries: Vec<(Value, Value)>) -> Self {
        // Decorated sort: keys fetched once, and the original index keeps
        // the unstable sort stable per key for the last-wins dedup.
        let mut dec: Vec<(u64, u32, Value, Value)> = Vec::with_capacity(entries.len());
        {
            let s = &*GLOBAL;
            dec.extend(
                entries
                    .iter()
                    .enumerate()
                    .map(|(i, &(k, v))| (s.entry(k).key, i as u32, k, v)),
            );
        }
        dec.sort_unstable_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| value_cmp(a.2, b.2))
                .then(a.1.cmp(&b.1))
        });
        entries.clear();
        for &(_, _, k, v) in &dec {
            if let Some(last) = entries.last_mut() {
                if last.0 == k {
                    last.1 = v;
                    continue;
                }
            }
            entries.push((k, v));
        }
        GLOBAL.intern_map(&entries)
    }

    /// Map from pairs already sorted by key with unique keys.
    pub fn map_sorted(entries: Vec<(Value, Value)>) -> Self {
        debug_assert!(entries
            .windows(2)
            .all(|w| value_cmp(w[0].0, w[1].0) == Ordering::Less));
        GLOBAL.intern_map(&entries)
    }

    pub fn variant(label: QuintName, payload: Value) -> Result<Self, QuintError> {
        Ok(intern(ValueData::Variant(label, payload.normalize()?)))
    }

    pub fn lambda(slots: Box<[u32]>, fnid: crate::vm::FnId) -> Self {
        let idx = {
            let mut len = GLOBAL.lambdas_len.lock().unwrap();
            let idx = *len;
            // Safety: idx claimed under the lock; published by the intern
            // of the `Lambda(idx)` value below before the id escapes.
            unsafe {
                GLOBAL
                    .lambdas
                    .write(idx, Box::leak(Box::new(LambdaVal { slots, fnid })))
            };
            *len += 1;
            idx
        };
        intern(ValueData::Lambda(idx))
    }

    pub fn interval(start: i64, end: i64) -> Self {
        intern(ValueData::Interval(start, end))
    }

    pub fn cross_product(sets: Vec<Value>) -> Self {
        intern_seq(SeqKind::CrossProduct, &sets)
    }

    pub fn power_set(base: Value) -> Self {
        intern(ValueData::PowerSet(base))
    }

    pub fn map_set(domain: Value, range: Value) -> Self {
        intern(ValueData::MapSet(domain, range))
    }

    pub fn infinite_int() -> Self {
        intern(ValueData::InfiniteInt)
    }

    pub fn infinite_nat() -> Self {
        intern(ValueData::InfiniteNat)
    }
}

// ---------------------------------------------------------------------------
// Accessors (panic on type mismatch: input is type-checked by quint).
// ---------------------------------------------------------------------------

impl Value {
    pub fn as_int(self) -> i64 {
        match self.data() {
            ValueData::Int(n) => *n,
            v => panic!("expected int, got {v:?}"),
        }
    }

    /// Bools have reserved ids: no store access on the hot path
    /// (`JumpIfFalse` is the most frequent instruction).
    #[inline]
    pub fn as_bool(self) -> bool {
        match self.0.get() {
            TRUE_ID => true,
            FALSE_ID => false,
            _ => panic!("expected bool, got {:?}", self.data()),
        }
    }

    pub fn as_str(self) -> Symbol {
        match self.data() {
            ValueData::Str(s) => *s,
            v => panic!("expected str, got {v:?}"),
        }
    }

    /// Map entries, sorted by key.
    pub fn as_map(self) -> &'static [(Value, Value)] {
        match self.data() {
            ValueData::Map(m) => m,
            v => panic!("expected map, got {v:?}"),
        }
    }

    pub fn as_record(self) -> (&'static RecordShape, &'static [Value]) {
        match self.data() {
            ValueData::Record(shape, values) => (shape, values),
            v => panic!("expected record, got {v:?}"),
        }
    }

    /// Tuples and lists share representation semantics for indexing builtins.
    pub fn as_elems(self) -> &'static [Value] {
        match self.data() {
            ValueData::Tuple(vs) | ValueData::List(vs) => vs,
            v => panic!("expected tuple/list, got {v:?}"),
        }
    }

    pub fn as_variant(self) -> (Symbol, Value) {
        match self.data() {
            ValueData::Variant(label, payload) => (*label, *payload),
            v => panic!("expected variant, got {v:?}"),
        }
    }

    pub fn as_tuple2(self) -> (Value, Value) {
        let elems = self.as_elems();
        (elems[0], elems[1])
    }

    /// Set elements, sorted by [`value_cmp`]. Panics on symbolic forms —
    /// use [`Value::enumerate`] for those.
    pub fn as_set(self) -> &'static [Value] {
        match self.data() {
            ValueData::Set(s) => s,
            v => panic!("expected concrete set, got {v:?}"),
        }
    }

    /// Tag-bit test: no store access (see [`SYMBOLIC_BIT`]).
    #[inline]
    pub fn is_symbolic(self) -> bool {
        self.0.get() & SYMBOLIC_BIT != 0
    }

    pub fn is_set(self) -> bool {
        matches!(self.data(), ValueData::Set(_)) || self.is_symbolic()
    }

    /// Record field lookup by symbol (u32 compares; fields are few).
    pub fn record_field(self, field: Symbol) -> Option<Value> {
        let (shape, values) = self.as_record();
        shape
            .fields
            .iter()
            .position(|&f| f == field)
            .map(|i| values[i])
    }

    /// Map lookup by (normalized) key: id scan / binary search.
    pub fn map_get(self, key: Value) -> Option<Value> {
        let entries = self.as_map();
        map_find_by_id(entries, key).map(|i| entries[i].1)
    }
}

// ---------------------------------------------------------------------------
// Symbolic set operations: cardinality / membership / subset / enumeration /
// indexed access — all without enumerating when structurally answerable.
// ---------------------------------------------------------------------------

impl Value {
    pub fn cardinality(self) -> Result<u64, QuintError> {
        match self.data() {
            ValueData::Set(s) => Ok(s.len() as u64),
            ValueData::Tuple(vs) | ValueData::List(vs) => Ok(vs.len() as u64),
            ValueData::Record(shape, _) => Ok(shape.fields.len() as u64),
            ValueData::Map(m) => Ok(m.len() as u64),
            ValueData::Interval(start, end) => end
                .checked_sub(*start)
                .and_then(|d| d.checked_add(1))
                .and_then(|n| u64::try_from(n).ok())
                .ok_or_else(|| overflow("interval cardinality")),
            ValueData::CrossProduct(sets) => sets.iter().try_fold(1u64, |acc, s| {
                acc.checked_mul(s.cardinality()?)
                    .ok_or_else(|| overflow("cross product cardinality"))
            }),
            ValueData::PowerSet(base) => {
                let n = base.cardinality()?;
                let exp = u32::try_from(n).map_err(|_| overflow("powerset cardinality"))?;
                2u64
                    .checked_pow(exp)
                    .ok_or_else(|| overflow("powerset cardinality"))
            }
            ValueData::MapSet(domain, range) => {
                let d = domain.cardinality()?;
                let r = range.cardinality()?;
                let exp = u32::try_from(d).map_err(|_| overflow("setOfMaps cardinality"))?;
                r.checked_pow(exp)
                    .ok_or_else(|| overflow("setOfMaps cardinality"))
            }
            ValueData::InfiniteInt => Err(unsupported("infinite set Int")),
            ValueData::InfiniteNat => Err(unsupported("infinite set Nat")),
            v => panic!("cardinality: not a set: {v:?}"),
        }
    }

    /// Set membership, structural where possible. `elem` must be normalized
    /// (guaranteed when it came out of evaluation of a non-set expression or
    /// any container).
    pub fn contains(self, elem: Value) -> Result<bool, QuintError> {
        Ok(match (self.data(), elem.data()) {
            (ValueData::Set(s), _) => set_find(s, elem).is_ok(),
            (ValueData::Interval(start, end), ValueData::Int(n)) => start <= n && n <= end,
            (ValueData::Interval(_, _), _) => false,
            (ValueData::CrossProduct(sets), ValueData::Tuple(elems)) => {
                sets.len() == elems.len()
                    && sets.iter().zip(elems.iter()).try_fold(true, |acc, (s, e)| {
                        Ok::<_, QuintError>(acc && s.contains(*e)?)
                    })?
            }
            (ValueData::CrossProduct(_), _) => false,
            (ValueData::PowerSet(base), ValueData::Set(elems)) => elems
                .iter()
                .try_fold(true, |acc, e| Ok::<_, QuintError>(acc && base.contains(*e)?))?,
            (ValueData::PowerSet(_), _) => false,
            (ValueData::MapSet(domain, range), ValueData::Map(m)) => {
                let map_domain = Value::set_sorted(m.iter().map(|(k, _)| *k).collect());
                value_eq(map_domain, *domain)?
                    && m.iter().try_fold(true, |acc, (_, v)| {
                        Ok::<_, QuintError>(acc && range.contains(*v)?)
                    })?
            }
            (ValueData::MapSet(_, _), _) => false,
            (ValueData::InfiniteInt, ValueData::Int(_)) => true,
            (ValueData::InfiniteInt, _) => false,
            (ValueData::InfiniteNat, ValueData::Int(n)) => *n >= 0,
            (ValueData::InfiniteNat, _) => false,
            (v, _) => panic!("contains: not a set: {v:?}"),
        })
    }

    pub fn subseteq(self, superset: Value) -> Result<bool, QuintError> {
        Ok(match (self.data(), superset.data()) {
            (ValueData::Set(a), ValueData::Set(_)) => a
                .iter()
                .try_fold(true, |acc, e| Ok::<_, QuintError>(acc && superset.contains(*e)?))?,
            (ValueData::Interval(a1, a2), ValueData::Interval(b1, b2)) => {
                a1 > a2 || (a1 >= b1 && a2 <= b2)
            }
            (ValueData::CrossProduct(a), ValueData::CrossProduct(b)) => {
                a.len() == b.len()
                    && a.iter().zip(b.iter()).try_fold(true, |acc, (x, y)| {
                        Ok::<_, QuintError>(acc && x.subseteq(*y)?)
                    })?
            }
            (ValueData::PowerSet(a), ValueData::PowerSet(b)) => a.subseteq(*b)?,
            (ValueData::MapSet(ad, ar), ValueData::MapSet(bd, br)) => {
                value_eq(*ad, *bd)? && ar.subseteq(*br)?
            }
            (ValueData::InfiniteNat, ValueData::InfiniteNat | ValueData::InfiniteInt) => true,
            (ValueData::InfiniteInt, ValueData::InfiniteInt) => true,
            (ValueData::InfiniteInt | ValueData::InfiniteNat, _) => false,
            (_, ValueData::InfiniteInt | ValueData::InfiniteNat) => self
                .enumerate()?
                .iter()
                .try_fold(true, |acc, v| Ok::<_, QuintError>(acc && superset.contains(*v)?))?,
            _ => {
                let a = self.enumerate()?;
                a.iter()
                    .try_fold(true, |acc, v| Ok::<_, QuintError>(acc && superset.contains(*v)?))?
            }
        })
    }

    /// Materialize a (possibly symbolic) set as a canonical sorted slice,
    /// guarded by [`MAX_ENUM`]. Concrete sets borrow from the store.
    pub fn enumerate(self) -> Result<Cow<'static, [Value]>, QuintError> {
        match self.data() {
            ValueData::Set(s) => Ok(Cow::Borrowed(&s[..])),
            ValueData::InfiniteInt => Err(unsupported("enumerating infinite set Int")),
            ValueData::InfiniteNat => Err(unsupported("enumerating infinite set Nat")),
            _ => {
                let n = self.cardinality()?;
                if n > MAX_ENUM {
                    return Err(QuintError::new(
                        "QNT501",
                        format!(
                            "set of cardinality {n} is too large to enumerate (max {MAX_ENUM})"
                        ),
                    ));
                }
                if n == 0 {
                    return Ok(Cow::Owned(Vec::new()));
                }
                let mut out = Vec::with_capacity(n as usize);
                let bounds = self.bounds()?;
                let mut indices = U64Buf::zeros(bounds.len());
                let indices = indices.as_mut_slice();
                'outer: loop {
                    out.push(self.pick(&mut indices.iter().copied())?.normalize()?);
                    // mixed-radix increment
                    let mut i = 0;
                    loop {
                        if i == bounds.len() {
                            break 'outer;
                        }
                        indices[i] += 1;
                        if indices[i] < bounds[i] {
                            break;
                        }
                        indices[i] = 0;
                        i += 1;
                    }
                }
                sort_values_structural(&mut out);
                out.dedup();
                Ok(Cow::Owned(out))
            }
        }
    }

    /// The mixed-radix bounds for indexed access (`pick`). One entry per
    /// independent index dimension. Stack-allocated for the common case
    /// (one dimension) — this runs on every `oneOf` choice.
    pub fn bounds(self) -> Result<U64Buf, QuintError> {
        let mut out = U64Buf::new();
        self.bounds_into(&mut out)?;
        Ok(out)
    }

    fn bounds_into(self, out: &mut U64Buf) -> Result<(), QuintError> {
        match self.data() {
            ValueData::Set(s) => out.push(s.len() as u64),
            ValueData::Interval(_, _) => out.push(self.cardinality()?),
            ValueData::CrossProduct(sets) => {
                for s in sets.iter() {
                    out.push(s.cardinality()?);
                }
            }
            ValueData::PowerSet(base) => {
                let n = base.cardinality()?;
                if n >= 63 {
                    return Err(unsupported(format!(
                        "powerset of a {n}-element set (2^{n} elements)"
                    )));
                }
                out.push(1u64 << n);
            }
            ValueData::MapSet(domain, range) => {
                let d = domain.cardinality()? as usize;
                let mut range_bounds = U64Buf::new();
                range.bounds_into(&mut range_bounds)?;
                for _ in 0..d {
                    for &b in range_bounds.iter() {
                        out.push(b);
                    }
                }
            }
            ValueData::InfiniteInt => return Err(unsupported("picking from infinite set Int")),
            ValueData::InfiniteNat => return Err(unsupported("picking from infinite set Nat")),
            v => panic!("bounds: not a set: {v:?}"),
        }
        Ok(())
    }

    /// Pick the element identified by `indexes` (one index per bound, in
    /// order). Deterministic thanks to the canonical sorted order.
    pub fn pick<T: Iterator<Item = u64>>(self, indexes: &mut T) -> Result<Value, QuintError> {
        Ok(match self.data() {
            ValueData::Set(s) => {
                let i = indexes.next().expect("too few pick indices (bug)");
                *s.get(i as usize).expect("pick index out of bounds (bug)")
            }
            ValueData::Interval(start, _) => {
                let i = indexes.next().expect("too few pick indices (bug)");
                Value::int(start + i as i64)
            }
            ValueData::CrossProduct(sets) => {
                let elems = sets
                    .iter()
                    .map(|s| s.pick(indexes))
                    .collect::<Result<Vec<_>, _>>()?;
                Value::tuple(elems)?
            }
            ValueData::PowerSet(base) => {
                let i = indexes.next().expect("too few pick indices (bug)");
                let base = base.enumerate()?;
                let elems = base
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| i & (1u64 << j) != 0)
                    .map(|(_, e)| *e)
                    .collect();
                // base is sorted, so the filtered subsequence is sorted too
                Value::set_sorted(elems)
            }
            ValueData::MapSet(domain, range) => {
                if domain.cardinality()? == 0 {
                    // TLC behavior: setOfMaps with empty domain = Set(Map())
                    return Ok(Value::map_sorted(Vec::new()));
                }
                let range = if matches!(range.data(), ValueData::MapSet(_, _)) {
                    Value::set_of_normalized(range.enumerate()?.into_owned())
                } else {
                    *range
                };
                let keys = domain.enumerate()?;
                let entries = keys
                    .iter()
                    .map(|k| Ok::<_, QuintError>((*k, range.pick(indexes)?)))
                    .collect::<Result<Vec<_>, _>>()?;
                Value::map(entries)?
            }
            ValueData::InfiniteInt | ValueData::InfiniteNat => {
                return Err(unsupported("picking from an infinite set"))
            }
            v => panic!("pick: not a set: {v:?}"),
        })
    }

    /// Canonical form: symbolic set forms are materialized (recursively).
    /// Values built from containers are already canonical, so this is a
    /// tag-bit test in the common case.
    #[inline]
    pub fn normalize(self) -> Result<Value, QuintError> {
        if self.is_symbolic() {
            Ok(Value::set_sorted(self.enumerate()?.into_owned()))
        } else {
            Ok(self)
        }
    }
}

/// Fallible equality that copes with symbolic operands (`S == Set(...)`,
/// `Interval(1,3) == 1.to(3)`), matching the reference semantics: mixed
/// representations are compared by enumeration; infinite sets are equal
/// only to themselves.
pub fn value_eq(a: Value, b: Value) -> Result<bool, QuintError> {
    if a == b {
        return Ok(true);
    }
    // hash-consed: distinct non-symbolic ids ⇒ unequal (tag-bit tests
    // only, no store access — this is the `eq` builtin's hot path)
    if !a.is_symbolic() && !b.is_symbolic() {
        return Ok(false);
    }
    let (ia, ib) = (a.data(), b.data());
    match (ia.is_symbolic(), ib.is_symbolic()) {
        (false, false) => Ok(false),
        _ => match (ia, ib) {
            (ValueData::InfiniteInt | ValueData::InfiniteNat, _)
            | (_, ValueData::InfiniteInt | ValueData::InfiniteNat) => Ok(false),
            (ValueData::Interval(a1, a2), ValueData::Interval(b1, b2)) => {
                // Empty intervals are all equal
                Ok((a1 > a2 && b1 > b2) || (a1, a2) == (b1, b2))
            }
            _ => {
                if a.is_set() && b.is_set() {
                    if a.cardinality()? != b.cardinality()? {
                        return Ok(false);
                    }
                    Ok(*a.enumerate()? == *b.enumerate()?)
                } else {
                    Ok(false)
                }
            }
        },
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.data() {
            ValueData::Int(n) => write!(f, "{n}"),
            ValueData::Bool(b) => write!(f, "{b}"),
            ValueData::Str(s) => write!(f, "{:?}", s.as_str()),
            ValueData::Set(_)
            | ValueData::Interval(_, _)
            | ValueData::CrossProduct(_)
            | ValueData::PowerSet(_)
            | ValueData::MapSet(_, _) => {
                write!(f, "Set(")?;
                match self.enumerate() {
                    Ok(set) => {
                        for (i, e) in set.iter().enumerate() {
                            if i > 0 {
                                write!(f, ", ")?;
                            }
                            write!(f, "{e}")?;
                        }
                    }
                    Err(_) => write!(f, "<too large>")?,
                }
                write!(f, ")")
            }
            ValueData::InfiniteInt => write!(f, "Int"),
            ValueData::InfiniteNat => write!(f, "Nat"),
            ValueData::Tuple(vs) => {
                write!(f, "(")?;
                for (i, e) in vs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{e}")?;
                }
                write!(f, ")")
            }
            ValueData::List(vs) => {
                write!(f, "[")?;
                for (i, e) in vs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{e}")?;
                }
                write!(f, "]")
            }
            ValueData::Record(shape, values) => {
                write!(f, "{{ ")?;
                for (i, (k, v)) in shape.fields.iter().zip(values.iter()).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k}: {v}")?;
                }
                write!(f, " }}")
            }
            ValueData::Map(m) => {
                write!(f, "Map(")?;
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k} -> {v}")?;
                }
                write!(f, ")")
            }
            ValueData::Variant(label, payload) => {
                if matches!(payload.data(), ValueData::Tuple(vs) if vs.is_empty()) {
                    write!(f, "{label}")
                } else {
                    write!(f, "{label}({payload})")
                }
            }
            ValueData::Lambda(_) => write!(f, "<lambda>"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(v: Value) -> u64 {
        GLOBAL.entry(v).key
    }

    /// Key soundness: for every pair with distinct valid keys, the key
    /// order must equal the structural order.
    #[test]
    fn order_keys_agree_with_value_cmp() {
        let s = |t: &str| Value::str(Symbol::intern(t));
        let set = |vs: &[Value]| Value::set(vs.iter().copied()).unwrap();
        let tup = |vs: &[Value]| Value::tuple(vs.iter().copied()).unwrap();
        let rec = |fs: &[(&str, Value)]| {
            Value::record(fs.iter().map(|(k, v)| (Symbol::intern(k), *v))).unwrap()
        };
        let map = |es: &[(Value, Value)]| Value::map(es.iter().copied()).unwrap();

        let values = vec![
            Value::bool(false),
            Value::bool(true),
            Value::int(-5000),
            Value::int(-1),
            Value::int(0),
            Value::int(1),
            Value::int(1 << 40),
            Value::int((1 << 40) + 1), // differs only in the low byte
            s(""),
            s("a"),
            s("ab"),
            s("prefix_shared_x"),
            s("prefix_shared_y"), // shares a 7-byte prefix
            set(&[]),
            set(&[Value::bool(false)]),
            set(&[Value::int(1), Value::int(2)]),
            set(&[Value::int(3)]),
            tup(&[]),
            tup(&[Value::int(1), s("a")]),
            tup(&[Value::int(2)]),
            Value::list([Value::int(9)]).unwrap(),
            rec(&[("a", Value::int(1))]),
            rec(&[("a", Value::int(2))]),
            rec(&[("b", Value::int(1))]),
            map(&[]),
            map(&[(s("k1"), Value::int(1))]),
            map(&[(s("k2"), Value::int(1))]),
            Value::variant(Symbol::intern("Some"), Value::int(1)).unwrap(),
            Value::variant(Symbol::intern("None"), tup(&[])).unwrap(),
        ];

        for &a in &values {
            for &b in &values {
                let (ka, kb) = (key(a), key(b));
                if ka != kb && ka != NO_ORDER_KEY && kb != NO_ORDER_KEY {
                    assert_eq!(
                        ka.cmp(&kb),
                        value_cmp(a, b),
                        "key order diverges from structural order: {a:?} vs {b:?}"
                    );
                }
                if a == b {
                    assert_eq!(ka, kb);
                }
            }
        }
    }

    #[test]
    fn unorderable_values_have_no_order_key() {
        assert_eq!(key(Value::interval(1, 3)), NO_ORDER_KEY);
        assert_eq!(key(Value::power_set(Value::interval(1, 2))), NO_ORDER_KEY);
        assert_eq!(key(Value::infinite_int()), NO_ORDER_KEY);
        assert_eq!(
            key(Value::map_set(Value::interval(1, 2), Value::interval(1, 2))),
            NO_ORDER_KEY
        );
    }
}
