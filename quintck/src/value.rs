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
use rustc_hash::FxHashMap;
use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::fmt;
use std::num::NonZeroU32;
use std::rc::Rc;

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

/// A compiled lambda: shared parameter registers plus the VM function id
/// of the body. Created once per `Lambda` IR node at lowering time.
pub struct LambdaVal {
    pub registers: Vec<Rc<Cell<Option<Value>>>>,
    pub fnid: crate::vm::FnId,
}

// ---------------------------------------------------------------------------
// The store
// ---------------------------------------------------------------------------

struct ValueStore {
    map: FxHashMap<&'static ValueData, Value>,
    entries: Vec<&'static ValueData>,
    shapes: FxHashMap<&'static [Symbol], &'static RecordShape>,
    lambdas: Vec<&'static LambdaVal>,
}

// Reserved ids, in ValueStore::new interning order.
const FALSE_ID: u32 = 1;
const TRUE_ID: u32 = 2;
const SMALL_INT_MIN: i64 = -1024;
const SMALL_INT_MAX: i64 = 1023;
const INT_BASE: u32 = 3; // id of SMALL_INT_MIN

impl ValueStore {
    fn new() -> Self {
        let mut store = ValueStore {
            map: FxHashMap::default(),
            entries: Vec::new(),
            shapes: FxHashMap::default(),
            lambdas: Vec::new(),
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

    fn intern(&mut self, data: ValueData) -> Value {
        if let Some(&id) = self.map.get(&data) {
            return id;
        }
        let leaked: &'static ValueData = Box::leak(Box::new(data));
        let id = Value(NonZeroU32::new(self.entries.len() as u32 + 1).expect("id overflow"));
        self.entries.push(leaked);
        self.map.insert(leaked, id);
        id
    }

    fn shape(&mut self, fields: &[Symbol]) -> &'static RecordShape {
        if let Some(&s) = self.shapes.get(fields) {
            return s;
        }
        let shape: &'static RecordShape = Box::leak(Box::new(RecordShape {
            fields: fields.into(),
        }));
        self.shapes.insert(&shape.fields, shape);
        shape
    }
}

thread_local! {
    static VALUES: RefCell<ValueStore> = RefCell::new(ValueStore::new());
}

fn intern(data: ValueData) -> Value {
    VALUES.with_borrow_mut(|s| s.intern(data))
}

fn intern_shape(fields: &[Symbol]) -> &'static RecordShape {
    VALUES.with_borrow_mut(|s| s.shape(fields))
}

/// Number of interned values (diagnostics).
pub fn store_len() -> usize {
    VALUES.with_borrow(|s| s.entries.len())
}

impl Value {
    /// The interned node. `&'static`: the store is append-only and leaked.
    #[inline]
    pub fn data(self) -> &'static ValueData {
        VALUES.with_borrow(|s| s.entries[self.0.get() as usize - 1])
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
        match self.data() {
            ValueData::Lambda(i) => VALUES.with_borrow(|s| s.lambdas[*i as usize]),
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

/// Total structural order over *normalized* values — the same order as the
/// pre-hash-consing `Ord`, so all user-visible enumeration is unchanged.
/// Symbolic forms and lambdas must never be ordered (they never enter
/// containers); doing so is a bug. Hash-consing gives `Equal ⇔ same id`, so
/// comparison never recurses into equal subtrees.
pub fn value_cmp(a: Value, b: Value) -> Ordering {
    if a == b {
        return Ordering::Equal;
    }
    let (da, db) = (a.data(), b.data());
    let ord = match (da, db) {
        (ValueData::Bool(x), ValueData::Bool(y)) => x.cmp(y),
        (ValueData::Int(x), ValueData::Int(y)) => x.cmp(y),
        (ValueData::Str(x), ValueData::Str(y)) => x.cmp(y),
        (ValueData::Set(x), ValueData::Set(y)) => cmp_slices(x, y),
        (ValueData::Tuple(x), ValueData::Tuple(y)) => cmp_slices(x, y),
        (ValueData::List(x), ValueData::List(y)) => cmp_slices(x, y),
        (ValueData::Record(sa, va), ValueData::Record(sb, vb)) => {
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
    };
    debug_assert!(
        ord != Ordering::Equal,
        "distinct ids compared equal: hash-consing invariant broken"
    );
    ord
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
        let elems = elems
            .into_iter()
            .map(|v| v.normalize())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::set_of_normalized(elems))
    }

    /// Set from already-normalized elements (sorts and dedups).
    pub fn set_of_normalized(mut elems: Vec<Value>) -> Self {
        elems.sort_unstable_by(|a, b| value_cmp(*a, *b));
        elems.dedup();
        intern(ValueData::Set(elems.into_boxed_slice()))
    }

    /// Set from elements already sorted by [`value_cmp`] with no duplicates.
    pub fn set_sorted(elems: Vec<Value>) -> Self {
        debug_assert!(elems.windows(2).all(|w| value_cmp(w[0], w[1]) == Ordering::Less));
        intern(ValueData::Set(elems.into_boxed_slice()))
    }

    pub fn tuple(elems: impl IntoIterator<Item = Value>) -> Result<Self, QuintError> {
        let vs = elems
            .into_iter()
            .map(|v| v.normalize())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(intern(ValueData::Tuple(vs.into_boxed_slice())))
    }

    pub fn list(elems: impl IntoIterator<Item = Value>) -> Result<Self, QuintError> {
        let vs = elems
            .into_iter()
            .map(|v| v.normalize())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::list_of_normalized(vs))
    }

    pub fn list_of_normalized(vs: Vec<Value>) -> Self {
        intern(ValueData::List(vs.into_boxed_slice()))
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
        let values: Box<[Value]> = fs.into_iter().map(|(_, v)| v).collect();
        Ok(intern(ValueData::Record(shape, values)))
    }

    /// Record with a known shape and values in shape order (all normalized).
    pub fn record_shaped(shape: &'static RecordShape, values: Box<[Value]>) -> Self {
        debug_assert_eq!(shape.fields.len(), values.len());
        intern(ValueData::Record(shape, values))
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
        entries.sort_by(|a, b| value_cmp(a.0, b.0));
        entries.dedup_by(|later, earlier| {
            if later.0 == earlier.0 {
                earlier.1 = later.1;
                true
            } else {
                false
            }
        });
        intern(ValueData::Map(entries.into_boxed_slice()))
    }

    /// Map from pairs already sorted by key with unique keys.
    pub fn map_sorted(entries: Vec<(Value, Value)>) -> Self {
        debug_assert!(entries
            .windows(2)
            .all(|w| value_cmp(w[0].0, w[1].0) == Ordering::Less));
        intern(ValueData::Map(entries.into_boxed_slice()))
    }

    pub fn variant(label: QuintName, payload: Value) -> Result<Self, QuintError> {
        Ok(intern(ValueData::Variant(label, payload.normalize()?)))
    }

    pub fn lambda(registers: Vec<Rc<Cell<Option<Value>>>>, fnid: crate::vm::FnId) -> Self {
        VALUES.with_borrow_mut(|s| {
            let idx = s.lambdas.len() as u32;
            s.lambdas
                .push(Box::leak(Box::new(LambdaVal { registers, fnid })));
            s.intern(ValueData::Lambda(idx))
        })
    }

    pub fn interval(start: i64, end: i64) -> Self {
        intern(ValueData::Interval(start, end))
    }

    pub fn cross_product(sets: Vec<Value>) -> Self {
        intern(ValueData::CrossProduct(sets.into_boxed_slice()))
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

    pub fn as_bool(self) -> bool {
        match self.data() {
            ValueData::Bool(b) => *b,
            v => panic!("expected bool, got {v:?}"),
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

    pub fn is_symbolic(self) -> bool {
        self.data().is_symbolic()
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

    /// Map lookup by (normalized) key: binary search over the sorted entries.
    pub fn map_get(self, key: Value) -> Option<Value> {
        let entries = self.as_map();
        entries
            .binary_search_by(|(k, _)| value_cmp(*k, key))
            .ok()
            .map(|i| entries[i].1)
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
            (ValueData::Set(s), _) => s
                .binary_search_by(|&e| value_cmp(e, elem))
                .is_ok(),
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
                let mut indices = vec![0u64; bounds.len()];
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
                out.sort_unstable_by(|a, b| value_cmp(*a, *b));
                out.dedup();
                Ok(Cow::Owned(out))
            }
        }
    }

    /// The mixed-radix bounds for indexed access (`pick`). One entry per
    /// independent index dimension.
    pub fn bounds(self) -> Result<Vec<u64>, QuintError> {
        Ok(match self.data() {
            ValueData::Set(s) => vec![s.len() as u64],
            ValueData::Interval(_, _) => vec![self.cardinality()?],
            ValueData::CrossProduct(sets) => sets
                .iter()
                .map(|s| s.cardinality())
                .collect::<Result<Vec<_>, _>>()?,
            ValueData::PowerSet(base) => {
                let n = base.cardinality()?;
                if n >= 63 {
                    return Err(unsupported(format!(
                        "powerset of a {n}-element set (2^{n} elements)"
                    )));
                }
                vec![1u64 << n]
            }
            ValueData::MapSet(domain, range) => {
                let d = domain.cardinality()? as usize;
                let range_bounds = range.bounds()?;
                range_bounds.repeat(d)
            }
            ValueData::InfiniteInt => return Err(unsupported("picking from infinite set Int")),
            ValueData::InfiniteNat => return Err(unsupported("picking from infinite set Nat")),
            v => panic!("bounds: not a set: {v:?}"),
        })
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
    /// no-op in the common case.
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
    let (ia, ib) = (a.data(), b.data());
    match (ia.is_symbolic(), ib.is_symbolic()) {
        (false, false) => Ok(false), // hash-consed: distinct ids ⇒ unequal
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
