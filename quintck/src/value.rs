//! Runtime values.
//!
//! Design differences from the reference `quint_evaluator`:
//! - Ordered containers (`BTreeSet`/`BTreeMap`) instead of hash containers,
//!   so every *normalized* value has one canonical form and a total `Ord` —
//!   states can be deduplicated and hashed with no separate normalize pass,
//!   and enumeration order is deterministic without a seeded hasher.
//! - Symbolic ("lazy") set forms (`Interval`, `CrossProduct`, `PowerSet`,
//!   `MapSet`, infinite sets) never enter containers or state variables:
//!   every container constructor normalizes its elements. Symbolic values
//!   exist only transiently as operands, where membership/cardinality are
//!   answered structurally (`TypeOK`-style `x.in(S.setOfMaps(T))` never
//!   enumerates).

use crate::error::{overflow, unsupported, QuintError};
use quint_ast::QuintName;
use std::borrow::Cow;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::rc::Rc;

use crate::eval::CompiledExpr;

/// Enumeration guard: symbolic sets larger than this refuse to materialize.
/// TODO(v2): make configurable via CLI (--max-enum-size).
pub const MAX_ENUM: u64 = 1 << 20;

pub type EvalResult = Result<Value, QuintError>;

#[derive(Clone, Debug)]
pub struct Value(pub Rc<ValueInner>);

#[derive(Debug)]
pub enum ValueInner {
    Int(i64),
    Bool(bool),
    Str(QuintName),
    Set(BTreeSet<Value>),
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Record(BTreeMap<QuintName, Value>),
    Map(BTreeMap<Value, Value>),
    Variant(QuintName, Value),
    Lambda(Vec<Rc<RefCell<Option<Value>>>>, CompiledExpr),
    // Symbolic set forms (never stored in containers/states):
    Interval(i64, i64),
    CrossProduct(Vec<Value>),
    PowerSet(Value),
    MapSet(Value, Value),
    InfiniteInt,
    InfiniteNat,
}

impl std::ops::Deref for Value {
    type Target = ValueInner;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn discriminant_rank(v: &ValueInner) -> u8 {
    match v {
        ValueInner::Bool(_) => 0,
        ValueInner::Int(_) => 1,
        ValueInner::Str(_) => 2,
        ValueInner::Set(_) => 3,
        ValueInner::Tuple(_) => 4,
        ValueInner::List(_) => 5,
        ValueInner::Record(_) => 6,
        ValueInner::Map(_) => 7,
        ValueInner::Variant(_, _) => 8,
        ValueInner::Lambda(_, _) => 9,
        ValueInner::Interval(_, _) => 10,
        ValueInner::CrossProduct(_) => 11,
        ValueInner::PowerSet(_) => 12,
        ValueInner::MapSet(_, _) => 13,
        ValueInner::InfiniteInt => 14,
        ValueInner::InfiniteNat => 15,
    }
}

/// Total order over *normalized* values. Symbolic forms and lambdas must
/// never be compared (they never enter containers); doing so is a bug.
impl Ord for Value {
    fn cmp(&self, other: &Self) -> Ordering {
        if Rc::ptr_eq(&self.0, &other.0) {
            return Ordering::Equal;
        }
        let (a, b) = (self.0.as_ref(), other.0.as_ref());
        match (a, b) {
            (ValueInner::Bool(x), ValueInner::Bool(y)) => x.cmp(y),
            (ValueInner::Int(x), ValueInner::Int(y)) => x.cmp(y),
            (ValueInner::Str(x), ValueInner::Str(y)) => x.cmp(y),
            (ValueInner::Set(x), ValueInner::Set(y)) => x.cmp(y),
            (ValueInner::Tuple(x), ValueInner::Tuple(y)) => x.cmp(y),
            (ValueInner::List(x), ValueInner::List(y)) => x.cmp(y),
            (ValueInner::Record(x), ValueInner::Record(y)) => x.cmp(y),
            (ValueInner::Map(x), ValueInner::Map(y)) => x.cmp(y),
            (ValueInner::Variant(lx, vx), ValueInner::Variant(ly, vy)) => {
                lx.cmp(ly).then_with(|| vx.cmp(vy))
            }
            (ValueInner::Lambda(_, _), _) | (_, ValueInner::Lambda(_, _)) => {
                panic!("cannot compare lambdas")
            }
            _ if a.is_symbolic() || b.is_symbolic() => {
                panic!("cannot order symbolic set forms; normalize first (bug)")
            }
            _ => discriminant_rank(a).cmp(&discriminant_rank(b)),
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Value {}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let inner = self.0.as_ref();
        discriminant_rank(inner).hash(state);
        match inner {
            ValueInner::Bool(b) => b.hash(state),
            ValueInner::Int(n) => n.hash(state),
            ValueInner::Str(s) => s.hash(state),
            ValueInner::Set(s) => {
                for v in s {
                    v.hash(state);
                }
            }
            ValueInner::Tuple(vs) | ValueInner::List(vs) => {
                for v in vs {
                    v.hash(state);
                }
            }
            ValueInner::Record(fields) => {
                for (k, v) in fields {
                    k.hash(state);
                    v.hash(state);
                }
            }
            ValueInner::Map(m) => {
                for (k, v) in m {
                    k.hash(state);
                    v.hash(state);
                }
            }
            ValueInner::Variant(label, v) => {
                label.hash(state);
                v.hash(state);
            }
            _ => panic!("cannot hash symbolic/lambda values (bug)"),
        }
    }
}

impl ValueInner {
    pub fn is_symbolic(&self) -> bool {
        matches!(
            self,
            ValueInner::Interval(_, _)
                | ValueInner::CrossProduct(_)
                | ValueInner::PowerSet(_)
                | ValueInner::MapSet(_, _)
                | ValueInner::InfiniteInt
                | ValueInner::InfiniteNat
        )
    }
}

// ---------------------------------------------------------------------------
// Constructors. Container constructors normalize their (potentially
// symbolic) elements so that everything reachable from a container is
// canonical.
// ---------------------------------------------------------------------------

impl Value {
    pub fn int(n: i64) -> Self {
        Value(Rc::new(ValueInner::Int(n)))
    }

    pub fn bool(b: bool) -> Self {
        Value(Rc::new(ValueInner::Bool(b)))
    }

    pub fn str(s: QuintName) -> Self {
        Value(Rc::new(ValueInner::Str(s)))
    }

    pub fn set(elems: impl IntoIterator<Item = Value>) -> Result<Self, QuintError> {
        let set = elems
            .into_iter()
            .map(|v| v.normalize())
            .collect::<Result<BTreeSet<_>, _>>()?;
        Ok(Value(Rc::new(ValueInner::Set(set))))
    }

    pub fn tuple(elems: impl IntoIterator<Item = Value>) -> Result<Self, QuintError> {
        let vs = elems
            .into_iter()
            .map(|v| v.normalize())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Value(Rc::new(ValueInner::Tuple(vs))))
    }

    pub fn list(elems: impl IntoIterator<Item = Value>) -> Result<Self, QuintError> {
        let vs = elems
            .into_iter()
            .map(|v| v.normalize())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Value(Rc::new(ValueInner::List(vs))))
    }

    pub fn record(
        fields: impl IntoIterator<Item = (QuintName, Value)>,
    ) -> Result<Self, QuintError> {
        let fs = fields
            .into_iter()
            .map(|(k, v)| Ok((k, v.normalize()?)))
            .collect::<Result<BTreeMap<_, _>, QuintError>>()?;
        Ok(Value(Rc::new(ValueInner::Record(fs))))
    }

    pub fn map(entries: impl IntoIterator<Item = (Value, Value)>) -> Result<Self, QuintError> {
        let m = entries
            .into_iter()
            .map(|(k, v)| Ok((k.normalize()?, v.normalize()?)))
            .collect::<Result<BTreeMap<_, _>, QuintError>>()?;
        Ok(Value(Rc::new(ValueInner::Map(m))))
    }

    /// Map constructor for keys/values that are already normalized.
    pub fn map_normalized(m: BTreeMap<Value, Value>) -> Self {
        Value(Rc::new(ValueInner::Map(m)))
    }

    pub fn set_normalized(s: BTreeSet<Value>) -> Self {
        Value(Rc::new(ValueInner::Set(s)))
    }

    pub fn variant(label: QuintName, payload: Value) -> Result<Self, QuintError> {
        Ok(Value(Rc::new(ValueInner::Variant(
            label,
            payload.normalize()?,
        ))))
    }

    pub fn lambda(registers: Vec<Rc<RefCell<Option<Value>>>>, body: CompiledExpr) -> Self {
        Value(Rc::new(ValueInner::Lambda(registers, body)))
    }

    pub fn interval(start: i64, end: i64) -> Self {
        Value(Rc::new(ValueInner::Interval(start, end)))
    }

    pub fn cross_product(sets: Vec<Value>) -> Self {
        Value(Rc::new(ValueInner::CrossProduct(sets)))
    }

    pub fn power_set(base: Value) -> Self {
        Value(Rc::new(ValueInner::PowerSet(base)))
    }

    pub fn map_set(domain: Value, range: Value) -> Self {
        Value(Rc::new(ValueInner::MapSet(domain, range)))
    }

    pub fn infinite_int() -> Self {
        Value(Rc::new(ValueInner::InfiniteInt))
    }

    pub fn infinite_nat() -> Self {
        Value(Rc::new(ValueInner::InfiniteNat))
    }
}

// ---------------------------------------------------------------------------
// Accessors (panic on type mismatch: input is type-checked by quint).
// ---------------------------------------------------------------------------

impl Value {
    pub fn as_int(&self) -> i64 {
        match self.0.as_ref() {
            ValueInner::Int(n) => *n,
            v => panic!("expected int, got {v:?}"),
        }
    }

    pub fn as_bool(&self) -> bool {
        match self.0.as_ref() {
            ValueInner::Bool(b) => *b,
            v => panic!("expected bool, got {v:?}"),
        }
    }

    pub fn as_str(&self) -> &QuintName {
        match self.0.as_ref() {
            ValueInner::Str(s) => s,
            v => panic!("expected str, got {v:?}"),
        }
    }

    pub fn as_map(&self) -> &BTreeMap<Value, Value> {
        match self.0.as_ref() {
            ValueInner::Map(m) => m,
            v => panic!("expected map, got {v:?}"),
        }
    }

    pub fn as_record(&self) -> &BTreeMap<QuintName, Value> {
        match self.0.as_ref() {
            ValueInner::Record(r) => r,
            v => panic!("expected record, got {v:?}"),
        }
    }

    /// Tuples and lists share representation semantics for indexing builtins.
    pub fn as_elems(&self) -> &Vec<Value> {
        match self.0.as_ref() {
            ValueInner::Tuple(vs) | ValueInner::List(vs) => vs,
            v => panic!("expected tuple/list, got {v:?}"),
        }
    }

    pub fn as_variant(&self) -> (&QuintName, &Value) {
        match self.0.as_ref() {
            ValueInner::Variant(label, payload) => (label, payload),
            v => panic!("expected variant, got {v:?}"),
        }
    }

    pub fn as_tuple2(&self) -> (&Value, &Value) {
        let elems = self.as_elems();
        (&elems[0], &elems[1])
    }

    pub fn is_set(&self) -> bool {
        matches!(self.0.as_ref(), ValueInner::Set(_)) || self.0.is_symbolic()
    }
}

// ---------------------------------------------------------------------------
// Symbolic set operations: cardinality / membership / subset / enumeration /
// indexed access — all without enumerating when structurally answerable.
// ---------------------------------------------------------------------------

impl Value {
    pub fn cardinality(&self) -> Result<u64, QuintError> {
        match self.0.as_ref() {
            ValueInner::Set(s) => Ok(s.len() as u64),
            ValueInner::Tuple(vs) | ValueInner::List(vs) => Ok(vs.len() as u64),
            ValueInner::Record(fields) => Ok(fields.len() as u64),
            ValueInner::Map(m) => Ok(m.len() as u64),
            ValueInner::Interval(start, end) => end
                .checked_sub(*start)
                .and_then(|d| d.checked_add(1))
                .and_then(|n| u64::try_from(n).ok())
                .ok_or_else(|| overflow("interval cardinality")),
            ValueInner::CrossProduct(sets) => sets.iter().try_fold(1u64, |acc, s| {
                acc.checked_mul(s.cardinality()?)
                    .ok_or_else(|| overflow("cross product cardinality"))
            }),
            ValueInner::PowerSet(base) => {
                let n = base.cardinality()?;
                let exp = u32::try_from(n).map_err(|_| overflow("powerset cardinality"))?;
                2u64
                    .checked_pow(exp)
                    .ok_or_else(|| overflow("powerset cardinality"))
            }
            ValueInner::MapSet(domain, range) => {
                let d = domain.cardinality()?;
                let r = range.cardinality()?;
                let exp = u32::try_from(d).map_err(|_| overflow("setOfMaps cardinality"))?;
                r.checked_pow(exp)
                    .ok_or_else(|| overflow("setOfMaps cardinality"))
            }
            ValueInner::InfiniteInt => Err(unsupported("infinite set Int")),
            ValueInner::InfiniteNat => Err(unsupported("infinite set Nat")),
            v => panic!("cardinality: not a set: {v:?}"),
        }
    }

    /// Set membership, structural where possible. `elem` must be normalized
    /// (guaranteed when it came out of evaluation of a non-set expression or
    /// any container).
    pub fn contains(&self, elem: &Value) -> Result<bool, QuintError> {
        Ok(match (self.0.as_ref(), elem.0.as_ref()) {
            (ValueInner::Set(s), _) => s.contains(elem),
            (ValueInner::Interval(start, end), ValueInner::Int(n)) => start <= n && n <= end,
            (ValueInner::Interval(_, _), _) => false,
            (ValueInner::CrossProduct(sets), ValueInner::Tuple(elems)) => {
                sets.len() == elems.len()
                    && sets
                        .iter()
                        .zip(elems)
                        .try_fold(true, |acc, (s, e)| {
                            Ok::<_, QuintError>(acc && s.contains(e)?)
                        })?
            }
            (ValueInner::CrossProduct(_), _) => false,
            (ValueInner::PowerSet(base), ValueInner::Set(elems)) => elems
                .iter()
                .try_fold(true, |acc, e| Ok::<_, QuintError>(acc && base.contains(e)?))?,
            (ValueInner::PowerSet(_), _) => false,
            (ValueInner::MapSet(domain, range), ValueInner::Map(m)) => {
                let map_domain = Value::set_normalized(m.keys().cloned().collect());
                value_eq(&map_domain, domain)?
                    && m.values()
                        .try_fold(true, |acc, v| Ok::<_, QuintError>(acc && range.contains(v)?))?
            }
            (ValueInner::MapSet(_, _), _) => false,
            (ValueInner::InfiniteInt, ValueInner::Int(_)) => true,
            (ValueInner::InfiniteInt, _) => false,
            (ValueInner::InfiniteNat, ValueInner::Int(n)) => *n >= 0,
            (ValueInner::InfiniteNat, _) => false,
            (v, _) => panic!("contains: not a set: {v:?}"),
        })
    }

    pub fn subseteq(&self, superset: &Value) -> Result<bool, QuintError> {
        Ok(match (self.0.as_ref(), superset.0.as_ref()) {
            (ValueInner::Set(a), ValueInner::Set(b)) => a.is_subset(b),
            (ValueInner::Interval(a1, a2), ValueInner::Interval(b1, b2)) => {
                a1 > a2 || (a1 >= b1 && a2 <= b2)
            }
            (ValueInner::CrossProduct(a), ValueInner::CrossProduct(b)) => {
                a.len() == b.len()
                    && a.iter().zip(b).try_fold(true, |acc, (x, y)| {
                        Ok::<_, QuintError>(acc && x.subseteq(y)?)
                    })?
            }
            (ValueInner::PowerSet(a), ValueInner::PowerSet(b)) => a.subseteq(b)?,
            (ValueInner::MapSet(ad, ar), ValueInner::MapSet(bd, br)) => {
                value_eq(ad, bd)? && ar.subseteq(br)?
            }
            (ValueInner::InfiniteNat, ValueInner::InfiniteNat | ValueInner::InfiniteInt) => true,
            (ValueInner::InfiniteInt, ValueInner::InfiniteInt) => true,
            (ValueInner::InfiniteInt | ValueInner::InfiniteNat, _) => false,
            (_, ValueInner::InfiniteInt | ValueInner::InfiniteNat) => self
                .enumerate()?
                .iter()
                .try_fold(true, |acc, v| Ok::<_, QuintError>(acc && superset.contains(v)?))?,
            _ => {
                let a = self.enumerate()?;
                let b = superset.enumerate()?;
                a.iter().all(|v| b.contains(v))
            }
        })
    }

    /// Materialize a (possibly symbolic) set as a canonical `BTreeSet`,
    /// guarded by [`MAX_ENUM`].
    pub fn enumerate(&self) -> Result<Cow<'_, BTreeSet<Value>>, QuintError> {
        match self.0.as_ref() {
            ValueInner::Set(s) => Ok(Cow::Borrowed(s)),
            ValueInner::InfiniteInt => Err(unsupported("enumerating infinite set Int")),
            ValueInner::InfiniteNat => Err(unsupported("enumerating infinite set Nat")),
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
                    return Ok(Cow::Owned(BTreeSet::new()));
                }
                let mut out = BTreeSet::new();
                let bounds = self.bounds()?;
                let mut indices = vec![0u64; bounds.len()];
                loop {
                    out.insert(self.pick(&mut indices.iter().copied())?.normalize()?);
                    // mixed-radix increment
                    let mut i = 0;
                    loop {
                        if i == bounds.len() {
                            return Ok(Cow::Owned(out));
                        }
                        indices[i] += 1;
                        if indices[i] < bounds[i] {
                            break;
                        }
                        indices[i] = 0;
                        i += 1;
                    }
                }
            }
        }
    }

    /// The mixed-radix bounds for indexed access (`pick`). One entry per
    /// independent index dimension.
    pub fn bounds(&self) -> Result<Vec<u64>, QuintError> {
        Ok(match self.0.as_ref() {
            ValueInner::Set(s) => vec![s.len() as u64],
            ValueInner::Interval(_, _) => vec![self.cardinality()?],
            ValueInner::CrossProduct(sets) => sets
                .iter()
                .map(|s| s.cardinality())
                .collect::<Result<Vec<_>, _>>()?,
            ValueInner::PowerSet(base) => {
                let n = base.cardinality()?;
                if n >= 63 {
                    return Err(unsupported(format!(
                        "powerset of a {n}-element set (2^{n} elements)"
                    )));
                }
                vec![1u64 << n]
            }
            ValueInner::MapSet(domain, range) => {
                let d = domain.cardinality()? as usize;
                let range_bounds = range.bounds()?;
                range_bounds.repeat(d)
            }
            ValueInner::InfiniteInt => return Err(unsupported("picking from infinite set Int")),
            ValueInner::InfiniteNat => return Err(unsupported("picking from infinite set Nat")),
            v => panic!("bounds: not a set: {v:?}"),
        })
    }

    /// Pick the element identified by `indexes` (one index per bound, in
    /// order). Deterministic thanks to BTree iteration order.
    pub fn pick<T: Iterator<Item = u64>>(&self, indexes: &mut T) -> Result<Value, QuintError> {
        Ok(match self.0.as_ref() {
            ValueInner::Set(s) => {
                let i = indexes.next().expect("too few pick indices (bug)");
                s.iter().nth(i as usize).cloned().expect("pick index out of bounds (bug)")
            }
            ValueInner::Interval(start, _) => {
                let i = indexes.next().expect("too few pick indices (bug)");
                Value::int(start + i as i64)
            }
            ValueInner::CrossProduct(sets) => {
                let elems = sets
                    .iter()
                    .map(|s| s.pick(indexes))
                    .collect::<Result<Vec<_>, _>>()?;
                Value::tuple(elems)?
            }
            ValueInner::PowerSet(base) => {
                let i = indexes.next().expect("too few pick indices (bug)");
                let base = base.enumerate()?;
                let mut elems = BTreeSet::new();
                for (j, e) in base.iter().enumerate() {
                    if i & (1u64 << j) != 0 {
                        elems.insert(e.clone());
                    }
                }
                Value::set_normalized(elems)
            }
            ValueInner::MapSet(domain, range) => {
                if domain.cardinality()? == 0 {
                    // TLC behavior: setOfMaps with empty domain = Set(Map())
                    return Ok(Value::map_normalized(BTreeMap::new()));
                }
                let range = if matches!(range.0.as_ref(), ValueInner::MapSet(_, _)) {
                    Value::set_normalized(range.enumerate()?.into_owned())
                } else {
                    range.clone()
                };
                let keys = domain.enumerate()?;
                let entries = keys
                    .iter()
                    .map(|k| Ok::<_, QuintError>((k.clone(), range.pick(indexes)?)))
                    .collect::<Result<Vec<_>, _>>()?;
                Value::map(entries)?
            }
            ValueInner::InfiniteInt | ValueInner::InfiniteNat => {
                return Err(unsupported("picking from an infinite set"))
            }
            v => panic!("pick: not a set: {v:?}"),
        })
    }

    /// Canonical form: symbolic set forms are materialized (recursively).
    /// Values built from containers are already canonical, so this is a
    /// cheap no-op in the common case.
    pub fn normalize(&self) -> Result<Value, QuintError> {
        if self.0.is_symbolic() {
            Ok(Value::set_normalized(self.enumerate()?.into_owned()))
        } else {
            Ok(self.clone())
        }
    }
}

/// Fallible equality that copes with symbolic operands (`S == Set(...)`,
/// `Interval(1,3) == 1.to(3)`), matching the reference semantics: mixed
/// representations are compared by enumeration; infinite sets are equal
/// only to themselves.
pub fn value_eq(a: &Value, b: &Value) -> Result<bool, QuintError> {
    let (ia, ib) = (a.0.as_ref(), b.0.as_ref());
    match (ia.is_symbolic(), ib.is_symbolic()) {
        (false, false) => Ok(a == b),
        _ => match (ia, ib) {
            (ValueInner::InfiniteInt, ValueInner::InfiniteInt) => Ok(true),
            (ValueInner::InfiniteNat, ValueInner::InfiniteNat) => Ok(true),
            (ValueInner::InfiniteInt | ValueInner::InfiniteNat, _)
            | (_, ValueInner::InfiniteInt | ValueInner::InfiniteNat) => Ok(false),
            (ValueInner::Interval(a1, a2), ValueInner::Interval(b1, b2)) => {
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
        match self.0.as_ref() {
            ValueInner::Int(n) => write!(f, "{n}"),
            ValueInner::Bool(b) => write!(f, "{b}"),
            ValueInner::Str(s) => write!(f, "{s:?}"),
            ValueInner::Set(_)
            | ValueInner::Interval(_, _)
            | ValueInner::CrossProduct(_)
            | ValueInner::PowerSet(_)
            | ValueInner::MapSet(_, _) => {
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
            ValueInner::InfiniteInt => write!(f, "Int"),
            ValueInner::InfiniteNat => write!(f, "Nat"),
            ValueInner::Tuple(vs) => {
                write!(f, "(")?;
                for (i, e) in vs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{e}")?;
                }
                write!(f, ")")
            }
            ValueInner::List(vs) => {
                write!(f, "[")?;
                for (i, e) in vs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{e}")?;
                }
                write!(f, "]")
            }
            ValueInner::Record(fields) => {
                write!(f, "{{ ")?;
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k}: {v}")?;
                }
                write!(f, " }}")
            }
            ValueInner::Map(m) => {
                write!(f, "Map(")?;
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k} -> {v}")?;
                }
                write!(f, ")")
            }
            ValueInner::Variant(label, payload) => {
                if matches!(payload.0.as_ref(), ValueInner::Tuple(vs) if vs.is_empty()) {
                    write!(f, "{label}")
                } else {
                    write!(f, "{label}({payload})")
                }
            }
            ValueInner::Lambda(_, _) => write!(f, "<lambda>"),
        }
    }
}
