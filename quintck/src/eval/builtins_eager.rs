//! First-order eager builtin operators: arguments are evaluated before
//! application. Semantics and error codes ported from the reference
//! `builtins.rs`. Higher-order builtins (fold/map/filter/...) live in
//! `vm::builtins`, which dispatches here for everything first-order.
//!
//! Values are interned ids: container updates copy flat id slices (memcpy)
//! and re-intern; equality is an id compare; membership is a binary search
//! over the canonical sorted order.

use super::Env;
use crate::error::{unsupported, QuintError};
use crate::value::{value_cmp, value_eq, EvalResult, Value};

pub type EagerFn = fn(&mut Env, &[Value]) -> EvalResult;

fn at_index(list: &[Value], index: i64) -> EvalResult {
    if index < 0 || index as usize >= list.len() {
        return Err(QuintError::new(
            "QNT510",
            format!("Out of bounds, nth({index})"),
        ));
    }
    Ok(list[index as usize])
}

fn checked(op: &str, a: i64, b: i64, r: Option<i64>) -> EvalResult {
    r.map(Value::int).ok_or_else(|| {
        QuintError::new(
            "QNT601",
            format!("Integer overflow in arithmetic operations: {a} {op} {b}"),
        )
    })
}

/// Look up the implementation of an eager builtin by opcode.
/// Returns None for unknown opcodes (then it must be a user-defined op).
pub fn eager_op(op: &str) -> Option<EagerFn> {
    Some(match op {
        "Set" => |_, args| Value::set(args.iter().copied()),
        "Rec" => |_, args| {
            Value::record(
                args.chunks_exact(2)
                    .map(|kv| (kv[0].as_str(), kv[1])),
            )
        },
        "Tup" => |_, args| Value::tuple(args.iter().copied()),
        "List" => |_, args| Value::list(args.iter().copied()),
        "Map" => |_, args| Value::map(args.iter().map(|kv| kv.as_tuple2())),
        "variant" => |_, args| Value::variant(args[0].as_str(), args[1]),
        "not" => |_, args| Ok(Value::bool(!args[0].as_bool())),
        "iff" => |_, args| Ok(Value::bool(args[0].as_bool() == args[1].as_bool())),
        "eq" => |_, args| Ok(Value::bool(value_eq(args[0], args[1])?)),
        "neq" => |_, args| Ok(Value::bool(!value_eq(args[0], args[1])?)),
        "iadd" => |_, args| {
            let (a, b) = (args[0].as_int(), args[1].as_int());
            checked("+", a, b, a.checked_add(b))
        },
        "isub" => |_, args| {
            let (a, b) = (args[0].as_int(), args[1].as_int());
            checked("-", a, b, a.checked_sub(b))
        },
        "imul" => |_, args| {
            let (a, b) = (args[0].as_int(), args[1].as_int());
            checked("*", a, b, a.checked_mul(b))
        },
        "idiv" => |_, args| {
            let (a, b) = (args[0].as_int(), args[1].as_int());
            if b == 0 {
                return Err(QuintError::new("QNT503", "Division by zero"));
            }
            checked("/", a, b, a.checked_div(b))
        },
        "imod" => |_, args| {
            let (a, b) = (args[0].as_int(), args[1].as_int());
            checked("%", a, b, a.checked_rem(b))
        },
        "ipow" => |_, args| {
            let (base, exp) = (args[0].as_int(), args[1].as_int());
            if base == 0 && exp == 0 {
                return Err(QuintError::new("QNT503", "0^0 is undefined"));
            }
            if exp < 0 {
                return Err(QuintError::new("QNT503", "i^j is undefined for j < 0"));
            }
            let r = u32::try_from(exp).ok().and_then(|e| base.checked_pow(e));
            checked("^", base, exp, r)
        },
        "iuminus" => |_, args| {
            let a = args[0].as_int();
            checked("-", 0, a, a.checked_neg())
        },
        "ilt" => |_, args| Ok(Value::bool(args[0].as_int() < args[1].as_int())),
        "ilte" => |_, args| Ok(Value::bool(args[0].as_int() <= args[1].as_int())),
        "igt" => |_, args| Ok(Value::bool(args[0].as_int() > args[1].as_int())),
        "igte" => |_, args| Ok(Value::bool(args[0].as_int() >= args[1].as_int())),
        // Tuples are 1-indexed (_1, _2, ...)
        "item" => |_, args| at_index(args[0].as_elems(), args[1].as_int() - 1),
        "tuples" => |_, args| Ok(Value::cross_product(args.to_vec())),
        "range" => |_, args| {
            let (start, end) = (args[0].as_int(), args[1].as_int());
            Ok(Value::list_of_normalized((start..end).map(Value::int).collect()))
        },
        "nth" => |_, args| at_index(args[0].as_elems(), args[1].as_int()),
        "replaceAt" => |_, args| {
            let list = args[0].as_elems();
            let index = args[1].as_int();
            if index < 0 || index as usize >= list.len() {
                return Err(QuintError::new(
                    "QNT510",
                    format!("Out of bounds, replaceAt({index})"),
                ));
            }
            let mut out = list.to_vec();
            out[index as usize] = args[2].normalize()?;
            Ok(Value::list_of_normalized(out))
        },
        "head" => |_, args| match args[0].as_elems().first() {
            Some(h) => Ok(*h),
            None => Err(QuintError::new("QNT505", "Called 'head' on an empty list")),
        },
        "tail" => |_, args| {
            let list = args[0].as_elems();
            if list.is_empty() {
                Err(QuintError::new("QNT505", "Called 'tail' on an empty list"))
            } else {
                Ok(Value::list_of_normalized(list[1..].to_vec()))
            }
        },
        "slice" => |_, args| {
            let list = args[0].as_elems();
            let (start, end) = (args[1].as_int(), args[2].as_int());
            if start >= 0 && end >= start && (end as usize) <= list.len() {
                Ok(Value::list_of_normalized(
                    list[start as usize..end as usize].to_vec(),
                ))
            } else {
                Err(QuintError::new(
                    "QNT506",
                    format!(
                        "slice(..., {start}, {end}) applied to a list of size {}",
                        list.len()
                    ),
                ))
            }
        },
        "length" => |_, args| int_from_card(args[0].cardinality()?),
        "append" => |_, args| {
            let list = args[0].as_elems();
            let mut out = Vec::with_capacity(list.len() + 1);
            out.extend_from_slice(list);
            out.push(args[1].normalize()?);
            Ok(Value::list_of_normalized(out))
        },
        "concat" => |_, args| {
            let (a, b) = (args[0].as_elems(), args[1].as_elems());
            let mut out = Vec::with_capacity(a.len() + b.len());
            out.extend_from_slice(a);
            out.extend_from_slice(b);
            Ok(Value::list_of_normalized(out))
        },
        "indices" => |_, args| {
            let size = i64::try_from(args[0].cardinality()?)
                .map_err(|_| crate::error::overflow("indices"))?;
            Ok(Value::interval(0, size - 1))
        },
        "field" => |_, args| {
            Ok(args[0]
                .record_field(args[1].as_str())
                .expect("field: no such field (type checker bug?)"))
        },
        "fieldNames" => |_, args| {
            let (shape, _) = args[0].as_record();
            Value::set(shape.fields.iter().map(|k| Value::str(*k)))
        },
        "with" => |_, args| {
            let (shape, values) = args[0].as_record();
            let field = args[1].as_str();
            let i = shape
                .fields
                .iter()
                .position(|&f| f == field)
                .expect("with: no such field (type checker bug?)");
            let mut out = crate::value::ValueBuf::from_slice(values);
            out.as_mut_slice()[i] = args[2].normalize()?;
            Ok(Value::record_shaped(shape, out.as_slice()))
        },
        "powerset" => |_, args| Ok(Value::power_set(args[0])),
        "contains" => |_, args| Ok(Value::bool(args[0].contains(args[1].normalize()?)?)),
        "in" => |_, args| Ok(Value::bool(args[1].contains(args[0].normalize()?)?)),
        "subseteq" => |_, args| Ok(Value::bool(args[0].subseteq(args[1])?)),
        "exclude" => |_, args| {
            let a = args[0].enumerate()?;
            let mut out = Vec::new();
            for v in a.iter() {
                if !args[1].contains(*v)? {
                    out.push(*v);
                }
            }
            // filtering a sorted slice keeps it sorted
            Ok(Value::set_sorted(out))
        },
        "union" => |_, args| {
            let a = args[0].enumerate()?;
            let b = args[1].enumerate()?;
            Ok(Value::set_sorted(merge_sorted(&a, &b)))
        },
        "intersect" => |_, args| {
            let a = args[0].enumerate()?;
            let mut out = Vec::new();
            for v in a.iter() {
                if args[1].contains(*v)? {
                    out.push(*v);
                }
            }
            Ok(Value::set_sorted(out))
        },
        "size" => |_, args| int_from_card(args[0].cardinality()?),
        // Only finite sets are supported, so this is constant.
        "isFinite" => |_, _| Ok(Value::bool(true)),
        "to" => |_, args| {
            let (start, end) = (args[0].as_int(), args[1].as_int());
            if start > end {
                // canonical empty set instead of an empty interval
                Value::set(std::iter::empty())
            } else {
                Ok(Value::interval(start, end))
            }
        },
        "flatten" => |_, args| {
            let outer = args[0].enumerate()?;
            let mut result = Vec::new();
            for inner in outer.iter() {
                result.extend_from_slice(&inner.enumerate()?);
            }
            Ok(Value::set_of_normalized(result))
        },
        "get" => |_, args| {
            let key = args[1].normalize()?;
            args[0].map_get(key).ok_or_else(|| {
                QuintError::new(
                    "QNT507",
                    format!(
                        "Called 'get' with a non-existing key. Key is {key}. Map has keys: {}",
                        args[0]
                            .as_map()
                            .iter()
                            .map(|(k, _)| k.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                )
            })
        },
        "set" => |_, args| {
            let entries = args[0].as_map();
            let key = args[1].normalize()?;
            match entries.binary_search_by(|(k, _)| value_cmp(*k, key)) {
                Ok(i) => {
                    let mut out = entries.to_vec();
                    out[i].1 = args[2].normalize()?;
                    Ok(Value::map_sorted(out))
                }
                Err(_) => Err(QuintError::new(
                    "QNT507",
                    "Called 'set' with a non-existing key",
                )),
            }
        },
        "put" => |_, args| {
            let entries = args[0].as_map();
            let key = args[1].normalize()?;
            let value = args[2].normalize()?;
            let mut out = entries.to_vec();
            match out.binary_search_by(|(k, _)| value_cmp(*k, key)) {
                Ok(i) => out[i].1 = value,
                Err(i) => out.insert(i, (key, value)),
            }
            Ok(Value::map_sorted(out))
        },
        "keys" => |_, args| {
            // map keys are sorted by value_cmp already
            Ok(Value::set_sorted(
                args[0].as_map().iter().map(|(k, _)| *k).collect(),
            ))
        },
        "setToMap" => |_, args| {
            Value::map(args[0].enumerate()?.iter().map(|kv| kv.as_tuple2()))
        },
        "setOfMaps" => |_, args| Ok(Value::map_set(args[0], args[1])),
        "fail" => |_, args| Ok(Value::bool(!args[0].as_bool())),
        "assert" => |_, args| {
            if !args[0].as_bool() {
                return Err(QuintError::new("QNT508", "Assertion failed"));
            }
            Ok(Value::bool(true))
        },
        "getOnlyElement" => |_, args| {
            let set = args[0].enumerate()?;
            if set.len() != 1 {
                return Err(QuintError::new(
                    "QNT505",
                    format!(
                        "Called 'getOnlyElement' on a set with {} elements. \
                         Make sure the set has exactly one element.",
                        set.len()
                    ),
                ));
            }
            Ok(set[0])
        },
        "q::debug" => |_, args| {
            eprintln!("> {} {}", args[0].as_str(), args[1]);
            Ok(args[1])
        },
        // All lists over a set with length <= n.
        "allListsUpTo" => |_, args| {
            let elems = args[0].enumerate()?;
            let max_len = args[1].as_int().max(0) as u32;
            let count = (elems.len() as u64)
                .checked_pow(max_len)
                .filter(|c| *c <= crate::value::MAX_ENUM)
                .ok_or_else(|| unsupported("allListsUpTo of this size"))?;
            let _ = count;
            let mut lists: Vec<Vec<Value>> = vec![Vec::new()];
            let mut frontier: Vec<Vec<Value>> = vec![Vec::new()];
            for _ in 0..max_len {
                let mut next_frontier = Vec::new();
                for list in &frontier {
                    for e in elems.iter() {
                        let mut l = list.clone();
                        l.push(*e);
                        next_frontier.push(l);
                    }
                }
                lists.extend(next_frontier.iter().cloned());
                frontier = next_frontier;
            }
            Value::set(
                lists
                    .into_iter()
                    .map(Value::list_of_normalized)
                    .collect::<Vec<_>>(),
            )
        },
        // Deterministic pick: the canonical (structurally least) element.
        "chooseSome" => |_, args| {
            args[0]
                .enumerate()?
                .first()
                .copied()
                .ok_or_else(|| QuintError::new("QNT505", "Called 'chooseSome' on an empty set"))
        },
        "allLists" | "always" | "eventually" | "enabled" | "orKeep" | "mustChange"
        | "weakFair" | "strongFair" | "leadsTo" => {
            |_, _| Err(unsupported("this built-in operator"))
        }
        _ => return None,
    })
}

/// Merge two value_cmp-sorted dedup'd slices into one (set union).
fn merge_sorted(a: &[Value], b: &[Value]) -> Vec<Value> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match value_cmp(a[i], b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

fn int_from_card(card: u64) -> EvalResult {
    i64::try_from(card)
        .map(Value::int)
        .map_err(|_| crate::error::overflow("size conversion"))
}
