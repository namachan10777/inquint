//! Eager builtin operators: arguments are evaluated before application.
//! Semantics and error codes ported from the reference `builtins.rs`.

use super::{apply_lambda, Env};
use crate::error::{unsupported, QuintError};
use crate::value::{value_eq, EvalResult, Value, ValueInner};
use std::collections::BTreeMap;

pub type EagerFn = fn(&mut Env, Vec<Value>) -> EvalResult;

fn at_index(list: &[Value], index: i64) -> EvalResult {
    if index < 0 || index as usize >= list.len() {
        return Err(QuintError::new(
            "QNT510",
            format!("Out of bounds, nth({index})"),
        ));
    }
    Ok(list[index as usize].clone())
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
        "Set" => |_, args| Value::set(args),
        "Rec" => |_, args| {
            Value::record(
                args.chunks_exact(2)
                    .map(|kv| (kv[0].as_str().clone(), kv[1].clone())),
            )
        },
        "Tup" => |_, args| Value::tuple(args),
        "List" => |_, args| Value::list(args),
        "Map" => |_, args| {
            Value::map(args.iter().map(|kv| {
                let (k, v) = kv.as_tuple2();
                (k.clone(), v.clone())
            }))
        },
        "variant" => |_, args| Value::variant(args[0].as_str().clone(), args[1].clone()),
        "not" => |_, args| Ok(Value::bool(!args[0].as_bool())),
        "iff" => |_, args| Ok(Value::bool(args[0].as_bool() == args[1].as_bool())),
        "eq" => |_, args| Ok(Value::bool(value_eq(&args[0], &args[1])?)),
        "neq" => |_, args| Ok(Value::bool(!value_eq(&args[0], &args[1])?)),
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
        "tuples" => |_, args| Ok(Value::cross_product(args)),
        "range" => |_, args| {
            let (start, end) = (args[0].as_int(), args[1].as_int());
            Value::list((start..end).map(Value::int))
        },
        "nth" => |_, args| at_index(args[0].as_elems(), args[1].as_int()),
        "replaceAt" => |_, args| {
            let mut list = args[0].as_elems().clone();
            let index = args[1].as_int();
            if index < 0 || index as usize >= list.len() {
                return Err(QuintError::new(
                    "QNT510",
                    format!("Out of bounds, replaceAt({index})"),
                ));
            }
            list[index as usize] = args[2].clone().normalize()?;
            Value::list(list)
        },
        "head" => |_, args| match args[0].as_elems().first() {
            Some(h) => Ok(h.clone()),
            None => Err(QuintError::new("QNT505", "Called 'head' on an empty list")),
        },
        "tail" => |_, args| {
            let list = args[0].as_elems();
            if list.is_empty() {
                Err(QuintError::new("QNT505", "Called 'tail' on an empty list"))
            } else {
                Value::list(list[1..].iter().cloned())
            }
        },
        "slice" => |_, args| {
            let list = args[0].as_elems();
            let (start, end) = (args[1].as_int(), args[2].as_int());
            if start >= 0 && end >= start && (end as usize) <= list.len() {
                Value::list(list[start as usize..end as usize].iter().cloned())
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
            let mut list = args[0].as_elems().clone();
            list.push(args[1].clone().normalize()?);
            Value::list(list)
        },
        "concat" => |_, args| {
            let mut list = args[0].as_elems().clone();
            list.extend(args[1].as_elems().iter().cloned());
            Value::list(list)
        },
        "indices" => |_, args| {
            let size = i64::try_from(args[0].cardinality()?)
                .map_err(|_| crate::error::overflow("indices"))?;
            Ok(Value::interval(0, size - 1))
        },
        "field" => |_, args| {
            Ok(args[0]
                .as_record()
                .get(args[1].as_str())
                .expect("field: no such field (type checker bug?)")
                .clone())
        },
        "fieldNames" => |_, args| {
            Value::set(args[0].as_record().keys().map(|k| Value::str(k.clone())))
        },
        "with" => |_, args| {
            let mut record = args[0].as_record().clone();
            record.insert(args[1].as_str().clone(), args[2].clone().normalize()?);
            Ok(Value(std::rc::Rc::new(ValueInner::Record(record))))
        },
        "powerset" => |_, args| Ok(Value::power_set(args[0].clone())),
        "contains" => |_, args| Ok(Value::bool(args[0].contains(&args[1].normalize()?)?)),
        "in" => |_, args| Ok(Value::bool(args[1].contains(&args[0].normalize()?)?)),
        "subseteq" => |_, args| Ok(Value::bool(args[0].subseteq(&args[1])?)),
        "exclude" => |_, args| {
            let a = args[0].enumerate()?.into_owned();
            let b = args[1].enumerate()?;
            Ok(Value::set_normalized(
                a.into_iter().filter(|v| !b.contains(v)).collect(),
            ))
        },
        "union" => |_, args| {
            let mut a = args[0].enumerate()?.into_owned();
            a.extend(args[1].enumerate()?.iter().cloned());
            Ok(Value::set_normalized(a))
        },
        "intersect" => |_, args| {
            let a = args[0].enumerate()?.into_owned();
            let b = args[1].enumerate()?;
            Ok(Value::set_normalized(
                a.into_iter().filter(|v| b.contains(v)).collect(),
            ))
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
        "fold" => |env, args| {
            let set = args[0].enumerate()?.into_owned();
            let mut acc = args[1].clone();
            for v in set {
                acc = apply_lambda(&args[2], env, vec![acc, v])?;
            }
            Ok(acc)
        },
        "foldl" => |env, args| {
            let mut acc = args[1].clone();
            for v in args[0].as_elems().clone() {
                acc = apply_lambda(&args[2], env, vec![acc, v])?;
            }
            Ok(acc)
        },
        "foldr" => |env, args| {
            let mut acc = args[1].clone();
            for v in args[0].as_elems().clone().into_iter().rev() {
                acc = apply_lambda(&args[2], env, vec![v, acc])?;
            }
            Ok(acc)
        },
        "flatten" => |_, args| {
            let outer = args[0].enumerate()?.into_owned();
            let mut result = std::collections::BTreeSet::new();
            for inner in outer {
                result.extend(inner.enumerate()?.iter().cloned());
            }
            Ok(Value::set_normalized(result))
        },
        "get" => |_, args| {
            let map = args[0].as_map();
            let key = args[1].normalize()?;
            map.get(&key).cloned().ok_or_else(|| {
                QuintError::new(
                    "QNT507",
                    format!(
                        "Called 'get' with a non-existing key. Key is {key}. Map has keys: {}",
                        map.keys()
                            .map(|k| k.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                )
            })
        },
        "set" => |_, args| {
            let mut map = args[0].as_map().clone();
            let key = args[1].normalize()?;
            if !map.contains_key(&key) {
                return Err(QuintError::new(
                    "QNT507",
                    "Called 'set' with a non-existing key",
                ));
            }
            map.insert(key, args[2].clone().normalize()?);
            Ok(Value::map_normalized(map))
        },
        "put" => |_, args| {
            let mut map = args[0].as_map().clone();
            map.insert(args[1].normalize()?, args[2].clone().normalize()?);
            Ok(Value::map_normalized(map))
        },
        "setBy" => |env, args| {
            let mut map = args[0].as_map().clone();
            let key = args[1].normalize()?;
            match map.get(&key).cloned() {
                Some(old) => {
                    let new = apply_lambda(&args[2], env, vec![old])?.normalize()?;
                    map.insert(key, new);
                    Ok(Value::map_normalized(map))
                }
                None => Err(QuintError::new(
                    "QNT507",
                    format!("Called 'setBy' with a non-existing key {key}"),
                )),
            }
        },
        "keys" => |_, args| Ok(Value::set_normalized(args[0].as_map().keys().cloned().collect())),
        "exists" => |env, args| {
            for v in args[0].enumerate()?.into_owned() {
                if apply_lambda(&args[1], env, vec![v])?.as_bool() {
                    return Ok(Value::bool(true));
                }
            }
            Ok(Value::bool(false))
        },
        "forall" => |env, args| {
            for v in args[0].enumerate()?.into_owned() {
                if !apply_lambda(&args[1], env, vec![v])?.as_bool() {
                    return Ok(Value::bool(false));
                }
            }
            Ok(Value::bool(true))
        },
        "map" => |env, args| {
            let set = args[0].enumerate()?.into_owned();
            let mut out = std::collections::BTreeSet::new();
            for v in set {
                out.insert(apply_lambda(&args[1], env, vec![v])?.normalize()?);
            }
            Ok(Value::set_normalized(out))
        },
        "filter" => |env, args| {
            let set = args[0].enumerate()?.into_owned();
            let mut out = std::collections::BTreeSet::new();
            for v in set {
                if apply_lambda(&args[1], env, vec![v.clone()])?.as_bool() {
                    out.insert(v);
                }
            }
            Ok(Value::set_normalized(out))
        },
        "select" => |env, args| {
            let mut out = Vec::new();
            for v in args[0].as_elems().clone() {
                if apply_lambda(&args[1], env, vec![v.clone()])?.as_bool() {
                    out.push(v);
                }
            }
            Value::list(out)
        },
        "mapBy" => |env, args| {
            let keys = args[0].enumerate()?.into_owned();
            let mut out = BTreeMap::new();
            for k in keys {
                let v = apply_lambda(&args[1], env, vec![k.clone()])?.normalize()?;
                out.insert(k, v);
            }
            Ok(Value::map_normalized(out))
        },
        "setToMap" => |_, args| {
            Value::map(args[0].enumerate()?.iter().map(|kv| {
                let (k, v) = kv.as_tuple2();
                (k.clone(), v.clone())
            }))
        },
        "setOfMaps" => |_, args| Ok(Value::map_set(args[0].clone(), args[1].clone())),
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
            Ok(set.iter().next().cloned().unwrap())
        },
        "q::debug" => |_, args| Ok(args[1].clone()),
        "allLists" | "allListsUpTo" | "chooseSome" | "always" | "eventually" | "enabled"
        | "orKeep" | "mustChange" | "weakFair" | "strongFair" | "leadsTo" => {
            |_, _| Err(unsupported("this built-in operator"))
        }
        _ => return None,
    })
}

fn int_from_card(card: u64) -> EvalResult {
    i64::try_from(card)
        .map(Value::int)
        .map_err(|_| crate::error::overflow("size conversion"))
}
