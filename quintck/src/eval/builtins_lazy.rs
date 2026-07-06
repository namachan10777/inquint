//! Lazy builtin operators: control flow, actions, and choice points.
//! Arguments arrive as unevaluated closures.

use super::{apply_lambda, CompiledExpr, Env};
use crate::error::{unsupported, QuintError};
use crate::value::{EvalResult, Value};

pub type LazyFn = fn(&mut Env, &[CompiledExpr]) -> EvalResult;

pub const LAZY_OPS: &[&str] = &[
    "and",
    "or",
    "implies",
    "ite",
    "matchVariant",
    "actionAll",
    "actionAny",
    "oneOf",
    // rejected in v1 (run/temporal layer):
    "next",
    "then",
    "reps",
    "expect",
];

pub fn lazy_op(op: &str) -> LazyFn {
    match op {
        "and" => |env, args| {
            for arg in args {
                if !arg.execute(env)?.as_bool() {
                    return Ok(Value::bool(false));
                }
            }
            Ok(Value::bool(true))
        },
        "or" => |env, args| {
            for arg in args {
                if arg.execute(env)?.as_bool() {
                    return Ok(Value::bool(true));
                }
            }
            Ok(Value::bool(false))
        },
        "implies" => |env, args| {
            if !args[0].execute(env)?.as_bool() {
                return Ok(Value::bool(true));
            }
            args[1].execute(env)
        },
        "ite" => |env, args| {
            if args[0].execute(env)?.as_bool() {
                args[1].execute(env)
            } else {
                args[2].execute(env)
            }
        },
        "matchVariant" => |env, args| {
            let matched = args[0].execute(env)?;
            let (label, payload) = matched.as_variant();
            let matching_case = args[1..].chunks_exact(2).find_map(|case| {
                let case_label = case[0].execute(env).ok()?;
                if case_label.as_str() == label || case_label.as_str().as_ref() == "_" {
                    Some(&case[1])
                } else {
                    None
                }
            });
            match matching_case {
                Some(elim) => {
                    let closure = elim.execute(env)?;
                    apply_lambda(&closure, env, vec![payload.clone()])
                }
                None => Err(QuintError::new(
                    "QNT505",
                    format!("No match for variant {label}"),
                )),
            }
        },
        // Execute all conjunct actions; roll back next-state writes if any
        // of them is disabled.
        "actionAll" => |env, args| {
            let snapshot = env.storage.borrow().snapshot_next();
            for action in args {
                if !action.execute(env)?.as_bool() {
                    env.storage.borrow().restore_next(&snapshot);
                    return Ok(Value::bool(false));
                }
            }
            Ok(Value::bool(true))
        },
        // Choice point: execute exactly ONE branch, selected by the oracle.
        // Every enabled branch becomes a distinct successor via a distinct
        // choice trail (the reference simulator instead shuffles and takes
        // the first enabled branch).
        "actionAny" => |env, args| {
            let Some(i) = env.choose(args.len() as u64)? else {
                return Ok(Value::bool(false));
            };
            let snapshot = env.storage.borrow().snapshot_next();
            let result = args[i as usize].execute(env)?;
            if result.as_bool() {
                Ok(Value::bool(true))
            } else {
                env.storage.borrow().restore_next(&snapshot);
                Ok(Value::bool(false))
            }
        },
        // Bare oneOf (outside a nondet let): a choice point over the set.
        // Empty set is a hard error, matching the reference (QNT509).
        "oneOf" => |env, args| {
            let set = args[0].execute(env)?;
            let bounds = set.bounds()?;
            let mut indices = Vec::with_capacity(bounds.len());
            for bound in bounds {
                match env.choose(bound)? {
                    Some(i) => indices.push(i),
                    None => {
                        return Err(QuintError::new("QNT509", "Applied oneOf on an empty set"))
                    }
                }
            }
            set.pick(&mut indices.into_iter())
        },
        "next" | "then" | "reps" | "expect" => {
            |_, _| Err(unsupported("run/temporal operator (then/reps/expect/next)"))
        }
        _ => panic!("unknown lazy op: {op}"),
    }
}
