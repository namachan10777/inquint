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
        // Choice point: the oracle picks a branch. In checking mode a
        // disabled pick is simply a failed path (fail-fast; the successor
        // set is unaffected). In run-test mode (`any_fallthrough`) the
        // first enabled branch from the pick onward (cyclically) executes,
        // so `any` returns false only when NO branch is enabled — matching
        // quint's "choose among enabled branches" semantics. Exhaustive
        // enumeration covers every enabled branch in both modes.
        "actionAny" => |env, args| {
            let n = args.len();
            let Some(start) = env.choose(n as u64)? else {
                return Ok(Value::bool(false));
            };
            let snapshot = env.storage.borrow().snapshot_next();
            let tries = if env.any_fallthrough { n } else { 1 };
            for k in 0..tries {
                let branch = (start as usize + k) % n;
                let result = args[branch].execute(env)?;
                if result.as_bool() {
                    return Ok(Value::bool(true));
                }
                env.storage.borrow().restore_next(&snapshot);
            }
            Ok(Value::bool(false))
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
        // next(x): evaluate the argument against the next-state register
        // bank. Only legal inside temporal edge-atom evaluation.
        "next" => |env, args| {
            if !env.next_allowed {
                return Err(unsupported("next() outside a temporal property"));
            }
            let saved = env.next_mode;
            env.next_mode = true;
            let result = args[0].execute(env);
            env.next_mode = saved;
            result
        },
        // Run-test operators. `a.then(b)`: run a, commit the state, run b.
        "then" => |env, args| {
            let first = args[0].execute(env)?;
            if !first.as_bool() {
                return Err(QuintError::new(
                    "QNT513",
                    "Cannot continue in `then` because the highlighted expression evaluated to false",
                ));
            }
            env.storage.borrow().shift();
            args[1].execute(env)
        },
        // n.reps(i => A(i)): run A n times, committing between iterations.
        "reps" => |env, args| {
            let reps = args[0].execute(env)?.as_int();
            let mut result = Value::bool(true);
            for i in 0..reps {
                let closure = args[1].execute(env)?;
                result = apply_lambda(&closure, env, vec![Value::int(i)])?;
                if !result.as_bool() {
                    return Err(QuintError::new(
                        "QNT513",
                        format!(
                            "Reps loop could not continue after iteration #{} evaluated to false",
                            i + 1
                        ),
                    ));
                }
                if i < reps - 1 {
                    env.storage.borrow().shift();
                }
            }
            Ok(result)
        },
        // a.expect(p): a must be enabled; p must hold in a's post-state;
        // the net effect on the state is exactly a's.
        "expect" => |env, args| {
            let action_result = args[0].execute(env)?;
            if !action_result.as_bool() {
                return Err(QuintError::new("QNT508", "Cannot continue to \"expect\""));
            }
            let snapshot = env.storage.borrow().snapshot_next();
            env.storage.borrow().shift();
            let predicate = args[1].execute(env)?;
            env.storage.borrow().restore_next(&snapshot);
            if !predicate.as_bool() {
                return Err(QuintError::new(
                    "QNT508",
                    "Expect condition does not hold true",
                ));
            }
            Ok(Value::bool(true))
        },
        _ => panic!("unknown lazy op: {op}"),
    }
}
