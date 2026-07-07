//! Builtin dispatch for the VM.
//!
//! First-order builtins reuse the eager table (`eval::builtins_eager`)
//! directly — they never apply lambdas, so they are engine-agnostic.
//! Higher-order builtins are reimplemented on top of [`Vm::call_lambda`]
//! (the closure engine's `apply_lambda` would re-enter the VM's `RefCell`).

use super::Vm;
use crate::error::QuintError;
use crate::eval::builtins_eager::{eager_op, EagerFn};
use crate::eval::Env;
use crate::value::{value_cmp, EvalResult, Value};

pub type HoFn = fn(&mut Vm, &mut Env, &[Value]) -> EvalResult;

#[derive(Clone, Copy)]
pub enum VmBuiltin {
    Simple(EagerFn),
    Ho(HoFn),
}

/// Look up a VM builtin by opcode (None → user-defined op).
pub fn vm_builtin(op: &str) -> Option<VmBuiltin> {
    let ho: HoFn = match op {
        "fold" => |vm, env, args| {
            let set = args[0].enumerate()?;
            let mut acc = args[1];
            for v in set.iter() {
                acc = vm.call_lambda(env, args[2], &[acc, *v])?;
            }
            Ok(acc)
        },
        "foldl" => |vm, env, args| {
            let mut acc = args[1];
            for v in args[0].as_elems() {
                acc = vm.call_lambda(env, args[2], &[acc, *v])?;
            }
            Ok(acc)
        },
        "foldr" => |vm, env, args| {
            let mut acc = args[1];
            for v in args[0].as_elems().iter().rev() {
                acc = vm.call_lambda(env, args[2], &[*v, acc])?;
            }
            Ok(acc)
        },
        "exists" => |vm, env, args| {
            for v in args[0].enumerate()?.iter() {
                if vm.call_lambda(env, args[1], &[*v])?.as_bool() {
                    return Ok(Value::bool(true));
                }
            }
            Ok(Value::bool(false))
        },
        "forall" => |vm, env, args| {
            for v in args[0].enumerate()?.iter() {
                if !vm.call_lambda(env, args[1], &[*v])?.as_bool() {
                    return Ok(Value::bool(false));
                }
            }
            Ok(Value::bool(true))
        },
        "map" => |vm, env, args| {
            let set = args[0].enumerate()?;
            let mut out = Vec::with_capacity(set.len());
            for v in set.iter() {
                out.push(vm.call_lambda(env, args[1], &[*v])?.normalize()?);
            }
            Ok(Value::set_of_normalized(out))
        },
        "filter" => |vm, env, args| {
            let set = args[0].enumerate()?;
            let mut out = Vec::new();
            for v in set.iter() {
                if vm.call_lambda(env, args[1], &[*v])?.as_bool() {
                    out.push(*v);
                }
            }
            // filtering a sorted slice keeps it sorted
            Ok(Value::set_sorted(out))
        },
        "select" => |vm, env, args| {
            let mut out = Vec::new();
            for v in args[0].as_elems() {
                if vm.call_lambda(env, args[1], &[*v])?.as_bool() {
                    out.push(*v);
                }
            }
            Ok(Value::list_of_normalized(out))
        },
        "mapBy" => |vm, env, args| {
            let keys = args[0].enumerate()?;
            let mut out = Vec::with_capacity(keys.len());
            for k in keys.iter() {
                let v = vm.call_lambda(env, args[1], &[*k])?.normalize()?;
                out.push((*k, v));
            }
            // keys come sorted out of enumerate
            Ok(Value::map_sorted(out))
        },
        "setBy" => |vm, env, args| {
            let entries = args[0].as_map();
            let key = args[1].normalize()?;
            match entries.binary_search_by(|(k, _)| value_cmp(*k, key)) {
                Ok(i) => {
                    let new = vm.call_lambda(env, args[2], &[entries[i].1])?.normalize()?;
                    let mut out = entries.to_vec();
                    out[i].1 = new;
                    Ok(Value::map_sorted(out))
                }
                Err(_) => Err(QuintError::new(
                    "QNT507",
                    format!("Called 'setBy' with a non-existing key {key}"),
                )),
            }
        },
        _ => return eager_op(op).map(VmBuiltin::Simple),
    };
    Some(VmBuiltin::Ho(ho))
}
