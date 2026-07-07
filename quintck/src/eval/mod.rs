//! The evaluation runtime shared surface: the per-run environment and
//! quantifier-bound expressions used by the temporal layer.
//!
//! Nondeterminism (`any`, `nondet`/`oneOf`) goes through the
//! [`crate::choice::ChoiceCtl`] oracle instead of a RNG, so successors can
//! be enumerated exhaustively.

pub mod builtins_eager;

use crate::choice::ChoiceCtl;
use crate::error::QuintError;
use crate::value::{EvalResult, Value};
use crate::vm::{FnId, Vm};

/// Per-run evaluation environment. Carries only run-scoped flags and a
/// mutable borrow of the choice oracle — no shared ownership.
pub struct Env<'c> {
    /// The choice oracle. `None` during invariant evaluation, where any
    /// nondeterminism is an error.
    pub choices: Option<&'c mut ChoiceCtl>,
    /// When true, state-variable reads take the `next` bank — this is how
    /// `next(x)` is evaluated inside temporal edge atoms.
    pub next_mode: bool,
    /// `next(...)` is only legal while evaluating a temporal edge atom.
    pub next_allowed: bool,
    /// Run-test mode: `any` falls through to the next enabled branch when
    /// the chosen one is disabled (quint's "choose among enabled"
    /// semantics). The model checker keeps disabled picks as cheap failed
    /// paths instead — the successor sets are identical either way, but
    /// fail-fast is significantly faster on specs with many guards.
    pub any_fallthrough: bool,
}

impl<'c> Env<'c> {
    pub fn new(choices: Option<&'c mut ChoiceCtl>) -> Self {
        Env {
            choices,
            next_mode: false,
            next_allowed: false,
            any_fallthrough: false,
        }
    }

    /// Make a choice among `bound` alternatives, or fail if nondeterminism
    /// is not allowed in this context. `Ok(None)` means the choice is empty
    /// (branch disabled).
    pub fn choose(&mut self, bound: u64) -> Result<Option<u64>, QuintError> {
        match &mut self.choices {
            Some(ctl) => Ok(ctl.choose(bound)),
            None => Err(QuintError::new(
                "QNT501",
                "nondeterministic choice (oneOf/any) is not allowed in this context \
                 (e.g. inside an invariant)",
            )),
        }
    }
}

/// A compiled expression plus captured values for quantifier-bound
/// parameters (used by temporal atoms instantiated under `Set.forall`).
/// Evaluation binds the parameter slots around the call.
#[derive(Clone)]
pub struct BoundExpr {
    pub fnid: FnId,
    /// (parameter slot, captured value)
    pub bindings: Vec<(u32, Value)>,
}

impl BoundExpr {
    pub fn eval(&self, vm: &mut Vm, env: &mut Env) -> EvalResult {
        let saved = self.set_bindings(vm);
        let result = vm.run(env, self.fnid);
        self.restore_bindings(vm, saved);
        result
    }

    /// Install the captured bindings; returns the previous slot values.
    pub fn set_bindings(&self, vm: &mut Vm) -> Vec<Option<Value>> {
        self.bindings
            .iter()
            .map(|&(slot, value)| vm.param_replace(slot, Some(value)))
            .collect()
    }

    pub fn restore_bindings(&self, vm: &mut Vm, saved: Vec<Option<Value>>) {
        for (&(slot, _), old) in self.bindings.iter().zip(saved) {
            vm.param_replace(slot, old);
        }
    }
}
