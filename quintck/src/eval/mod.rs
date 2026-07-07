//! The evaluation runtime shared surface: the environment, compiled
//! expressions (bytecode functions of the [`crate::vm`] engine), and
//! quantifier-bound expressions used by the temporal layer.
//!
//! Nondeterminism (`any`, `nondet`/`oneOf`) goes through the
//! [`crate::choice::ChoiceCtl`] oracle instead of a RNG, so successors can
//! be enumerated exhaustively.

pub mod builtins_eager;

use crate::choice::ChoiceCtl;
use crate::error::QuintError;
use crate::state::{Register, VarStorage};
use crate::value::{EvalResult, Value};
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

pub struct Env {
    pub storage: Rc<RefCell<VarStorage>>,
    /// The choice oracle. `None` during invariant evaluation, where any
    /// nondeterminism is an error.
    pub choices: Option<Rc<RefCell<ChoiceCtl>>>,
    /// When true, state-variable reads take the `next` register bank —
    /// this is how `next(x)` is evaluated inside temporal edge atoms.
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

impl Env {
    pub fn new(
        storage: Rc<RefCell<VarStorage>>,
        choices: Option<Rc<RefCell<ChoiceCtl>>>,
    ) -> Self {
        Env {
            storage,
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
        match &self.choices {
            Some(ctl) => Ok(ctl.borrow_mut().choose(bound)),
            None => Err(QuintError::new(
                "QNT501",
                "nondeterministic choice (oneOf/any) is not allowed in this context \
                 (e.g. inside an invariant)",
            )),
        }
    }
}

/// A compiled expression: a bytecode function of a shared [`crate::vm::Vm`].
/// `execute` is a top-level entry only — VM execution never re-enters
/// through `CompiledExpr` (internal calls go through `FnId` directly).
#[derive(Clone)]
pub struct CompiledExpr {
    pub vm: Rc<RefCell<crate::vm::Vm>>,
    pub fnid: crate::vm::FnId,
}

impl CompiledExpr {
    pub fn execute(&self, env: &mut Env) -> EvalResult {
        self.vm.borrow_mut().run(env, self.fnid)
    }
}

impl fmt::Debug for CompiledExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<fn {}>", self.fnid)
    }
}

/// A compiled expression plus captured values for quantifier-bound
/// parameters (used by temporal atoms instantiated under `Set.forall`).
/// Evaluation sets the shared param registers around the call.
#[derive(Clone)]
pub struct BoundExpr {
    pub expr: CompiledExpr,
    pub bindings: Vec<(Register, Value)>,
}

impl BoundExpr {
    pub fn eval(&self, env: &mut Env) -> EvalResult {
        let saved = self.set_bindings();
        let result = self.expr.execute(env);
        self.restore_bindings(saved);
        result
    }

    /// Install the captured bindings; returns the previous register values.
    pub fn set_bindings(&self) -> Vec<Option<Value>> {
        self.bindings
            .iter()
            .map(|(reg, value)| reg.replace(Some(*value)))
            .collect()
    }

    pub fn restore_bindings(&self, saved: Vec<Option<Value>>) {
        for ((reg, _), old) in self.bindings.iter().zip(saved) {
            reg.set(old);
        }
    }
}
