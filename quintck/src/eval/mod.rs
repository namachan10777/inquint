//! The evaluator: closure compilation of Quint IR expressions.
//!
//! Architecture follows the reference `quint_evaluator` (compile each
//! expression once into an `Rc<dyn Fn>` graph; name references resolve at
//! compile time to shared registers), with two deliberate differences:
//! - nondeterminism (`any`, `nondet`/`oneOf`) goes through the
//!   [`crate::choice::ChoiceCtl`] oracle instead of a RNG, so successors can
//!   be enumerated exhaustively;
//! - no instance/namespace machinery: the input module is flattened.

pub mod builtins_eager;
pub mod builtins_lazy;
pub mod compile;

pub use compile::Compiler;

use crate::choice::ChoiceCtl;
use crate::error::QuintError;
use crate::state::VarStorage;
use crate::value::{EvalResult, Value, ValueInner};
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

pub struct Env {
    pub storage: Rc<RefCell<VarStorage>>,
    /// The choice oracle. `None` during invariant evaluation, where any
    /// nondeterminism is an error.
    pub choices: Option<Rc<RefCell<ChoiceCtl>>>,
}

impl Env {
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

#[derive(Clone)]
pub struct CompiledExpr(Rc<dyn Fn(&mut Env) -> EvalResult>);

impl CompiledExpr {
    pub fn new(f: impl Fn(&mut Env) -> EvalResult + 'static) -> Self {
        CompiledExpr(Rc::new(f))
    }

    pub fn execute(&self, env: &mut Env) -> EvalResult {
        self.0(env)
    }
}

impl fmt::Debug for CompiledExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<compiled>")
    }
}

/// Apply a lambda value to arguments, saving and restoring the parameter
/// registers so that shadowing and reentrant calls are correct.
pub fn apply_lambda(lambda: &Value, env: &mut Env, args: Vec<Value>) -> EvalResult {
    let (registers, body) = match lambda.0.as_ref() {
        ValueInner::Lambda(registers, body) => (registers, body),
        v => panic!("expected lambda, got {v:?}"),
    };
    debug_assert_eq!(registers.len(), args.len());
    let saved: Vec<Option<Value>> = registers.iter().map(|r| r.borrow().clone()).collect();
    for (register, arg) in registers.iter().zip(args) {
        *register.borrow_mut() = Some(arg);
    }
    let result = body.execute(env);
    for (register, old) in registers.iter().zip(saved) {
        *register.borrow_mut() = old;
    }
    result
}
