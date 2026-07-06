//! Exhaustive successor enumeration: run an action once per path through
//! its choice tree, collecting every distinct next state.

use crate::choice::ChoiceCtl;
use crate::error::QuintError;
use crate::eval::{CompiledExpr, Env};
use crate::state::{State, VarStorage};
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;

/// Backstop against runaway choice trees within a single transition.
const MAX_RUNS_PER_STATE: u64 = 1 << 24;

/// Enumerate all distinct successor states of `from` under `action`
/// (or all initial states when `from` is `None`).
pub fn enumerate(
    action: &CompiledExpr,
    storage: &Rc<RefCell<VarStorage>>,
    from: Option<&State>,
) -> Result<BTreeSet<State>, QuintError> {
    match from {
        Some(state) => storage.borrow().load(state),
        None => storage.borrow().clear_current(),
    }

    let ctl = Rc::new(RefCell::new(ChoiceCtl::new()));
    let mut out = BTreeSet::new();
    let mut runs: u64 = 0;

    loop {
        runs += 1;
        if runs > MAX_RUNS_PER_STATE {
            return Err(QuintError::new(
                "QNT501",
                format!(
                    "more than {MAX_RUNS_PER_STATE} choice combinations in a single \
                     transition; reduce nondeterminism or instance bounds"
                ),
            ));
        }

        storage.borrow().reset_next();
        let mut env = Env {
            storage: storage.clone(),
            choices: Some(ctl.clone()),
        };
        let enabled = action.execute(&mut env)?;
        if enabled.as_bool() {
            out.insert(storage.borrow().take_next_state()?);
        }
        if !ctl.borrow_mut().advance() {
            break;
        }
    }

    Ok(out)
}
