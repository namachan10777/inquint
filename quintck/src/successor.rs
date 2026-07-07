//! Exhaustive successor enumeration: run an action once per path through
//! its choice tree, collecting every distinct next state.

use crate::choice::ChoiceCtl;
use crate::error::QuintError;
use crate::eval::{CompiledExpr, Env};
use crate::state::{State, VarStorage};
use crate::value::Value;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;

/// Backstop against runaway choice trees within a single transition.
const MAX_RUNS_PER_STATE: u64 = 1 << 24;

/// The transition predicate of a sub-action at a source state: the set of
/// partial next-state assignments its successful runs produce. Variables
/// the action did not assign are `None` — unconstrained, per TLA action
/// semantics (they may take any value in the next state).
pub struct SubActionRuns {
    pub partials: BTreeSet<Vec<Option<Value>>>,
}

impl SubActionRuns {
    /// Does the concrete edge (from, to) satisfy the action? True iff some
    /// run's assigned variables all agree with `to`.
    pub fn matches(&self, to: &[Value]) -> bool {
        self.partials.iter().any(|p| {
            p.iter()
                .zip(to.iter())
                .all(|(assigned, actual)| match assigned {
                    Some(v) => v == actual,
                    None => true,
                })
        })
    }

    /// The frame-filled successor for a partial run: unassigned = keep the
    /// source value. Used for ENABLED ⟨A⟩_v checks.
    pub fn framed<'a>(&'a self, from: &'a [Value]) -> impl Iterator<Item = State> + 'a {
        self.partials.iter().map(move |p| {
            let values: Vec<Value> = p
                .iter()
                .zip(from.iter())
                .map(|(assigned, cur)| assigned.unwrap_or(*cur))
                .collect();
            Rc::from(values.into_boxed_slice())
        })
    }

    pub fn is_enabled(&self) -> bool {
        !self.partials.is_empty()
    }
}

/// Enumerate all runs of a *sub-action* (e.g. a fairness target or an
/// orKeep/mustChange action) from `from`, as partial assignments.
pub fn enumerate_partial(
    action: &crate::eval::BoundExpr,
    storage: &Rc<RefCell<VarStorage>>,
    from: &[Value],
) -> Result<SubActionRuns, QuintError> {
    let saved = action.set_bindings();
    let result = enumerate_partial_inner(action, storage, from);
    action.restore_bindings(saved);
    result
}

fn enumerate_partial_inner(
    action: &crate::eval::BoundExpr,
    storage: &Rc<RefCell<VarStorage>>,
    from: &[Value],
) -> Result<SubActionRuns, QuintError> {
    storage.borrow().load(from);

    let ctl = Rc::new(RefCell::new(ChoiceCtl::new()));
    let mut partials = BTreeSet::new();
    let mut runs: u64 = 0;

    loop {
        runs += 1;
        if runs > MAX_RUNS_PER_STATE {
            return Err(QuintError::new(
                "QNT501",
                format!("more than {MAX_RUNS_PER_STATE} choice combinations in one sub-action"),
            ));
        }
        storage.borrow().reset_next();
        let mut env = Env::new(storage.clone(), Some(ctl.clone()));
        let enabled = action.expr.execute(&mut env)?;
        if enabled.as_bool() {
            partials.insert(storage.borrow().take_partial());
        }
        if !ctl.borrow_mut().advance() {
            break;
        }
    }

    Ok(SubActionRuns { partials })
}

/// Enumerate all distinct successor states of `from` under `action`
/// (or all initial states when `from` is `None`). The result is sorted
/// (by id) and deduplicated.
pub fn enumerate(
    action: &CompiledExpr,
    storage: &Rc<RefCell<VarStorage>>,
    from: Option<&[Value]>,
) -> Result<Vec<State>, QuintError> {
    match from {
        Some(state) => storage.borrow().load(state),
        None => storage.borrow().clear_current(),
    }

    let ctl = Rc::new(RefCell::new(ChoiceCtl::new()));
    let mut out: Vec<State> = Vec::new();
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
        let mut env = Env::new(storage.clone(), Some(ctl.clone()));
        let enabled = action.execute(&mut env)?;
        if enabled.as_bool() {
            out.push(storage.borrow().take_next_state()?);
        }
        if !ctl.borrow_mut().advance() {
            break;
        }
    }

    out.sort_unstable();
    out.dedup();
    Ok(out)
}
