//! Exhaustive successor enumeration: run an action once per path through
//! its choice tree, collecting every distinct next state.

use crate::choice::ChoiceCtl;
use crate::error::QuintError;
use crate::eval::{BoundExpr, Env};
use crate::state::State;
use crate::value::Value;
use crate::vm::{FnId, Vm};
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
    vm: &mut Vm,
    action: &BoundExpr,
    from: &[Value],
) -> Result<SubActionRuns, QuintError> {
    let saved = action.set_bindings(vm);
    let result = enumerate_partial_inner(vm, action.fnid, from);
    action.restore_bindings(vm, saved);
    result
}

fn enumerate_partial_inner(
    vm: &mut Vm,
    action: FnId,
    from: &[Value],
) -> Result<SubActionRuns, QuintError> {
    vm.load(from);

    let mut ctl = ChoiceCtl::new();
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
        vm.reset_next();
        let mut env = Env::new(Some(&mut ctl));
        let enabled = vm.run(&mut env, action)?;
        if enabled.as_bool() {
            partials.insert(vm.take_partial());
        }
        if !ctl.advance() {
            break;
        }
    }

    Ok(SubActionRuns { partials })
}

/// Enumerate all distinct successor states of `from` under `action`
/// (or all initial states when `from` is `None`). The result is sorted
/// (by id) and deduplicated.
pub fn enumerate(
    vm: &mut Vm,
    action: FnId,
    from: Option<&[Value]>,
) -> Result<Vec<State>, QuintError> {
    match from {
        Some(state) => vm.load(state),
        None => vm.clear_current(),
    }

    let mut ctl = ChoiceCtl::new();
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

        vm.reset_next();
        let mut env = Env::new(Some(&mut ctl));
        let enabled = vm.run(&mut env, action)?;
        if enabled.as_bool() {
            out.push(vm.take_next_state()?);
        }
        if !ctl.advance() {
            break;
        }
    }

    out.sort_unstable();
    out.dedup();
    Ok(out)
}
