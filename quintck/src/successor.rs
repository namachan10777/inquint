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

/// A run's dynamic access footprint (POR probe).
pub struct RunFootprint {
    /// Var-granularity read/write bitsets.
    pub reads: u64,
    pub writes: u64,
    /// Element accesses: (container value id, element id, is_write).
    pub elems: Vec<crate::value::ProbeElem>,
}

/// Like [`enumerate`] but records each successful run's footprint
/// (deduplicated per successor by merging). Sequential probe use only.
pub fn enumerate_probe(
    vm: &mut Vm,
    action: FnId,
    from: Option<&[Value]>,
) -> Result<Vec<(State, RunFootprint)>, QuintError> {
    match from {
        Some(state) => vm.load(state),
        None => vm.clear_current(),
    }

    let mut ctl = ChoiceCtl::new();
    let mut out: Vec<(State, RunFootprint)> = Vec::new();
    let mut runs: u64 = 0;

    loop {
        runs += 1;
        if runs > MAX_RUNS_PER_STATE {
            return Err(QuintError::new(
                "QNT501",
                format!("more than {MAX_RUNS_PER_STATE} choice combinations in a single transition"),
            ));
        }
        vm.reset_next();
        vm.probe_vars = Some((0, 0, 0));
        crate::value::probe_begin();
        let mut env = Env::new(Some(&mut ctl));
        let enabled = vm.run(&mut env, action)?;
        let (mut reads, writes, frames) = vm.probe_vars.take().unwrap();
        // optimistic: reads of frame-copied vars are treated as pure
        // copies (the copy commutes with writers); over-optimistic when
        // the var is ALSO genuinely read — acceptable for an upper bound
        reads &= !(frames & !writes);
        let elems = crate::value::probe_take();
        if enabled.as_bool() {
            let succ = vm.take_next_state()?;
            match out.iter_mut().find(|(s, _)| *s == succ) {
                Some((_, fp)) => {
                    // several runs reach the same successor: merge
                    fp.reads |= reads;
                    fp.writes |= writes;
                    fp.elems.extend_from_slice(&elems);
                }
                None => out.push((
                    succ,
                    RunFootprint {
                        reads,
                        writes,
                        elems,
                    },
                )),
            }
        }
        if !ctl.advance() {
            break;
        }
    }

    Ok(out)
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
