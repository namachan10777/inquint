//! Breadth-first state-space exploration (TLC-style): every reachable state
//! within the step bound is visited exactly once; counterexamples are
//! shortest by construction.
//!
//! Two storage modes:
//! - **fingerprint** (default): the seen-set holds 64-bit fingerprints only
//!   and full states exist only on the BFS frontier (~14 bytes/state:
//!   fingerprint table + one parent index for trace reconstruction).
//!   Probabilistically sound, like TLC: a fingerprint collision would
//!   silently prune a state. Counterexample traces are materialized by a
//!   deterministic re-run that only keeps the states on the parent path.
//! - **exact** (`--exact-states`): full states in a flat arena, exact
//!   deduplication — the pre-fingerprint behavior.

use crate::error::QuintError;
use crate::spec::CompiledSpec;
use crate::state::{FpSet, SeenSet, State, StateArena};
use crate::successor::enumerate;
use crate::value::Value;
use quint_ast::QuintName;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

pub struct CheckConfig {
    /// Maximum trace length (number of steps from an initial state).
    /// `None` = fully exhaustive.
    pub max_steps: Option<u32>,
    /// Report states with no successors.
    pub deadlock: bool,
    /// Abort after this many states (safety valve, not a violation).
    pub max_states: Option<u64>,
    /// Keep full states for exact deduplication instead of fingerprints.
    pub exact_states: bool,
}

impl Default for CheckConfig {
    fn default() -> Self {
        CheckConfig {
            max_steps: Some(10),
            deadlock: true,
            max_states: None,
            exact_states: false,
        }
    }
}

pub enum CheckOutcome {
    Pass {
        states: u64,
        max_depth: u32,
    },
    InvariantViolation {
        invariant: QuintName,
        trace: Vec<State>,
    },
    Deadlock {
        trace: Vec<State>,
    },
    /// Search stopped early by max_states; no violation found so far.
    Incomplete {
        states: u64,
    },
}

pub struct CheckError {
    pub error: QuintError,
    /// The state being expanded/checked when the error occurred, if any.
    pub trace: Vec<State>,
}

pub fn check(spec: &CompiledSpec, cfg: &CheckConfig) -> Result<CheckOutcome, Box<CheckError>> {
    if cfg.exact_states {
        check_exact(spec, cfg)
    } else {
        check_fp(spec, cfg)
    }
}

// Check the invariants on a state; return the first violated one.
fn check_invs(spec: &CompiledSpec, state: &[Value]) -> Result<Option<QuintName>, QuintError> {
    for (name, inv) in &spec.invariants {
        if !spec.eval_invariant_at(state, inv)?.as_bool() {
            return Ok(Some(*name));
        }
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// Fingerprint mode
// ---------------------------------------------------------------------------

const NO_PARENT: u32 = u32::MAX;

/// BFS frontier with flat storage: states are `n_vars` consecutive values
/// in a ring buffer — no per-state allocation or `Rc` header. States enter
/// in discovery order, so the discovery index is recovered by a pop
/// counter.
struct FlatFrontier {
    n_vars: usize,
    data: VecDeque<Value>,
    depths: VecDeque<u32>,
    pop_idx: u32,
}

impl FlatFrontier {
    fn new(n_vars: usize) -> Self {
        FlatFrontier {
            n_vars,
            data: VecDeque::new(),
            depths: VecDeque::new(),
            pop_idx: 0,
        }
    }

    fn push(&mut self, state: &[Value], depth: u32) {
        debug_assert_eq!(state.len(), self.n_vars);
        self.data.extend(state.iter().copied());
        self.depths.push_back(depth);
    }

    /// Pop the next state into `buf`; returns its discovery index and depth.
    fn pop(&mut self, buf: &mut Vec<Value>) -> Option<(u32, u32)> {
        let depth = self.depths.pop_front()?;
        buf.clear();
        buf.extend(self.data.drain(..self.n_vars));
        let idx = self.pop_idx;
        self.pop_idx += 1;
        Some((idx, depth))
    }

    fn len(&self) -> usize {
        self.depths.len()
    }
}

fn check_fp(spec: &CompiledSpec, cfg: &CheckConfig) -> Result<CheckOutcome, Box<CheckError>> {
    let mut fps = FpSet::default();
    // discovery index -> parent discovery index (the only per-state data
    // besides the fingerprint)
    let mut parents: Vec<u32> = Vec::new();
    let mut frontier = FlatFrontier::new(spec.vars.len());
    let mut max_depth: u32 = 0;

    let err_with = |parents: &[u32], idx: Option<u32>, error: QuintError| {
        Box::new(CheckError {
            error,
            trace: idx.map(|i| reconstruct(spec, cfg, parents, i)).unwrap_or_default(),
        })
    };

    let initial = enumerate(&spec.init, &spec.storage, None)
        .map_err(|e| err_with(&parents, None, e))?;
    for state in initial {
        if !fps.insert(FpSet::fingerprint(&state)) {
            continue;
        }
        let idx = parents.len() as u32;
        parents.push(NO_PARENT);
        match check_invs(spec, &state) {
            Ok(Some(invariant)) => {
                return Ok(CheckOutcome::InvariantViolation {
                    invariant,
                    trace: reconstruct(spec, cfg, &parents, idx),
                })
            }
            Ok(None) => {}
            Err(e) => return Err(err_with(&parents, Some(idx), e)),
        }
        frontier.push(&state, 0);
    }

    let mut last_report = std::time::Instant::now();
    let mut state_buf: Vec<Value> = Vec::with_capacity(spec.vars.len());

    while let Some((idx, depth)) = frontier.pop(&mut state_buf) {
        max_depth = max_depth.max(depth);

        if cfg.max_steps.is_some_and(|max| depth >= max) {
            continue;
        }

        let successors = enumerate(&spec.step, &spec.storage, Some(&state_buf))
            .map_err(|e| err_with(&parents, Some(idx), e))?;

        if successors.is_empty() && cfg.deadlock {
            return Ok(CheckOutcome::Deadlock {
                trace: reconstruct(spec, cfg, &parents, idx),
            });
        }

        for succ in successors {
            if !fps.insert(FpSet::fingerprint(&succ)) {
                continue;
            }
            let succ_idx = parents.len() as u32;
            parents.push(idx);
            match check_invs(spec, &succ) {
                Ok(Some(invariant)) => {
                    return Ok(CheckOutcome::InvariantViolation {
                        invariant,
                        trace: reconstruct(spec, cfg, &parents, succ_idx),
                    })
                }
                Ok(None) => {}
                Err(e) => return Err(err_with(&parents, Some(succ_idx), e)),
            }
            frontier.push(&succ, depth + 1);

            if cfg
                .max_states
                .is_some_and(|max| parents.len() as u64 >= max)
            {
                return Ok(CheckOutcome::Incomplete {
                    states: parents.len() as u64,
                });
            }
        }

        if last_report.elapsed().as_secs() >= 1 {
            eprintln!(
                "  {} states, depth {}, frontier {}",
                parents.len(),
                depth,
                frontier.len()
            );
            last_report = std::time::Instant::now();
        }
    }

    Ok(CheckOutcome::Pass {
        states: parents.len() as u64,
        max_depth,
    })
}

/// Materialize the trace to discovery index `target` by re-running the BFS.
///
/// Exploration is fully deterministic (canonical ordered values, the
/// `ChoiceCtl` oracle, deduplicated successor lists), so a re-run assigns
/// identical discovery indices; we keep only the states on the parent path
/// and stop as soon as the target — the largest index on the path — is
/// discovered. Costs one extra traversal, paid only when reporting.
fn reconstruct(spec: &CompiledSpec, cfg: &CheckConfig, parents: &[u32], target: u32) -> Vec<State> {
    let mut path = vec![target];
    let mut cur = target;
    while parents[cur as usize] != NO_PARENT {
        cur = parents[cur as usize];
        path.push(cur);
    }
    path.reverse();
    let mut wanted: FxHashMap<u32, Option<State>> =
        path.iter().map(|&i| (i, None)).collect();
    let mut remaining = path.len();

    let mut record = |idx: u32, state: &State, remaining: &mut usize| {
        if let Some(slot) = wanted.get_mut(&idx) {
            *slot = Some(state.clone());
            *remaining -= 1;
        }
    };

    let mut fps = FpSet::default();
    let mut counter: u32 = 0;
    let mut frontier = FlatFrontier::new(spec.vars.len());
    let mut state_buf: Vec<Value> = Vec::with_capacity(spec.vars.len());

    let initial = enumerate(&spec.init, &spec.storage, None)
        .expect("reconstruction diverged from the original run (init)");
    'outer: {
        for state in initial {
            if !fps.insert(FpSet::fingerprint(&state)) {
                continue;
            }
            let idx = counter;
            counter += 1;
            record(idx, &state, &mut remaining);
            if remaining == 0 {
                break 'outer;
            }
            frontier.push(&state, 0);
        }
        while let Some((_, depth)) = frontier.pop(&mut state_buf) {
            if cfg.max_steps.is_some_and(|max| depth >= max) {
                continue;
            }
            let successors = enumerate(&spec.step, &spec.storage, Some(&state_buf))
                .expect("reconstruction diverged from the original run (step)");
            for succ in successors {
                if !fps.insert(FpSet::fingerprint(&succ)) {
                    continue;
                }
                let idx = counter;
                counter += 1;
                record(idx, &succ, &mut remaining);
                if remaining == 0 {
                    break 'outer;
                }
                frontier.push(&succ, depth + 1);
            }
        }
    }
    assert_eq!(remaining, 0, "reconstruction did not reach the target state");

    path.iter()
        .map(|i| wanted[i].clone().expect("path state materialized"))
        .collect()
}

// ---------------------------------------------------------------------------
// Exact mode (--exact-states): full states in a flat arena
// ---------------------------------------------------------------------------

fn check_exact(spec: &CompiledSpec, cfg: &CheckConfig) -> Result<CheckOutcome, Box<CheckError>> {
    let mut arena = StateArena::new(spec.vars.len());
    let mut seen = SeenSet::default();
    let mut frontier: VecDeque<u32> = VecDeque::new();
    let mut max_depth: u32 = 0;

    let err_with = |arena: &StateArena, id: Option<u32>, error: QuintError| {
        Box::new(CheckError {
            error,
            trace: id.map(|i| arena.trace(i)).unwrap_or_default(),
        })
    };

    // Initial states
    let initial = enumerate(&spec.init, &spec.storage, None)
        .map_err(|e| err_with(&arena, None, e))?;
    for state in initial {
        let (id, fresh) = seen.insert_or_get(&mut arena, &state, None, 0);
        debug_assert!(fresh, "enumerate returns deduplicated states");
        match check_invs(spec, &state) {
            Ok(Some(invariant)) => {
                return Ok(CheckOutcome::InvariantViolation {
                    invariant,
                    trace: arena.trace(id),
                })
            }
            Ok(None) => {}
            Err(e) => return Err(err_with(&arena, Some(id), e)),
        }
        frontier.push_back(id);
    }

    let mut last_report = std::time::Instant::now();
    let mut state_buf: Vec<Value> = Vec::with_capacity(spec.vars.len());

    while let Some(id) = frontier.pop_front() {
        let depth = arena.depth(id);
        max_depth = max_depth.max(depth);

        if cfg.max_steps.is_some_and(|max| depth >= max) {
            continue;
        }

        state_buf.clear();
        state_buf.extend_from_slice(arena.get(id));

        let successors = enumerate(&spec.step, &spec.storage, Some(&state_buf))
            .map_err(|e| err_with(&arena, Some(id), e))?;

        if successors.is_empty() && cfg.deadlock {
            return Ok(CheckOutcome::Deadlock {
                trace: arena.trace(id),
            });
        }

        for succ in successors {
            let (succ_id, fresh) = seen.insert_or_get(&mut arena, &succ, Some(id), depth + 1);
            if !fresh {
                continue;
            }
            match check_invs(spec, &succ) {
                Ok(Some(invariant)) => {
                    return Ok(CheckOutcome::InvariantViolation {
                        invariant,
                        trace: arena.trace(succ_id),
                    })
                }
                Ok(None) => {}
                Err(e) => return Err(err_with(&arena, Some(succ_id), e)),
            }
            frontier.push_back(succ_id);

            if cfg
                .max_states
                .is_some_and(|max| arena.len() as u64 >= max)
            {
                return Ok(CheckOutcome::Incomplete {
                    states: arena.len() as u64,
                });
            }
        }

        if last_report.elapsed().as_secs() >= 1 {
            eprintln!(
                "  {} states, depth {}, frontier {}",
                arena.len(),
                depth,
                frontier.len()
            );
            last_report = std::time::Instant::now();
        }
    }

    Ok(CheckOutcome::Pass {
        states: arena.len() as u64,
        max_depth,
    })
}
