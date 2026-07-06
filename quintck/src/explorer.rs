//! Breadth-first state-space exploration (TLC-style): every reachable state
//! within the step bound is visited exactly once; counterexamples are
//! shortest by construction.

use crate::error::QuintError;
use crate::spec::CompiledSpec;
use crate::state::State;
use crate::successor::enumerate;
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
}

impl Default for CheckConfig {
    fn default() -> Self {
        CheckConfig {
            max_steps: Some(10),
            deadlock: true,
            max_states: None,
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

struct Arena {
    entries: Vec<(State, Option<u32>, u32)>, // (state, parent, depth)
}

impl Arena {
    fn trace(&self, mut id: u32) -> Vec<State> {
        let mut trace = Vec::new();
        loop {
            let (state, parent, _) = &self.entries[id as usize];
            trace.push(state.clone());
            match parent {
                Some(p) => id = *p,
                None => break,
            }
        }
        trace.reverse();
        trace
    }
}

pub fn check(spec: &CompiledSpec, cfg: &CheckConfig) -> Result<CheckOutcome, Box<CheckError>> {
    let mut arena = Arena { entries: Vec::new() };
    let mut seen: FxHashMap<State, u32> = FxHashMap::default();
    let mut frontier: VecDeque<u32> = VecDeque::new();
    let mut max_depth: u32 = 0;

    let err_with = |arena: &Arena, id: Option<u32>, error: QuintError| {
        Box::new(CheckError {
            error,
            trace: id.map(|i| arena.trace(i)).unwrap_or_default(),
        })
    };

    // Check the invariants on a state; return the first violated one.
    let check_invs = |state: &State| -> Result<Option<QuintName>, QuintError> {
        for (name, inv) in &spec.invariants {
            if !spec.eval_invariant_at(state, inv)?.as_bool() {
                return Ok(Some(name.clone()));
            }
        }
        Ok(None)
    };

    // Initial states
    let initial = enumerate(&spec.init, &spec.storage, None)
        .map_err(|e| err_with(&arena, None, e))?;
    for state in initial {
        let id = arena.entries.len() as u32;
        arena.entries.push((state.clone(), None, 0));
        seen.insert(state.clone(), id);
        match check_invs(&state) {
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

    while let Some(id) = frontier.pop_front() {
        let (state, _, depth) = arena.entries[id as usize].clone();
        max_depth = max_depth.max(depth);

        if cfg.max_steps.is_some_and(|max| depth >= max) {
            continue;
        }

        let successors = enumerate(&spec.step, &spec.storage, Some(&state))
            .map_err(|e| err_with(&arena, Some(id), e))?;

        if successors.is_empty() && cfg.deadlock {
            return Ok(CheckOutcome::Deadlock {
                trace: arena.trace(id),
            });
        }

        for succ in successors {
            if seen.contains_key(&succ) {
                continue;
            }
            let succ_id = arena.entries.len() as u32;
            arena.entries.push((succ.clone(), Some(id), depth + 1));
            seen.insert(succ.clone(), succ_id);
            match check_invs(&succ) {
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
                .is_some_and(|max| arena.entries.len() as u64 >= max)
            {
                return Ok(CheckOutcome::Incomplete {
                    states: arena.entries.len() as u64,
                });
            }
        }

        if last_report.elapsed().as_secs() >= 1 {
            eprintln!(
                "  {} states, depth {}, frontier {}",
                arena.entries.len(),
                depth,
                frontier.len()
            );
            last_report = std::time::Instant::now();
        }
    }

    Ok(CheckOutcome::Pass {
        states: arena.entries.len() as u64,
        max_depth,
    })
}
