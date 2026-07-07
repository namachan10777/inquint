//! Breadth-first state-space exploration (TLC-style): every reachable state
//! within the step bound is visited exactly once; counterexamples are
//! shortest by construction.

use crate::error::QuintError;
use crate::spec::CompiledSpec;
use crate::state::{SeenSet, State, StateArena};
use crate::successor::enumerate;
use crate::value::Value;
use quint_ast::QuintName;
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

pub fn check(spec: &CompiledSpec, cfg: &CheckConfig) -> Result<CheckOutcome, Box<CheckError>> {
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

    // Check the invariants on a state; return the first violated one.
    let check_invs = |state: &[Value]| -> Result<Option<QuintName>, QuintError> {
        for (name, inv) in &spec.invariants {
            if !spec.eval_invariant_at(state, inv)?.as_bool() {
                return Ok(Some(*name));
            }
        }
        Ok(None)
    };

    // Initial states
    let initial = enumerate(&spec.init, &spec.storage, None)
        .map_err(|e| err_with(&arena, None, e))?;
    for state in initial {
        let (id, fresh) = seen.insert_or_get(&mut arena, &state, None, 0);
        debug_assert!(fresh, "enumerate returns deduplicated states");
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
