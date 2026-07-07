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
    /// Worker threads for the fingerprint BFS (exact mode, temporal
    /// checking and run tests are single-threaded regardless).
    pub threads: usize,
}

impl Default for CheckConfig {
    fn default() -> Self {
        CheckConfig {
            max_steps: Some(10),
            deadlock: true,
            max_states: None,
            exact_states: false,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
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
    } else if cfg.threads > 1 {
        match check_parallel(spec, cfg) {
            // Anything that needs a trace or an error report is redone
            // single-threaded: the sequential pass is deterministic, finds
            // the same (minimal) violation depth, and reuses the existing
            // trace-reconstruction machinery.
            ParOutcome::Rerun => check_fp(spec, cfg),
            ParOutcome::Pass { states, max_depth } => {
                Ok(CheckOutcome::Pass { states, max_depth })
            }
            ParOutcome::Incomplete { states } => Ok(CheckOutcome::Incomplete { states }),
        }
    } else {
        check_fp(spec, cfg)
    }
}

// Check the invariants on a state; return the first violated one.
fn check_invs(
    spec: &CompiledSpec,
    vm: &mut crate::vm::Vm,
    state: &[Value],
) -> Result<Option<QuintName>, QuintError> {
    for &(name, inv) in &spec.invariants {
        if !spec.eval_invariant_at(vm, state, inv)?.as_bool() {
            return Ok(Some(name));
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

// ---------------------------------------------------------------------------
// Parallel fingerprint mode
// ---------------------------------------------------------------------------

/// Fingerprint seen-set sharded by the fingerprint's top bits; one short
/// mutex acquisition per *fresh* state is the only cross-thread write on
/// the exploration hot path.
struct ShardedFpSet {
    shards: Box<[std::sync::Mutex<hashbrown::HashTable<u64>>]>,
}

const FP_SHARDS: usize = 4096;

impl ShardedFpSet {
    fn new() -> Self {
        ShardedFpSet {
            shards: (0..FP_SHARDS)
                .map(|_| std::sync::Mutex::new(hashbrown::HashTable::new()))
                .collect(),
        }
    }

    /// Insert; returns true if the fingerprint was fresh.
    fn insert(&self, fp: u64) -> bool {
        let mut shard = self.shards[(fp >> 52) as usize & (FP_SHARDS - 1)].lock().unwrap();
        if shard.find(fp, |&e| e == fp).is_some() {
            return false;
        }
        shard.insert_unique(fp, fp, |&e| e);
        true
    }

    fn len(&self) -> u64 {
        self.shards.iter().map(|s| s.lock().unwrap().len() as u64).sum()
    }
}

enum ParOutcome {
    Pass { states: u64, max_depth: u32 },
    Incomplete { states: u64 },
    /// A violation, deadlock or evaluation error was detected — redo
    /// sequentially for the deterministic trace/report.
    Rerun,
}

/// Level-synchronized parallel BFS. Work stealing: each level's frontier
/// is a flat state array split into fixed-size chunks; workers claim
/// chunks with one `fetch_add` (dynamic load balance without deques).
/// Everything a worker touches per state — Vm, ChoiceCtl, output buffer —
/// is thread-private; the shared surface is the immutable `Program`, the
/// sharded seen-set and the (read-only) current frontier.
fn check_parallel(spec: &CompiledSpec, cfg: &CheckConfig) -> ParOutcome {
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

    let n_vars = spec.vars.len();
    let fps = ShardedFpSet::new();
    let stop = AtomicBool::new(false);
    let capped = AtomicBool::new(false);
    let discovered = AtomicU64::new(0);

    // Initial states: sequential (the set is tiny).
    let mut frontier: Vec<Value> = Vec::new();
    {
        let mut vm = spec.make_vm();
        let initial = match enumerate(&mut vm, spec.init, None) {
            Ok(v) => v,
            Err(_) => return ParOutcome::Rerun,
        };
        for state in initial {
            if !fps.insert(FpSet::fingerprint(&state)) {
                continue;
            }
            match check_invs(spec, &mut vm, &state) {
                Ok(None) => {}
                _ => return ParOutcome::Rerun,
            }
            frontier.extend_from_slice(&state);
            discovered.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// States per work unit: small enough to balance uneven successor
    /// costs, large enough that the claim `fetch_add` is noise.
    const CHUNK: usize = 256;

    let mut depth: u32 = 0;
    let mut last_report = std::time::Instant::now();

    while !frontier.is_empty() {
        if cfg.max_steps.is_some_and(|max| depth >= max) {
            break;
        }
        let n_states = frontier.len() / n_vars;
        let cursor = AtomicUsize::new(0);
        let frontier_ref = &frontier;

        let outs: Vec<Vec<Value>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..cfg.threads)
                .map(|_| {
                    s.spawn(|| {
                        let mut vm = spec.make_vm();
                        let mut out: Vec<Value> = Vec::new();
                        'chunks: loop {
                            if stop.load(Ordering::Relaxed) {
                                break;
                            }
                            let c = cursor.fetch_add(1, Ordering::Relaxed);
                            let lo = c * CHUNK;
                            if lo >= n_states {
                                break;
                            }
                            let hi = (lo + CHUNK).min(n_states);
                            for i in lo..hi {
                                let state = &frontier_ref[i * n_vars..(i + 1) * n_vars];
                                let succs = match enumerate(&mut vm, spec.step, Some(state)) {
                                    Ok(v) => v,
                                    Err(_) => {
                                        stop.store(true, Ordering::Relaxed);
                                        break 'chunks;
                                    }
                                };
                                if succs.is_empty() && cfg.deadlock {
                                    stop.store(true, Ordering::Relaxed);
                                    break 'chunks;
                                }
                                let mut fresh = 0u64;
                                for succ in succs {
                                    if !fps.insert(FpSet::fingerprint(&succ)) {
                                        continue;
                                    }
                                    fresh += 1;
                                    match check_invs(spec, &mut vm, &succ) {
                                        Ok(None) => {}
                                        _ => {
                                            stop.store(true, Ordering::Relaxed);
                                            break 'chunks;
                                        }
                                    }
                                    out.extend_from_slice(&succ);
                                }
                                if fresh > 0 {
                                    let total = discovered.fetch_add(fresh, Ordering::Relaxed) + fresh;
                                    if cfg.max_states.is_some_and(|max| total >= max) {
                                        capped.store(true, Ordering::Relaxed);
                                        stop.store(true, Ordering::Relaxed);
                                        break 'chunks;
                                    }
                                }
                            }
                        }
                        out
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        if stop.load(Ordering::Relaxed) {
            if capped.load(Ordering::Relaxed) {
                return ParOutcome::Incomplete {
                    states: discovered.load(Ordering::Relaxed),
                };
            }
            return ParOutcome::Rerun;
        }

        frontier.clear();
        for out in outs {
            frontier.extend_from_slice(&out);
        }
        if !frontier.is_empty() {
            depth += 1;
        }

        if last_report.elapsed().as_secs() >= 1 {
            eprintln!(
                "  {} states, depth {}, frontier {} ({} threads)",
                discovered.load(Ordering::Relaxed),
                depth,
                frontier.len() / n_vars,
                cfg.threads,
            );
            last_report = std::time::Instant::now();
        }
    }

    ParOutcome::Pass {
        states: fps.len(),
        max_depth: depth,
    }
}

fn check_fp(spec: &CompiledSpec, cfg: &CheckConfig) -> Result<CheckOutcome, Box<CheckError>> {
    let mut vm = spec.make_vm();
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

    let initial = enumerate(&mut vm, spec.init, None)
        .map_err(|e| err_with(&parents, None, e))?;
    for state in initial {
        if !fps.insert(FpSet::fingerprint(&state)) {
            continue;
        }
        let idx = parents.len() as u32;
        parents.push(NO_PARENT);
        match check_invs(spec, &mut vm, &state) {
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

        let successors = enumerate(&mut vm, spec.step, Some(&state_buf))
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
            match check_invs(spec, &mut vm, &succ) {
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
    let mut vm = spec.make_vm();
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

    let initial = enumerate(&mut vm, spec.init, None)
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
            let successors = enumerate(&mut vm, spec.step, Some(&state_buf))
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
    let mut vm = spec.make_vm();
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
    let initial = enumerate(&mut vm, spec.init, None)
        .map_err(|e| err_with(&arena, None, e))?;
    for state in initial {
        let (id, fresh) = seen.insert_or_get(&mut arena, &state, None, 0);
        debug_assert!(fresh, "enumerate returns deduplicated states");
        match check_invs(spec, &mut vm, &state) {
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

        let successors = enumerate(&mut vm, spec.step, Some(&state_buf))
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
            match check_invs(spec, &mut vm, &succ) {
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
