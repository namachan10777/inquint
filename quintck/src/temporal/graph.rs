//! The full reachable state graph, closed under stuttering, with all
//! temporal atoms evaluated eagerly.
//!
//! TLA behaviors satisfy □[Next]_vars — stuttering steps are allowed
//! anywhere — so every state gets a self-loop edge. A real `step`
//! self-loop and the stutter loop coincide as a graph edge; the edge-atom
//! definitions make this sound: `⟨A⟩_v` is false on (s,s) via the
//! v-changed conjunct, `[A]_v` is true via the v-unchanged disjunct.

use crate::error::QuintError;
use crate::eval::Env;
use crate::explorer::{CheckConfig, CheckError};
use crate::spec::CompiledSpec;
use crate::state::State;
use crate::successor::{enumerate, enumerate_partial, SubActionRuns};
use crate::temporal::parse::{AtomTable, EdgeAtom, StateAtom};
use crate::value::Value;
use quint_ast::QuintName;
use rustc_hash::FxHashMap;
use std::collections::{BTreeMap, BTreeSet};

pub struct StateGraph {
    pub states: Vec<State>,
    pub init_count: u32,
    pub depth: Vec<u32>,
    pub parent: Vec<Option<u32>>,
    /// CSR adjacency: edges of state s are targets[offsets[s]..offsets[s+1]].
    /// Includes the stutter self-loop.
    pub offsets: Vec<u32>,
    pub targets: Vec<u32>,
    /// state_bits[atom][state]
    pub state_bits: Vec<Vec<bool>>,
    /// edge_bits[atom][csr edge index]
    pub edge_bits: Vec<Vec<bool>>,
}

impl StateGraph {
    pub fn edges_of(&self, s: u32) -> impl Iterator<Item = (u32, u32)> + '_ {
        // yields (csr index, target)
        let lo = self.offsets[s as usize] as usize;
        let hi = self.offsets[s as usize + 1] as usize;
        (lo..hi).map(move |i| (i as u32, self.targets[i]))
    }

    pub fn trace_to(&self, mut id: u32) -> Vec<State> {
        let mut trace = Vec::new();
        loop {
            trace.push(self.states[id as usize].clone());
            match self.parent[id as usize] {
                Some(p) => id = p,
                None => break,
            }
        }
        trace.reverse();
        trace
    }
}

pub enum GraphOutcome {
    Graph(StateGraph),
    InvariantViolation { invariant: QuintName, trace: Vec<State> },
    Deadlock { trace: Vec<State> },
    Incomplete { states: u64 },
}

/// Build the full stutter-closed reachable graph, evaluating invariants
/// and all atoms along the way.
pub fn build(
    spec: &CompiledSpec,
    atoms: &AtomTable,
    cfg: &CheckConfig,
) -> Result<GraphOutcome, Box<CheckError>> {
    let mut g = StateGraph {
        states: Vec::new(),
        init_count: 0,
        depth: Vec::new(),
        parent: Vec::new(),
        offsets: vec![0],
        targets: Vec::new(),
        state_bits: vec![Vec::new(); atoms.state_atoms.len()],
        edge_bits: vec![Vec::new(); atoms.edge_atoms.len()],
    };
    // per-state values of the `v` expressions (for ⟨A⟩_v / [A]_v)
    let mut vars_vals: Vec<Vec<Value>> = vec![Vec::new(); atoms.vars_exprs.len()];
    let mut seen: FxHashMap<State, u32> = FxHashMap::default();

    let storage = &spec.storage;

    let fail = |g: &StateGraph, id: Option<u32>, error: QuintError| {
        Box::new(CheckError {
            error,
            trace: id.map(|i| g.trace_to(i)).unwrap_or_default(),
        })
    };

    // Intern a state: assign id, evaluate its state atoms / vars values /
    // invariants. Returns Err(violated invariant name) on violation.
    macro_rules! intern {
        ($g:expr, $state:expr, $parent:expr, $depth:expr) => {{
            let state: State = $state;
            match seen.get(&state) {
                Some(id) => Ok::<u32, Box<CheckError>>(*id),
                None => {
                    let id = $g.states.len() as u32;
                    $g.states.push(state.clone());
                    $g.depth.push($depth);
                    $g.parent.push($parent);
                    seen.insert(state.clone(), id);

                    storage.borrow().load(&state);
                    let mut env = Env::new(storage.clone(), None);
                    for (i, atom) in atoms.state_atoms.iter().enumerate() {
                        let bit = match atom {
                            StateAtom::Pred(p) => p
                                .eval(&mut env)
                                .map_err(|e| fail(&$g, Some(id), e))?
                                .as_bool(),
                            // Enabled atoms need successor enumeration;
                            // filled in during expansion below (placeholder).
                            StateAtom::Enabled { .. } => false,
                        };
                        $g.state_bits[i].push(bit);
                    }
                    for (i, v) in atoms.vars_exprs.iter().enumerate() {
                        let value = v.eval(&mut env).map_err(|e| fail(&$g, Some(id), e))?;
                        vars_vals[i].push(value);
                    }
                    Ok(id)
                }
            }
        }};
    }

    // Initial states
    let initial =
        enumerate(&spec.init, storage, None).map_err(|e| fail(&g, None, e))?;
    for state in initial {
        intern!(g, state, None, 0)?;
    }
    g.init_count = g.states.len() as u32;

    // Which state atoms are Enabled-atoms, grouped by action.
    let enabled_atoms: Vec<(usize, u32, Option<u32>)> = atoms
        .state_atoms
        .iter()
        .enumerate()
        .filter_map(|(i, a)| match a {
            StateAtom::Enabled { action, changed } => Some((i, *action, *changed)),
            _ => None,
        })
        .collect();
    // Actions referenced by any atom.
    let used_actions: BTreeSet<u32> = enabled_atoms
        .iter()
        .map(|(_, a, _)| *a)
        .chain(atoms.edge_atoms.iter().filter_map(|e| match e {
            EdgeAtom::Taken { action, .. } | EdgeAtom::Kept { action, .. } => Some(*action),
            EdgeAtom::NextPred(_) => None,
        }))
        .collect();

    let mut last_report = std::time::Instant::now();

    // BFS by index (ids are assigned in discovery order).
    let mut i: usize = 0;
    while i < g.states.len() {
        let s_id = i as u32;
        let s = g.states[i].clone();
        let depth = g.depth[i];

        // 1. step successors (strict)
        let successors =
            enumerate(&spec.step, storage, Some(&s)).map_err(|e| fail(&g, Some(s_id), e))?;
        if successors.is_empty() && cfg.deadlock {
            return Ok(GraphOutcome::Deadlock {
                trace: g.trace_to(s_id),
            });
        }

        // 2. sub-action transition predicates for atoms
        let mut t_a: BTreeMap<u32, SubActionRuns> = BTreeMap::new();
        for &a in &used_actions {
            let runs = enumerate_partial(&atoms.actions[a as usize], storage, &s)
                .map_err(|e| fail(&g, Some(s_id), e))?;
            t_a.insert(a, runs);
        }

        // 3. Enabled bits for s. For ENABLED ⟨A⟩_v the candidate successor
        // is the frame-filled run (unassigned = unchanged).
        for (atom_idx, action, changed) in &enabled_atoms {
            let runs = &t_a[action];
            let bit = match changed {
                None => runs.is_enabled(),
                Some(v) => {
                    let v = *v as usize;
                    let mut any = false;
                    for t in runs.framed(&s) {
                        // v(t): evaluate against t (throwaway states are legal)
                        storage.borrow().load(&t);
                        let mut env = Env::new(storage.clone(), None);
                        let vt = atoms.vars_exprs[v]
                            .eval(&mut env)
                            .map_err(|e| fail(&g, Some(s_id), e))?;
                        if vt != vars_vals[v][i] {
                            any = true;
                            break;
                        }
                    }
                    any
                }
            };
            g.state_bits[*atom_idx][i] = bit;
        }

        // 4. targets = successors ∪ stutter self-loop (merged if present)
        let mut target_states: Vec<State> = successors.into_iter().collect();
        if !target_states.contains(&s) {
            target_states.push(s.clone());
        }

        // 5. intern targets, then evaluate edge bits
        let mut target_ids = Vec::with_capacity(target_states.len());
        for t in &target_states {
            let t_id = intern!(g, t.clone(), Some(s_id), depth + 1)?;
            target_ids.push(t_id);
        }

        for (t, &t_id) in target_states.iter().zip(&target_ids) {
            // NextPred atoms need current = s, next bank = t
            storage.borrow().load(&s);
            storage.borrow().load_next(t);
            let mut env = Env::new(storage.clone(), None);
            env.next_allowed = true;
            for (ai, atom) in atoms.edge_atoms.iter().enumerate() {
                let bit = match atom {
                    EdgeAtom::Taken { action, vars } => {
                        t_a[action].matches(t)
                            && vars_vals[*vars as usize][t_id as usize]
                                != vars_vals[*vars as usize][i]
                    }
                    EdgeAtom::Kept { action, vars } => {
                        t_a[action].matches(t)
                            || vars_vals[*vars as usize][t_id as usize]
                                == vars_vals[*vars as usize][i]
                    }
                    EdgeAtom::NextPred(p) => p
                        .eval(&mut env)
                        .map_err(|e| fail(&g, Some(s_id), e))?
                        .as_bool(),
                };
                g.edge_bits[ai].push(bit);
            }
            g.targets.push(t_id);
        }
        g.offsets.push(g.targets.len() as u32);

        if cfg
            .max_states
            .is_some_and(|max| g.states.len() as u64 >= max)
        {
            return Ok(GraphOutcome::Incomplete {
                states: g.states.len() as u64,
            });
        }
        if last_report.elapsed().as_secs() >= 1 {
            eprintln!("  {} states (building graph)", g.states.len());
            last_report = std::time::Instant::now();
        }
        i += 1;
    }

    // Invariants: check on every state (after the fact, states are loaded
    // once more; done here to keep the intern macro simple).
    for (id, state) in g.states.iter().enumerate() {
        for (name, inv) in &spec.invariants {
            let holds = spec
                .eval_invariant_at(state, inv)
                .map_err(|e| fail(&g, Some(id as u32), e))?
                .as_bool();
            if !holds {
                return Ok(GraphOutcome::InvariantViolation {
                    invariant: name.clone(),
                    trace: g.trace_to(id as u32),
                });
            }
        }
    }

    Ok(GraphOutcome::Graph(g))
}
