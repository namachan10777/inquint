//! Safety fast path: properties whose body is a conjunction of
//! propositional formulas and `always(propositional)` are checked as
//! state/edge invariants over the stutter-closed graph — no automaton,
//! and counterexamples are shortest (BFS state order).

use super::graph::StateGraph;
use super::ltl::{Prop, SafetyConjunct};
use crate::state::State;

pub enum SafetyOutcome {
    Pass,
    /// Violated at a state (trace ends at that state) or on an edge
    /// (trace ends with the edge's source and target states).
    Violation { trace: Vec<State> },
}

fn eval_prop(prop: &Prop, g: &StateGraph, state: u32, edge: Option<u32>) -> bool {
    match prop {
        Prop::True => true,
        Prop::False => false,
        Prop::SAtom(i) => g.state_bits[*i as usize][state as usize],
        Prop::EAtom(i) => {
            let e = edge.expect("edge atom outside edge context (classifier bug)");
            g.edge_bits[*i as usize][e as usize]
        }
        Prop::Not(p) => !eval_prop(p, g, state, edge),
        Prop::And(ps) => ps.iter().all(|p| eval_prop(p, g, state, edge)),
        Prop::Or(ps) => ps.iter().any(|p| eval_prop(p, g, state, edge)),
    }
}

pub fn check_safety(conjuncts: &[SafetyConjunct], g: &StateGraph) -> SafetyOutcome {
    // Initial-state conjuncts
    for c in conjuncts {
        if let SafetyConjunct::Initial(prop) = c {
            if prop.has_edge_atoms() {
                // An initial-position edge predicate constrains the first
                // step; check it on every edge out of every initial state.
                for s in 0..g.init_count {
                    for (e, _) in g.edges_of(s) {
                        if !eval_prop(prop, g, s, Some(e)) {
                            return violation_edge(g, s, e);
                        }
                    }
                }
            } else {
                for s in 0..g.init_count {
                    if !eval_prop(prop, g, s, None) {
                        return SafetyOutcome::Violation {
                            trace: g.trace_to(s),
                        };
                    }
                }
            }
        }
    }

    // always(prop) conjuncts: states in BFS order → shortest counterexample
    for s in 0..g.len() as u32 {
        for c in conjuncts {
            if let SafetyConjunct::AlwaysProp(prop) = c {
                if prop.has_edge_atoms() {
                    for (e, _) in g.edges_of(s) {
                        if !eval_prop(prop, g, s, Some(e)) {
                            return violation_edge(g, s, e);
                        }
                    }
                } else if !eval_prop(prop, g, s, None) {
                    return SafetyOutcome::Violation {
                        trace: g.trace_to(s),
                    };
                }
            }
        }
    }
    SafetyOutcome::Pass
}

fn violation_edge(g: &StateGraph, s: u32, e: u32) -> SafetyOutcome {
    let mut trace = g.trace_to(s);
    let t = g.targets[e as usize];
    trace.push(g.state_rc(t));
    SafetyOutcome::Violation { trace }
}
