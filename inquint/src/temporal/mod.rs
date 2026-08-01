//! Temporal-property (LTL/liveness) checking: automata-theoretic model
//! checking with TLC-style fairness handling.
//!
//! Pipeline: parse (`temporal` def → fairness premises + LTL body over
//! opaque atoms) → safety fast path where possible, otherwise GPVW
//! tableau (¬body → generalized Büchi automaton) × stutter-closed state
//! graph → Tarjan SCC with acceptance + fairness side conditions →
//! lasso counterexample (ITF `loop_index`).

pub mod buchi;
pub mod graph;
pub mod ltl;
pub mod parse;
pub mod product;
pub mod safety;
pub mod scc;
pub mod validate;
pub mod witness;

pub use parse::{AtomTable, Fairness, Property};

use crate::explorer::{CheckConfig, CheckError};
use crate::spec::CompiledSpec;
use crate::state::State;
use quint_ast::QuintName;

pub struct Lasso {
    pub states: Vec<State>,
    /// The successor of the last state is `states[loop_index]` (ITF ADR 015).
    pub loop_index: usize,
}

pub enum TemporalOutcome {
    Pass {
        states: u64,
    },
    /// A liveness violation: an infinite lasso-shaped counterexample.
    Violation {
        property: QuintName,
        lasso: Lasso,
    },
    /// A safety-shaped temporal property violated by a finite prefix.
    SafetyViolation {
        property: QuintName,
        trace: Vec<State>,
    },
    /// An invariant violated or deadlock found while building the graph.
    InvariantViolation {
        invariant: QuintName,
        trace: Vec<State>,
    },
    Deadlock {
        trace: Vec<State>,
    },
    Incomplete {
        states: u64,
    },
}

/// Check all temporal properties of the spec (plus its invariants, which
/// are evaluated during graph construction).
pub fn check_temporal(
    spec: &CompiledSpec,
    cfg: &CheckConfig,
) -> Result<TemporalOutcome, Box<CheckError>> {
    let g = match graph::build(spec, &spec.atoms, cfg)? {
        graph::GraphOutcome::Graph(g) => g,
        graph::GraphOutcome::InvariantViolation { invariant, trace } => {
            return Ok(TemporalOutcome::InvariantViolation { invariant, trace })
        }
        graph::GraphOutcome::Deadlock { trace } => {
            return Ok(TemporalOutcome::Deadlock { trace })
        }
        graph::GraphOutcome::Incomplete { states } => {
            return Ok(TemporalOutcome::Incomplete { states })
        }
    };

    for prop in &spec.temporal {
        if let Some(conjuncts) = ltl::safety_fragment(&prop.body) {
            // Fairness premises are irrelevant for safety bodies: a safety
            // violation is a finite prefix, and any finite prefix extends
            // to a fair behavior (WF/SF are machine-closed).
            match safety::check_safety(&conjuncts, &g) {
                safety::SafetyOutcome::Pass => continue,
                safety::SafetyOutcome::Violation { trace } => {
                    return Ok(TemporalOutcome::SafetyViolation {
                        property: prop.name,
                        trace,
                    })
                }
            }
        }
        // General (liveness) path: implemented in the buchi/product/scc
        // modules (M3/M4).
        match check_liveness(spec, prop, &g)? {
            None => continue,
            Some(lasso) => {
                return Ok(TemporalOutcome::Violation {
                    property: prop.name,
                    lasso,
                })
            }
        }
    }

    Ok(TemporalOutcome::Pass {
        states: g.len() as u64,
    })
}

fn check_liveness(
    _spec: &CompiledSpec,
    prop: &Property,
    g: &graph::StateGraph,
) -> Result<Option<Lasso>, Box<CheckError>> {
    // Counterexample search: a fair lasso satisfying ¬body.
    let neg = ltl::nnf(&prop.body, true);
    let gba = buchi::build_gba(&neg);
    let product = product::build_product(g, &gba);
    let Some(fair_scc) = scc::find_fair_accepting(&product, &gba, g, &prop.fairness) else {
        return Ok(None);
    };
    let lasso = witness::extract_lasso(&product, g, &fair_scc);
    if !lasso_satisfies_negation(&prop.body, &lasso, g) {
        return Err(Box::new(CheckError {
            error: crate::error::QuintError::new(
                "QNT599",
                "internal error: extracted lasso does not violate the temporal property",
            ),
            trace: lasso.states,
        }));
    }
    Ok(Some(lasso))
}

/// Self-check: the extracted lasso must satisfy ¬body under the graph's
/// atom valuations. Requires mapping lasso states back to graph ids and
/// consecutive pairs to CSR edges.
fn lasso_satisfies_negation(body: &ltl::Ltl, lasso: &Lasso, g: &graph::StateGraph) -> bool {
    use crate::value::Value;
    use rustc_hash::FxHashMap;
    let index: FxHashMap<&[Value], u32> = (0..g.len() as u32)
        .map(|i| (g.state(i), i))
        .collect();
    let ids: Vec<u32> = lasso
        .states
        .iter()
        .map(|s| *index.get(&s[..]).expect("lasso state in graph"))
        .collect();
    let n = ids.len();
    let succ_of = |i: usize| if i + 1 < n { i + 1 } else { lasso.loop_index };
    // CSR edge index per position
    let edges: Vec<u32> = (0..n)
        .map(|i| {
            let (s, t) = (ids[i], ids[succ_of(i)]);
            g.edges_of(s)
                .find(|(_, tt)| *tt == t)
                .map(|(e, _)| e)
                .expect("lasso step is a real edge")
        })
        .collect();

    struct W<'a> {
        g: &'a graph::StateGraph,
        ids: &'a [u32],
        edges: &'a [u32],
        loop_index: usize,
    }
    impl validate::LassoAtoms for W<'_> {
        fn n(&self) -> usize {
            self.ids.len()
        }
        fn loop_index(&self) -> usize {
            self.loop_index
        }
        fn state_atom(&self, atom: u32, pos: usize) -> bool {
            self.g.state_bits[atom as usize][self.ids[pos] as usize]
        }
        fn edge_atom(&self, atom: u32, pos: usize) -> bool {
            self.g.edge_bits[atom as usize][self.edges[pos] as usize]
        }
    }
    let w = W {
        g,
        ids: &ids,
        edges: &edges,
        loop_index: lasso.loop_index,
    };
    !validate::holds(body, &w)
}
