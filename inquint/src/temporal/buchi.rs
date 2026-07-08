//! LTL → generalized Büchi automaton via the classic GPVW tableau
//! (Gerth, Peled, Vardi, Wolper: "Simple On-the-fly Automatic Verification
//! of Linear Temporal Logic").
//!
//! Convention: a run visiting node `q` at position `i` requires
//! `state_lits(q)` to hold on state `s_i` and `edge_lits(q)` to hold on
//! the step `(s_i, s_{i+1})`. There is one acceptance set per `Until`
//! subformula: F_u = { q : u ∉ old(q) or ψ_u ∈ old(q) } — a run is
//! accepting iff it visits every F_u infinitely often (generalized
//! acceptance; the SCC check handles it without degeneralization).

use super::ltl::{AtomRef, Lit, Nnf};
use std::collections::{BTreeMap, BTreeSet};

type FormulaId = u32;
const INIT: u32 = u32::MAX;

pub struct Gba {
    pub init: Vec<u32>,
    pub edges: Vec<Vec<u32>>,
    pub state_lits: Vec<Vec<Lit>>,
    pub edge_lits: Vec<Vec<Lit>>,
    /// acc_sets[i][q]: node q belongs to acceptance set i.
    /// Non-empty: with no Until subformulas a single all-nodes set is used.
    pub acc_sets: Vec<Vec<bool>>,
}

impl Gba {
    pub fn n_nodes(&self) -> usize {
        self.edges.len()
    }
}

struct Builder {
    formulas: Vec<Nnf>,
    ids: BTreeMap<Nnf, FormulaId>,
    /// done nodes: (old, next) -> node id
    done_index: BTreeMap<(BTreeSet<FormulaId>, BTreeSet<FormulaId>), u32>,
    old_sets: Vec<BTreeSet<FormulaId>>,
    next_sets: Vec<BTreeSet<FormulaId>>,
    incoming: Vec<BTreeSet<u32>>,
}

impl Builder {
    fn intern(&mut self, f: &Nnf) -> FormulaId {
        if let Some(id) = self.ids.get(f) {
            return *id;
        }
        let id = self.formulas.len() as FormulaId;
        self.formulas.push(f.clone());
        self.ids.insert(f.clone(), id);
        id
    }

    fn expand(
        &mut self,
        mut new: BTreeSet<FormulaId>,
        mut old: BTreeSet<FormulaId>,
        next: BTreeSet<FormulaId>,
        incoming: BTreeSet<u32>,
    ) {
        let Some(&eta) = new.iter().next() else {
            // Fully expanded: merge with an existing node or create one.
            if let Some(&id) = self.done_index.get(&(old.clone(), next.clone())) {
                self.incoming[id as usize].extend(incoming);
                return;
            }
            let id = self.old_sets.len() as u32;
            self.old_sets.push(old.clone());
            self.next_sets.push(next.clone());
            self.incoming.push(incoming);
            self.done_index.insert((old, next), id);
            // Expand the successor obligation.
            let new_next = self.next_sets[id as usize].clone();
            self.expand(
                new_next,
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::from([id]),
            );
            return;
        };
        new.remove(&eta);
        let formula = self.formulas[eta as usize].clone();
        match formula {
            Nnf::True => {
                old.insert(eta);
                self.expand(new, old, next, incoming);
            }
            Nnf::False => { /* contradiction: drop this branch */ }
            Nnf::Lit(lit) => {
                let neg = Nnf::Lit(lit.negated());
                if let Some(neg_id) = self.ids.get(&neg) {
                    if old.contains(neg_id) {
                        return; // contradiction
                    }
                }
                old.insert(eta);
                self.expand(new, old, next, incoming);
            }
            Nnf::And(a, b) => {
                let (ia, ib) = (self.intern(&a), self.intern(&b));
                let mut new2 = new;
                if !old.contains(&ia) {
                    new2.insert(ia);
                }
                if !old.contains(&ib) {
                    new2.insert(ib);
                }
                old.insert(eta);
                self.expand(new2, old, next, incoming);
            }
            Nnf::Or(a, b) => {
                let (ia, ib) = (self.intern(&a), self.intern(&b));
                let mut old2 = old.clone();
                old2.insert(eta);
                let mut new_a = new.clone();
                if !old2.contains(&ia) {
                    new_a.insert(ia);
                }
                self.expand(new_a, old2.clone(), next.clone(), incoming.clone());
                let mut new_b = new;
                if !old2.contains(&ib) {
                    new_b.insert(ib);
                }
                self.expand(new_b, old2, next, incoming);
            }
            Nnf::Until(a, b) => {
                let (ia, ib) = (self.intern(&a), self.intern(&b));
                let mut old2 = old.clone();
                old2.insert(eta);
                // wait branch: a holds now, Until carried to the next state
                let mut new_a = new.clone();
                if !old2.contains(&ia) {
                    new_a.insert(ia);
                }
                let mut next_a = next.clone();
                next_a.insert(eta);
                self.expand(new_a, old2.clone(), next_a, incoming.clone());
                // fulfil branch: b holds now
                let mut new_b = new;
                if !old2.contains(&ib) {
                    new_b.insert(ib);
                }
                self.expand(new_b, old2, next, incoming);
            }
            Nnf::Release(a, b) => {
                let (ia, ib) = (self.intern(&a), self.intern(&b));
                let mut old2 = old.clone();
                old2.insert(eta);
                // hold branch: b holds now, Release carried
                let mut new_b = new.clone();
                if !old2.contains(&ib) {
                    new_b.insert(ib);
                }
                let mut next_r = next.clone();
                next_r.insert(eta);
                self.expand(new_b, old2.clone(), next_r, incoming.clone());
                // fulfil branch: a and b hold now
                let mut new_ab = new;
                if !old2.contains(&ia) {
                    new_ab.insert(ia);
                }
                if !old2.contains(&ib) {
                    new_ab.insert(ib);
                }
                self.expand(new_ab, old2, next, incoming);
            }
        }
    }
}

pub fn build_gba(formula: &Nnf) -> Gba {
    let mut b = Builder {
        formulas: Vec::new(),
        ids: BTreeMap::new(),
        done_index: BTreeMap::new(),
        old_sets: Vec::new(),
        next_sets: Vec::new(),
        incoming: Vec::new(),
    };
    let root = b.intern(formula);
    b.expand(
        BTreeSet::from([root]),
        BTreeSet::new(),
        BTreeSet::new(),
        BTreeSet::from([INIT]),
    );

    let n = b.old_sets.len();
    let mut init = Vec::new();
    let mut edges: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (id, incoming) in b.incoming.iter().enumerate() {
        for &src in incoming {
            if src == INIT {
                init.push(id as u32);
            } else {
                edges[src as usize].push(id as u32);
            }
        }
    }

    let mut state_lits: Vec<Vec<Lit>> = vec![Vec::new(); n];
    let mut edge_lits: Vec<Vec<Lit>> = vec![Vec::new(); n];
    for (id, old) in b.old_sets.iter().enumerate() {
        for &f in old {
            if let Nnf::Lit(lit) = &b.formulas[f as usize] {
                match lit.atom {
                    AtomRef::S(_) => state_lits[id].push(*lit),
                    AtomRef::E(_) => edge_lits[id].push(*lit),
                }
            }
        }
    }

    // Acceptance sets: one per Until subformula.
    let mut acc_sets = Vec::new();
    for (fid, f) in b.formulas.iter().enumerate() {
        if let Nnf::Until(_, psi) = f {
            let psi_id = *b.ids.get(psi).expect("interned");
            let set: Vec<bool> = b
                .old_sets
                .iter()
                .map(|old| !old.contains(&(fid as FormulaId)) || old.contains(&psi_id))
                .collect();
            acc_sets.push(set);
        }
    }
    if acc_sets.is_empty() {
        acc_sets.push(vec![true; n]);
    }

    Gba {
        init,
        edges,
        state_lits,
        edge_lits,
        acc_sets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal::ltl::{nnf, Ltl};

    fn sa(i: u32) -> Ltl {
        Ltl::SAtom(i)
    }

    /// Explicitly check a tiny Kripke structure against the GBA by brute
    /// force: states 0..n with given edges and one boolean atom valuation;
    /// search for an accepting lasso in the product.
    fn gba_accepts_some_lasso(gba: &Gba, atom_true: &[bool], edges: &[(usize, usize)]) -> bool {
        let n_states = atom_true.len();
        // product nodes (s, q) valid if state lits hold
        let ok_state = |q: usize, s: usize| {
            gba.state_lits[q].iter().all(|lit| {
                let val = atom_true[s];
                match lit.atom {
                    AtomRef::S(0) => val == lit.pos,
                    _ => true,
                }
            })
        };
        // build product adjacency
        let mut pnodes = Vec::new();
        let mut pindex = std::collections::BTreeMap::new();
        for s in 0..n_states {
            for q in 0..gba.n_nodes() {
                if ok_state(q, s) {
                    pindex.insert((s, q), pnodes.len());
                    pnodes.push((s, q));
                }
            }
        }
        let mut padj = vec![Vec::new(); pnodes.len()];
        for (pi, &(s, q)) in pnodes.iter().enumerate() {
            for &(es, et) in edges {
                if es != s {
                    continue;
                }
                for &q2 in &gba.edges[q] {
                    if let Some(&pj) = pindex.get(&(et, q2 as usize)) {
                        padj[pi].push(pj);
                    }
                }
            }
        }
        // reachable from initial product nodes
        let mut reach = vec![false; pnodes.len()];
        let mut stack: Vec<usize> = pnodes
            .iter()
            .enumerate()
            .filter(|(_, &(s, q))| s == 0 && gba.init.contains(&(q as u32)))
            .map(|(i, _)| i)
            .collect();
        for &i in &stack {
            reach[i] = true;
        }
        while let Some(i) = stack.pop() {
            for &j in &padj[i] {
                if !reach[j] {
                    reach[j] = true;
                    stack.push(j);
                }
            }
        }
        // brute force: some cycle through a reachable node hitting all acc sets
        // (tiny graphs: check pairs via Floyd-Warshall reachability)
        let np = pnodes.len();
        let mut path = vec![vec![false; np]; np];
        for (i, adj) in padj.iter().enumerate() {
            for &j in adj {
                path[i][j] = true;
            }
        }
        for k in 0..np {
            for i in 0..np {
                for j in 0..np {
                    path[i][j] |= path[i][k] && path[k][j];
                }
            }
        }
        // an SCC containing i exists if path[i][i]; collect its members
        for i in 0..np {
            if !reach[i] || !path[i][i] {
                continue;
            }
            let members: Vec<usize> = (0..np)
                .filter(|&j| j == i || (path[i][j] && path[j][i]))
                .collect();
            let all_acc = gba.acc_sets.iter().all(|f| {
                members.iter().any(|&m| f[pnodes[m].1])
            });
            if all_acc {
                return true;
            }
        }
        false
    }

    /// Kripke: 2 states, 0 -> 1 -> 1; atom p false at 0, true at 1.
    /// F p must accept, G p must reject (p false at initial state).
    #[test]
    fn fp_and_gp() {
        let fp = build_gba(&nnf(&Ltl::Eventually(Box::new(sa(0))), false));
        assert!(gba_accepts_some_lasso(&fp, &[false, true], &[(0, 1), (1, 1)]));

        let gp = build_gba(&nnf(&Ltl::Always(Box::new(sa(0))), false));
        assert!(!gba_accepts_some_lasso(&gp, &[false, true], &[(0, 1), (1, 1)]));
        // and G p accepts when p holds everywhere
        assert!(gba_accepts_some_lasso(&gp, &[true, true], &[(0, 1), (1, 1)]));
    }

    /// GF p on 0 <-> 1 with p only at 1: accepts (visit 1 infinitely often).
    /// FG p on the same structure with a p-less sink loop at 0 only: rejects.
    #[test]
    fn gfp_and_fgp() {
        let gfp = build_gba(&nnf(
            &Ltl::Always(Box::new(Ltl::Eventually(Box::new(sa(0))))),
            false,
        ));
        assert!(gba_accepts_some_lasso(
            &gfp,
            &[false, true],
            &[(0, 1), (1, 0)]
        ));

        let fgp = build_gba(&nnf(
            &Ltl::Eventually(Box::new(Ltl::Always(Box::new(sa(0))))),
            false,
        ));
        // only lasso is 0->0 with p false: FG p rejects
        assert!(!gba_accepts_some_lasso(&fgp, &[false], &[(0, 0)]));
        // 0 -> 1 -> 1 with p at 1: accepts
        assert!(gba_accepts_some_lasso(
            &fgp,
            &[false, true],
            &[(0, 1), (1, 1)]
        ));
    }

    /// ¬(p ~> q) = F(p & G ¬q): on a structure where p happens then q
    /// never does, the negation accepts (counterexample exists).
    #[test]
    fn negated_leads_to() {
        let leads = Ltl::Always(Box::new(Ltl::Or(vec![
            Ltl::Not(Box::new(sa(0))),
            Ltl::Eventually(Box::new(sa(0))), // p leadsTo p: trivially true
        ])));
        let neg = build_gba(&nnf(&leads, true));
        assert!(!gba_accepts_some_lasso(&neg, &[true], &[(0, 0)]));
    }
}
