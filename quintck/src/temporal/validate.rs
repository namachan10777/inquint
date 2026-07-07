//! Oracle: evaluate an LTL formula on an ultimately periodic word (lasso).
//! Used to self-check extracted counterexamples (debug builds and tests)
//! and for randomized differential testing of the GBA/product/SCC pipeline.

use super::ltl::Ltl;

/// Atom valuations along a lasso of `n` positions where the successor of
/// position `n-1` is `loop_index`.
pub trait LassoAtoms {
    fn n(&self) -> usize;
    fn loop_index(&self) -> usize;
    fn state_atom(&self, atom: u32, pos: usize) -> bool;
    /// Edge atom on the step from `pos` to its successor.
    fn edge_atom(&self, atom: u32, pos: usize) -> bool;
}

/// Evaluate `f` at every position; returns the valuation vector.
/// Fixpoint iteration over the loop handles Always/Eventually.
fn eval(f: &Ltl, w: &dyn LassoAtoms) -> Vec<bool> {
    let n = w.n();
    let succ = |i: usize| if i + 1 < n { i + 1 } else { w.loop_index() };
    match f {
        Ltl::True => vec![true; n],
        Ltl::False => vec![false; n],
        Ltl::SAtom(a) => (0..n).map(|i| w.state_atom(*a, i)).collect(),
        Ltl::EAtom(a) => (0..n).map(|i| w.edge_atom(*a, i)).collect(),
        Ltl::Not(p) => eval(p, w).into_iter().map(|b| !b).collect(),
        Ltl::And(ps) => {
            let evs: Vec<Vec<bool>> = ps.iter().map(|p| eval(p, w)).collect();
            (0..n).map(|i| evs.iter().all(|e| e[i])).collect()
        }
        Ltl::Or(ps) => {
            let evs: Vec<Vec<bool>> = ps.iter().map(|p| eval(p, w)).collect();
            (0..n).map(|i| evs.iter().any(|e| e[i])).collect()
        }
        Ltl::Always(p) => {
            let inner = eval(p, w);
            // G p at i: p at every position reachable from i.
            // Fixpoint: v[i] = inner[i] && v[succ(i)], iterate to stability.
            let mut v = vec![true; n];
            loop {
                let mut changed = false;
                for i in (0..n).rev() {
                    let nv = inner[i] && v[succ(i)];
                    if nv != v[i] {
                        v[i] = nv;
                        changed = true;
                    }
                }
                if !changed {
                    return v;
                }
            }
        }
        Ltl::Eventually(p) => {
            let inner = eval(p, w);
            let mut v = vec![false; n];
            loop {
                let mut changed = false;
                for i in (0..n).rev() {
                    let nv = inner[i] || v[succ(i)];
                    if nv != v[i] {
                        v[i] = nv;
                        changed = true;
                    }
                }
                if !changed {
                    return v;
                }
            }
        }
    }
}

/// Does the formula hold at position 0 of the lasso?
pub fn holds(f: &Ltl, w: &dyn LassoAtoms) -> bool {
    eval(f, w)[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Simple {
        p: Vec<bool>,
        loop_index: usize,
    }
    impl LassoAtoms for Simple {
        fn n(&self) -> usize {
            self.p.len()
        }
        fn loop_index(&self) -> usize {
            self.loop_index
        }
        fn state_atom(&self, _atom: u32, pos: usize) -> bool {
            self.p[pos]
        }
        fn edge_atom(&self, _atom: u32, _pos: usize) -> bool {
            false
        }
    }

    fn p() -> Ltl {
        Ltl::SAtom(0)
    }

    #[test]
    fn fixpoints_on_lasso() {
        // word: p ¬p (¬p p)^ω, loop at 2 over positions [2,3]
        let w = Simple {
            p: vec![true, false, false, true],
            loop_index: 2,
        };
        assert!(holds(&Ltl::Eventually(Box::new(p())), &w));
        assert!(!holds(&Ltl::Always(Box::new(p())), &w));
        // GF p: p at position 3 in the loop — infinitely often
        assert!(holds(
            &Ltl::Always(Box::new(Ltl::Eventually(Box::new(p())))),
            &w
        ));
        // FG p: p never holds forever
        assert!(!holds(
            &Ltl::Eventually(Box::new(Ltl::Always(Box::new(p())))),
            &w
        ));

        // stutter lasso: p true then loops at ¬p forever
        let w2 = Simple {
            p: vec![true, false],
            loop_index: 1,
        };
        assert!(!holds(
            &Ltl::Always(Box::new(Ltl::Eventually(Box::new(p())))),
            &w2
        ));
        assert!(holds(
            &Ltl::Eventually(Box::new(Ltl::Always(Box::new(Ltl::Not(Box::new(p())))))),
            &w2
        ));
    }
}
