//! The enumeration oracle: a recorded trail of nondeterministic choices.
//!
//! The state-space search itself is BFS (see `explorer`); this controls the
//! *within-transition* enumeration. One evaluation run of an action makes a
//! sequence of choices (`any` branch picks, `nondet oneOf` picks). We record
//! them in a trail; re-running the action replays the trail prefix and
//! [`ChoiceCtl::advance`] moves to the next unexplored combination —
//! a depth-first walk over the choice tree of a single transition.
//! Evaluation is deterministic (ordered containers, no RNG), so replaying a
//! prefix always reaches the same choice points.

#[derive(Debug, Clone, Copy)]
pub struct ChoicePoint {
    pub bound: u64,
    pub chosen: u64,
}

#[derive(Debug, Default)]
pub struct ChoiceCtl {
    trail: Vec<ChoicePoint>,
    cursor: usize,
}

impl ChoiceCtl {
    pub fn new() -> Self {
        Self::default()
    }

    /// Called at every choice point in evaluation order. Returns the index
    /// chosen for this point on the current run, or `None` iff `bound == 0`
    /// (empty choice: the caller treats the branch as disabled).
    pub fn choose(&mut self, bound: u64) -> Option<u64> {
        if bound == 0 {
            return None;
        }
        if self.cursor < self.trail.len() {
            debug_assert_eq!(
                self.trail[self.cursor].bound, bound,
                "nondeterministic replay divergence (evaluation must be deterministic)"
            );
        } else {
            self.trail.push(ChoicePoint { bound, chosen: 0 });
        }
        let chosen = self.trail[self.cursor].chosen;
        self.cursor += 1;
        Some(chosen)
    }

    /// Move to the next unexplored path through the choice tree.
    /// Returns false when the whole tree is exhausted.
    pub fn advance(&mut self) -> bool {
        // Drop stale entries beyond what the last run actually visited:
        // they belong to abandoned subtrees.
        self.trail.truncate(self.cursor);
        while let Some(last) = self.trail.last_mut() {
            if last.chosen + 1 < last.bound {
                last.chosen += 1;
                self.cursor = 0;
                return true;
            }
            self.trail.pop();
        }
        false
    }

    /// Number of choice points visited on the last run.
    pub fn visited(&self) -> usize {
        self.cursor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Enumerate a static 2x3 choice tree.
    #[test]
    fn enumerates_product() {
        let mut ctl = ChoiceCtl::new();
        let mut seen = Vec::new();
        loop {
            let a = ctl.choose(2).unwrap();
            let b = ctl.choose(3).unwrap();
            seen.push((a, b));
            if !ctl.advance() {
                break;
            }
        }
        assert_eq!(
            seen,
            vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        );
    }

    /// A dependent tree: the second choice only exists on branch 1.
    #[test]
    fn enumerates_dependent_tree() {
        let mut ctl = ChoiceCtl::new();
        let mut seen = Vec::new();
        loop {
            let a = ctl.choose(2).unwrap();
            if a == 1 {
                let b = ctl.choose(2).unwrap();
                seen.push((a, Some(b)));
            } else {
                seen.push((a, None));
            }
            if !ctl.advance() {
                break;
            }
        }
        assert_eq!(seen, vec![(0, None), (1, Some(0)), (1, Some(1))]);
    }

    #[test]
    fn empty_bound_is_none() {
        let mut ctl = ChoiceCtl::new();
        assert_eq!(ctl.choose(0), None);
        assert!(!ctl.advance());
    }
}
