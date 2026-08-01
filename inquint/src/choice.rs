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

/// Pure mixed-radix increment over the visited prefix of a choice trail.
/// Kept slice-based so the omission-critical part of `advance` can be
/// verified without modeling allocation.
fn advance_trail(
    trail: &mut [ChoicePoint],
    len: &mut usize,
    cursor: &mut usize,
) -> bool {
    *len = (*len).min(*cursor);
    while *len > 0 {
        let last = &mut trail[*len - 1];
        if last.chosen + 1 < last.bound {
            last.chosen += 1;
            *cursor = 0;
            return true;
        }
        *len -= 1;
    }
    false
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
        let mut len = self.trail.len();
        let advanced = advance_trail(&mut self.trail, &mut len, &mut self.cursor);
        // Drop stale entries beyond what the last run actually visited and
        // exhausted suffix digits. They belong to abandoned subtrees.
        self.trail.truncate(len);
        advanced
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

    /// Exhaustively cover small dependent choice trees. This is the concrete
    /// counterpart of the Kani proof below and guards ordinary builds too.
    #[test]
    fn all_small_dependent_trees_are_complete_and_unique() {
        for first_bound in 1..=3u64 {
            for encoded in 0..4u64.pow(first_bound as u32) {
                let mut rest = encoded;
                let mut second = [0u64; 3];
                for bound in second.iter_mut().take(first_bound as usize) {
                    *bound = rest % 4;
                    rest /= 4;
                }
                check_dependent_tree(first_bound, second);
            }
        }
    }

    fn check_dependent_tree(first_bound: u64, second: [u64; 3]) {
        let mut ctl = ChoiceCtl::new();
        let mut counts = [0u8; 12];
        loop {
            let a = ctl.choose(first_bound).unwrap();
            let b = ctl.choose(second[a as usize]);
            let slot = a as usize * 4 + b.map_or(0, |v| v as usize + 1);
            counts[slot] += 1;
            if !ctl.advance() {
                break;
            }
        }
        for a in 0..first_bound as usize {
            if second[a] == 0 {
                assert_eq!(counts[a * 4], 1);
            } else {
                assert_eq!(counts[a * 4], 0);
                for b in 0..second[a] as usize {
                    assert_eq!(counts[a * 4 + b + 1], 1);
                }
            }
        }
    }
}

#[cfg(kani)]
mod verification {
    use super::*;

    /// The production mixed-radix increment visits every pair exactly once
    /// for all radices in 1..=2.
    #[kani::proof]
    #[kani::unwind(6)]
    fn mixed_radix_advance_is_complete_and_unique() {
        let first: u8 = kani::any();
        let second: u8 = kani::any();
        kani::assume((1..=2).contains(&first));
        kani::assume((1..=2).contains(&second));
        let mut trail = [
            ChoicePoint { bound: first as u64, chosen: 0 },
            ChoicePoint { bound: second as u64, chosen: 0 },
        ];
        let mut len = 2usize;
        let mut cursor = 2usize;
        let mut counts = [0u8; 4];
        loop {
            let slot = trail[0].chosen as usize * 2 + trail[1].chosen as usize;
            assert_eq!(counts[slot], 0, "choice leaf repeated");
            counts[slot] = 1;
            if !advance_trail(&mut trail, &mut len, &mut cursor) {
                break;
            }
            if len == 1 {
                // The next deterministic replay reaches its second choice
                // and appends a fresh zero digit after `advance` truncated it.
                trail[1] = ChoicePoint {
                    bound: second as u64,
                    chosen: 0,
                };
                len = 2;
            }
            cursor = len;
        }
        for a in 0..first as usize {
            for b in 0..second as usize {
                assert_eq!(counts[a * 2 + b], 1, "choice leaf omitted");
            }
        }
        kani::cover!(first == 2 && second == 2);
    }

    /// A dependent evaluation that skipped the second point cannot retain
    /// that stale suffix for the next branch.
    #[kani::proof]
    fn advance_discards_unvisited_suffix() {
        let chosen: u64 = kani::any();
        kani::assume(chosen < 2);
        let mut trail = [
            ChoicePoint { bound: 2, chosen },
            ChoicePoint { bound: 2, chosen: 1 },
        ];
        let mut len = 2usize;
        let mut cursor = 1usize;
        let advanced = advance_trail(&mut trail, &mut len, &mut cursor);
        assert_eq!(len, if chosen == 0 { 1 } else { 0 });
        assert_eq!(advanced, chosen == 0);
    }
}
