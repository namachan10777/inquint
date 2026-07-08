/**
 * Abstract clock synchronization with bounded drift, written from scratch.
 *
 * N processes each own a discrete clock. A clock may tick forward by 1 or 2,
 * but only while staying within MaxSkew of the slowest clock (drift bound
 * enforced by the environment). A process may also resynchronize by jumping
 * to the fastest clock. The skew between any two clocks never exceeds
 * MaxSkew, and this is provable by a simple inductive invariant.
 *
 * Feature showcase: inductive invariants that hold by construction,
 * pure def fold helpers (min/max), assume declarations, bounded nondet,
 * a known-depth broken invariant (depth 1).
 *
 * Expected-pass:
 *   quint verify specs/ClockSync.qnt --main=main --invariant=skewOK
 *   quint verify specs/ClockSync.qnt --main=main \
 *     --invariant=skewOK --inductive-invariant=IndInv
 *
 * Expected-violation:
 *   quint verify specs/ClockSync.qnt --main=main \
 *     --invariant=brokenSkewTight --max-steps=2       (counterexample depth 1)
 *   quint verify specs/ClockSync.qnt --main=main \
 *     --invariant=skewOK --inductive-invariant=TypeOK (implication check fails)
 */
module clock_sync {
  const N: int
  const MaxSkew: int
  /// Global clock ceiling, so the reachable state space is finite.
  const MaxClock: int

  assume _ = and {
    N >= 2,
    MaxSkew >= 2,
    MaxClock > MaxSkew,
  }

  val Procs = 1.to(N)

  var clocks: int -> int

  /// Maximum of a non-empty set of clock values (clocks are in 0..MaxClock).
  pure def maxOf(s: Set[int]): int =
    s.fold(0, (m, x) => if (x > m) x else m)

  /// Minimum of a set of clock values, folding down from `ceiling`.
  pure def minOf(s: Set[int], ceiling: int): int =
    s.fold(ceiling, (m, x) => if (x < m) x else m)

  def slowest = minOf(Procs.map(p => clocks.get(p)), MaxClock)
  def fastest = maxOf(Procs.map(p => clocks.get(p)))

  action init =
    clocks' = Procs.mapBy(_ => 0)

  /// Clock p drifts forward, but the environment guarantees it never gets
  /// more than MaxSkew ahead of the slowest clock, nor past MaxClock.
  action tick(p: int): bool =
    nondet d = 1.to(2).oneOf()
    all {
      clocks.get(p) + d <= slowest + MaxSkew,
      clocks.get(p) + d <= MaxClock,
      clocks' = clocks.set(p, clocks.get(p) + d),
    }

  /// Clock p resynchronizes to the fastest clock.
  action resync(p: int): bool =
    clocks' = clocks.set(p, fastest)

  action step =
    nondet p = Procs.oneOf()
    any {
      tick(p),
      resync(p),
    }

  /// Safety: pairwise skew is bounded (holds).
  val skewOK =
    tuples(Procs, Procs).forall(((p, q)) =>
      clocks.get(p) - clocks.get(q) <= MaxSkew)

  /// Domain constraint over the state, needed for induction.
  val TypeOK =
    clocks.in(Procs.setOfMaps(0.to(MaxClock)))

  /// Inductive invariant: the tick guard re-establishes skewOK directly,
  /// and resync only moves a clock up to the current maximum (holds).
  val IndInv = and {
    TypeOK,
    skewOK,
  }

  /// Too tight by one: a single tick with d = 2 exceeds MaxSkew - 1.
  /// Counterexample at depth 1.
  val brokenSkewTight =
    tuples(Procs, Procs).forall(((p, q)) =>
      clocks.get(p) - clocks.get(q) <= MaxSkew - 1)
}

module main {
  import clock_sync(N = 3, MaxSkew = 2, MaxClock = 6).*
}

// Larger instance for benchmarking explicit-state checkers.
module bench {
  import clock_sync(N = 7, MaxSkew = 5, MaxClock = 40).*
}
