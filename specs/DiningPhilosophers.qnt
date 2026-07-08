/**
 * Dining philosophers, written from scratch.
 *
 * N philosophers around a table, one fork between each pair. Each
 * philosopher picks up a first fork, then a second, eats, and puts both
 * down. The naive strategy (everyone left-first) deadlocks; the asymmetric
 * fix (the last philosopher goes right-first) is deadlock-free.
 *
 * One parameterized module, two instances:
 *   dining_fixed — fixLast = true,  deadlock-free
 *   dining_naive — fixLast = false, reaches a deadlock at depth 6 (N = 3)
 *
 * Feature showcase: polymorphic Option[a], enabled, strongFair,
 * qualified instances (import ... as naive + naive:: CLI flags).
 *
 * Expected-pass:
 *   quint verify specs/DiningPhilosophers.qnt --main=dining_fixed \
 *     --invariant=consistent
 *   quint verify specs/DiningPhilosophers.qnt --main=dining_fixed \
 *     --backend=tlc --temporal=noDeadlock,someoneEats
 *
 * Expected-violation:
 *   quint verify specs/DiningPhilosophers.qnt --main=dining_fixed \
 *     --invariant=brokenNeverEating --max-steps=4        (depth 3)
 *   quint verify specs/DiningPhilosophers.qnt --main=dining_naive \
 *     --backend=tlc --init=naive::init --step=naive::step \
 *     --temporal=naive::noDeadlock                       (deadlock, depth 6)
 *   quint verify specs/DiningPhilosophers.qnt --main=dining_fixed \
 *     --backend=tlc --temporal=brokenSomeoneEatsNoFairness
 */
module dining {
  const N: int
  /// If true, the last philosopher picks the right fork first.
  const fixLast: bool

  assume _ = N >= 2

  type Option[a] = Some(a) | None

  type PhilState = Thinking | WaitFirst | WaitSecond | Eating

  val Ids = 0.to(N - 1)

  var phil: int -> PhilState
  /// fork.get(f) is the philosopher currently holding fork f, if any.
  var fork: int -> Option[int]

  val allVars = (phil, fork)

  pure def left(i: int): int = i
  pure def right(i: int): int = (i + 1) % N

  def firstFork(i: int): int =
    if (fixLast and i == N - 1) right(i) else left(i)

  def secondFork(i: int): int =
    if (fixLast and i == N - 1) left(i) else right(i)

  def holds(i: int, f: int): bool = fork.get(f) == Some(i)

  action init = all {
    phil' = Ids.mapBy(_ => Thinking),
    fork' = Ids.mapBy(_ => None),
  }

  action getHungry(i: int): bool = all {
    phil.get(i) == Thinking,
    phil' = phil.set(i, WaitFirst),
    fork' = fork,
  }

  action takeFirst(i: int): bool = all {
    phil.get(i) == WaitFirst,
    fork.get(firstFork(i)) == None,
    fork' = fork.set(firstFork(i), Some(i)),
    phil' = phil.set(i, WaitSecond),
  }

  action takeSecond(i: int): bool = all {
    phil.get(i) == WaitSecond,
    fork.get(secondFork(i)) == None,
    fork' = fork.set(secondFork(i), Some(i)),
    phil' = phil.set(i, Eating),
  }

  action putDown(i: int): bool = all {
    phil.get(i) == Eating,
    fork' = fork.setBy(firstFork(i), _ => None).setBy(secondFork(i), _ => None),
    phil' = phil.set(i, Thinking),
  }

  // Deliberately no stutter branch: in the naive configuration the
  // all-hold-one-fork state has no successor, so noDeadlock is falsifiable.
  action step =
    nondet i = Ids.oneOf()
    any {
      getHungry(i),
      takeFirst(i),
      takeSecond(i),
      putDown(i),
    }

  /// Fork ownership is consistent with philosopher states (holds).
  val consistent = and {
    Ids.forall(f =>
      match fork.get(f) {
        | Some(p) => and {
            f == firstFork(p) or f == secondFork(p),
            phil.get(p).in(Set(WaitSecond, Eating)),
          }
        | None => true
      }),
    Ids.forall(p =>
      phil.get(p) == Eating implies
        (holds(p, firstFork(p)) and holds(p, secondFork(p)))),
  }

  /// Some transition is always possible (holds for fixed, fails for naive).
  temporal noDeadlock = always(enabled(step))

  /// Under fairness, the table keeps making progress: infinitely often
  /// somebody eats. takeSecond/takeFirst are only intermittently enabled
  /// (a neighbor may grab the fork), hence strong fairness (holds for fixed).
  temporal someoneEats =
    and {
      Ids.forall(i => weakFair(getHungry(i), allVars)),
      Ids.forall(i => strongFair(takeFirst(i), allVars)),
      Ids.forall(i => strongFair(takeSecond(i), allVars)),
      Ids.forall(i => weakFair(putDown(i), allVars)),
    } implies always(eventually(Ids.exists(i => phil.get(i) == Eating)))

  /// Without any fairness nobody is forced to act (fails: stuttering-free
  /// lasso where philosophers cycle without ever eating does not exist —
  /// the violating lasso is the one where nothing beyond thinking happens).
  temporal brokenSomeoneEatsNoFairness =
    always(eventually(Ids.exists(i => phil.get(i) == Eating)))

  /// Eating is clearly reachable: depth 3 (fails).
  val brokenNeverEating = Ids.forall(i => phil.get(i) != Eating)
}

module dining_fixed {
  import dining(N = 3, fixLast = true).*
}

module dining_naive {
  import dining(N = 3, fixLast = false) as naive
}

// Larger instance for benchmarking explicit-state checkers.
module bench {
  import dining(N = 13, fixLast = true).*
}

// Very large instance: 96-entry maps, for persistent-container
// benchmarks (measured with --max-states as a throughput probe).
module bench_large {
  import dining(N = 96, fixLast = true).*
}
