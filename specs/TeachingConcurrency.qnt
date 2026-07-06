/**
 * Lamport's "Teaching Concurrency" algorithm, written from scratch.
 *
 * N processes, each executing atomically:
 *   x[i] := 1;
 *   y[i] := x[(i - 1) mod N]
 * and stopping. Initially every x[i] = 0 and y[i] is arbitrary.
 *
 * Property: once every process has stopped, at least one y[i] equals 1.
 * The proof is an inductive invariant: any process past its first statement
 * has x[i] = 1, so the last process to execute its second statement reads 1.
 *
 * Feature showcase: inductive invariants (--inductive-invariant),
 * sum types + match, record spread update, nondet initial state.
 *
 * Expected-pass:
 *   quint test   specs/TeachingConcurrency.qnt --main=main
 *   quint verify specs/TeachingConcurrency.qnt --main=main \
 *     --invariant=correctness --inductive-invariant=IndInv
 *
 * Expected-violation:
 *   quint verify specs/TeachingConcurrency.qnt --main=main \
 *     --invariant=brokenAllYOne
 *   quint verify specs/TeachingConcurrency.qnt --main=main \
 *     --invariant=correctness --inductive-invariant=brokenIndInv
 */
module teaching_concurrency {
  const N: int

  assume _ = N >= 2

  /// Program counter of one process.
  type Pc =
    | WriteX
    | WriteY
    | Done

  type Proc = {
    x: int,
    y: int,
    pc: Pc,
  }

  /// The y registers start with an arbitrary value from this small domain.
  pure val Domain = 0.to(2)

  val Idx = 0.to(N - 1)

  var procs: int -> Proc

  /// One atomic statement of process i, given its left neighbor's state.
  pure def exec(p: Proc, left: Proc): Proc =
    match p.pc {
      | WriteX => { ...p, x: 1, pc: WriteY }
      | WriteY => { ...p, y: left.x, pc: Done }
      | Done => p
    }

  action init =
    nondet ys = Idx.setOfMaps(Domain).oneOf()
    procs' = Idx.mapBy(i => { x: 0, y: ys.get(i), pc: WriteX })

  action step =
    val active = Idx.filter(i => procs.get(i).pc != Done)
    if (active != Set())
      nondet i = active.oneOf()
      procs' = procs.set(i, exec(procs.get(i), procs.get((i + N - 1) % N)))
    else
      // all processes stopped: stutter
      procs' = procs

  /// The algorithm's correctness property (holds).
  val correctness =
    Idx.forall(i => procs.get(i).pc == Done)
      implies Idx.exists(i => procs.get(i).y == 1)

  /// Domain constraint over the single state variable, required for
  /// inductive-invariant checking.
  val TypeOK =
    val procStates =
      tuples(Set(0, 1), Domain.union(Set(0, 1)), Set(WriteX, WriteY, Done))
        .map(((x, y, pc)) => { x: x, y: y, pc: pc })
    procs.in(Idx.setOfMaps(procStates))

  /// Inductive invariant proving correctness (holds).
  val IndInv = and {
    TypeOK,
    correctness,
    Idx.forall(i => procs.get(i).pc != WriteX implies procs.get(i).x == 1),
  }

  /// Too strong: only *some* y is guaranteed to be 1, not all of them.
  /// Counterexample: any complete schedule where some process reads its
  /// neighbor's x before that neighbor wrote it (depth 2*N).
  val brokenAllYOne =
    Idx.forall(i => procs.get(i).pc == Done)
      implies Idx.forall(i => procs.get(i).y == 1)

  /// Not inductive: without the x-strengthening conjunct the induction
  /// step fails (a WriteY process may read x = 0 from a Done neighbor).
  val brokenIndInv = and {
    TypeOK,
    correctness,
  }
}

module main {
  import teaching_concurrency(N = 3).*

  /// Any 6 steps finish all 3 processes (2 statements each), after which
  /// correctness must hold in every completed schedule.
  run completesTest =
    init.then(6.reps(_ => step)).expect(correctness)

  run correctnessAfterStutterTest =
    init.then(8.reps(_ => step)).expect(correctness)
}
