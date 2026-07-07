/**
 * Two-phase commit (Gray & Lamport, "Consensus on Transaction Commit"),
 * written from scratch.
 *
 * A transaction manager (TM) coordinates a set of resource managers (RMs).
 * RMs prepare spontaneously; the TM commits once all RMs have prepared, or
 * may abort at any time before deciding. Committed and Aborted must never
 * coexist across RMs.
 *
 * Feature showcase: sum types + match, run tests (.then/.expect/.fail),
 * weakFair + leadsTo checked with the TLC backend, tuple destructuring.
 *
 * Expected-pass:
 *   quint test   specs/TwoPhaseCommit.qnt --main=main
 *   quint verify specs/TwoPhaseCommit.qnt --main=main --invariant=consistency
 *   quint verify specs/TwoPhaseCommit.qnt --main=main --backend=tlc \
 *     --temporal=decisionReached,committedPropagates
 *
 * Expected-violation:
 *   quint verify specs/TwoPhaseCommit.qnt --main=main \
 *     --invariant=brokenNoAbort --max-steps=2            (depth 1)
 *   quint verify specs/TwoPhaseCommit.qnt --main=main --backend=tlc \
 *     --temporal=brokenDecisionNoFairness                (stuttering lasso)
 */
module two_phase_commit {
  const RMs: Set[str]

  assume _ = RMs != Set()

  type RMState = Working | Prepared | Committed | Aborted
  type TMState = TMInit | TMCommitted | TMAborted
  type Msg = PreparedMsg(str) | DoCommit | DoAbort

  var rmState: str -> RMState
  var tmState: TMState
  /// RMs from which the TM has received a PreparedMsg.
  var tmPrepared: Set[str]
  /// Message soup: messages are never removed, only added.
  var msgs: Set[Msg]

  val allVars = (rmState, tmState, tmPrepared, msgs)

  action init = all {
    rmState' = RMs.mapBy(_ => Working),
    tmState' = TMInit,
    tmPrepared' = Set(),
    msgs' = Set(),
  }

  /// An RM spontaneously prepares and notifies the TM.
  action rmPrepare(r: str): bool = all {
    rmState.get(r) == Working,
    rmState' = rmState.set(r, Prepared),
    msgs' = msgs.union(Set(PreparedMsg(r))),
    tmState' = tmState,
    tmPrepared' = tmPrepared,
  }

  /// An RM spontaneously aborts while still working.
  action rmChooseAbort(r: str): bool = all {
    rmState.get(r) == Working,
    rmState' = rmState.set(r, Aborted),
    tmState' = tmState,
    tmPrepared' = tmPrepared,
    msgs' = msgs,
  }

  /// The TM registers a PreparedMsg.
  action tmRcvPrepared(r: str): bool = all {
    tmState == TMInit,
    msgs.contains(PreparedMsg(r)),
    tmPrepared' = tmPrepared.union(Set(r)),
    rmState' = rmState,
    tmState' = tmState,
    msgs' = msgs,
  }

  /// The TM decides to commit once every RM has prepared.
  action tmCommit = all {
    tmState == TMInit,
    tmPrepared == RMs,
    tmState' = TMCommitted,
    msgs' = msgs.union(Set(DoCommit)),
    rmState' = rmState,
    tmPrepared' = tmPrepared,
  }

  /// The TM decides to abort (allowed any time before deciding).
  action tmAbort = all {
    tmState == TMInit,
    tmState' = TMAborted,
    msgs' = msgs.union(Set(DoAbort)),
    rmState' = rmState,
    tmPrepared' = tmPrepared,
  }

  /// An RM learns the commit decision.
  action rmRcvCommit(r: str): bool = all {
    msgs.contains(DoCommit),
    rmState.get(r) != Committed,
    rmState' = rmState.set(r, Committed),
    tmState' = tmState,
    tmPrepared' = tmPrepared,
    msgs' = msgs,
  }

  /// An RM learns the abort decision.
  action rmRcvAbort(r: str): bool = all {
    msgs.contains(DoAbort),
    rmState.get(r) != Aborted,
    rmState' = rmState.set(r, Aborted),
    tmState' = tmState,
    tmPrepared' = tmPrepared,
    msgs' = msgs,
  }

  action step =
    nondet r = RMs.oneOf()
    any {
      rmPrepare(r),
      rmChooseAbort(r),
      tmRcvPrepared(r),
      tmCommit,
      tmAbort,
      rmRcvCommit(r),
      rmRcvAbort(r),
      // Explicit stutter, so the state machine never deadlocks
      // (all decisions delivered is a legal resting state).
      all {
        rmState' = rmState,
        tmState' = tmState,
        tmPrepared' = tmPrepared,
        msgs' = msgs,
      },
    }

  /// Safety: no RM commits while another aborts (holds).
  val consistency =
    tuples(RMs, RMs).forall(((r1, r2)) =>
      not(rmState.get(r1) == Committed and rmState.get(r2) == Aborted))

  /// Once all RMs are prepared, a fair TM eventually decides (holds).
  temporal decisionReached =
    weakFair(tmCommit, allVars)
      implies ((tmPrepared == RMs and tmState == TMInit) leadsTo (tmState != TMInit))

  /// A commit decision fairly delivered reaches every RM (holds).
  temporal committedPropagates =
    RMs.forall(r => weakFair(rmRcvCommit(r), allVars))
      implies (tmState == TMCommitted leadsTo RMs.forall(r => rmState.get(r) == Committed))

  /// Without fairness the TM may never decide: stuttering lasso (fails).
  temporal brokenDecisionNoFairness =
    (tmPrepared == RMs and tmState == TMInit) leadsTo (tmState != TMInit)

  /// The TM can abort immediately: counterexample at depth 1 (fails).
  val brokenNoAbort = tmState != TMAborted
}

module main {
  import two_phase_commit(RMs = Set("rm1", "rm2", "rm3")).*

  /// Happy path: everyone prepares, TM commits, everyone commits.
  run happyPathTest =
    init
      .then(rmPrepare("rm1"))
      .then(rmPrepare("rm2"))
      .then(rmPrepare("rm3"))
      .then(tmRcvPrepared("rm1"))
      .then(tmRcvPrepared("rm2"))
      .then(tmRcvPrepared("rm3"))
      .then(tmCommit)
      .then(rmRcvCommit("rm1"))
      .then(rmRcvCommit("rm2"))
      .then(rmRcvCommit("rm3"))
      .expect(RMs.forall(r => rmState.get(r) == Committed))

  /// TM aborts early; consistency still holds after delivery.
  run abortTest =
    init
      .then(rmPrepare("rm1"))
      .then(tmAbort)
      .then(rmRcvAbort("rm2"))
      .expect(consistency)

  /// An RM cannot commit before the TM sent the decision.
  run commitWithoutDecisionTest =
    init.then(rmRcvCommit("rm1")).fail()

  /// Random exploration preserves consistency.
  run randomWalkTest =
    init.then(12.reps(_ => step)).expect(consistency)
}

// Larger instance for benchmarking explicit-state checkers.
module bench {
  import two_phase_commit(RMs = Set("rm1", "rm2", "rm3", "rm4", "rm5", "rm6", "rm7", "rm8")).*
}
