#!/usr/bin/env bash
# Verification harness for the specs/ suite.
#
#   bash specs/check.sh              run every check
#   bash specs/check.sh --only Raft  run only checks whose spec name matches
#
# Every check is either
#   pass      — the verifier must exit 0, or
#   violation — the verifier must exit non-zero AND report a counterexample
#               (output contains "violation"); a tool crash does not count.
set -u

cd "$(dirname "$0")/.."

ONLY="${2:-}"
if [[ "${1:-}" == "--only" ]]; then
  ONLY="$2"
fi

# expected|spec|description|command
CHECKS=$(cat <<'EOF'
pass|TeachingConcurrency|run tests|quint test specs/TeachingConcurrency.qnt --main=main
pass|TeachingConcurrency|inductive invariant proves correctness|quint verify specs/TeachingConcurrency.qnt --main=main --invariant=correctness --inductive-invariant=IndInv
violation|TeachingConcurrency|all-y-one is too strong (depth 6)|quint verify specs/TeachingConcurrency.qnt --main=main --invariant=brokenAllYOne
violation|TeachingConcurrency|weakened IndInv is not inductive|quint verify specs/TeachingConcurrency.qnt --main=main --invariant=correctness --inductive-invariant=brokenIndInv
pass|ClockSync|bounded skew invariant|quint verify specs/ClockSync.qnt --main=main --invariant=skewOK
pass|ClockSync|inductive invariant proves skew bound|quint verify specs/ClockSync.qnt --main=main --invariant=skewOK --inductive-invariant=IndInv
violation|ClockSync|skew bound minus one (depth 1)|quint verify specs/ClockSync.qnt --main=main --invariant=brokenSkewTight --max-steps=2
violation|ClockSync|TypeOK alone does not imply skewOK|quint verify specs/ClockSync.qnt --main=main --invariant=skewOK --inductive-invariant=TypeOK
pass|TwoPhaseCommit|run tests (happy path, abort, .fail)|quint test specs/TwoPhaseCommit.qnt --main=main
pass|TwoPhaseCommit|commit/abort consistency|quint verify specs/TwoPhaseCommit.qnt --main=main --invariant=consistency
violation|TwoPhaseCommit|abort is reachable (depth 1)|quint verify specs/TwoPhaseCommit.qnt --main=main --invariant=brokenNoAbort --max-steps=2
pass|TwoPhaseCommit|fair TM decides; commits propagate (TLC)|quint verify specs/TwoPhaseCommit.qnt --main=main --backend=tlc --temporal=decisionReached,committedPropagates
violation|TwoPhaseCommit|no decision without fairness (TLC lasso)|quint verify specs/TwoPhaseCommit.qnt --main=main --backend=tlc --temporal=brokenDecisionNoFairness
pass|ReadersWriters|reader/writer exclusion|quint verify specs/ReadersWriters.qnt --main=main --invariant=safety
violation|ReadersWriters|concurrent readers allowed (depth 4)|quint verify specs/ReadersWriters.qnt --main=main --invariant=brokenOneReader --max-steps=5
pass|ReadersWriters|FIFO + fairness = no starvation (TLC)|quint verify specs/ReadersWriters.qnt --main=main --backend=tlc --temporal=noStarvation
violation|ReadersWriters|starvation without fairness (TLC lasso)|quint verify specs/ReadersWriters.qnt --main=main --backend=tlc --temporal=brokenNoStarvationNoFairness
pass|TwoLayeredCache|write-back consistency invariants|quint verify specs/TwoLayeredCache.qnt --main=main --invariants cleanConsistency dirtyInL1
violation|TwoLayeredCache|L1 not always backed by L2 (depth 1)|quint verify specs/TwoLayeredCache.qnt --main=main --invariant=brokenL1Backed --max-steps=2
pass|TwoLayeredCache|version monotone (orKeep) + fair flush drains (TLC)|quint verify specs/TwoLayeredCache.qnt --main=main --backend=tlc --temporal=verMonotone,eventuallyClean
pass|TwoLayeredCache|version monotone via next (Apalache temporal)|echo y | quint verify specs/TwoLayeredCache.qnt --main=main --temporal=verNeverDecreases
violation|TwoLayeredCache|cannot progress forever, mustChange (Apalache temporal)|echo y | quint verify specs/TwoLayeredCache.qnt --main=main --temporal=brokenAlwaysProgress
violation|TwoLayeredCache|dirty may linger without fairness (TLC lasso)|quint verify specs/TwoLayeredCache.qnt --main=main --backend=tlc --temporal=brokenEventuallyCleanNoFairness
pass|DiningPhilosophers|fork-ownership consistency|quint verify specs/DiningPhilosophers.qnt --main=dining_fixed --invariant=consistent
violation|DiningPhilosophers|eating is reachable (depth 3)|quint verify specs/DiningPhilosophers.qnt --main=dining_fixed --invariant=brokenNeverEating --max-steps=4
pass|DiningPhilosophers|asymmetric fix: no deadlock + progress (TLC, enabled/strongFair)|quint verify specs/DiningPhilosophers.qnt --main=dining_fixed --backend=tlc --temporal=noDeadlock,someoneEats
violation|DiningPhilosophers|naive left-first deadlocks (TLC, depth 6)|quint verify specs/DiningPhilosophers.qnt --main=dining_naive --backend=tlc --init=naive::init --step=naive::step --temporal=naive::noDeadlock
violation|DiningPhilosophers|no progress without fairness (TLC lasso)|quint verify specs/DiningPhilosophers.qnt --main=dining_fixed --backend=tlc --temporal=brokenSomeoneEatsNoFairness
pass|ReliableBroadcast|validity + relay-before-deliver|quint verify specs/ReliableBroadcast.qnt --main=main --invariants validity relayedBeforeDelivered
pass|ReliableBroadcast|inductive invariant proves validity|quint verify specs/ReliableBroadcast.qnt --main=main --invariant=validity --inductive-invariant=IndInv
violation|ReliableBroadcast|delivery is reachable (depth 2)|quint verify specs/ReliableBroadcast.qnt --main=main --invariant=brokenNobodyDelivers --max-steps=3
pass|ReliableBroadcast|totality under fair relaying (TLC)|quint verify specs/ReliableBroadcast.qnt --main=main --backend=tlc --temporal=totality
violation|ReliableBroadcast|no totality without fairness (TLC lasso)|quint verify specs/ReliableBroadcast.qnt --main=main --backend=tlc --temporal=brokenTotalityNoFairness
pass|LamportMutex|run tests (round trip, .reps random walk)|quint test specs/LamportMutex.qnt --main=main
pass|LamportMutex|mutual exclusion + request consistency|quint verify specs/LamportMutex.qnt --main=main --invariants mutex requestConsistency --max-steps=8
violation|LamportMutex|critical section is reachable (depth 6)|quint verify specs/LamportMutex.qnt --main=main --invariant=brokenNooneCritical --max-steps=6
pass|Paxos|agreement + one value per ballot|quint verify specs/Paxos.qnt --main=main --invariants agreement oneValuePerBallot --max-steps=8
violation|Paxos|a value is chosen (depth 6)|quint verify specs/Paxos.qnt --main=main --invariant=brokenNothingChosen --max-steps=6
pass|Raft|election safety + log matching + vote integrity|quint verify specs/Raft.qnt --main=raft_3 --invariants electionSafety logMatching voteIntegrity --max-steps=8
violation|Raft|a leader is reachable (depth 3)|quint verify specs/Raft.qnt --main=raft_3 --invariant=brokenNoLeader --max-steps=4
violation|Raft|concurrent candidates (depth 2)|quint verify specs/Raft.qnt --main=raft_3 --invariant=brokenAtMostOneCandidate --max-steps=3
pass|Raft|terms monotone (orKeep) + quorum candidate progress (TLC)|quint verify specs/Raft.qnt --main=raft_election --backend=tlc --temporal=termsMonotone,quorumCandidateProgress
violation|Raft|no leader without fairness (TLC lasso)|quint verify specs/Raft.qnt --main=raft_election --backend=tlc --temporal=brokenEventuallyLeaderNoFairness
EOF
)

SPECS=(TeachingConcurrency ClockSync TwoPhaseCommit ReadersWriters TwoLayeredCache
       DiningPhilosophers ReliableBroadcast LamportMutex Paxos Raft)

failures=0
total=0

echo "== typecheck =="
for s in "${SPECS[@]}"; do
  if [[ -n "$ONLY" && "$s" != *"$ONLY"* ]]; then continue; fi
  total=$((total + 1))
  if out=$(quint typecheck "specs/$s.qnt" 2>&1); then
    echo "[ok]   typecheck $s"
  else
    echo "[FAIL] typecheck $s"
    echo "$out"
    failures=$((failures + 1))
  fi
done

# The TLC backend reuses the Apalache distribution jar; make sure it is
# downloaded (and the Apalache server warmed up) before any TLC check runs.
echo "== warm-up (downloads Apalache on first run) =="
quint verify specs/ClockSync.qnt --main=main --invariant=skewOK --max-steps=1 >/dev/null 2>&1 \
  || echo "warning: warm-up verify failed; subsequent checks may fail"

echo "== checks =="
while IFS='|' read -r expected spec desc cmd; do
  [[ -z "$expected" ]] && continue
  if [[ -n "$ONLY" && "$spec" != *"$ONLY"* ]]; then continue; fi
  total=$((total + 1))

  out=$(eval "$cmd" 2>&1)
  code=$?

  ok=false
  case "$expected" in
    pass)
      [[ $code -eq 0 ]] && ok=true
      ;;
    violation)
      [[ $code -ne 0 ]] && grep -q "violation" <<<"$out" && ok=true
      ;;
  esac

  if $ok; then
    echo "[ok]   $spec: $desc (expected $expected)"
  else
    echo "[FAIL] $spec: $desc (expected $expected, exit $code)"
    echo "----- command: $cmd"
    echo "$out" | tail -20
    echo "-----"
    failures=$((failures + 1))
  fi
done <<<"$CHECKS"

echo
if [[ $failures -eq 0 ]]; then
  echo "all $total checks behaved as expected"
else
  echo "$failures of $total checks FAILED"
  exit 1
fi
