#!/usr/bin/env bash
# Runs quintck over the specs corpus fixtures and asserts the expected
# outcome of every check — the invariant/deadlock subset of specs/check.sh,
# checked natively (no Apalache/TLC).
#
#   bash quintck-check.sh [--only <Spec>]
#
# pass      => exit 0
# violation => exit 1 AND output contains "[violation]" at the documented depth
set -u

cd "$(dirname "$0")"

ONLY=""
if [[ "${1:-}" == "--only" ]]; then
  ONLY="$2"
fi

echo "== building =="
cargo build -q --release || exit 1
Q=target/release/quintck

# expected|spec|depth(empty for pass)|args...
CHECKS=$(cat <<'EOF'
pass|TeachingConcurrency||fixtures/TeachingConcurrency.json --invariant=correctness --exhaustive
violation|TeachingConcurrency|6|fixtures/TeachingConcurrency.json --invariant=brokenAllYOne --exhaustive
pass|ClockSync||fixtures/ClockSync.json --invariant=skewOK --exhaustive
violation|ClockSync|1|fixtures/ClockSync.json --invariant=brokenSkewTight --max-steps=2
pass|TwoPhaseCommit||fixtures/TwoPhaseCommit.json --invariant=consistency --exhaustive
violation|TwoPhaseCommit|1|fixtures/TwoPhaseCommit.json --invariant=brokenNoAbort --max-steps=2
pass|ReadersWriters||fixtures/ReadersWriters.json --invariant=safety --exhaustive
violation|ReadersWriters|4|fixtures/ReadersWriters.json --invariant=brokenOneReader --max-steps=5
pass|TwoLayeredCache||fixtures/TwoLayeredCache.json --invariant=cleanConsistency,dirtyInL1 --exhaustive
violation|TwoLayeredCache|1|fixtures/TwoLayeredCache.json --invariant=brokenL1Backed --max-steps=2
pass|DiningPhilosophers||fixtures/DiningPhilosophers_fixed.json --invariant=consistent --exhaustive
violation|DiningPhilosophers|3|fixtures/DiningPhilosophers_fixed.json --invariant=brokenNeverEating --max-steps=4
violation|DiningPhilosophers|6|fixtures/DiningPhilosophers_naive.json --exhaustive
pass|ReliableBroadcast||fixtures/ReliableBroadcast.json --invariant=validity,relayedBeforeDelivered --exhaustive
violation|ReliableBroadcast|2|fixtures/ReliableBroadcast.json --invariant=brokenNobodyDelivers --max-steps=3
pass|LamportMutex||fixtures/LamportMutex.json --invariant=mutex,requestConsistency --max-steps=8
violation|LamportMutex|4|fixtures/LamportMutex.json --invariant=brokenNooneCritical --max-steps=6
pass|Paxos||fixtures/Paxos.json --invariant=agreement,oneValuePerBallot --max-steps=8
violation|Paxos|6|fixtures/Paxos.json --invariant=brokenNothingChosen --max-steps=6
pass|Raft||fixtures/Raft.json --invariant=electionSafety,logMatching,voteIntegrity --max-steps=8
violation|Raft|3|fixtures/Raft.json --invariant=brokenNoLeader --max-steps=4
violation|Raft|2|fixtures/Raft.json --invariant=brokenAtMostOneCandidate --max-steps=3
EOF
)

failures=0
total=0

echo "== checks =="
while IFS='|' read -r expected spec depth args; do
  [[ -z "$expected" ]] && continue
  if [[ -n "$ONLY" && "$spec" != *"$ONLY"* ]]; then continue; fi
  total=$((total + 1))

  out=$($Q $args 2>&1)
  code=$?

  ok=false
  case "$expected" in
    pass)
      [[ $code -eq 0 ]] && ok=true
      ;;
    violation)
      if [[ $code -eq 1 ]] && grep -q '\[violation\]' <<<"$out"; then
        if [[ -n "$depth" ]]; then
          grep -qE "at depth $depth\$" <<<"$out" && ok=true
        else
          ok=true
        fi
      fi
      ;;
  esac

  if $ok; then
    echo "[ok]   $spec: $args (expected $expected${depth:+ @ depth $depth})"
  else
    echo "[FAIL] $spec: $args (expected $expected${depth:+ @ depth $depth}, exit $code)"
    echo "$out" | tail -8
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
