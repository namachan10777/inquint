#!/usr/bin/env bash
# Regenerates the committed `quint compile --target=json` fixtures from
# specs/*.qnt. Pinned to quint 0.32.x — the IR shapes are versioned.
set -eu

cd "$(dirname "$0")/.."

version=$(quint --version)
case "$version" in
  0.32.*) ;;
  *) echo "error: fixtures are pinned to quint 0.32.x, found $version" >&2; exit 1 ;;
esac

gen() { # gen <outname> <file> <args...>
  local out="fixtures/$1"; shift
  echo "generating $out"
  quint compile --target=json "$@" > "$out"
}

gen TeachingConcurrency.json specs/TeachingConcurrency.qnt --main=main \
  --invariant=correctness,brokenAllYOne
gen ClockSync.json specs/ClockSync.qnt --main=main \
  --invariant=skewOK,brokenSkewTight
gen TwoPhaseCommit.json specs/TwoPhaseCommit.qnt --main=main \
  --invariant=consistency,brokenNoAbort
gen ReadersWriters.json specs/ReadersWriters.qnt --main=main \
  --invariant=safety,brokenOneReader
gen TwoLayeredCache.json specs/TwoLayeredCache.qnt --main=main \
  --invariant=cleanConsistency,dirtyInL1,brokenL1Backed
gen DiningPhilosophers_fixed.json specs/DiningPhilosophers.qnt --main=dining_fixed \
  --invariant=consistent,brokenNeverEating
gen DiningPhilosophers_naive.json specs/DiningPhilosophers.qnt --main=dining_naive \
  --init=naive::init --step=naive::step
gen ReliableBroadcast.json specs/ReliableBroadcast.qnt --main=main \
  --invariant=validity,relayedBeforeDelivered,brokenNobodyDelivers
gen LamportMutex.json specs/LamportMutex.qnt --main=main \
  --invariant=mutex,requestConsistency,brokenNooneCritical
gen Paxos.json specs/Paxos.qnt --main=main \
  --invariant=agreement,oneValuePerBallot,brokenNothingChosen
gen Raft.json specs/Raft.qnt --main=raft_3 \
  --invariant=electionSafety,logMatching,voteIntegrity,brokenNoLeader,brokenAtMostOneCandidate

# Benchmark instances (module bench in each spec): much larger parameters,
# sized so each takes on the order of a minute with the v1 checker. Only
# checked by quintck (quintck-bench.sh) — too large for Apalache/TLC.
benchgen() { # benchgen <name> <invariants>
  local out="fixtures/bench_$1"
  echo "generating $out.json"
  quint compile --target=json "specs/$1.qnt" --main=bench --invariant="$2" > "$out.json"
}
benchgen TeachingConcurrency correctness
benchgen ClockSync skewOK
benchgen TwoPhaseCommit consistency
benchgen ReadersWriters safety
benchgen TwoLayeredCache cleanConsistency,dirtyInL1
benchgen DiningPhilosophers consistent
benchgen ReliableBroadcast validity,relayedBeforeDelivered
benchgen LamportMutex mutex,requestConsistency
benchgen Paxos agreement,oneValuePerBallot
benchgen Raft electionSafety,logMatching,voteIntegrity

# Upstream evaluator fixtures, for parser coverage of specs we didn't write.
for f in simple tictactoe ewd426 ewd840; do
  echo "copying $f.json from quint/evaluator/fixtures"
  cp "quint/evaluator/fixtures/$f.json" "fixtures/$f.json"
done

echo "done ($version)"
