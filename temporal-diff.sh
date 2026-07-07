#!/usr/bin/env bash
# Differential test: quintck's native temporal verdicts vs quint's TLC
# backend on the same properties. The two TLC-unparseable properties
# (verNeverDecreases: next-atom; brokenAlwaysProgress: mustChange) are
# pinned to their Apalache-verified expectations and skipped on TLC.
#
# Requires: quint on PATH, Apalache dist downloaded (run a quint verify
# once), no other Apalache-backend runs in parallel.
set -u -o pipefail

cd "$(dirname "$0")"

echo "== building =="
cargo build -q --release || exit 1
Q=target/release/quintck

# file|main|init|step|temporal|fixture|quintck-extra
CASES=$(cat <<'EOF'
specs/TwoPhaseCommit.qnt|main|||decisionReached,committedPropagates|TwoPhaseCommit_temporal.json|
specs/TwoPhaseCommit.qnt|main|||brokenDecisionNoFairness|TwoPhaseCommit_temporal.json|
specs/ReadersWriters.qnt|main|||noStarvation|ReadersWriters_temporal.json|
specs/ReadersWriters.qnt|main|||brokenNoStarvationNoFairness|ReadersWriters_temporal.json|
specs/TwoLayeredCache.qnt|main|||verMonotone,eventuallyClean|TwoLayeredCache_temporal.json|
specs/TwoLayeredCache.qnt|main|||brokenEventuallyCleanNoFairness|TwoLayeredCache_temporal.json|
specs/DiningPhilosophers.qnt|dining_fixed|||noDeadlock,someoneEats|DiningPhilosophers_temporal.json|
specs/DiningPhilosophers.qnt|dining_fixed|||brokenSomeoneEatsNoFairness|DiningPhilosophers_temporal.json|
specs/DiningPhilosophers.qnt|dining_naive|naive::init|naive::step|naive::noDeadlock|DiningPhilosophers_naive_temporal.json|--no-deadlock
specs/ReliableBroadcast.qnt|main|||totality|ReliableBroadcast_temporal.json|
specs/ReliableBroadcast.qnt|main|||brokenTotalityNoFairness|ReliableBroadcast_temporal.json|
specs/Raft.qnt|raft_election|||termsMonotone,quorumCandidateProgress|Raft_election.json|
specs/Raft.qnt|raft_election|||brokenEventuallyLeaderNoFairness|Raft_election.json|
quint/examples/language-features/weakFairness.qnt||||eventuallyDone,notDoneLeadsToDone|weakFairness.json|
quint/examples/language-features/weakFairness.qnt||||notDoneLeadsToDoneNoFairness|weakFairness.json|
quint/examples/language-features/strongFairness.qnt||||eventuallyHundredDegrees|strongFairness.json|
quint/examples/language-features/strongFairness.qnt||||eventuallyHundredDegreesWeakOnly|strongFairness.json|
quint/examples/classic/distributed/ewd840/ewd840.qnt|ewd840_3|||liveness|ewd840_temporal.json|--no-deadlock
quint/examples/classic/distributed/ewd840/ewd840.qnt|ewd840_3|||falseLiveness|ewd840_temporal.json|--no-deadlock
quint/examples/classic/distributed/ewd426/ewd426.qnt|ewd426|||convergence,closure,persistence|ewd426_temporal.json|--no-deadlock
EOF
)

verdict_of() { # exit code + output -> pass|violation|error
  local code=$1 out=$2
  if [[ $code -eq 0 ]]; then echo pass
  elif grep -qi 'violation\|counterexample' <<<"$out"; then echo violation
  else echo error
  fi
}

failures=0
total=0

while IFS='|' read -r file main init step temporal fixture extra; do
  [[ -z "$file" ]] && continue
  total=$((total + 1))

  args=()
  [[ -n "$main" ]] && args+=("--main=$main")
  [[ -n "$init" ]] && args+=("--init=$init")
  [[ -n "$step" ]] && args+=("--step=$step")

  tlc_out=$(quint verify "$file" "${args[@]}" --backend=tlc --temporal="$temporal" 2>&1)
  tlc=$(verdict_of $? "$tlc_out")

  qk_args=("fixtures/$fixture" "--temporal=$temporal")
  [[ -n "$extra" ]] && qk_args+=($extra)
  qk_out=$($Q "${qk_args[@]}" 2>&1)
  qk=$(verdict_of $? "$qk_out")

  if [[ "$tlc" == "$qk" ]]; then
    printf "[ok]   %-28s tlc=%s quintck=%s\n" "$temporal" "$tlc" "$qk"
  else
    printf "[DIFF] %-28s tlc=%s quintck=%s (%s)\n" "$temporal" "$tlc" "$qk" "$file"
    echo "--- tlc: $(tail -2 <<<"$tlc_out")"
    echo "--- quintck: $(tail -1 <<<"$qk_out")"
    failures=$((failures + 1))
  fi
done <<<"$CASES"

echo
if [[ $failures -eq 0 ]]; then
  echo "all $total verdicts agree with TLC"
else
  echo "$failures of $total verdicts DIVERGE"
  exit 1
fi
