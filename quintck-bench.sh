#!/usr/bin/env bash
# Benchmark suite: one heavyweight check per spec, each sized to take on
# the order of a minute with the v1 (unoptimized) checker. Use this to
# measure optimization work; correctness is gated by quintck-check.sh.
#
#   bash quintck-bench.sh [--only <Spec>]
#
# The bench instances live in each spec's `module bench` and are compiled
# into fixtures/bench_*.json by fixtures/regen.sh. They are NOT verified by
# Apalache/TLC (specs/check.sh) — far too large for those backends.
set -u -o pipefail

cd "$(dirname "$0")"

ONLY=""
if [[ "${1:-}" == "--only" ]]; then
  ONLY="$2"
fi

echo "== building (release) =="
cargo build -q --release || exit 1
Q=target/release/quintck

# spec|args   (all expected to pass; timing is the point)
BENCHES=$(cat <<'EOF'
TeachingConcurrency|fixtures/bench_TeachingConcurrency.json --exhaustive
ClockSync|fixtures/bench_ClockSync.json --exhaustive
TwoPhaseCommit|fixtures/bench_TwoPhaseCommit.json --exhaustive
ReadersWriters|fixtures/bench_ReadersWriters.json --exhaustive
TwoLayeredCache|fixtures/bench_TwoLayeredCache.json --exhaustive
DiningPhilosophers|fixtures/bench_DiningPhilosophers.json --exhaustive
ReliableBroadcast|fixtures/bench_ReliableBroadcast.json --exhaustive
LamportMutex|fixtures/bench_LamportMutex.json --exhaustive
Paxos|fixtures/bench_Paxos.json --exhaustive
Raft|fixtures/bench_Raft.json --exhaustive
EOF
)

failures=0
total_time=0

while IFS='|' read -r spec args; do
  [[ -z "$spec" ]] && continue
  if [[ -n "$ONLY" && "$spec" != *"$ONLY"* ]]; then continue; fi

  start=$(python3 -c 'import time; print(time.time())')
  out=$($Q $args 2>&1 | tail -1)
  code=$?
  end=$(python3 -c 'import time; print(time.time())')
  elapsed=$(echo "$end - $start" | bc)
  total_time=$(echo "$total_time + $elapsed" | bc)

  if [[ $code -eq 0 ]]; then
    printf "[ok]   %7.2fs  %-20s %s\n" "$elapsed" "$spec" "$out"
  else
    printf "[FAIL] %7.2fs  %-20s exit=%d %s\n" "$elapsed" "$spec" "$code" "$out"
    failures=$((failures + 1))
  fi
done <<<"$BENCHES"

echo
printf "total: %.1fs\n" "$total_time"
if [[ $failures -gt 0 ]]; then
  echo "$failures benches FAILED"
  exit 1
fi
