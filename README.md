# inquint

Fast explicit-state (TLC-like) model checker for [Quint](https://quint-lang.org/).

inquint consumes the flattened JSON IR produced by `quint compile --target=json`,
explores the state space breadth-first with exhaustive enumeration of all
nondeterminism (`any` branches, `nondet ... oneOf` picks), checks invariants
and **temporal (LTL/liveness) properties**, detects deadlocks, executes `run`
test definitions, and emits counterexample traces in
[ITF format](https://apalache-mc.org/docs/adr/015adr-trace.html) — including
lassos with `loop_index` for liveness violations. BFS guarantees shortest
safety counterexamples.

Temporal checking is automata-theoretic: properties are translated to LTL
over state/edge atoms (with `Set.forall`-quantifiers over constant sets
expanded), safety-shaped bodies (`always` of propositional/action formulas,
incl. `orKeep`, `mustChange`, `enabled`, `next(...)` predicates) are checked
as invariants over the stutter-closed state graph, and liveness goes through
a GPVW tableau (¬property → generalized Büchi automaton), a product with the
state graph, and Tarjan SCC analysis. `weakFair`/`strongFair` premises
(`F1 and ... and Fn implies Body`) are handled TLC-style as SCC acceptance
side conditions, including strong-fairness refinement. Every extracted lasso
is self-validated against the negated property in debug builds.

## Getting started

```sh
cargo build --release            # binary at target/release/inquint
export PATH="$PWD/target/release:$PATH"
```

Checking a `.qnt` file directly requires the `quint` CLI on PATH (inquint
compiles it internally). Alternatively, pre-compile once and hand inquint
the JSON — no `quint` needed at check time:

```sh
quint compile --target=json --main=main spec.qnt > spec.json
inquint spec.json --invariant=myInvariant
```

## Usage

### Checking invariants (the default mode)

```sh
# check `agreement` up to 8 steps from the initial states
inquint specs/Paxos.qnt --main=main --invariant=agreement --max-steps=8

# explore the complete (finite) state space instead of a depth bound
inquint specs/Paxos.qnt --main=main --invariant=agreement --exhaustive

# several invariants at once
inquint spec.qnt --invariant=safety,consistency
```

Exploration is breadth-first over every nondeterministic choice (`any`
branches, `nondet ... oneOf` picks), so a reported counterexample is
always a shortest one. Deadlocks (states with no successor) are reported
by default; suppress with `--no-deadlock` for specs whose executions
legitimately terminate.

On success inquint prints the result scope and state-deduplication assurance.
`--exhaustive --exact-states` produces `[ok:assured-exact]`; a bounded run says
only that no violation was found through its stated depth, and the default
fingerprint mode is labeled `[ok:non-assured:fingerprint]`. On a violation it
prints the invariant name and the trace:

```text
State 0:
  ...          # initial state of the trace
State 1:
  ...          # first step
[violation] invariant 'agreement' violated at depth 1
```

### Temporal (LTL / liveness) properties

```sh
inquint specs/ReadersWriters.qnt --main=main --temporal=noStarvation
inquint specs/TwoLayeredCache.qnt --main=main \
  --temporal=verMonotone,eventuallyClean --out-itf lasso.itf.json
```

Temporal checking always explores the full state space (a depth bound
would be unsound; `--max-steps` is ignored with a warning). Liveness
counterexamples are lassos — with `--out-itf` they carry `loop_index` in
the ITF trace.

### Run tests

```sh
# execute `run` definitions; every path through their nondeterminism must pass
inquint specs/TwoPhaseCommit.qnt --main=main --test=happyPathTest,abortTest
inquint spec.qnt --test          # no names = all runs in the module
```

### Common options

| Option | Meaning |
|---|---|
| `--main=M` | main module (forwarded to `quint compile`) |
| `--invariant=I1,I2` | invariants to check (default: `q::inv` when the spec was compiled with one) |
| `--max-steps=N` | depth bound, default 10; `--exhaustive` removes it |
| `--init=A --step=A` | non-default action names (e.g. qualified instances: `--init=naive::init --step=naive::step`) |
| `--threads=N` | worker threads (default: all cores). Results are independent of the thread count |
| `--max-states=N` | abort without a verdict after ~N states (safety valve) |
| `--out-itf=PATH` | write the counterexample trace/lasso as [ITF](https://apalache-mc.org/docs/adr/015adr-trace.html) |
| `--exact-states` | exact deduplication (full states), required for the deterministic no-omission claim |
| `--test [T1,T2]` | execute `run` tests instead of model checking |

Exit codes: `0` = all properties hold, `1` = violation found (output
contains `[violation]` and the trace/lasso), `2` = tool error /
unsupported feature.

### Notes on semantics

- **Fingerprint mode (default)**: the seen-set stores 64-bit state
  fingerprints, like TLC. A hash collision would silently prune a state
  with probability ~n²/2⁶⁵ — use `--exact-states` when you want exact
  deduplication at a higher memory cost.
- **Verdict scope**: a bounded success means only “no violation through depth
  N”. Only an exhaustive, exact-state success claims the property over the
  complete reachable finite graph. Resource limits and unsupported operations
  return no verdict.
- **Parallelism**: state counts, depths and verdicts are deterministic
  and identical for every `--threads` value; when a parallel run finds a
  violation, the trace is reproduced by a deterministic single-threaded
  pass, so it is a shortest counterexample there too. Temporal checking,
  run tests and `--exact-states` are single-threaded.
- **Known limits**: `weakFair`/`strongFair` only as top-level premises;
  integers are i64; `allLists`, `Int`/`Nat` enumeration and
  `apalache::generate` are rejected. Unbounded random-walk `run` tests
  (e.g. `20.reps(_ => step)`) can exceed the path-enumeration cap — use
  the model checker for those.

## Workspace

- `quint-ast` — serde types for the quint compiler's flattened JSON IR
- `inquint` — checker core: values, evaluator, choice enumeration, BFS
- `inquint-cli` — the `inquint` binary
- `inquint-reference` — independent BTree/closure evaluator used only by the
  semantic differential verification tests

## Test corpus

`specs/` holds ten classic distributed algorithms (Paxos, Raft, Lamport
mutex, 2PC, ...) written in Quint, each with verified-true properties and
deliberately broken properties whose counterexample depths are known — see
`specs/README.md`. Two harnesses check them:

- `bash specs/check.sh` — verifies the corpus with quint's own backends
  (Apalache, TLC), the ground truth;
- `bash scripts/inquint-check.sh` — verifies the invariant/deadlock subset with
  inquint, asserting exact counterexample depths.

`cargo test --workspace` runs the same corpus assertions against committed
fixtures (`fixtures/`, regenerated by `fixtures/regen.sh`, pinned to quint
0.32.x) plus unit tests.

The precise no-omission claim, its finite proof bounds, independent reference
evaluator, and Kani commands are documented in
[`docs/verification.md`](docs/verification.md).

For performance work there is additionally `bash scripts/inquint-bench.sh`: one
heavyweight instance per algorithm (each spec's `module bench`,
1.4M–54M states), sized so every bench takes ~20–35s with inquint on
all cores. `bench/run.py` runs the same corpus against quint's TLC and
Apalache backends (TLC: complete explicit search, cross-validating state
counts; Apalache: bounded symbolic checking to the same depth) and
records everything to `bench/results.parquet`; `bench/plot.py` renders
the comparison charts.

## Harnesses

- `bash scripts/inquint-check.sh` — the correctness gate: 53 checks over the corpus
  (invariants with exact counterexample depths, temporal verdicts, run
  tests, detection-lab lassos).
- `bash scripts/temporal-diff.sh` — differential test: inquint's temporal verdicts
  vs `quint verify --backend=tlc` on the same properties (20 cases, all
  agreeing; the two TLA forms TLC cannot parse are covered natively).
- `bash scripts/inquint-bench.sh` — the ~1-minute-per-spec performance suite.

## Architecture (v2)

Parallel BFS over hash-consed values on a bytecode VM:

- **Interned symbols and values** (`quint-ast::Symbol`, `inquint::value`):
  every runtime value is a `Copy` 32-bit id in a leaked, append-only
  store. Equality is one `u32` compare, state hashing is a flat hash over
  ids, and structural sharing is maximal by construction. Set/Map contents
  are flat sorted id slices (the structural order `value_cmp`), so the
  canonical form — and every user-visible enumeration order — matches the
  earlier BTree representation exactly. The store is process-global and
  concurrent: reads are lock-free (append-only segment slab), interning
  goes through a per-thread cache and only takes a sharded mutex on a
  miss — ids are shared across worker threads with near-zero contention.
- **TLC-style fingerprint exploration** (default): the seen-set holds
  64-bit fingerprints only and full states exist only on the (flat) BFS
  frontier — ~14 bytes per explored state (fingerprint table + one parent
  index). Probabilistically sound like TLC: a fingerprint collision would
  silently prune a state (~n²/2⁶⁵). Counterexample traces are rebuilt by a
  deterministic re-run that keeps just the states on the parent path.
  `--exact-states` switches to exact deduplication over a flat state arena
  (`state::StateArena` + `state::SeenSet`) — no per-state allocation, no
  duplicate keys. Temporal checking always keeps the full graph.
- **Bytecode VM** (`vm::{lower, Vm}`): the IR is lowered once into 8-byte
  register-machine instructions. Short-circuit operators are conditional
  jumps, builtins are resolved call sites taking their arguments from a
  register window, `val`/`pureval`/let caching are indexed cache cells
  (per-state caches keyed on a storage generation counter), and errors are
  annotated from a pc→node side table — zero cost on the happy path.
  Nondeterminism still goes through the `ChoiceCtl` replay oracle; lowering
  preserves the evaluation order that makes replay deterministic.
  Guard-critical operations are fused into superinstructions: two-argument
  comparisons in conjunctions compile to single guard-branch opcodes,
  `eq`/`neq`/`not` and record-field access get dedicated opcodes (equality
  is an id compare thanks to hash-consing), and static call sites carry
  pre-resolved parameter slots. Lowering also inlines: single-use `let`
  bindings evaluate in place (no cell), and small choice-free operator
  bodies are expanded at their call sites with parameters resolved to the
  argument registers — no call frame, no parameter-bank traffic.

- **Parallel BFS** (default: all cores, `--threads N`): level-synchronized
  exploration with chunk-granularity work stealing — each level's frontier
  is a flat state array split into fixed-size chunks that workers claim
  with one `fetch_add`. Everything a worker touches per state (Vm,
  parameter/variable banks, choice oracle, output buffer) is
  thread-private; the shared surface is the immutable program, the
  lock-free value store and a fingerprint set sharded 256 ways (one short
  mutex acquisition per fresh state). No `Arc` anywhere on the hot path —
  workers borrow through `std::thread::scope`. The state set is
  schedule-independent, so state counts and depths are deterministic; on
  a violation/deadlock/error the (deterministic) single-threaded pass is
  re-run to produce the trace, so reports and counterexample depths are
  identical to `--threads 1`. Temporal checking, run tests and
  `--exact-states` remain single-threaded.
