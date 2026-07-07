# specs — verified Quint specifications of classic distributed algorithms

Ten classic distributed algorithms, written from scratch in
[Quint](https://quint-lang.org/) (v0.32), designed as a test corpus for
model checkers. Every property in every file is actually verified:

- **invariants** and **inductive invariants** with `quint verify` (Apalache,
  bounded model checking, default `--max-steps=10` unless noted);
- **temporal properties** with `quint verify --backend=tlc` (TLC explores
  the full — deliberately finite — state space);
- a few action-level temporal properties with Apalache's bounded temporal
  mode (see *toolchain notes* below).

Each algorithm additionally defines deliberately **broken properties**
(prefix `broken*`) with a counterexample at a known, small depth. These are
detection tests: a checker that fails to find those counterexamples is
unsound. `check.sh` runs every check and asserts the expected outcome:

```sh
bash specs/check.sh              # everything (~40 checks)
bash specs/check.sh --only Raft  # a single spec
```

Requirements: `quint` (0.32.x) and Java 17+. The first Apalache-backend run
downloads the Apalache distribution; the TLC backend reuses the same jar
(so an Apalache run must happen first — `check.sh` warms it up). Do not run
two Apalache-backend verifications concurrently: they race to start the
shared Apalache server on port 8822 and can crash.

## The specs

| Spec | Instance size | True properties | Broken properties (counterexample depth) |
|---|---|---|---|
| `TeachingConcurrency` | 3 procs | `correctness` (+ inductive `IndInv`) | `brokenAllYOne` (6), `brokenIndInv` (induction step) |
| `ClockSync` | 3 procs, skew 2 | `skewOK` (+ inductive `IndInv`) | `brokenSkewTight` (1), `TypeOK` as ind. inv. (implication) |
| `TwoPhaseCommit` | 3 RMs | `consistency`; TLC: `decisionReached`, `committedPropagates` | `brokenNoAbort` (1), `brokenDecisionNoFairness` (lasso) |
| `ReadersWriters` | 3 procs | `safety`; TLC: `noStarvation` | `brokenOneReader` (4), `brokenNoStarvationNoFairness` (lasso) |
| `TwoLayeredCache` | 2 keys, 3 writes | `cleanConsistency`, `dirtyInL1`; TLC: `verMonotone`, `eventuallyClean`; Apalache temporal: `verNeverDecreases` | `brokenL1Backed` (1), `brokenAlwaysProgress` (lasso), `brokenEventuallyCleanNoFairness` (lasso) |
| `DiningPhilosophers` | 3 phils × {fixed, naive} | `consistent`; TLC (fixed): `noDeadlock`, `someoneEats` | `brokenNeverEating` (3), naive `noDeadlock` (deadlock at 6), `brokenSomeoneEatsNoFairness` (lasso) |
| `ReliableBroadcast` | 3 procs, faulty sender | `validity`, `relayedBeforeDelivered` (+ inductive `IndInv`); TLC: `totality` | `brokenNobodyDelivers` (2), `brokenTotalityNoFairness` (lasso) |
| `LamportMutex` | 2 procs, clock ≤ 3 | `mutex`, `requestConsistency` | `brokenNooneCritical` (4) |
| `Paxos` | 3 acceptors, 2 values, 3 ballots | `agreement`, `oneValuePerBallot` | `brokenNothingChosen` (6) |
| `Raft` | 3 servers (`raft_3` full, `raft_election` election-only) | `electionSafety`, `logMatching`, `voteIntegrity`; TLC: `termsMonotone`, `quorumCandidateProgress` | `brokenNoLeader` (3), `brokenAtMostOneCandidate` (2), `brokenEventuallyLeaderNoFairness` (lasso) |

Notes on scope:

- **Paxos** replaces the classic unbounded `Ballot = Nat` (which makes the
  upstream quint example unverifiable,
  [quint#1284](https://github.com/informalsystems/quint/issues/1284)) with
  `Ballots = 0.to(MaxBallot)`. Safety is bounded-checked (`--max-steps=8`);
  a Voting-style inductive proof is out of scope.
- **Raft** is an abstract shared-state model (no message network): votes
  and log replication read peer state directly, preserving the safety
  arguments while keeping the state space small. There is no Raft spec in
  the upstream quint examples. TLC properties run on the election-only
  instance (`MaxLogLen = 0`).

## TemporalLab

`TemporalLab.qnt` holds three tiny machines with *known* liveness
counterexample shapes, used as deterministic detection tests for temporal
checkers (quintck-only, not part of `check.sh`): a pure stuttering lasso, a
real-cycle lasso (mod-3 counter), and a weak-vs-strong-fairness
discriminator (an action enabled only intermittently along the fair cycle).

## Bench instances

Each spec additionally defines a `module bench` — a much larger instance
(e.g. Raft with 5 terms, dining with 12 philosophers, 2PC with 8 RMs) used
as a performance benchmark by the quintck explicit-state checker
(`quintck-bench.sh` at the repo root; each bench is sized to take roughly
a minute unoptimized). These instances are **not** part of `check.sh`:
they are far too large for Apalache/TLC.

## Language feature coverage

| Quint feature | Showcased in |
|---|---|
| Sum types + `match` | TwoPhaseCommit (`Msg`), Paxos (`Msg`), everywhere |
| Polymorphic types (`Option[a]`) | Raft (`votedFor`), DiningPhilosophers (fork owner) |
| `nondet` + `oneOf` | all; over `powerset` in ReliableBroadcast |
| `pure def` / `pure val` vs `def` / `val` | ClockSync (fold min/max), Paxos (message filters) |
| `all` / `any` actions | all |
| Parameterized modules + instances (`import M(N = 3).*`) | all |
| Qualified instances (`import M(..) as naive` + `naive::` CLI flags) | DiningPhilosophers |
| `assume` | Paxos (quorum intersection), ClockSync, Raft |
| Type aliases | TwoLayeredCache (`CacheLayer`), Raft (`LogEntry`) |
| Higher-order operators (`fold`, `filter`, `map`, `select`) | Paxos (fold over message soup), ReadersWriters (List `select`) |
| Maps with tuple keys, List FIFO channels | LamportMutex |
| Record spread `{ ...r, f: v }`, tuple destructuring `((a, b)) =>` | TeachingConcurrency, Paxos, ClockSync |
| `run` tests (`.then`, `.expect`, `.fail`, `.reps`) | TwoPhaseCommit, LamportMutex, TeachingConcurrency |
| Invariants | all |
| Inductive invariants (`--inductive-invariant`) | TeachingConcurrency, ClockSync, ReliableBroadcast |
| `always`, `eventually` | all temporal specs |
| `orKeep` (`[A]_v`) | TwoLayeredCache (`verMonotone`), Raft (`termsMonotone`) |
| `mustChange` (`<A>_v`) | TwoLayeredCache (`brokenAlwaysProgress`) |
| `next` | TwoLayeredCache (`verNeverDecreases`) |
| `enabled` | DiningPhilosophers (`noDeadlock`) |
| `weakFair` | TwoPhaseCommit, ReadersWriters, TwoLayeredCache, ReliableBroadcast, Raft |
| `strongFair` | DiningPhilosophers (`someoneEats`) |
| `leadsTo` (`~>`) | TwoPhaseCommit, ReadersWriters, TwoLayeredCache, ReliableBroadcast, Raft |

## Toolchain notes (quint 0.32 / Apalache 0.56.1 / TLC 2.19)

Pitfalls discovered while making everything verifiable — useful both for
writing more specs and as behaviors a new checker should get right:

- **Temporal properties go to TLC** (`--backend=tlc`). Apalache's temporal
  support is partial and `--temporal` triggers an interactive confirmation
  prompt (pipe `echo y |` to script it). TLC ignores `--max-steps` and
  explores the full state space, so every spec with TLC-checked properties
  bounds its state space by construction (clock/term/version ceilings,
  explicit resting-state stutter actions instead of deadlocks).
- **TLC rejects `[]` over bare actions**: `always(next(x) >= x)` compiles
  to `[](x' >= x)`, which is not of the form `[][A]_v`. Wrap the action in
  `orKeep` (as in `verMonotone`/`termsMonotone`) — or check the `next`-based
  form with Apalache's bounded temporal mode instead.
- **`mustChange` emits malformed TLA for TLC** in quint 0.32 (SANY parse
  error). Properties using it are checked with Apalache temporal mode.
- **`size()` applied to state variables inside a temporal formula** also
  emits broken TLA for TLC ("identifier ... is either undefined or not an
  operator"). `size()` in action guards is fine. Raft's
  `quorumCandidateProgress` spells out "two distinct voters" instead.
- **`nondet x = (if (c) S1 else S2).oneOf()` is silently unsatisfiable
  under Apalache** — the action never fires and invariant checks pass
  vacuously. Hoist the conditional out of the domain (see
  `ReliableBroadcast::bcast`): pick from the larger set and constrain with
  a separate conjunct.
- `run` tests must live in the concrete instance module (e.g. `main`), not
  the const-parameterized module. `.then(assert(p))` does not typecheck
  (no state update); use `.expect(p)`.
- Inductive invariants need every state variable constrained by a
  `TypeOK`-style domain conjunct, and are Apalache-only.
