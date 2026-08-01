# Verification claims

inquint's verification work is aimed at verdict correctness, not general
panic-freedom. The failure to prevent is a successful result produced after a
reachable state, transition, or accepting lasso was omitted.

## Claims

For a well-typed, flattened JSON IR using inquint's supported finite fragment,
and provided evaluation terminates without a tool error or resource limit:

- `SearchScope::Exhaustive` with `Assurance::Exact` means every state reachable
  from every initial state was explored. An invariant success therefore means
  the invariant holds throughout that reachable state graph.
- `SearchScope::Bounded { max_steps: k }` with `Assurance::Exact` means no
  invariant violation or deadlock exists at a depth at most `k`. States at
  exactly `k` are evaluated for both invariants and deadlock, but their
  successors are not admitted to the search.
- A safety violation contains a real initial-to-violation path. Because the
  exploration is breadth-first, its depth is minimal.
- Temporal checking uses an exact, stutter-closed state graph. A liveness
  violation contains a real lasso and is checked against the negated property
  before it is returned. Failure of this self-check is a tool error, never a
  violation verdict.

`Incomplete`, evaluation errors, unsupported operators, state limits, and
choice-enumeration limits are not successful verdicts.

## Deliberately excluded modes

Fingerprint exploration stores only a 64-bit hash per state. A collision can
prune a distinct state, so `Assurance::ProbabilisticFingerprint` is explicitly
not part of the deterministic no-omission claim. The CLI renders it as
`[ok:non-assured:fingerprint]`. Use `--exact-states` when the exact claim is
required.

Parallel exploration currently exists only for fingerprint mode and is also
outside the deterministic claim. Its state counts and verdicts are checked
against sequential exploration in the corpus.

## Independent checks

The `inquint-reference` workspace crate is the final closure-based evaluator
from before the bytecode VM rewrite. It deliberately retains BTree values and
does not share bytecode lowering, hash-consing, canonical-container code, or
register-machine execution with production. Differential tests compare:

- complete initial and successor sets;
- invariant values;
- bounded reachable graphs for corpus models;
- a generated family of well-typed counter/action IR programs; and
- safety counterexample edges and the final failed invariant.

Temporal algorithms have separate finite oracles:

- Tarjan is compared with mutual reachability for every directed graph of at
  most four vertices;
- GPVW generalized Büchi acceptance is compared with direct LTL-on-lasso
  evaluation for every total two-state graph and atom valuation; and
- every emitted lasso is re-evaluated at runtime.

Kani proves the allocation-independent core of nondeterministic trail advance:

- every two-digit mixed-radix combination for radices in `1..=2` is visited
  exactly once; and
- a suffix not visited during deterministic replay is removed before the next
  branch.

Run these checks with:

```sh
cargo test --workspace
cargo kani --package inquint --output-format=terse
bash scripts/inquint-check.sh
```

## Bounds and trusted base

Kani's symbolic proof has the explicit radix and trail-length bounds above.
The unit-test oracles exhaust their stated finite graph sizes. Differential
tests are strong regression evidence, not a proof for every possible Quint
program.

The trusted base includes the Quint typechecker/flattener, Rust compiler,
standard library and dependencies, Kani/CBMC, and the operating system. A
universal theorem for arbitrary IR size would additionally require a
deductively verified IR semantics and evaluator; that is not claimed here.
