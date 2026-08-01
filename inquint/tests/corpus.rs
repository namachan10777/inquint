//! Correctness gate: check the specs corpus fixtures and assert exact
//! outcomes, including counterexample depths and successor counts.

use inquint::explorer::{
    check, Assurance, CheckConfig, CheckOutcome, SearchScope,
};
use inquint::spec::{CompiledSpec, EntryPoints};
use inquint::successor::enumerate;
use std::path::PathBuf;

fn load(name: &str) -> quint_ast::CompiledOutput {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures").join(name);
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e} (run fixtures/regen.sh)", path.display()));
    quint_ast::CompiledOutput::load(&json).unwrap()
}

fn build(name: &str, invariants: &[&str]) -> CompiledSpec {
    let out = load(name);
    let entry = EntryPoints {
        invariants: invariants.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    };
    CompiledSpec::build(&out, &entry).unwrap()
}

fn cfg(max_steps: Option<u32>) -> CheckConfig {
    CheckConfig {
        max_steps,
        deadlock: true,
        max_states: None,
        exact_states: false,
        ..CheckConfig::default()
    }
}

fn cfg_exact(max_steps: Option<u32>) -> CheckConfig {
    CheckConfig {
        exact_states: true,
        ..cfg(max_steps)
    }
}

/// The exact (full-state) mode must agree with the default fingerprint mode.
#[test]
fn exact_mode_agrees() {
    let spec = build("TeachingConcurrency.json", &["correctness"]);
    match check(&spec, &cfg_exact(None)).map_err(|e| e.error).unwrap() {
        CheckOutcome::Pass {
            states,
            scope: SearchScope::Exhaustive,
            assurance: Assurance::Exact,
            ..
        } => assert_eq!(states, 439),
        _ => panic!("expected pass"),
    }
    let broken = build("TwoPhaseCommit.json", &["brokenNoAbort"]);
    match check(&broken, &cfg_exact(Some(2))).map_err(|e| e.error).unwrap() {
        CheckOutcome::InvariantViolation { invariant, trace } => {
            assert_eq!(invariant.as_str(), "brokenNoAbort");
            assert_eq!(trace.len() - 1, 1);
        }
        _ => panic!("expected violation"),
    }
}

#[track_caller]
fn expect_pass(spec: &CompiledSpec, max_steps: Option<u32>) -> u64 {
    match check(spec, &cfg(max_steps)).map_err(|e| e.error).unwrap() {
        CheckOutcome::Pass {
            states,
            scope,
            assurance,
            ..
        } => {
            assert_eq!(
                scope,
                max_steps.map_or(SearchScope::Exhaustive, |max_steps| {
                    SearchScope::Bounded { max_steps }
                })
            );
            assert_eq!(assurance, Assurance::ProbabilisticFingerprint);
            states
        }
        CheckOutcome::InvariantViolation { invariant, trace } => {
            panic!("unexpected violation of {invariant} at depth {}", trace.len() - 1)
        }
        CheckOutcome::Deadlock { trace } => {
            panic!("unexpected deadlock at depth {}", trace.len() - 1)
        }
        CheckOutcome::Incomplete { .. } => panic!("unexpected incomplete"),
    }
}

#[track_caller]
fn expect_violation(spec: &CompiledSpec, max_steps: Option<u32>, name: &str, depth: usize) {
    match check(spec, &cfg(max_steps)).map_err(|e| e.error).unwrap() {
        CheckOutcome::InvariantViolation { invariant, trace } => {
            assert_eq!(invariant.as_ref(), name);
            assert_eq!(trace.len() - 1, depth, "counterexample depth");
        }
        _ => panic!("expected a violation of {name} at depth {depth}"),
    }
}

// -------------------------------------------------------------------------
// Successor-count assertions (M2 gate)
// -------------------------------------------------------------------------

/// TeachingConcurrency N=3: init nondet over setOfMaps(0..2) = 3^3 = 27
/// initial states; each fresh state has 3 enabled processes; the all-Done
/// state has exactly the stutter self-loop.
#[test]
fn teaching_concurrency_successor_counts() {
    let spec = build("TeachingConcurrency.json", &["correctness"]);
    let mut vm = spec.make_vm();
    let initial = enumerate(&mut vm, spec.init, None).unwrap();
    assert_eq!(initial.len(), 27);

    let some_init = initial.first().unwrap();
    let successors = enumerate(&mut vm, spec.step, Some(some_init)).unwrap();
    // 3 processes can each take a step; results are distinct states
    assert_eq!(successors.len(), 3);
}

#[test]
fn state_limit_during_initialization_never_returns_pass() {
    let spec = build("TeachingConcurrency.json", &["correctness"]);
    for (exact_states, threads) in [(true, 1), (false, 1), (false, 4)] {
        let config = CheckConfig {
            max_steps: Some(0),
            deadlock: false,
            max_states: Some(1),
            exact_states,
            threads,
        };
        match check(&spec, &config).map_err(|e| e.error).unwrap() {
            CheckOutcome::Incomplete { states } => assert!(states >= 1),
            _ => panic!("state limit must not produce a successful verdict"),
        }
    }
}

/// Determinism: two enumerations of the same state are identical.
#[test]
fn enumeration_is_deterministic() {
    let spec = build("TwoPhaseCommit.json", &["consistency"]);
    let mut vm = spec.make_vm();
    let initial = enumerate(&mut vm, spec.init, None).unwrap();
    let s = initial.first().unwrap();
    let a = enumerate(&mut vm, spec.step, Some(s)).unwrap();
    let b = enumerate(&mut vm, spec.step, Some(s)).unwrap();
    assert_eq!(a, b);
}

// -------------------------------------------------------------------------
// Corpus outcomes (M3 gate). Depths must match specs/README.md
// (LamportMutex corrected to 4: BFS proves the minimal counterexample).
// -------------------------------------------------------------------------

#[test]
fn teaching_concurrency() {
    let spec = build("TeachingConcurrency.json", &["correctness"]);
    assert_eq!(expect_pass(&spec, None), 439);
    let broken = build("TeachingConcurrency.json", &["brokenAllYOne"]);
    expect_violation(&broken, None, "brokenAllYOne", 6);
}

#[test]
fn clock_sync() {
    let spec = build("ClockSync.json", &["skewOK"]);
    expect_pass(&spec, None);
    let broken = build("ClockSync.json", &["brokenSkewTight"]);
    expect_violation(&broken, Some(2), "brokenSkewTight", 1);
}

#[test]
fn two_phase_commit() {
    let spec = build("TwoPhaseCommit.json", &["consistency"]);
    expect_pass(&spec, None);
    let broken = build("TwoPhaseCommit.json", &["brokenNoAbort"]);
    expect_violation(&broken, Some(2), "brokenNoAbort", 1);
}

#[test]
fn readers_writers() {
    let spec = build("ReadersWriters.json", &["safety"]);
    expect_pass(&spec, None);
    let broken = build("ReadersWriters.json", &["brokenOneReader"]);
    expect_violation(&broken, Some(5), "brokenOneReader", 4);
}

#[test]
fn two_layered_cache() {
    let spec = build("TwoLayeredCache.json", &["cleanConsistency", "dirtyInL1"]);
    expect_pass(&spec, None);
    let broken = build("TwoLayeredCache.json", &["brokenL1Backed"]);
    expect_violation(&broken, Some(2), "brokenL1Backed", 1);
}

#[test]
fn dining_philosophers_fixed() {
    let spec = build("DiningPhilosophers_fixed.json", &["consistent"]);
    expect_pass(&spec, None);
    let broken = build("DiningPhilosophers_fixed.json", &["brokenNeverEating"]);
    expect_violation(&broken, Some(4), "brokenNeverEating", 3);
}

#[test]
fn dining_philosophers_naive_deadlocks() {
    let out = load("DiningPhilosophers_naive.json");
    let spec = CompiledSpec::build(&out, &EntryPoints::default()).unwrap();
    match check(&spec, &cfg(None)).map_err(|e| e.error).unwrap() {
        CheckOutcome::Deadlock { trace } => assert_eq!(trace.len() - 1, 6),
        _ => panic!("expected a deadlock at depth 6"),
    }
}

/// A deadlock is a property of a state at the bound, so that state must be
/// expanded far enough to establish that it has no successors. This guards
/// against reporting a false bounded success at exactly the deadlock depth.
#[test]
fn deadlock_at_depth_bound_is_not_missed() {
    let out = load("DiningPhilosophers_naive.json");
    let spec = CompiledSpec::build(&out, &EntryPoints::default()).unwrap();
    let bounded = |max_steps| CheckConfig {
        max_steps: Some(max_steps),
        deadlock: true,
        max_states: None,
        exact_states: true,
        threads: 1,
    };

    match check(&spec, &bounded(5)).map_err(|e| e.error).unwrap() {
        CheckOutcome::Pass {
            scope: inquint::explorer::SearchScope::Bounded { max_steps: 5 },
            assurance: inquint::explorer::Assurance::Exact,
            ..
        } => {}
        _ => panic!("expected no deadlock through depth 5"),
    }
    match check(&spec, &bounded(6)).map_err(|e| e.error).unwrap() {
        CheckOutcome::Deadlock { trace } => assert_eq!(trace.len() - 1, 6),
        _ => panic!("expected deadlock at the depth bound"),
    }
}

#[test]
fn reliable_broadcast() {
    let spec = build(
        "ReliableBroadcast.json",
        &["validity", "relayedBeforeDelivered"],
    );
    expect_pass(&spec, None);
    let broken = build("ReliableBroadcast.json", &["brokenNobodyDelivers"]);
    expect_violation(&broken, Some(3), "brokenNobodyDelivers", 2);
}

#[test]
fn lamport_mutex() {
    let spec = build("LamportMutex.json", &["mutex", "requestConsistency"]);
    expect_pass(&spec, Some(8));
    let broken = build("LamportMutex.json", &["brokenNooneCritical"]);
    // BFS-minimal counterexample (specs/README originally estimated 6)
    expect_violation(&broken, Some(6), "brokenNooneCritical", 4);
}

#[test]
fn paxos() {
    let spec = build("Paxos.json", &["agreement", "oneValuePerBallot"]);
    expect_pass(&spec, Some(8));
    let broken = build("Paxos.json", &["brokenNothingChosen"]);
    expect_violation(&broken, Some(6), "brokenNothingChosen", 6);
}

#[test]
fn raft() {
    let spec = build(
        "Raft.json",
        &["electionSafety", "logMatching", "voteIntegrity"],
    );
    expect_pass(&spec, Some(8));
    let b1 = build("Raft.json", &["brokenNoLeader"]);
    expect_violation(&b1, Some(4), "brokenNoLeader", 3);
    let b2 = build("Raft.json", &["brokenAtMostOneCandidate"]);
    expect_violation(&b2, Some(3), "brokenAtMostOneCandidate", 2);
}

#[test]
fn itf_roundtrip() {
    let broken = build("TwoPhaseCommit.json", &["brokenNoAbort"]);
    let outcome = check(&broken, &cfg(Some(2))).map_err(|e| e.error).unwrap();
    let CheckOutcome::InvariantViolation { trace, .. } = outcome else {
        panic!("expected violation");
    };
    let itf = inquint::itf_out::trace_to_itf(&broken.vars.names, &trace, true, "test");
    let json = serde_json::to_string(&itf).unwrap();
    let parsed: itf::Trace<itf::Value> = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.states.len(), trace.len());
}

/// Parallel exploration must agree with single-threaded on the state count
/// and verdict (the state set is schedule-independent by construction).
#[test]
fn parallel_agrees_with_sequential() {
    for (fixture, inv) in [("TwoPhaseCommit.json", "consistency"), ("Paxos.json", "agreement")] {
        let spec = build(fixture, &[inv]);
        let seq = CheckConfig {
            threads: 1,
            deadlock: false,
            ..cfg(Some(6))
        };
        let par = CheckConfig {
            threads: 4,
            deadlock: false,
            ..cfg(Some(6))
        };
        let unpack = |r: Result<inquint::explorer::CheckOutcome, Box<inquint::explorer::CheckError>>, mode: &str| match r {
            Ok(inquint::explorer::CheckOutcome::Pass { states, max_depth, .. }) => (states, max_depth),
            Ok(inquint::explorer::CheckOutcome::InvariantViolation { .. }) => {
                panic!("{fixture} ({mode}): unexpected violation")
            }
            Ok(inquint::explorer::CheckOutcome::Deadlock { .. }) => {
                panic!("{fixture} ({mode}): unexpected deadlock")
            }
            Ok(inquint::explorer::CheckOutcome::Incomplete { .. }) => {
                panic!("{fixture} ({mode}): unexpected incomplete")
            }
            Err(e) => panic!("{fixture} ({mode}): error {}", e.error),
        };
        let (sa, da) = unpack(inquint::explorer::check(&spec, &seq), "seq");
        let (sb, db) = unpack(inquint::explorer::check(&spec, &par), "par");
        assert_eq!(sa, sb, "{fixture}: state count differs");
        assert_eq!(da, db, "{fixture}: depth differs");
    }
}
