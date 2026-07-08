//! Temporal correctness gate: verdicts for every temporal property in the
//! corpus and the upstream quint examples. Lasso shapes are non-canonical,
//! so only verdicts and lasso sanity are asserted; the debug-build
//! self-check inside check_liveness validates every extracted lasso
//! against the negated property.

use inquint::explorer::CheckConfig;
use inquint::spec::{CompiledSpec, EntryPoints};
use inquint::temporal::{check_temporal, TemporalOutcome};
use std::path::PathBuf;

fn build(fixture: &str, temporal: &[&str], init: Option<&str>, step: Option<&str>) -> CompiledSpec {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join(fixture);
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e} (run fixtures/regen.sh)", path.display()));
    let out = quint_ast::CompiledOutput::load(&json).unwrap();
    let entry = EntryPoints {
        init: init.map(str::to_string),
        step: step.map(str::to_string),
        temporal: temporal.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    };
    CompiledSpec::build(&out, &entry).unwrap_or_else(|e| panic!("{fixture}: {e}"))
}

fn cfg(deadlock: bool) -> CheckConfig {
    CheckConfig {
        max_steps: None,
        deadlock,
        max_states: None,
        exact_states: false,
        ..CheckConfig::default()
    }
}

#[track_caller]
fn expect_pass(fixture: &str, temporal: &[&str], deadlock: bool) {
    let spec = build(fixture, temporal, None, None);
    match check_temporal(&spec, &cfg(deadlock)).map_err(|e| e.error).unwrap() {
        TemporalOutcome::Pass { .. } => {}
        TemporalOutcome::Violation { property, .. } => {
            panic!("{fixture}: unexpected violation of {property}")
        }
        TemporalOutcome::SafetyViolation { property, .. } => {
            panic!("{fixture}: unexpected safety violation of {property}")
        }
        _ => panic!("{fixture}: unexpected outcome"),
    }
}

#[track_caller]
fn expect_liveness_violation(fixture: &str, temporal: &[&str], deadlock: bool) {
    let spec = build(fixture, temporal, None, None);
    match check_temporal(&spec, &cfg(deadlock)).map_err(|e| e.error).unwrap() {
        TemporalOutcome::Violation { property, lasso } => {
            assert_eq!(property.as_ref(), temporal[0]);
            assert!(lasso.loop_index < lasso.states.len(), "loop_index sanity");
        }
        TemporalOutcome::Pass { .. } => panic!("{fixture}: expected a violation, got pass"),
        TemporalOutcome::SafetyViolation { .. } => {
            panic!("{fixture}: expected a liveness violation, got safety")
        }
        _ => panic!("{fixture}: unexpected outcome"),
    }
}

#[test]
fn two_phase_commit_temporal() {
    expect_pass(
        "TwoPhaseCommit_temporal.json",
        &["decisionReached", "committedPropagates"],
        true,
    );
    expect_liveness_violation(
        "TwoPhaseCommit_temporal.json",
        &["brokenDecisionNoFairness"],
        true,
    );
}

#[test]
fn readers_writers_temporal() {
    expect_pass("ReadersWriters_temporal.json", &["noStarvation"], true);
    expect_liveness_violation(
        "ReadersWriters_temporal.json",
        &["brokenNoStarvationNoFairness"],
        true,
    );
}

#[test]
fn two_layered_cache_temporal() {
    expect_pass(
        "TwoLayeredCache_temporal.json",
        &["verMonotone", "verNeverDecreases", "eventuallyClean"],
        true,
    );
    expect_liveness_violation(
        "TwoLayeredCache_temporal.json",
        &["brokenAlwaysProgress"],
        true,
    );
    expect_liveness_violation(
        "TwoLayeredCache_temporal.json",
        &["brokenEventuallyCleanNoFairness"],
        true,
    );
}

#[test]
fn dining_temporal() {
    expect_pass(
        "DiningPhilosophers_temporal.json",
        &["noDeadlock", "someoneEats"],
        true,
    );
    expect_liveness_violation(
        "DiningPhilosophers_temporal.json",
        &["brokenSomeoneEatsNoFairness"],
        true,
    );
    // naive: always(enabled(step)) fails at the depth-6 deadlock state
    let spec = build(
        "DiningPhilosophers_naive_temporal.json",
        &["naive::noDeadlock"],
        None,
        None,
    );
    match check_temporal(&spec, &cfg(false)).map_err(|e| e.error).unwrap() {
        TemporalOutcome::SafetyViolation { trace, .. } => assert_eq!(trace.len() - 1, 6),
        _ => panic!("expected safety violation at depth 6"),
    }
}

#[test]
fn reliable_broadcast_temporal() {
    expect_pass("ReliableBroadcast_temporal.json", &["totality"], true);
    expect_liveness_violation(
        "ReliableBroadcast_temporal.json",
        &["brokenTotalityNoFairness"],
        true,
    );
}

#[test]
fn raft_election_temporal() {
    expect_pass(
        "Raft_election.json",
        &["termsMonotone", "quorumCandidateProgress"],
        true,
    );
    expect_liveness_violation(
        "Raft_election.json",
        &["brokenEventuallyLeaderNoFairness"],
        true,
    );
}

#[test]
fn quint_examples_fairness() {
    expect_pass(
        "weakFairness.json",
        &["eventuallyDone", "notDoneLeadsToDone"],
        true,
    );
    expect_liveness_violation("weakFairness.json", &["notDoneLeadsToDoneNoFairness"], true);
    // The kettle: WF+SF passes; WF alone is insufficient because the
    // sensor is only intermittently enabled — a real-cycle lasso.
    expect_pass("strongFairness.json", &["eventuallyHundredDegrees"], true);
    expect_liveness_violation(
        "strongFairness.json",
        &["eventuallyHundredDegreesWeakOnly"],
        true,
    );
}

#[test]
fn quint_examples_ewd() {
    expect_pass(
        "ewd426_temporal.json",
        &["convergence", "closure", "persistence"],
        false,
    );
    expect_pass("ewd840_temporal.json", &["liveness"], false);
    expect_liveness_violation("ewd840_temporal.json", &["falseLiveness"], false);
}
