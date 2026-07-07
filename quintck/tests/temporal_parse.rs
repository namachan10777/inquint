//! M1 gate: every temporal property in the corpus translates to the
//! expected fairness + LTL structure.

use quintck::spec::{CompiledSpec, EntryPoints};
use std::path::PathBuf;

fn build(fixture: &str, temporal: &[&str]) -> CompiledSpec {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join(fixture);
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e} (run fixtures/regen.sh)", path.display()));
    let out = quint_ast::CompiledOutput::load(&json).unwrap();
    let entry = EntryPoints {
        temporal: temporal.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    };
    CompiledSpec::build(&out, &entry).unwrap_or_else(|e| panic!("{fixture}: {e}"))
}

fn render(spec: &CompiledSpec) -> Vec<String> {
    spec.temporal
        .iter()
        .map(|p| {
            let fairness = p
                .fairness
                .iter()
                .map(|f| f.label.clone())
                .collect::<Vec<_>>()
                .join(" & ");
            if fairness.is_empty() {
                format!("{}: {}", p.name, p.body)
            } else {
                format!("{}: {} ⊢ {}", p.name, fairness, p.body)
            }
        })
        .collect()
}

#[test]
fn two_phase_commit_translations() {
    let spec = build(
        "TwoPhaseCommit_temporal.json",
        &["decisionReached", "committedPropagates", "brokenDecisionNoFairness"],
    );
    let rendered = render(&spec);
    // decisionReached: WF(tmCommit) ⊢ G(!P | F Q)
    assert_eq!(spec.temporal[0].fairness.len(), 1);
    assert!(!spec.temporal[0].fairness[0].strong);
    assert!(
        rendered[0].contains("G((!(s") && rendered[0].contains("| F(s"),
        "unexpected: {}",
        rendered[0]
    );
    // committedPropagates: 3 WF (forall-expanded over rm1..rm3)
    assert_eq!(spec.temporal[1].fairness.len(), 3);
    // broken: no fairness
    assert!(spec.temporal[2].fairness.is_empty());
    for line in &rendered {
        println!("{line}");
    }
}

#[test]
fn dining_translations() {
    let spec = build(
        "DiningPhilosophers_temporal.json",
        &["noDeadlock", "someoneEats", "brokenSomeoneEatsNoFairness"],
    );
    let rendered = render(&spec);
    // noDeadlock = G(enabled(step)): one state atom, always
    assert!(rendered[0].ends_with(": G(s0)"), "unexpected: {}", rendered[0]);
    // someoneEats: 3 WF(getHungry) + 3 SF(takeFirst) + 3 SF(takeSecond) + 3 WF(putDown)
    let f = &spec.temporal[1].fairness;
    assert_eq!(f.len(), 12);
    assert_eq!(f.iter().filter(|x| x.strong).count(), 6);
    // body: G F (exists ... Eating) — an Or of per-philosopher atoms or a
    // single collapsed atom (exists over states is pure → single atom)
    assert!(rendered[1].contains("G(F("), "unexpected: {}", rendered[1]);
    assert!(spec.temporal[2].fairness.is_empty());
    for line in &rendered {
        println!("{line}");
    }
}

#[test]
fn cache_translations() {
    let spec = build(
        "TwoLayeredCache_temporal.json",
        &[
            "verMonotone",
            "verNeverDecreases",
            "eventuallyClean",
            "brokenAlwaysProgress",
            "brokenEventuallyCleanNoFairness",
        ],
    );
    let rendered = render(&spec);
    // verMonotone = G([verAdvance]_allVars): single Kept edge atom
    assert!(rendered[0].ends_with(": G(e0)"), "unexpected: {}", rendered[0]);
    // verNeverDecreases = G(next(ver) >= ver): single NextPred edge atom
    assert!(rendered[1].ends_with(": G(e1)"), "unexpected: {}", rendered[1]);
    // eventuallyClean: WF(flushOne) ⊢ G(!P | F Q)
    assert_eq!(spec.temporal[2].fairness.len(), 1);
    // brokenAlwaysProgress = G F ⟨step⟩_allVars
    assert!(
        rendered[3].contains("G(F(e"),
        "unexpected: {}",
        rendered[3]
    );
    for line in &rendered {
        println!("{line}");
    }
}

#[test]
fn raft_election_translations() {
    let spec = build(
        "Raft_election.json",
        &[
            "termsMonotone",
            "quorumCandidateProgress",
            "brokenEventuallyLeaderNoFairness",
        ],
    );
    let rendered = render(&spec);
    assert!(rendered[0].ends_with(": G(e0)"), "unexpected: {}", rendered[0]);
    // 3 WF(becomeLeader(s)) ⊢ And of 3 leadsTo
    assert_eq!(spec.temporal[1].fairness.len(), 3);
    // broken: F(exists leader) — plain eventually
    assert!(rendered[2].contains(": F(s"), "unexpected: {}", rendered[2]);
    for line in &rendered {
        println!("{line}");
    }
}

#[test]
fn readers_writers_translations() {
    let spec = build(
        "ReadersWriters_temporal.json",
        &["noStarvation", "brokenNoStarvationNoFairness"],
    );
    // WF(serveHead) + 3 WF(release(p)) = 4
    assert_eq!(spec.temporal[0].fairness.len(), 4);
    assert!(spec.temporal[1].fairness.is_empty());
    for line in render(&spec) {
        println!("{line}");
    }
}

#[test]
fn reliable_broadcast_translations() {
    let spec = build(
        "ReliableBroadcast_temporal.json",
        &["totality", "brokenTotalityNoFairness"],
    );
    // Correct = {2, 3} (faulty sender 1 excluded): 2 WF(relay(p))
    assert_eq!(spec.temporal[0].fairness.len(), 2);
    for line in render(&spec) {
        println!("{line}");
    }
}

#[test]
fn naive_dining_no_deadlock() {
    let spec = build("DiningPhilosophers_naive_temporal.json", &["naive::noDeadlock"]);
    assert!(render(&spec)[0].ends_with(": G(s0)"));
}
