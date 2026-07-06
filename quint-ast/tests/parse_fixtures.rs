//! Every committed `quint compile` fixture must parse, contain exactly one
//! flattened module, and expose the expected entry points.

use quint_ast::{CompiledOutput, Declaration, OpQualifier};
use std::path::PathBuf;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures")
}

fn load(name: &str) -> CompiledOutput {
    let path = fixtures_dir().join(name);
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e} (run fixtures/regen.sh)", path.display()));
    CompiledOutput::load(&json).unwrap_or_else(|e| panic!("cannot load {name}: {e}"))
}

fn def_names(out: &CompiledOutput) -> Vec<&str> {
    out.module()
        .declarations
        .iter()
        .filter_map(|d| match d {
            Declaration::OpDef(op) => Some(op.name.as_ref()),
            _ => None,
        })
        .collect()
}

fn var_count(out: &CompiledOutput) -> usize {
    out.module()
        .declarations
        .iter()
        .filter(|d| matches!(d, Declaration::Var { .. }))
        .count()
}

#[test]
fn parses_all_fixtures() {
    for entry in std::fs::read_dir(fixtures_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "json") {
            let name = path.file_name().unwrap().to_str().unwrap().to_owned();
            load(&name);
        }
    }
}

#[test]
fn specs_have_synthetic_entry_points() {
    // DiningPhilosophers_naive uses --init/--step overrides, so q::init
    // still gets injected pointing at the default-named action if present;
    // check the common ones that always pass --invariant.
    for name in [
        "TeachingConcurrency.json",
        "ClockSync.json",
        "TwoPhaseCommit.json",
        "ReadersWriters.json",
        "TwoLayeredCache.json",
        "ReliableBroadcast.json",
        "LamportMutex.json",
        "Paxos.json",
        "Raft.json",
        "DiningPhilosophers_fixed.json",
    ] {
        let out = load(name);
        let names = def_names(&out);
        for entry in ["q::init", "q::step", "q::inv"] {
            assert!(
                names.contains(&entry),
                "{name}: missing {entry}; defs = {names:?}"
            );
        }
    }
}

#[test]
fn teaching_concurrency_shape() {
    let out = load("TeachingConcurrency.json");
    assert_eq!(var_count(&out), 1, "single `procs` variable");
    // Note: defs not reachable from the entry points (e.g. IndInv, TypeOK)
    // are pruned by the flattener — only what --invariant referenced remains.
    let names = def_names(&out);
    for expected in ["correctness", "brokenAllYOne", "step", "init"] {
        assert!(
            names.iter().any(|n| n.ends_with(expected)),
            "missing {expected}; defs = {names:?}"
        );
    }
}

#[test]
fn raft_shape() {
    let out = load("Raft.json");
    assert_eq!(var_count(&out), 5, "currentTerm/role/votedFor/votesGranted/log");
}

#[test]
fn dining_naive_keeps_qualified_names() {
    let out = load("DiningPhilosophers_naive.json");
    let names = def_names(&out);
    assert!(
        names.iter().any(|n| n.contains("init")),
        "expected a naive::init-ish def; defs = {names:?}"
    );
}

#[test]
fn temporal_defs_are_identifiable() {
    let out = load("ewd840.json");
    let has_temporal = out.module().declarations.iter().any(|d| {
        matches!(d, Declaration::OpDef(op) if op.qualifier == OpQualifier::Temporal)
    });
    assert!(has_temporal, "ewd840 defines temporal properties");
}
