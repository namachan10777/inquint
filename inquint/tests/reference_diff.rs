//! Differential semantics gate between the production bytecode VM and the
//! independent, closure-based pre-VM evaluator in `inquint-reference`.

#![allow(clippy::mutable_key_type)]

use inquint::spec::{CompiledSpec, EntryPoints};
use quint_ast::{CompiledOutput, Declaration, OpDef};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

fn load(name: &str) -> CompiledOutput {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join(name);
    let json = std::fs::read_to_string(&path).unwrap();
    CompiledOutput::load(&json).unwrap()
}

fn find_def<'a>(out: &'a CompiledOutput, name: &str) -> &'a OpDef {
    let suffix = format!("::{name}");
    out.module()
        .declarations
        .iter()
        .find_map(|decl| match decl {
            Declaration::OpDef(op)
                if op.name.as_ref() == name || op.name.ends_with(&suffix) =>
            {
                Some(op)
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("definition {name} not found"))
}

fn prod_key(state: &inquint::state::State) -> String {
    format!(
        "{:?}",
        state.iter().map(ToString::to_string).collect::<Vec<_>>()
    )
}

fn reference_key(state: &inquint_reference::State) -> String {
    format!(
        "{:?}",
        state.iter().map(ToString::to_string).collect::<Vec<_>>()
    )
}

/// Compare every successor set in the reachable graph through `max_depth`.
/// Each implementation advances using its own values and state storage; only
/// canonical renderings cross the boundary.
fn compare_output(out: CompiledOutput, label: &str, max_depth: u32) {
    let prod = CompiledSpec::build(&out, &EntryPoints::default()).unwrap();
    let mut prod_vm = prod.make_vm();
    let prod_init = inquint::successor::enumerate(&mut prod_vm, prod.init, None).unwrap();

    let reference_vars = inquint_reference::VarTable::from_module(out.module());
    let mut reference_compiler = inquint_reference::Compiler::new(&out.table, &reference_vars);
    let reference_init = reference_compiler.compile(&find_def(&out, "q::init").expr);
    let reference_step = reference_compiler.compile(&find_def(&out, "q::step").expr);
    let reference_invariants: Vec<_> = prod
        .invariants
        .iter()
        .map(|(name, _)| reference_compiler.compile(&find_def(&out, name.as_ref()).expr))
        .collect();
    let reference_storage = reference_compiler.storage.clone();
    let reference_init =
        inquint_reference::enumerate(&reference_init, &reference_storage, None).unwrap();

    let mut prod_frontier: BTreeMap<String, inquint::state::State> = prod_init
        .into_iter()
        .map(|state| (prod_key(&state), state))
        .collect();
    let mut reference_frontier: BTreeMap<String, inquint_reference::State> = reference_init
        .into_iter()
        .map(|state| (reference_key(&state), state))
        .collect();
    assert_eq!(
        prod_frontier.keys().collect::<Vec<_>>(),
        reference_frontier.keys().collect::<Vec<_>>(),
        "{label}: initial states differ"
    );

    let mut prod_seen: BTreeSet<String> = prod_frontier.keys().cloned().collect();
    let mut reference_seen: BTreeSet<String> = reference_frontier.keys().cloned().collect();

    for depth in 0..=max_depth {
        assert_eq!(
            prod_frontier.keys().collect::<Vec<_>>(),
            reference_frontier.keys().collect::<Vec<_>>(),
            "{label}: frontier differs at depth {depth}"
        );

        let mut prod_next = BTreeMap::new();
        let mut reference_next = BTreeMap::new();
        for key in prod_frontier.keys() {
            let prod_state = &prod_frontier[key];
            let reference_state = &reference_frontier[key];

            prod_vm.load(prod_state);
            reference_storage.borrow().load(reference_state);
            for ((name, prod_inv), reference_inv) in
                prod.invariants.iter().zip(&reference_invariants)
            {
                let prod_holds = prod_vm
                    .run(&mut inquint::eval::Env::new(None), *prod_inv)
                    .unwrap()
                    .as_bool();
                let reference_holds = reference_inv
                    .execute(&mut inquint_reference::eval::Env::new(
                        reference_storage.clone(),
                        None,
                    ))
                    .unwrap()
                    .as_bool();
                assert_eq!(
                    prod_holds, reference_holds,
                    "{label}: invariant {name} differs at depth {depth} for {key}"
                );
            }

            if depth == max_depth {
                continue;
            }

            let prod_succ =
                inquint::successor::enumerate(&mut prod_vm, prod.step, Some(prod_state)).unwrap();
            let reference_succ = inquint_reference::enumerate(
                &reference_step,
                &reference_storage,
                Some(reference_state),
            )
            .unwrap();

            let prod_keys: BTreeSet<_> = prod_succ.iter().map(prod_key).collect();
            let reference_keys: BTreeSet<_> = reference_succ.iter().map(reference_key).collect();
            assert_eq!(
                prod_keys, reference_keys,
                "{label}: successors differ at depth {depth} for {key}"
            );

            for state in prod_succ {
                let key = prod_key(&state);
                if prod_seen.insert(key.clone()) {
                    prod_next.insert(key, state);
                }
            }
            for state in reference_succ {
                let key = reference_key(&state);
                if reference_seen.insert(key.clone()) {
                    reference_next.insert(key, state);
                }
            }
        }
        prod_frontier = prod_next;
        reference_frontier = reference_next;
    }

    assert_eq!(prod_seen, reference_seen, "{label}: reachable states differ");
}

fn compare_graph(fixture: &str, max_depth: u32) {
    compare_output(load(fixture), fixture, max_depth);
}

/// Generate a flattened, well-typed IR family without going through either
/// evaluator. The step is an `actionAny` of `x' = x + delta` branches.
fn counter_ir(initial: i64, deltas: &[i64], invariant_limit: i64) -> CompiledOutput {
    use serde_json::{json, Map, Value};

    let var = json!({ "id": 1, "kind": "var", "name": "x" });
    let init = json!({
        "id": 10, "kind": "def", "name": "q::init", "qualifier": "action",
        "expr": {
            "id": 11, "kind": "app", "opcode": "assign", "args": [
                { "id": 12, "kind": "name", "name": "x" },
                { "id": 13, "kind": "int", "value": initial }
            ]
        }
    });

    let mut table = Map::new();
    table.insert("12".into(), var.clone());
    let mut branches = Vec::new();
    let mut id = 100u64;
    for &delta in deltas {
        branches.push(json!({
            "id": id, "kind": "app", "opcode": "assign", "args": [
                { "id": id + 1, "kind": "name", "name": "x" },
                { "id": id + 2, "kind": "app", "opcode": "iadd", "args": [
                    { "id": id + 3, "kind": "name", "name": "x" },
                    { "id": id + 4, "kind": "int", "value": delta }
                ] }
            ]
        }));
        table.insert((id + 1).to_string(), var.clone());
        table.insert((id + 3).to_string(), var.clone());
        id += 10;
    }
    let step = json!({
        "id": 20, "kind": "def", "name": "q::step", "qualifier": "action",
        "expr": { "id": 21, "kind": "app", "opcode": "actionAny", "args": branches }
    });
    let invariant = json!({
        "id": 30, "kind": "def", "name": "q::inv", "qualifier": "val",
        "expr": { "id": 31, "kind": "app", "opcode": "ilte", "args": [
            { "id": 32, "kind": "name", "name": "x" },
            { "id": 33, "kind": "int", "value": invariant_limit }
        ] }
    });
    table.insert("32".into(), var.clone());

    let output = json!({
        "stage": "compiling",
        "warnings": [],
        "modules": [{
            "id": 1,
            "name": "generated",
            "declarations": [var, init, step, invariant]
        }],
        "table": Value::Object(table),
        "errors": [],
        "main": "generated"
    });
    CompiledOutput::load(&serde_json::to_string(&output).unwrap()).unwrap()
}

#[test]
fn vm_matches_reference_on_small_state_graphs() {
    compare_graph("TwoPhaseCommit.json", 3);
    compare_graph("TeachingConcurrency.json", 2);
    compare_graph("DiningPhilosophers_fixed.json", 2);
}

#[test]
fn vm_matches_reference_on_generated_ir_family() {
    let cases: &[(i64, &[i64], i64)] = &[
        (0, &[-1, 0, 1], 2),
        (-2, &[1, 2], 4),
        (2, &[-2, -1], 3),
        (0, &[0], 0),
        (1, &[-1, 1, 2], 5),
    ];
    for (i, &(initial, deltas, limit)) in cases.iter().enumerate() {
        compare_output(
            counter_ir(initial, deltas, limit),
            &format!("generated-{i}"),
            4,
        );
    }
}

#[test]
fn safety_counterexample_is_real_in_reference_semantics() {
    use inquint::explorer::{check, CheckConfig, CheckOutcome};

    let out = load("TwoPhaseCommit.json");
    let prod = CompiledSpec::build(
        &out,
        &EntryPoints {
            invariants: vec!["brokenNoAbort".into()],
            ..Default::default()
        },
    )
    .unwrap();
    let outcome = check(
        &prod,
        &CheckConfig {
            max_steps: Some(2),
            deadlock: false,
            max_states: None,
            exact_states: true,
            threads: 1,
        },
    )
    .map_err(|e| e.error)
    .unwrap();
    let CheckOutcome::InvariantViolation { trace, .. } = outcome else {
        panic!("expected invariant violation");
    };

    let vars = inquint_reference::VarTable::from_module(out.module());
    let mut compiler = inquint_reference::Compiler::new(&out.table, &vars);
    let init = compiler.compile(&find_def(&out, "q::init").expr);
    let step = compiler.compile(&find_def(&out, "q::step").expr);
    let invariant = compiler.compile(&find_def(&out, "brokenNoAbort").expr);
    let storage = compiler.storage.clone();

    let initial = inquint_reference::enumerate(&init, &storage, None).unwrap();
    let mut current = initial
        .into_iter()
        .find(|state| reference_key(state) == prod_key(&trace[0]))
        .expect("trace starts in a reference initial state");
    for target in trace.iter().skip(1) {
        current = inquint_reference::enumerate(&step, &storage, Some(&current))
            .unwrap()
            .into_iter()
            .find(|state| reference_key(state) == prod_key(target))
            .expect("every trace edge exists in reference semantics");
    }
    storage.borrow().load(&current);
    let holds = invariant
        .execute(&mut inquint_reference::eval::Env::new(storage, None))
        .unwrap()
        .as_bool();
    assert!(!holds, "reference invariant must fail at the final state");
}
