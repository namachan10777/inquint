//! `run` definition execution (quint's unit tests: init.then(...).expect(...)).
//!
//! Unlike `quint test`, which samples nondeterministic choices randomly,
//! quintck enumerates the run's entire choice tree: the test passes only
//! if EVERY path through its nondeterminism succeeds.

use crate::choice::ChoiceCtl;
use crate::error::QuintError;
use crate::eval::Env;
use crate::state::VarTable;
use crate::vm::Lowerer;
use quint_ast::{CompiledOutput, Declaration, OpQualifier, QuintName};

/// Backstop against runaway choice trees.
const MAX_PATHS: u64 = 1 << 22;

pub struct TestReport {
    pub name: QuintName,
    /// Number of enumerated paths (all passed) or the failure.
    pub result: Result<u64, QuintError>,
}

/// Run the named tests (or all `run` defs when `names` is empty).
pub fn run_tests(out: &CompiledOutput, names: &[String]) -> Result<Vec<TestReport>, QuintError> {
    let module = out.module();
    let vars = VarTable::from_module(module);

    let selected: Vec<_> = module
        .declarations
        .iter()
        .filter_map(|d| match d {
            Declaration::OpDef(op) if op.qualifier == OpQualifier::Run => Some(op),
            _ => None,
        })
        .filter(|op| {
            names.is_empty()
                || names
                    .iter()
                    .any(|n| op.name.as_ref() == n || op.name.ends_with(&format!("::{n}")))
        })
        .collect();

    if selected.is_empty() {
        return Err(QuintError::new(
            "QNT000",
            "no run definitions found (compile the spec without --temporal/--invariant \
             pruning them, or check the names)",
        ));
    }

    let mut compiler = Lowerer::new(&out.table, &vars);
    let compiled: Vec<(QuintName, crate::vm::FnId)> = selected
        .iter()
        .map(|op| (op.name, compiler.compile(&op.expr)))
        .collect();
    let program = compiler.finish();
    let mut vm = crate::vm::Vm::new(&program);

    let mut reports = Vec::new();
    for (name, fnid) in compiled {
        let mut ctl = ChoiceCtl::new();
        let mut paths: u64 = 0;
        let result = loop {
            paths += 1;
            if paths > MAX_PATHS {
                break Err(QuintError::new(
                    "QNT501",
                    format!("more than {MAX_PATHS} nondeterministic paths in run {name}"),
                ));
            }
            vm.clear_current();
            vm.reset_next();
            let mut env = Env::new(Some(&mut ctl));
            env.any_fallthrough = true;
            match vm.run(&mut env, fnid) {
                Err(e) => break Err(e),
                Ok(v) if !v.as_bool() => {
                    break Err(QuintError::new("QNT511", "Test returned false"))
                }
                Ok(_) => {}
            }
            if !ctl.advance() {
                break Ok(paths);
            }
        };
        reports.push(TestReport { name, result });
    }
    Ok(reports)
}
