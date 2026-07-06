//! Entry-point resolution and compilation of a checkable spec.

use crate::eval::{CompiledExpr, Compiler, Env};
use crate::state::{State, VarStorage, VarTable};
use crate::value::EvalResult;
use quint_ast::{CompiledOutput, Declaration, OpDef, OpQualifier, QuintName};
use std::cell::RefCell;
use std::rc::Rc;

pub struct CompiledSpec {
    pub vars: VarTable,
    pub storage: Rc<RefCell<VarStorage>>,
    pub init: CompiledExpr,
    pub step: CompiledExpr,
    pub invariants: Vec<(QuintName, CompiledExpr)>,
}

#[derive(Default)]
pub struct EntryPoints {
    pub init: Option<String>,
    pub step: Option<String>,
    /// Invariant names; when empty, `q::inv` is used if present.
    pub invariants: Vec<String>,
}

#[derive(thiserror::Error, Debug)]
pub enum BuildError {
    #[error(
        "cannot find {kind} '{name}' in module; available actions: {available}"
    )]
    NoSuchDef {
        kind: &'static str,
        name: String,
        available: String,
    },
    #[error("temporal properties are not supported by quintck v1 (found {0})")]
    Temporal(String),
    #[error("the module declares no state variables")]
    NoVars,
}

/// Find a def whose name is `name` or ends with `::name` (flattened names
/// are prefixed, e.g. `main::clock_sync::skewOK`).
fn find_def<'m>(module: &'m quint_ast::QuintModule, name: &str) -> Option<&'m OpDef> {
    let suffix = format!("::{name}");
    module.declarations.iter().find_map(|d| match d {
        Declaration::OpDef(op)
            if op.name.as_ref() == name || op.name.ends_with(&suffix) =>
        {
            Some(op)
        }
        _ => None,
    })
}

fn available_actions(module: &quint_ast::QuintModule) -> String {
    module
        .declarations
        .iter()
        .filter_map(|d| match d {
            Declaration::OpDef(op) if op.qualifier == OpQualifier::Action => {
                Some(op.name.to_string())
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(", ")
}

impl CompiledSpec {
    pub fn build(out: &CompiledOutput, entry: &EntryPoints) -> Result<Self, BuildError> {
        let module = out.module();

        // Reject temporal entry points explicitly.
        if let Some(t) = module.declarations.iter().find_map(|d| match d {
            Declaration::OpDef(op)
                if op.qualifier == OpQualifier::Temporal
                    && op.name.as_ref() == "q::temporalProps" =>
            {
                Some(op.name.to_string())
            }
            _ => None,
        }) {
            return Err(BuildError::Temporal(t));
        }

        let vars = VarTable::from_module(module);
        if vars.is_empty() {
            return Err(BuildError::NoVars);
        }

        let resolve = |kind: &'static str, explicit: &Option<String>, default: &str| {
            let name = explicit.as_deref().unwrap_or(default);
            find_def(module, name).ok_or_else(|| BuildError::NoSuchDef {
                kind,
                name: name.to_string(),
                available: available_actions(module),
            })
        };

        let init_def = resolve("init action", &entry.init, "q::init")?;
        let step_def = resolve("step action", &entry.step, "q::step")?;

        let mut invariant_defs: Vec<(QuintName, &OpDef)> = Vec::new();
        if entry.invariants.is_empty() {
            if let Some(inv) = find_def(module, "q::inv") {
                invariant_defs.push((inv.name.clone(), inv));
            }
        } else {
            for name in &entry.invariants {
                let def = find_def(module, name).ok_or_else(|| BuildError::NoSuchDef {
                    kind: "invariant",
                    name: name.clone(),
                    available: available_actions(module),
                })?;
                invariant_defs.push((QuintName::from(name.as_str()), def));
            }
        }

        let mut compiler = Compiler::new(&out.table, &vars);
        let init = compiler.compile(&init_def.expr);
        let step = compiler.compile(&step_def.expr);
        let invariants = invariant_defs
            .into_iter()
            .map(|(name, def)| (name, compiler.compile(&def.expr)))
            .collect();
        let storage = compiler.storage.clone();

        Ok(CompiledSpec {
            vars,
            storage,
            init,
            step,
            invariants,
        })
    }

    /// Evaluate one invariant against a state (no nondeterminism allowed).
    pub fn eval_invariant_at(&self, state: &State, inv: &CompiledExpr) -> EvalResult {
        self.storage.borrow().load(state);
        let mut env = Env {
            storage: self.storage.clone(),
            choices: None,
        };
        inv.execute(&mut env)
    }
}
