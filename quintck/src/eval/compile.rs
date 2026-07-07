//! Compilation of Quint IR expressions into closures.

use super::builtins_eager::eager_op;
use super::builtins_lazy::{lazy_op, LAZY_OPS};
use super::{apply_lambda, CompiledExpr};
use crate::error::QuintError;
use crate::state::{VarStorage, VarTable};
use crate::value::{EvalResult, Value};
use quint_ast::{Declaration, LambdaParam, LookupDefinition, LookupTable, OpDef, OpQualifier, QuintEx, QuintId};
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::rc::Rc;

pub struct Compiler<'t> {
    table: &'t LookupTable,
    pub storage: Rc<RefCell<VarStorage>>,
    /// var declaration id -> index into the storage registers
    var_index: FxHashMap<QuintId, usize>,
    param_registry: FxHashMap<QuintId, Rc<RefCell<Option<Value>>>>,
    /// let-scoped cached values (also the binding cell for nondet picks)
    scoped_cache: FxHashMap<QuintId, Rc<RefCell<Option<EvalResult>>>>,
    memo: FxHashMap<QuintId, CompiledExpr>,
    def_memo: FxHashMap<QuintId, CompiledExpr>,
}

impl<'t> Compiler<'t> {
    pub fn new(table: &'t LookupTable, vars: &VarTable) -> Self {
        Compiler {
            table,
            storage: Rc::new(RefCell::new(VarStorage::new(vars))),
            var_index: vars.by_def_id.clone(),
            param_registry: FxHashMap::default(),
            scoped_cache: FxHashMap::default(),
            memo: FxHashMap::default(),
            def_memo: FxHashMap::default(),
        }
    }

    pub fn param_register(&mut self, param: &LambdaParam) -> Rc<RefCell<Option<Value>>> {
        self.param_registry
            .entry(param.id)
            .or_insert_with(|| Rc::new(RefCell::new(None)))
            .clone()
    }

    fn scoped_cell(&mut self, id: QuintId) -> Rc<RefCell<Option<EvalResult>>> {
        self.scoped_cache
            .entry(id)
            .or_insert_with(|| Rc::new(RefCell::new(None)))
            .clone()
    }

    /// Compile a definition (the target of a name reference).
    pub fn compile_def(&mut self, def: &LookupDefinition) -> CompiledExpr {
        let def_id = match def {
            LookupDefinition::Definition(d) => match d {
                Declaration::OpDef(op) => op.id,
                Declaration::Var { id, .. } => *id,
                Declaration::Const { id, .. } => *id,
                Declaration::Assume { id, .. } => *id,
                Declaration::TypeDef { id } => *id,
                _ => u64::MAX,
            },
            LookupDefinition::Param(p) => p.id,
        };
        if let Some(cached) = self.def_memo.get(&def_id) {
            return cached.clone();
        }

        let compiled = match def {
            LookupDefinition::Definition(Declaration::OpDef(op)) => self.compile_opdef(op),
            LookupDefinition::Definition(Declaration::Var { id, name }) => {
                let index = *self
                    .var_index
                    .get(id)
                    .unwrap_or_else(|| panic!("unknown variable {name} (id {id})"));
                let current = self.storage.borrow().current[index].clone();
                let next = self.storage.borrow().next[index].clone();
                let name = name.clone();
                CompiledExpr::new(move |env| {
                    let register = if env.next_mode { &next } else { &current };
                    register.borrow().clone().ok_or_else(|| {
                        QuintError::new("QNT502", format!("Variable {name} not set"))
                    })
                })
            }
            LookupDefinition::Definition(Declaration::Const { name, .. }) => {
                let name = name.clone();
                CompiledExpr::new(move |_| {
                    Err(QuintError::new(
                        "QNT500",
                        format!(
                            "Uninitialized const {name}. Use: import <moduleName>({name}=<value>).*"
                        ),
                    ))
                })
            }
            LookupDefinition::Param(p) => {
                let register = self.param_register(p);
                let name = p.name.clone();
                CompiledExpr::new(move |_| {
                    register
                        .borrow()
                        .clone()
                        .ok_or_else(|| QuintError::new("QNT500", format!("Param {name} not set")))
                })
            }
            d => panic!("cannot compile reference to {d:?}"),
        };

        self.def_memo.insert(def_id, compiled.clone());
        compiled
    }

    fn compile_opdef(&mut self, op: &OpDef) -> CompiledExpr {
        let top_level = op.depth.is_none_or(|d| d == 0);

        let base = if matches!(op.expr, QuintEx::Lambda { .. }) || top_level {
            self.compile(&op.expr)
        } else {
            // Scoped (let-bound) definition: lazily evaluated once per
            // enclosing let scope, via the shared cache cell. For nondet
            // bindings the cell is written by the nondet-let closure before
            // the body runs.
            let cell = self.scoped_cell(op.id);
            let body = self.compile(&op.expr);
            CompiledExpr::new(move |env| {
                let cached = cell.borrow().clone();
                match cached {
                    Some(result) => result,
                    None => {
                        let result = body.execute(env);
                        *cell.borrow_mut() = Some(result.clone());
                        result
                    }
                }
            })
        };

        // Top-level caching by qualifier: `val` per state, `pureval` forever.
        match (op.qualifier, top_level) {
            (OpQualifier::Val, true) => {
                let cell: Rc<RefCell<Option<EvalResult>>> = Rc::new(RefCell::new(None));
                self.storage.borrow_mut().caches_to_clear.push(cell.clone());
                CompiledExpr::new(move |env| {
                    // The per-state cache is keyed on the *current* state;
                    // under next_mode the val reads the next state, so the
                    // cache must be bypassed entirely (read and write).
                    if env.next_mode {
                        return base.execute(env);
                    }
                    let cached = cell.borrow().clone();
                    if let Some(Ok(v)) = cached {
                        return Ok(v);
                    }
                    let result = base.execute(env)?;
                    *cell.borrow_mut() = Some(Ok(result.clone()));
                    Ok(result)
                })
            }
            (OpQualifier::PureVal, true) => {
                let cell: Rc<RefCell<Option<EvalResult>>> = Rc::new(RefCell::new(None));
                CompiledExpr::new(move |env| {
                    let cached = cell.borrow().clone();
                    if let Some(Ok(v)) = cached {
                        return Ok(v);
                    }
                    let result = base.execute(env)?;
                    *cell.borrow_mut() = Some(Ok(result.clone()));
                    Ok(result)
                })
            }
            _ => base,
        }
    }

    /// Compile an expression, memoized by node id, wrapping errors with the
    /// node id for diagnostics.
    pub fn compile(&mut self, expr: &QuintEx) -> CompiledExpr {
        let id = expr.id();
        if let Some(cached) = self.memo.get(&id) {
            return cached.clone();
        }
        let core = self.compile_core(expr);
        let wrapped = CompiledExpr::new(move |env| {
            core.execute(env).map_err(|err| {
                if err.trace.is_empty() {
                    err.with_id(id)
                } else {
                    err
                }
            })
        });
        self.memo.insert(id, wrapped.clone());
        wrapped
    }

    fn compile_core(&mut self, expr: &QuintEx) -> CompiledExpr {
        match expr {
            QuintEx::Int { value, .. } => {
                let v = Value::int(*value);
                CompiledExpr::new(move |_| Ok(v.clone()))
            }
            QuintEx::Bool { value, .. } => {
                let v = Value::bool(*value);
                CompiledExpr::new(move |_| Ok(v.clone()))
            }
            QuintEx::Str { value, .. } => {
                let v = Value::str(value.clone());
                CompiledExpr::new(move |_| Ok(v.clone()))
            }

            QuintEx::Name { id, name } => match self.table.get(id) {
                Some(def) => {
                    let def = def.clone();
                    self.compile_def(&def)
                }
                None => builtin_name(name.as_ref()),
            },

            QuintEx::Lambda { params, expr, .. } => {
                let body = self.compile(expr);
                let registers = params.iter().map(|p| self.param_register(p)).collect();
                let lambda = Value::lambda(registers, body);
                CompiledExpr::new(move |_| Ok(lambda.clone()))
            }

            QuintEx::App {
                id, opcode, args, ..
            } => self.compile_app(*id, opcode.as_ref(), args),

            QuintEx::Let { opdef, expr, .. } => self.compile_let(opdef, expr),
        }
    }

    fn compile_app(&mut self, id: QuintId, opcode: &str, args: &[QuintEx]) -> CompiledExpr {
        if opcode == "assign" {
            let var_def = self
                .table
                .get(&args[0].id())
                .unwrap_or_else(|| panic!("assign target not in lookup table"))
                .clone();
            let (var_id, var_name) = match &var_def {
                LookupDefinition::Definition(Declaration::Var { id, name }) => {
                    (*id, name.clone())
                }
                d => panic!("assign target is not a variable: {d:?}"),
            };
            let index = *self
                .var_index
                .get(&var_id)
                .unwrap_or_else(|| panic!("unknown variable {var_name}"));
            let register = self.storage.borrow().next[index].clone();
            let rhs = self.compile(&args[1]);
            return CompiledExpr::new(move |env| {
                let value = rhs.execute(env)?.normalize()?;
                *register.borrow_mut() = Some(value);
                Ok(Value::bool(true))
            });
        }

        if LAZY_OPS.contains(&opcode) {
            let op = lazy_op(opcode);
            let compiled_args: Vec<CompiledExpr> =
                args.iter().map(|a| self.compile(a)).collect();
            return CompiledExpr::new(move |env| op(env, &compiled_args));
        }

        let compiled_args: Vec<CompiledExpr> = args.iter().map(|a| self.compile(a)).collect();

        match self.table.get(&id) {
            // User-defined operator application
            Some(def) => {
                let def = def.clone();
                let op = self.compile_def(&def);
                CompiledExpr::new(move |env| {
                    let evaluated = compiled_args
                        .iter()
                        .map(|a| a.execute(env))
                        .collect::<Result<Vec<_>, _>>()?;
                    let lambda = op.execute(env)?;
                    apply_lambda(&lambda, env, evaluated)
                })
            }
            // Eager builtin
            None => match eager_op(opcode) {
                Some(op) => CompiledExpr::new(move |env| {
                    let evaluated = compiled_args
                        .iter()
                        .map(|a| a.execute(env))
                        .collect::<Result<Vec<_>, _>>()?;
                    op(env, evaluated)
                }),
                None => {
                    let opcode = opcode.to_owned();
                    CompiledExpr::new(move |_| {
                        Err(QuintError::new(
                            "QNT000",
                            format!("unknown operator: {opcode}"),
                        ))
                    })
                }
            },
        }
    }

    fn compile_let(&mut self, opdef: &OpDef, body: &QuintEx) -> CompiledExpr {
        // nondet x = S.oneOf() — the enumerated choice point
        if opdef.qualifier == OpQualifier::Nondet {
            if let QuintEx::App { opcode, args, .. } = &opdef.expr {
                if opcode.as_ref() == "oneOf" && args.len() == 1 {
                    let set_expr = self.compile(&args[0]);
                    let cell = self.scoped_cell(opdef.id);
                    let body = self.compile(body);
                    return CompiledExpr::new(move |env| {
                        let set = set_expr.execute(env)?;
                        let bounds = set.bounds()?;
                        let mut indices = Vec::with_capacity(bounds.len());
                        for bound in bounds {
                            match env.choose(bound)? {
                                Some(i) => indices.push(i),
                                // empty set: this branch is disabled
                                None => return Ok(Value::bool(false)),
                            }
                        }
                        let picked = set.pick(&mut indices.into_iter())?.normalize()?;
                        let saved = cell.replace(Some(Ok(picked)));
                        let result = body.execute(env);
                        cell.replace(saved);
                        result
                    });
                }
            }
        }

        // Regular let: the bound value is evaluated lazily on first use and
        // cached for the scope (call-by-need); save/restore for reentrancy.
        let cell = self.scoped_cell(opdef.id);
        let body = self.compile(body);
        CompiledExpr::new(move |env| {
            let saved = cell.replace(None);
            let result = body.execute(env);
            cell.replace(saved);
            result
        })
    }
}

fn builtin_name(name: &str) -> CompiledExpr {
    match name {
        "true" => CompiledExpr::new(|_| Ok(Value::bool(true))),
        "false" => CompiledExpr::new(|_| Ok(Value::bool(false))),
        "Bool" => CompiledExpr::new(|_| Value::set([Value::bool(false), Value::bool(true)])),
        "Int" => CompiledExpr::new(|_| Ok(Value::infinite_int())),
        "Nat" => CompiledExpr::new(|_| Ok(Value::infinite_nat())),
        _ => {
            let name = name.to_owned();
            CompiledExpr::new(move |_| {
                Err(QuintError::new(
                    "QNT000",
                    format!("unknown builtin name: {name}"),
                ))
            })
        }
    }
}
