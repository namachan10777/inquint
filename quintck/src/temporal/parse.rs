//! Translation of a `temporal` definition's QuintEx into a checkable
//! [`Property`]: fairness premises (WF/SF) split off as side conditions,
//! the body as an [`Ltl`] formula over opaque atoms.
//!
//! Quantifiers (`Set.forall/exists`) over temporal subformulas are expanded
//! at translation time: the quantified set must be state-independent, and
//! each instantiation captures the bound value in a [`BoundExpr`].

use super::ltl::{EdgeAtomId, Ltl, StateAtomId};
use crate::error::QuintError;
use crate::eval::{BoundExpr, Env};
use crate::state::Register;
use crate::vm::Lowerer;
use crate::value::Value;
use quint_ast::{Declaration, LookupDefinition, LookupTable, QuintEx, QuintId, QuintName};
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;

pub type ActionId = u32;
pub type VarsId = u32;

type Binding = (QuintId, Register, Value);

pub enum StateAtom {
    /// A plain state predicate.
    Pred(BoundExpr),
    /// ENABLED A (changed: None) or ENABLED ⟨A⟩_v (changed: Some(v)).
    Enabled {
        action: ActionId,
        changed: Option<VarsId>,
    },
}

pub enum EdgeAtom {
    /// ⟨A⟩_v on edge (s,t): t ∈ T_A(s) and v(t) != v(s).
    Taken { action: ActionId, vars: VarsId },
    /// [A]_v on edge (s,t): t ∈ T_A(s) or v(t) == v(s).
    Kept { action: ActionId, vars: VarsId },
    /// An arbitrary predicate containing next(...): evaluated with
    /// current = s and the next bank = t.
    NextPred(BoundExpr),
}

/// One WF/SF premise: `strong ? SF_v(A) : WF_v(A)`.
pub struct Fairness {
    pub strong: bool,
    /// ⟨A⟩_v edge atom ("an A step that changes v was taken").
    pub taken: EdgeAtomId,
    /// ENABLED ⟨A⟩_v state atom.
    pub enabled: StateAtomId,
    /// Display name for diagnostics.
    pub label: String,
}

pub struct Property {
    pub name: QuintName,
    pub fairness: Vec<Fairness>,
    pub body: Ltl,
}

#[derive(Default)]
pub struct AtomTable {
    pub actions: Vec<BoundExpr>,
    pub vars_exprs: Vec<BoundExpr>,
    pub state_atoms: Vec<StateAtom>,
    pub edge_atoms: Vec<EdgeAtom>,
    // interning keys: (expression node id, bound values in binding order)
    action_index: BTreeMap<(QuintId, Vec<Value>), ActionId>,
    vars_index: BTreeMap<(QuintId, Vec<Value>), VarsId>,
    satom_index: BTreeMap<SAtomKey, StateAtomId>,
    eatom_index: BTreeMap<EAtomKey, EdgeAtomId>,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum SAtomKey {
    Pred(QuintId, Vec<Value>),
    Enabled(ActionId, Option<VarsId>),
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum EAtomKey {
    Taken(ActionId, VarsId),
    Kept(ActionId, VarsId),
    NextPred(QuintId, Vec<Value>),
}

pub struct Parser<'c, 't> {
    compiler: &'c mut Lowerer<'t>,
    table: &'t LookupTable,
    pub atoms: AtomTable,
    /// Current quantifier bindings: (param id, register, value).
    bindings: Vec<Binding>,
    temporal_memo: FxHashMap<QuintId, TemporalKind>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TemporalKind {
    /// No temporal operator, no next().
    Pure,
    /// Contains next() but no other temporal operator.
    NextOnly,
    /// Contains always/eventually/leadsTo/weakFair/strongFair/orKeep/
    /// mustChange/enabled.
    Temporal,
}

const TEMPORAL_OPS: &[&str] = &[
    "always",
    "eventually",
    "leadsTo",
    "weakFair",
    "strongFair",
    "orKeep",
    "mustChange",
    "enabled",
];

fn err(msg: impl Into<String>) -> QuintError {
    QuintError::new("QNT501", msg)
}

impl<'c, 't> Parser<'c, 't> {
    pub fn new(compiler: &'c mut Lowerer<'t>, table: &'t LookupTable) -> Self {
        Parser {
            compiler,
            table,
            atoms: AtomTable::default(),
            bindings: Vec::new(),
            temporal_memo: FxHashMap::default(),
        }
    }

    pub fn parse_property(&mut self, name: QuintName, expr: &QuintEx) -> Result<Property, QuintError> {
        let expr = self.deref_names(expr);
        // Split fairness premises: `prem implies body` with all-WF/SF prem.
        let (fairness, body_expr) = match &expr {
            QuintEx::App { opcode, args, .. } if opcode.as_ref() == "implies" => {
                match self.collect_fairness(&args[0]) {
                    Ok(fs) => (fs, args[1].clone()),
                    // premise is not a pure fairness conjunction: treat the
                    // whole formula as the body (WF/SF inside will error)
                    Err(_) => (Vec::new(), expr.clone()),
                }
            }
            _ => (Vec::new(), expr.clone()),
        };
        let body = self.translate(&body_expr)?;
        Ok(Property {
            name,
            fairness,
            body,
        })
    }

    /// Follow a chain of Name references to the underlying expression.
    fn deref_names(&self, expr: &QuintEx) -> QuintEx {
        let mut current = expr.clone();
        loop {
            match &current {
                QuintEx::Name { id, .. } => {
                    match self.table.get(id) {
                        Some(LookupDefinition::Definition(Declaration::OpDef(op)))
                            if !matches!(op.expr, QuintEx::Lambda { .. }) =>
                        {
                            current = op.expr.clone();
                        }
                        _ => return current,
                    }
                }
                _ => return current,
            }
        }
    }

    // ---------------------------------------------------------------
    // Temporal-kind analysis (syntactic, memoized by node id)
    // ---------------------------------------------------------------

    fn kind_of(&mut self, expr: &QuintEx) -> TemporalKind {
        if let Some(k) = self.temporal_memo.get(&expr.id()) {
            return *k;
        }
        // Insert a provisional value to cut (impossible) cycles.
        self.temporal_memo.insert(expr.id(), TemporalKind::Pure);
        let kind = self.kind_of_core(expr);
        self.temporal_memo.insert(expr.id(), kind);
        kind
    }

    fn kind_of_core(&mut self, expr: &QuintEx) -> TemporalKind {
        match expr {
            QuintEx::Bool { .. } | QuintEx::Int { .. } | QuintEx::Str { .. } => TemporalKind::Pure,
            QuintEx::Name { id, .. } => match self.table.get(id) {
                Some(LookupDefinition::Definition(Declaration::OpDef(op))) => {
                    let body = op.expr.clone();
                    self.kind_of(&body)
                }
                _ => TemporalKind::Pure,
            },
            QuintEx::App { id, opcode, args } => {
                let own = if opcode.as_ref() == "next" {
                    TemporalKind::NextOnly
                } else if TEMPORAL_OPS.contains(&opcode.as_ref()) {
                    TemporalKind::Temporal
                } else {
                    TemporalKind::Pure
                };
                let mut kind = own;
                // User-defined op: also look at its body.
                if let Some(LookupDefinition::Definition(Declaration::OpDef(op))) =
                    self.table.get(id)
                {
                    let body = op.expr.clone();
                    kind = kind.max_with(self.kind_of(&body));
                }
                for arg in args {
                    kind = kind.max_with(self.kind_of(arg));
                }
                kind
            }
            QuintEx::Lambda { expr, .. } => self.kind_of(expr),
            QuintEx::Let { opdef, expr, .. } => {
                let a = {
                    let body = opdef.expr.clone();
                    self.kind_of(&body)
                };
                a.max_with(self.kind_of(expr))
            }
        }
    }

    // ---------------------------------------------------------------
    // Fairness premises
    // ---------------------------------------------------------------

    fn collect_fairness(&mut self, prem: &QuintEx) -> Result<Vec<Fairness>, QuintError> {
        let prem = self.deref_names(prem);
        match &prem {
            QuintEx::App { opcode, args, .. } if opcode.as_ref() == "and" => {
                let mut out = Vec::new();
                for arg in args {
                    out.extend(self.collect_fairness(arg)?);
                }
                Ok(out)
            }
            QuintEx::App { opcode, args, .. } if opcode.as_ref() == "forall" => {
                let elems = self.eval_constant_set(&args[0])?;
                let (params, body) = self.lambda_parts(&args[1])?;
                let mut out = Vec::new();
                for elem in elems {
                    self.push_binding(&params[0], elem)?;
                    let result = self.collect_fairness(&body);
                    self.bindings.pop();
                    out.extend(result?);
                }
                Ok(out)
            }
            QuintEx::App { opcode, args, .. }
                if opcode.as_ref() == "weakFair" || opcode.as_ref() == "strongFair" =>
            {
                let strong = opcode.as_ref() == "strongFair";
                let action = self.intern_action(&args[0])?;
                let vars = self.intern_vars(&args[1])?;
                let taken = self.atoms.intern_edge(EAtomKey::Taken(action, vars), || {
                    EdgeAtom::Taken { action, vars }
                });
                let enabled = self
                    .atoms
                    .intern_state(SAtomKey::Enabled(action, Some(vars)), || StateAtom::Enabled {
                        action,
                        changed: Some(vars),
                    });
                let label = format!(
                    "{}({})",
                    if strong { "strongFair" } else { "weakFair" },
                    self.describe(&args[0])
                );
                Ok(vec![Fairness {
                    strong,
                    taken,
                    enabled,
                    label,
                }])
            }
            _ => Err(err("premise is not a fairness conjunction")),
        }
    }

    fn describe(&self, expr: &QuintEx) -> String {
        let bound = self
            .bindings
            .iter()
            .map(|(_, _, v)| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let base = match expr {
            QuintEx::Name { name, .. } => name.to_string(),
            QuintEx::App { opcode, .. } => opcode.to_string(),
            _ => "<action>".to_string(),
        };
        if bound.is_empty() {
            base
        } else {
            format!("{base}[{bound}]")
        }
    }

    // ---------------------------------------------------------------
    // Body translation
    // ---------------------------------------------------------------

    pub fn translate(&mut self, expr: &QuintEx) -> Result<Ltl, QuintError> {
        match self.kind_of(expr) {
            TemporalKind::Pure => {
                let id = self.intern_state_pred(expr)?;
                return Ok(Ltl::SAtom(id));
            }
            TemporalKind::NextOnly => {
                let id = self.intern_next_pred(expr)?;
                return Ok(Ltl::EAtom(id));
            }
            TemporalKind::Temporal => {}
        }

        match expr {
            QuintEx::Name { .. } => {
                let inner = self.deref_names(expr);
                if matches!(inner, QuintEx::Name { .. }) {
                    return Err(err("cannot resolve temporal name"));
                }
                self.translate(&inner)
            }
            QuintEx::App { id, opcode, args } => {
                match opcode.as_ref() {
                    "always" => Ok(Ltl::Always(Box::new(self.translate(&args[0])?))),
                    "eventually" => Ok(Ltl::Eventually(Box::new(self.translate(&args[0])?))),
                    "leadsTo" => {
                        let p = self.translate(&args[0])?;
                        let q = self.translate(&args[1])?;
                        Ok(Ltl::Always(Box::new(Ltl::Or(vec![
                            Ltl::Not(Box::new(p)),
                            Ltl::Eventually(Box::new(q)),
                        ]))))
                    }
                    "not" => Ok(Ltl::Not(Box::new(self.translate(&args[0])?))),
                    "and" | "actionAll" => Ok(Ltl::And(
                        args.iter()
                            .map(|a| self.translate(a))
                            .collect::<Result<Vec<_>, _>>()?,
                    )),
                    "or" | "actionAny" => Ok(Ltl::Or(
                        args.iter()
                            .map(|a| self.translate(a))
                            .collect::<Result<Vec<_>, _>>()?,
                    )),
                    "implies" => {
                        let p = self.translate(&args[0])?;
                        let q = self.translate(&args[1])?;
                        Ok(Ltl::Or(vec![Ltl::Not(Box::new(p)), q]))
                    }
                    "iff" => {
                        let p = self.translate(&args[0])?;
                        let q = self.translate(&args[1])?;
                        Ok(Ltl::And(vec![
                            Ltl::Or(vec![Ltl::Not(Box::new(p.clone())), q.clone()]),
                            Ltl::Or(vec![Ltl::Not(Box::new(q)), p]),
                        ]))
                    }
                    "orKeep" => {
                        let action = self.intern_action(&args[0])?;
                        let vars = self.intern_vars(&args[1])?;
                        let atom = self.atoms.intern_edge(EAtomKey::Kept(action, vars), || {
                            EdgeAtom::Kept { action, vars }
                        });
                        Ok(Ltl::EAtom(atom))
                    }
                    "mustChange" => {
                        let action = self.intern_action(&args[0])?;
                        let vars = self.intern_vars(&args[1])?;
                        let atom = self.atoms.intern_edge(EAtomKey::Taken(action, vars), || {
                            EdgeAtom::Taken { action, vars }
                        });
                        Ok(Ltl::EAtom(atom))
                    }
                    "enabled" => {
                        // enabled(mustChange(A, v)) => ENABLED ⟨A⟩_v
                        if let QuintEx::App {
                            opcode: inner_op,
                            args: inner_args,
                            ..
                        } = &args[0]
                        {
                            if inner_op.as_ref() == "mustChange" {
                                let action = self.intern_action(&inner_args[0])?;
                                let vars = self.intern_vars(&inner_args[1])?;
                                let atom = self.atoms.intern_state(
                                    SAtomKey::Enabled(action, Some(vars)),
                                    || StateAtom::Enabled {
                                        action,
                                        changed: Some(vars),
                                    },
                                );
                                return Ok(Ltl::SAtom(atom));
                            }
                        }
                        let action = self.intern_action(&args[0])?;
                        let atom = self
                            .atoms
                            .intern_state(SAtomKey::Enabled(action, None), || StateAtom::Enabled {
                                action,
                                changed: None,
                            });
                        Ok(Ltl::SAtom(atom))
                    }
                    "weakFair" | "strongFair" => Err(err(
                        "weakFair/strongFair are only supported as top-level premises \
                         (`F1 and ... and Fn implies Body`); rewrite the property",
                    )),
                    "forall" | "exists" => {
                        let elems = self.eval_constant_set(&args[0])?;
                        let (params, body) = self.lambda_parts(&args[1])?;
                        let mut parts = Vec::new();
                        for elem in elems {
                            self.push_binding(&params[0], elem)?;
                            let result = self.translate(&body);
                            self.bindings.pop();
                            parts.push(result?);
                        }
                        if opcode.as_ref() == "forall" {
                            Ok(Ltl::And(parts))
                        } else {
                            Ok(Ltl::Or(parts))
                        }
                    }
                    "ite" => {
                        // if (c) p else q with temporal branches: c must be pure
                        let c = self.translate(&args[0])?;
                        let p = self.translate(&args[1])?;
                        let q = self.translate(&args[2])?;
                        Ok(Ltl::Or(vec![
                            Ltl::And(vec![c.clone(), p]),
                            Ltl::And(vec![Ltl::Not(Box::new(c)), q]),
                        ]))
                    }
                    _ => {
                        // A user-defined operator application whose body is
                        // temporal: inline it, binding parameters to
                        // pure-evaluated argument values.
                        match self.table.get(id) {
                            Some(LookupDefinition::Definition(Declaration::OpDef(op))) => {
                                let body = op.expr.clone();
                                if let QuintEx::Lambda { params, expr, .. } = &body {
                                    let mut values = Vec::new();
                                    for arg in args {
                                        if self.kind_of(arg) != TemporalKind::Pure {
                                            return Err(err(
                                                "temporal expression as operator argument is not supported",
                                            ));
                                        }
                                        values.push(self.eval_pure(arg)?);
                                    }
                                    for (p, v) in params.iter().zip(values) {
                                        self.push_binding_param(p, v);
                                    }
                                    let result = self.translate(expr);
                                    for _ in params {
                                        self.bindings.pop();
                                    }
                                    result
                                } else {
                                    self.translate(&body)
                                }
                            }
                            _ => Err(err(format!(
                                "temporal operator {opcode} is not supported"
                            ))),
                        }
                    }
                }
            }
            QuintEx::Let { .. } | QuintEx::Lambda { .. } => Err(err(
                "let/lambda around temporal subformulas is not supported",
            )),
            _ => unreachable!("pure literals handled by kind_of"),
        }
    }

    // ---------------------------------------------------------------
    // Atom interning helpers
    // ---------------------------------------------------------------

    fn bound_values(&self) -> Vec<Value> {
        self.bindings.iter().map(|(_, _, v)| *v).collect()
    }

    fn bound_expr(&mut self, expr: &QuintEx) -> BoundExpr {
        let compiled = self.compiler.compile(expr);
        BoundExpr {
            expr: compiled,
            bindings: self
                .bindings
                .iter()
                .map(|(_, reg, v)| (reg.clone(), *v))
                .collect(),
        }
    }

    fn intern_state_pred(&mut self, expr: &QuintEx) -> Result<StateAtomId, QuintError> {
        let key = SAtomKey::Pred(expr.id(), self.bound_values());
        if let Some(id) = self.atoms.satom_index.get(&key) {
            return Ok(*id);
        }
        let be = self.bound_expr(expr);
        Ok(self.atoms.intern_state(key, || StateAtom::Pred(be)))
    }

    fn intern_next_pred(&mut self, expr: &QuintEx) -> Result<EdgeAtomId, QuintError> {
        let key = EAtomKey::NextPred(expr.id(), self.bound_values());
        if let Some(id) = self.atoms.eatom_index.get(&key) {
            return Ok(*id);
        }
        let be = self.bound_expr(expr);
        Ok(self.atoms.intern_edge(key, || EdgeAtom::NextPred(be)))
    }

    fn intern_action(&mut self, expr: &QuintEx) -> Result<ActionId, QuintError> {
        let key = (expr.id(), self.bound_values());
        if let Some(id) = self.atoms.action_index.get(&key) {
            return Ok(*id);
        }
        let be = self.bound_expr(expr);
        let id = self.atoms.actions.len() as ActionId;
        self.atoms.actions.push(be);
        self.atoms.action_index.insert(key, id);
        Ok(id)
    }

    fn intern_vars(&mut self, expr: &QuintEx) -> Result<VarsId, QuintError> {
        let key = (expr.id(), self.bound_values());
        if let Some(id) = self.atoms.vars_index.get(&key) {
            return Ok(*id);
        }
        let be = self.bound_expr(expr);
        let id = self.atoms.vars_exprs.len() as VarsId;
        self.atoms.vars_exprs.push(be);
        self.atoms.vars_index.insert(key, id);
        Ok(id)
    }

    // ---------------------------------------------------------------
    // Quantifier machinery
    // ---------------------------------------------------------------

    fn lambda_parts(
        &self,
        expr: &QuintEx,
    ) -> Result<(Vec<quint_ast::LambdaParam>, QuintEx), QuintError> {
        match expr {
            QuintEx::Lambda { params, expr, .. } => Ok((params.clone(), (**expr).clone())),
            _ => Err(err("quantifier body must be a lambda")),
        }
    }

    fn push_binding(&mut self, param: &quint_ast::LambdaParam, value: Value) -> Result<(), QuintError> {
        self.push_binding_param(param, value);
        Ok(())
    }

    fn push_binding_param(&mut self, param: &quint_ast::LambdaParam, value: Value) {
        let register = self.compiler.param_register(param);
        self.bindings.push((param.id, register, value));
    }

    /// Evaluate a state-independent expression (e.g. a quantifier domain).
    fn eval_pure(&mut self, expr: &QuintEx) -> Result<Value, QuintError> {
        let compiled = self.compiler.compile(expr);
        let be = BoundExpr {
            expr: compiled,
            bindings: self
                .bindings
                .iter()
                .map(|(_, reg, v)| (reg.clone(), *v))
                .collect(),
        };
        let storage = self.compiler.storage();
        storage.borrow().clear_current();
        let mut env = Env::new(storage, None);
        be.eval(&mut env).map_err(|e| {
            if e.code == "QNT502" {
                err("temporal quantification over a state-dependent set is not supported")
            } else {
                e
            }
        })
    }

    fn eval_constant_set(&mut self, expr: &QuintEx) -> Result<Vec<Value>, QuintError> {
        let set = self.eval_pure(expr)?;
        Ok(set.enumerate()?.iter().cloned().collect())
    }
}

impl TemporalKind {
    fn max_with(self, other: TemporalKind) -> TemporalKind {
        use TemporalKind::*;
        match (self, other) {
            (Temporal, _) | (_, Temporal) => Temporal,
            (NextOnly, _) | (_, NextOnly) => NextOnly,
            _ => Pure,
        }
    }
}

impl AtomTable {
    fn intern_state(&mut self, key: SAtomKey, mk: impl FnOnce() -> StateAtom) -> StateAtomId {
        if let Some(id) = self.satom_index.get(&key) {
            return *id;
        }
        let id = self.state_atoms.len() as StateAtomId;
        self.state_atoms.push(mk());
        self.satom_index.insert(key, id);
        id
    }

    fn intern_edge(&mut self, key: EAtomKey, mk: impl FnOnce() -> EdgeAtom) -> EdgeAtomId {
        if let Some(id) = self.eatom_index.get(&key) {
            return *id;
        }
        let id = self.edge_atoms.len() as EdgeAtomId;
        self.edge_atoms.push(mk());
        self.eatom_index.insert(key, id);
        id
    }
}
