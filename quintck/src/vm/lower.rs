//! Lowering of Quint IR expressions to VM bytecode.
//!
//! Mirrors the closure compiler (`eval::compile`) construct for construct;
//! in particular the **emit order equals the closure engine's evaluation
//! order**, which is what keeps `ChoiceCtl` replay deterministic across
//! engines (see `choice.rs`). Do not reorder operand lowering.
//!
//! Register discipline: **every register is written before it is read, on
//! every control path** — `lower_into` always writes `dst`, argument blocks
//! are filled before `Call`/`Builtin`, and every jump target reading a
//! register is dominated by a write to it. The VM relies on this to reuse
//! register-file slots without clearing them between calls.

use super::builtins::vm_builtin;
use super::{BuiltinCall, CallSite, Chunk, FnId, Instr, MatchTable, Op, Program, Vm};
use crate::eval::{BoundExpr, Env};
use crate::state::VarTable;
use crate::value::{EvalResult, Value};
use quint_ast::{
    Declaration, LambdaParam, LookupDefinition, LookupTable, OpDef, OpQualifier, QuintEx, QuintId,
    Symbol,
};
use rustc_hash::{FxHashMap, FxHashSet};

pub struct Lowerer<'t> {
    table: &'t LookupTable,
    /// The program under construction (immutable once `finish`ed).
    program: Program,
    var_index: FxHashMap<QuintId, usize>,
    param_slots: FxHashMap<QuintId, u32>,
    let_cells: FxHashMap<QuintId, u32>,
    def_refs: FxHashMap<QuintId, DefRef>,
    /// Lambda literal node id → interned lambda value + body function.
    lambda_memo: FxHashMap<QuintId, (Value, FnId)>,
    /// Top-level (zero-param) function per expression node id.
    entry_memo: FxHashMap<QuintId, FnId>,
    /// Guard-hoisting analysis memo (expression node id → info).
    conj_memo: FxHashMap<QuintId, ConjInfo>,
}

/// What a conjunct may do, for guard hoisting: whether it is pure and
/// deterministic (no state writes, no choices, no run/temporal operators),
/// and which let-scoped definitions it (transitively) references.
#[derive(Clone, Default)]
struct ConjInfo {
    pure: bool,
    refs: FxHashSet<QuintId>,
}

/// How a definition is referenced in expression position.
#[derive(Clone)]
enum DefRef {
    /// Plain zero-arg call, evaluated per reference.
    Fn(FnId),
    /// Let-scoped call-by-need cell (also the nondet binding cell).
    CachedLet(FnId, u32),
    /// Per-state cache (bypassed under next_mode).
    CachedVal(FnId, u32),
    /// Persistent cache.
    CachedPure(FnId, u32),
    /// A lambda constant.
    Lambda(Value, FnId),
}

/// Per-function bytecode builder with a stack-discipline register
/// allocator.
struct FnB {
    code: Vec<Instr>,
    node_ids: Vec<QuintId>,
    cur_node: QuintId,
    next_reg: u16,
    max_reg: u16,
    param_slots: Vec<u32>,
    branch_tables: Vec<Box<[u16]>>,
    match_tables: Vec<MatchTable>,
    builtin_calls: Vec<BuiltinCall>,
    call_sites: Vec<CallSite>,
    field_syms: Vec<Symbol>,
}

impl FnB {
    fn new(param_slots: Vec<u32>, node: QuintId) -> Self {
        FnB {
            code: Vec::new(),
            node_ids: Vec::new(),
            cur_node: node,
            next_reg: 0,
            max_reg: 0,
            param_slots,
            branch_tables: Vec::new(),
            match_tables: Vec::new(),
            builtin_calls: Vec::new(),
            call_sites: Vec::new(),
            field_syms: Vec::new(),
        }
    }

    fn emit(&mut self, op: Op, a: u16, b: u16, c: u16) -> usize {
        self.code.push(Instr { op, a, b, c });
        self.node_ids.push(self.cur_node);
        self.code.len() - 1
    }

    fn pc(&self) -> u16 {
        u16::try_from(self.code.len()).expect("function too large for the VM")
    }

    fn patch_b(&mut self, idx: usize, pc: u16) {
        self.code[idx].b = pc;
    }

    fn patch_c(&mut self, idx: usize, pc: u16) {
        self.code[idx].c = pc;
    }

    fn alloc(&mut self, n: u16) -> u16 {
        let r = self.next_reg;
        self.next_reg = self
            .next_reg
            .checked_add(n)
            .expect("register overflow in the VM lowerer");
        self.max_reg = self.max_reg.max(self.next_reg);
        r
    }

    fn free_to(&mut self, mark: u16) {
        self.next_reg = mark;
    }

    fn finish(self) -> Chunk {
        assert!(self.code.len() <= u16::MAX as usize, "function too large");
        Chunk {
            code: self.code,
            node_ids: self.node_ids,
            nregs: self.max_reg,
            param_slots: self.param_slots.into_boxed_slice(),
            branch_tables: self.branch_tables,
            match_tables: self.match_tables,
            builtin_calls: self.builtin_calls,
            call_sites: self.call_sites,
            field_syms: self.field_syms,
        }
    }
}

fn u16_of(n: usize, what: &str) -> u16 {
    u16::try_from(n).unwrap_or_else(|_| panic!("{what} overflows the VM's 16-bit operand"))
}

/// CallValue packs (argbase, nargs) into the c operand so the JIT can
/// stage the arguments without runtime arity discovery.
fn pack_callvalue(argbase: u16, nargs: usize) -> u16 {
    assert!(argbase < (1 << 11), "argbase overflows CallValue encoding");
    assert!(nargs <= crate::vm::MAX_PARAMS, "too many call arguments");
    (argbase << 5) | nargs as u16
}

impl<'t> Lowerer<'t> {
    pub fn new(table: &'t LookupTable, vars: &VarTable) -> Self {
        let program = Program {
            funcs: Vec::new(),
            param_names: Vec::new(),
            var_names: vars.names.clone(),
            n_let_cells: 0,
            n_val_cells: 0,
            n_pure_cells: 0,
            errors: Vec::new(),
        };
        Lowerer {
            table,
            program,
            var_index: vars.by_def_id.clone(),
            param_slots: FxHashMap::default(),
            let_cells: FxHashMap::default(),
            def_refs: FxHashMap::default(),
            lambda_memo: FxHashMap::default(),
            entry_memo: FxHashMap::default(),
            conj_memo: FxHashMap::default(),
        }
    }

    /// Analyze a conjunct for guard hoisting (memoized by node id).
    /// Conservative: anything unresolvable or dynamic counts as impure.
    fn conj_info(&mut self, expr: &QuintEx) -> ConjInfo {
        let id = expr.id();
        if let Some(info) = self.conj_memo.get(&id) {
            return info.clone();
        }
        // provisional pessimistic entry (cycle guard; quint has no
        // recursion, this is defensive)
        self.conj_memo.insert(id, ConjInfo::default());
        let info = self.conj_info_core(expr);
        self.conj_memo.insert(id, info.clone());
        info
    }

    fn conj_info_core(&mut self, expr: &QuintEx) -> ConjInfo {
        /// Operators that make an expression non-hoistable: state writes,
        /// choice points, run-test control, next(), and output.
        const IMPURE_OPS: &[&str] = &[
            "assign",
            "actionAny",
            "oneOf",
            "next",
            "then",
            "reps",
            "expect",
            "q::debug",
        ];
        let mut info = ConjInfo {
            pure: true,
            refs: FxHashSet::default(),
        };
        let merge = |info: &mut ConjInfo, other: ConjInfo| {
            info.pure &= other.pure;
            info.refs.extend(other.refs);
        };
        match expr {
            QuintEx::Int { .. } | QuintEx::Bool { .. } | QuintEx::Str { .. } => {}
            QuintEx::Name { id, .. } => match self.table.get(id) {
                Some(LookupDefinition::Definition(Declaration::OpDef(op))) => {
                    let op = op.clone();
                    let scoped = op.depth.is_some_and(|d| d != 0);
                    merge(&mut info, self.conj_info(&op.expr));
                    if scoped {
                        info.refs.insert(op.id);
                    }
                }
                Some(LookupDefinition::Definition(
                    Declaration::Var { .. } | Declaration::Const { .. },
                ))
                | Some(LookupDefinition::Param(_))
                | None => {}
                _ => info.pure = false,
            },
            QuintEx::Lambda { expr, .. } => merge(&mut info, self.conj_info(expr)),
            QuintEx::App { id, opcode, args } => {
                if IMPURE_OPS.contains(&opcode.as_str()) {
                    info.pure = false;
                }
                match self.table.get(id) {
                    // user-defined operator: analyze its body
                    Some(LookupDefinition::Definition(Declaration::OpDef(op))) => {
                        let op = op.clone();
                        let scoped = op.depth.is_some_and(|d| d != 0);
                        merge(&mut info, self.conj_info(&op.expr));
                        if scoped {
                            info.refs.insert(op.id);
                        }
                    }
                    // applying a parameter-held operator: its body is
                    // unknown here — conservative
                    Some(LookupDefinition::Param(_)) => info.pure = false,
                    Some(_) => info.pure = false,
                    None => {} // builtin
                }
                for arg in args {
                    merge(&mut info, self.conj_info(arg));
                }
            }
            QuintEx::Let { opdef, expr, .. } => {
                if opdef.qualifier == OpQualifier::Nondet {
                    info.pure = false;
                }
                merge(&mut info, self.conj_info(&opdef.expr.clone()));
                merge(&mut info, self.conj_info(expr));
            }
        }
        info
    }

    /// Hand over the finished (immutable) program.
    pub fn finish(self) -> Program {
        self.program
    }

    /// Evaluate a state-independent expression during lowering (temporal
    /// quantifier domains): runs a throwaway Vm over the program built so
    /// far, with the given parameter bindings installed.
    pub fn eval_bound(&self, fnid: FnId, bindings: &[(u32, Value)]) -> EvalResult {
        let mut vm = Vm::new(&self.program);
        let be = BoundExpr {
            fnid,
            bindings: bindings.to_vec(),
        };
        let mut env = Env::new(None);
        be.eval(&mut vm, &mut env)
    }


    pub fn param_slot(&mut self, param: &LambdaParam) -> u32 {
        if let Some(&s) = self.param_slots.get(&param.id) {
            return s;
        }
        let slot = self.program.param_names.len() as u32;
        self.program.param_names.push(param.name);
        self.param_slots.insert(param.id, slot);
        slot
    }

    fn let_cell(&mut self, id: QuintId) -> u32 {
        if let Some(&c) = self.let_cells.get(&id) {
            return c;
        }
        let cell = self.program.n_let_cells as u32;
        self.program.n_let_cells += 1;
        self.let_cells.insert(id, cell);
        cell
    }

    fn error_idx(&mut self, code: &'static str, msg: String) -> u16 {
        self.program.errors.push((code, msg));
        u16_of(self.program.errors.len() - 1, "error table")
    }

    /// Lower an expression as a zero-parameter function (memoized).
    pub fn lower_entry(&mut self, expr: &QuintEx) -> FnId {
        if let Some(&f) = self.entry_memo.get(&expr.id()) {
            return f;
        }
        let f = self.lower_fn(expr, Vec::new());
        self.entry_memo.insert(expr.id(), f);
        f
    }

    fn lower_fn(&mut self, expr: &QuintEx, param_slots: Vec<u32>) -> FnId {
        let mut f = FnB::new(param_slots, expr.id());
        let dst = f.alloc(1);
        self.lower_into(&mut f, expr, dst);
        f.cur_node = expr.id();
        f.emit(Op::Ret, dst, 0, 0);
        let chunk = f.finish();
        self.program.funcs.push(chunk);
        u16_of(self.program.funcs.len() - 1, "function table") as FnId
    }

    /// The interned lambda constant for a `Lambda` literal (body lowered
    /// once; parameters are shared cells).
    fn lambda_const(&mut self, expr: &QuintEx) -> (Value, FnId) {
        let id = expr.id();
        if let Some(v) = self.lambda_memo.get(&id) {
            return *v;
        }
        let QuintEx::Lambda { params, expr: body, .. } = expr else {
            panic!("lambda_const on a non-lambda")
        };
        assert!(params.len() <= super::MAX_PARAMS, "too many lambda params");
        let slots: Vec<u32> = params.iter().map(|p| self.param_slot(p)).collect();
        let fnid = self.lower_fn(body, slots.clone());
        let value = Value::lambda(slots.into_boxed_slice(), fnid);
        self.lambda_memo.insert(id, (value, fnid));
        (value, fnid)
    }

    fn def_ref(&mut self, op: &OpDef) -> DefRef {
        if let Some(r) = self.def_refs.get(&op.id) {
            return r.clone();
        }
        let top_level = op.depth.is_none_or(|d| d == 0);
        let r = if matches!(op.expr, QuintEx::Lambda { .. }) {
            let (v, f) = self.lambda_const(&op.expr);
            DefRef::Lambda(v, f)
        } else if top_level {
            match op.qualifier {
                OpQualifier::Val => {
                    let f = self.lower_fn(&op.expr, Vec::new());
                    let cell = self.program.n_val_cells as u32;
                    self.program.n_val_cells += 1;
                    DefRef::CachedVal(f, cell)
                }
                OpQualifier::PureVal => {
                    let f = self.lower_fn(&op.expr, Vec::new());
                    let cell = self.program.n_pure_cells as u32;
                    self.program.n_pure_cells += 1;
                    DefRef::CachedPure(f, cell)
                }
                _ => DefRef::Fn(self.lower_fn(&op.expr, Vec::new())),
            }
        } else {
            // Scoped (let-bound) definition: call-by-need through the shared
            // cell. For nondet bindings the cell is written by OneOfBind
            // before the body runs.
            let f = self.lower_fn(&op.expr, Vec::new());
            DefRef::CachedLet(f, self.let_cell(op.id))
        };
        self.def_refs.insert(op.id, r.clone());
        r
    }

    fn emit_load_imm(&mut self, f: &mut FnB, dst: u16, v: Value) {
        let bits = v.to_bits();
        f.emit(Op::LoadImm, dst, bits as u16, (bits >> 16) as u16);
    }

    fn emit_fail(&mut self, f: &mut FnB, code: &'static str, msg: String) {
        let idx = self.error_idx(code, msg);
        f.emit(Op::Fail, idx, 0, 0);
    }

    /// Emit code producing a definition's value into `dst`.
    fn lower_def_value(&mut self, f: &mut FnB, def: &LookupDefinition, dst: u16) {
        match def {
            LookupDefinition::Definition(Declaration::OpDef(op)) => match self.def_ref(op) {
                DefRef::Lambda(v, _) => self.emit_load_imm(f, dst, v),
                DefRef::Fn(fnid) => {
                    let cs = self.call_site(f, fnid);
                    f.emit(Op::Call, dst, cs, 0);
                }
                DefRef::CachedLet(fnid, cell) => {
                    f.emit(
                        Op::CallCached,
                        dst,
                        u16_of(fnid as usize, "fn id"),
                        u16_of(cell as usize, "let cell"),
                    );
                }
                DefRef::CachedVal(fnid, cell) => {
                    f.emit(
                        Op::CallCachedVal,
                        dst,
                        u16_of(fnid as usize, "fn id"),
                        u16_of(cell as usize, "val cell"),
                    );
                }
                DefRef::CachedPure(fnid, cell) => {
                    f.emit(
                        Op::CallCachedPure,
                        dst,
                        u16_of(fnid as usize, "fn id"),
                        u16_of(cell as usize, "pure cell"),
                    );
                }
            },
            LookupDefinition::Definition(Declaration::Var { id, name }) => {
                let index = *self
                    .var_index
                    .get(id)
                    .unwrap_or_else(|| panic!("unknown variable {name} (id {id})"));
                f.emit(Op::LoadVar, dst, u16_of(index, "var index"), 0);
            }
            LookupDefinition::Definition(Declaration::Const { name, .. }) => {
                self.emit_fail(
                    f,
                    "QNT500",
                    format!(
                        "Uninitialized const {name}. Use: import <moduleName>({name}=<value>).*"
                    ),
                );
            }
            LookupDefinition::Param(p) => {
                let slot = self.param_slot(p);
                f.emit(Op::LoadParam, dst, u16_of(slot as usize, "param slot"), 0);
            }
            d => panic!("cannot compile reference to {d:?}"),
        }
    }

    fn lower_into(&mut self, f: &mut FnB, expr: &QuintEx, dst: u16) {
        let saved_node = f.cur_node;
        f.cur_node = expr.id();
        match expr {
            QuintEx::Int { value, .. } => self.emit_load_imm(f, dst, Value::int(*value)),
            QuintEx::Bool { value, .. } => self.emit_load_imm(f, dst, Value::bool(*value)),
            QuintEx::Str { value, .. } => self.emit_load_imm(f, dst, Value::str(*value)),

            QuintEx::Name { id, name } => match self.table.get(id) {
                Some(def) => {
                    let def = def.clone();
                    self.lower_def_value(f, &def, dst);
                }
                None => match name.as_str() {
                    "true" => self.emit_load_imm(f, dst, Value::bool(true)),
                    "false" => self.emit_load_imm(f, dst, Value::bool(false)),
                    "Bool" => {
                        let v = Value::set([Value::bool(false), Value::bool(true)])
                            .expect("Bool set");
                        self.emit_load_imm(f, dst, v);
                    }
                    "Int" => self.emit_load_imm(f, dst, Value::infinite_int()),
                    "Nat" => self.emit_load_imm(f, dst, Value::infinite_nat()),
                    other => {
                        self.emit_fail(f, "QNT000", format!("unknown builtin name: {other}"))
                    }
                },
            },

            QuintEx::Lambda { .. } => {
                let (v, _) = self.lambda_const(expr);
                self.emit_load_imm(f, dst, v);
            }

            QuintEx::App { id, opcode, args } => {
                self.lower_app(f, *id, opcode.as_str(), args, dst);
            }

            QuintEx::Let { opdef, expr, .. } => self.lower_let(f, opdef, expr, dst),
        }
        f.cur_node = saved_node;
    }

    fn lower_app(&mut self, f: &mut FnB, id: QuintId, opcode: &str, args: &[QuintEx], dst: u16) {
        match opcode {
            "assign" => {
                let var_def = self
                    .table
                    .get(&args[0].id())
                    .unwrap_or_else(|| panic!("assign target not in lookup table"))
                    .clone();
                let (var_id, var_name) = match &var_def {
                    LookupDefinition::Definition(Declaration::Var { id, name }) => (*id, *name),
                    d => panic!("assign target is not a variable: {d:?}"),
                };
                let index = *self
                    .var_index
                    .get(&var_id)
                    .unwrap_or_else(|| panic!("unknown variable {var_name}"));
                let mark = f.next_reg;
                let r = f.alloc(1);
                self.lower_into(f, &args[1], r);
                f.emit(Op::Assign, dst, u16_of(index, "var index"), r);
                f.free_to(mark);
            }

            "or" | "actionAny" if args.is_empty() => {
                self.emit_load_imm(f, dst, Value::bool(false));
            }

            "and" => self.lower_and(f, args, dst),

            "or" => {
                let mut jumps = Vec::new();
                for (i, arg) in args.iter().enumerate() {
                    self.lower_into(f, arg, dst);
                    if i + 1 < args.len() {
                        jumps.push(f.emit(Op::JumpIfTrue, dst, 0, 0));
                    }
                }
                let join = f.pc();
                for j in jumps {
                    f.patch_b(j, join);
                }
            }

            "implies" => {
                let mut guards = Vec::new();
                if !self.try_lower_guard(f, &args[0], &mut guards) {
                    self.lower_into(f, &args[0], dst);
                    guards.clear();
                    let j_false = f.emit(Op::JumpIfFalse, dst, 0, 0);
                    self.lower_into(f, &args[1], dst);
                    let j_join = f.emit(Op::Jump, 0, 0, 0);
                    f.patch_b(j_false, f.pc());
                    self.emit_load_imm(f, dst, Value::bool(true));
                    f.patch_b(j_join, f.pc());
                } else {
                    self.lower_into(f, &args[1], dst);
                    let j_join = f.emit(Op::Jump, 0, 0, 0);
                    let vac = f.pc();
                    for g in guards {
                        f.patch_c(g, vac);
                    }
                    self.emit_load_imm(f, dst, Value::bool(true));
                    f.patch_b(j_join, f.pc());
                }
            }

            "ite" => {
                let mut guards = Vec::new();
                if self.try_lower_guard(f, &args[0], &mut guards) {
                    self.lower_into(f, &args[1], dst);
                    let j_join = f.emit(Op::Jump, 0, 0, 0);
                    let els = f.pc();
                    for g in guards {
                        f.patch_c(g, els);
                    }
                    self.lower_into(f, &args[2], dst);
                    f.patch_b(j_join, f.pc());
                } else {
                    self.lower_into(f, &args[0], dst);
                    let j_else = f.emit(Op::JumpIfFalse, dst, 0, 0);
                    self.lower_into(f, &args[1], dst);
                    let j_join = f.emit(Op::Jump, 0, 0, 0);
                    f.patch_b(j_else, f.pc());
                    self.lower_into(f, &args[2], dst);
                    f.patch_b(j_join, f.pc());
                }
            }

            "actionAll" => self.lower_action_all(f, args, dst),

            "actionAny" => {
                let tbl = u16_of(f.branch_tables.len(), "branch table");
                f.branch_tables.push(Box::new([]));
                let begin = f.emit(Op::AnyBegin, dst, tbl, 0);
                let mut pcs = Vec::with_capacity(args.len());
                for arg in args {
                    pcs.push(f.pc());
                    self.lower_into(f, arg, dst);
                    f.emit(Op::AnyEnd, dst, 0, 0);
                }
                f.branch_tables[tbl as usize] = pcs.into_boxed_slice();
                let join = f.pc();
                f.patch_c(begin, join);
            }

            "oneOf" => {
                let mark = f.next_reg;
                let r = f.alloc(1);
                self.lower_into(f, &args[0], r);
                f.emit(Op::OneOfPick, dst, r, 0);
                f.free_to(mark);
            }

            "next" => {
                f.emit(Op::NextEnter, 0, 0, 0);
                self.lower_into(f, &args[0], dst);
                f.emit(Op::NextExit, 0, 0, 0);
            }

            "then" => {
                self.lower_into(f, &args[0], dst);
                let j_fail = f.emit(Op::JumpIfFalse, dst, 0, 0);
                f.emit(Op::Shift, 0, 0, 0);
                self.lower_into(f, &args[1], dst);
                let j_join = f.emit(Op::Jump, 0, 0, 0);
                f.patch_b(j_fail, f.pc());
                self.emit_fail(
                    f,
                    "QNT513",
                    "Cannot continue in `then` because the highlighted expression evaluated \
                     to false"
                        .to_string(),
                );
                let join = f.pc();
                f.patch_b(j_join, join);
            }

            "reps" => {
                let mark = f.next_reg;
                let r_n = f.alloc(1);
                let r_lam = f.alloc(1);
                self.lower_into(f, &args[0], r_n);
                self.lower_into(f, &args[1], r_lam);
                f.emit(Op::Reps, dst, r_n, r_lam);
                f.free_to(mark);
            }

            "expect" => {
                let mark = f.next_reg;
                self.lower_into(f, &args[0], dst);
                let j_fail1 = f.emit(Op::JumpIfFalse, dst, 0, 0);
                f.emit(Op::SnapNext, 0, 0, 0);
                f.emit(Op::Shift, 0, 0, 0);
                let r = f.alloc(1);
                self.lower_into(f, &args[1], r);
                f.emit(Op::RestoreSnapPop, 0, 0, 0);
                let j_fail2 = f.emit(Op::JumpIfFalse, r, 0, 0);
                self.emit_load_imm(f, dst, Value::bool(true));
                let j_join = f.emit(Op::Jump, 0, 0, 0);
                f.patch_b(j_fail1, f.pc());
                self.emit_fail(f, "QNT508", "Cannot continue to \"expect\"".to_string());
                f.patch_b(j_fail2, f.pc());
                self.emit_fail(f, "QNT508", "Expect condition does not hold true".to_string());
                let join = f.pc();
                f.patch_b(j_join, join);
                f.free_to(mark);
            }

            "matchVariant" => self.lower_match(f, args, dst),

            _ => self.lower_call_or_builtin(f, id, opcode, args, dst),
        }
    }

    /// Resolve a static call target into a call-site table entry with the
    /// callee's parameter slots pre-copied.
    fn call_site(&mut self, f: &mut FnB, fnid: FnId) -> u16 {
        let slots = self.program.funcs[fnid as usize].param_slots.clone();
        f.call_sites.push(CallSite { fnid, slots });
        u16_of(f.call_sites.len() - 1, "call site table")
    }

    /// Fused guard: if `expr` is a two-argument comparison builtin, lower
    /// its operands and emit a single Guard instruction that jumps to the
    /// (to-be-patched, via `patch_c`) else-target when the comparison is
    /// false. Returns false when the expression doesn't match.
    fn try_lower_guard(&mut self, f: &mut FnB, expr: &QuintEx, guard_jumps: &mut Vec<usize>) -> bool {
        let QuintEx::App { id, opcode, args } = expr else {
            return false;
        };
        if args.len() != 2 || self.table.get(id).is_some() {
            return false; // user-defined op shadowing a builtin name
        }
        let op = match opcode.as_str() {
            "eq" => Op::GuardEq,
            "neq" => Op::GuardNeq,
            "ilt" => Op::GuardLt,
            "ilte" => Op::GuardLte,
            "igt" => Op::GuardGt,
            "igte" => Op::GuardGte,
            _ => return false,
        };
        let saved_node = f.cur_node;
        f.cur_node = expr.id();
        let mark = f.next_reg;
        let rx = f.alloc(1);
        self.lower_into(f, &args[0], rx);
        let ry = f.alloc(1);
        self.lower_into(f, &args[1], ry);
        guard_jumps.push(f.emit(op, rx, ry, 0));
        f.free_to(mark);
        f.cur_node = saved_node;
        true
    }

    /// Short-circuit conjunction (empty ⇒ true).
    fn lower_and(&mut self, f: &mut FnB, args: &[QuintEx], dst: u16) {
        if args.is_empty() {
            self.emit_load_imm(f, dst, Value::bool(true));
            return;
        }
        let mut jumps = Vec::new(); // JumpIfFalse (dst already false)
        let mut guards = Vec::new(); // fused Guard* (dst not written)
        for (i, arg) in args.iter().enumerate() {
            if i + 1 < args.len() && self.try_lower_guard(f, arg, &mut guards) {
                continue;
            }
            self.lower_into(f, arg, dst);
            if i + 1 < args.len() {
                jumps.push(f.emit(Op::JumpIfFalse, dst, 0, 0));
            }
        }
        if guards.is_empty() {
            let join = f.pc();
            for j in jumps {
                f.patch_b(j, join);
            }
        } else {
            // fused guards need a false-trampoline (they don't write dst)
            let j_ok = f.emit(Op::Jump, 0, 0, 0);
            let false_pc = f.pc();
            for g in guards {
                f.patch_c(g, false_pc);
            }
            self.emit_load_imm(f, dst, Value::bool(false));
            let join = f.pc();
            f.patch_b(j_ok, join);
            for j in jumps {
                f.patch_b(j, join);
            }
        }
    }

    /// Action conjunction: roll back next-state writes if any conjunct is
    /// disabled (empty ⇒ true).
    fn lower_action_all(&mut self, f: &mut FnB, args: &[QuintEx], dst: u16) {
        if args.is_empty() {
            self.emit_load_imm(f, dst, Value::bool(true));
            return;
        }
        f.emit(Op::SnapNext, 0, 0, 0);
        let mut fails = Vec::new(); // JumpIfFalse
        let mut guards = Vec::new(); // fused Guard*
        let last = args.len() - 1;
        for (i, arg) in args.iter().enumerate() {
            if i < last && self.try_lower_guard(f, arg, &mut guards) {
                continue;
            }
            self.lower_into(f, arg, dst);
            fails.push(f.emit(Op::JumpIfFalse, dst, 0, 0));
        }
        f.emit(Op::DropSnap, 0, 0, 0);
        let j_join = f.emit(Op::Jump, 0, 0, 0);
        let rollback = f.pc();
        for j in fails {
            f.patch_b(j, rollback);
        }
        for g in guards {
            f.patch_c(g, rollback);
        }
        // fused guards don't write dst, so set false explicitly here
        self.emit_load_imm(f, dst, Value::bool(false));
        f.emit(Op::RestoreSnapPop, 0, 0, 0);
        let join = f.pc();
        f.patch_b(j_join, join);
    }

    fn lower_match(&mut self, f: &mut FnB, args: &[QuintEx], dst: u16) {
        let mark = f.next_reg;
        let r_subj = f.alloc(1);
        self.lower_into(f, &args[0], r_subj);
        let r_payload = f.alloc(1);
        let tbl = u16_of(f.match_tables.len(), "match table");
        f.match_tables.push(MatchTable { cases: Box::new([]) });
        f.emit(Op::MatchVariant, r_payload, r_subj, tbl);

        let mut cases = Vec::new();
        let mut joins = Vec::new();
        for case in args[1..].chunks_exact(2) {
            let QuintEx::Str { value: label, .. } = &case[0] else {
                // The quint compiler always emits literal labels.
                self.emit_fail(
                    f,
                    "QNT501",
                    "non-literal matchVariant label is not supported".to_string(),
                );
                continue;
            };
            let case_pc = f.pc();
            match &case[1] {
                QuintEx::Lambda { params, expr, .. } if params.len() == 1 => {
                    // Inline the eliminator: bind the payload to the
                    // parameter cell around the body.
                    let slot = self.param_slot(&params[0]);
                    let slot16 = u16_of(slot as usize, "param slot");
                    f.emit(Op::ParamEnter, slot16, r_payload, 0);
                    self.lower_into(f, expr, dst);
                    f.emit(Op::ParamExit, slot16, 0, 0);
                }
                elim => {
                    let m = f.next_reg;
                    let r_lam = f.alloc(1);
                    self.lower_into(f, elim, r_lam);
                    f.emit(Op::CallValue, dst, r_lam, pack_callvalue(r_payload, 1));
                    f.free_to(m);
                }
            }
            joins.push(f.emit(Op::Jump, 0, 0, 0));
            cases.push((*label, case_pc));
        }
        f.match_tables[tbl as usize].cases = cases.into_boxed_slice();
        let join = f.pc();
        for j in joins {
            f.patch_b(j, join);
        }
        f.free_to(mark);
    }

    fn lower_call_or_builtin(
        &mut self,
        f: &mut FnB,
        id: QuintId,
        opcode: &str,
        args: &[QuintEx],
        dst: u16,
    ) {
        match self.table.get(&id) {
            // User-defined operator application: arguments evaluate first,
            // then the operator (closure-engine order).
            Some(def) => {
                let def = def.clone();
                let mark = f.next_reg;
                let ab = f.alloc(u16_of(args.len(), "argument count"));
                for (i, arg) in args.iter().enumerate() {
                    self.lower_into(f, arg, ab + i as u16);
                }
                let direct = match &def {
                    LookupDefinition::Definition(Declaration::OpDef(op)) => {
                        match self.def_ref(op) {
                            DefRef::Lambda(_, fnid) => Some(fnid),
                            _ => None,
                        }
                    }
                    _ => None,
                };
                match direct {
                    Some(fnid) => {
                        let cs = self.call_site(f, fnid);
                        f.emit(Op::Call, dst, cs, ab);
                    }
                    None => {
                        let r_lam = f.alloc(1);
                        self.lower_def_value(f, &def, r_lam);
                        f.emit(Op::CallValue, dst, r_lam, pack_callvalue(ab, args.len()));
                    }
                }
                f.free_to(mark);
            }
            // Fused eager builtins: dedicated opcodes skip the call-site
            // machinery entirely.
            None if matches!(opcode, "eq" | "neq") && args.len() == 2 => {
                let mark = f.next_reg;
                let rx = f.alloc(1);
                self.lower_into(f, &args[0], rx);
                let ry = f.alloc(1);
                self.lower_into(f, &args[1], ry);
                let op = if opcode == "eq" { Op::Eq } else { Op::Neq };
                f.emit(op, dst, rx, ry);
                f.free_to(mark);
            }
            None if opcode == "not" && args.len() == 1 => {
                let mark = f.next_reg;
                let rx = f.alloc(1);
                self.lower_into(f, &args[0], rx);
                f.emit(Op::Not, dst, rx, 0);
                f.free_to(mark);
            }
            None if opcode == "field"
                && args.len() == 2
                && matches!(&args[1], QuintEx::Str { .. }) =>
            {
                let QuintEx::Str { value: sym, .. } = &args[1] else {
                    unreachable!()
                };
                let mark = f.next_reg;
                let r_rec = f.alloc(1);
                self.lower_into(f, &args[0], r_rec);
                let idx = u16_of(f.field_syms.len(), "field symbol table");
                f.field_syms.push(*sym);
                f.emit(Op::RecField, dst, r_rec, idx);
                f.free_to(mark);
            }
            // Eager builtin
            None => match vm_builtin(opcode) {
                Some(func) => {
                    let mark = f.next_reg;
                    let ab = f.alloc(u16_of(args.len(), "argument count"));
                    for (i, arg) in args.iter().enumerate() {
                        self.lower_into(f, arg, ab + i as u16);
                    }
                    let cs = u16_of(f.builtin_calls.len(), "builtin call table");
                    f.builtin_calls.push(BuiltinCall {
                        func,
                        argbase: ab,
                        nargs: u16_of(args.len(), "argument count"),
                    });
                    f.emit(Op::Builtin, dst, cs, 0);
                    f.free_to(mark);
                }
                None => {
                    self.emit_fail(f, "QNT000", format!("unknown operator: {opcode}"));
                }
            },
        }
    }

    fn lower_let(&mut self, f: &mut FnB, opdef: &OpDef, body: &QuintEx, dst: u16) {
        // nondet x = S.oneOf() — the enumerated choice point
        if opdef.qualifier == OpQualifier::Nondet {
            if let QuintEx::App { opcode, args, .. } = &opdef.expr {
                if opcode.as_str() == "oneOf" && args.len() == 1 {
                    let cell = self.let_cell(opdef.id);
                    let cell16 = u16_of(cell as usize, "let cell");
                    let mark = f.next_reg;
                    let r_set = f.alloc(1);
                    self.lower_into(f, &args[0], r_set);

                    // Guard hoisting: when the body is a conjunction, its
                    // leading pure conjuncts that don't reference this
                    // binding are evaluated *before* the choice point — a
                    // false guard then disables the whole action in one
                    // run instead of once per pick. Pure + choice-free, so
                    // the values and ChoiceCtl replay order are unchanged;
                    // evaluation order stays left-to-right (prefix only).
                    let (conj_kind, conj_args): (Option<&str>, &[QuintEx]) = match body {
                        QuintEx::App { opcode, args, .. }
                            if opcode.as_str() == "and" || opcode.as_str() == "actionAll" =>
                        {
                            (Some(opcode.as_str()), args)
                        }
                        _ => (None, &[]),
                    };
                    let mut hoist = 0;
                    if conj_kind.is_some() {
                        for arg in conj_args {
                            let info = self.conj_info(arg);
                            if info.pure && !info.refs.contains(&opdef.id) {
                                hoist += 1;
                            } else {
                                break;
                            }
                        }
                    }
                    let mut disabled_jumps = Vec::new();
                    let mut disabled_guards = Vec::new();
                    for arg in &conj_args[..hoist] {
                        if self.try_lower_guard(f, arg, &mut disabled_guards) {
                            continue;
                        }
                        self.lower_into(f, arg, dst);
                        disabled_jumps.push(f.emit(Op::JumpIfFalse, dst, 0, 0));
                    }

                    let bind = f.emit(Op::OneOfBind, cell16, r_set, 0);
                    match conj_kind {
                        Some("and") => self.lower_and(f, &conj_args[hoist..], dst),
                        Some("actionAll") => self.lower_action_all(f, &conj_args[hoist..], dst),
                        _ => self.lower_into(f, body, dst),
                    }
                    f.emit(Op::LetExit, cell16, 0, 0);
                    let j_join = f.emit(Op::Jump, 0, 0, 0);
                    // empty set or hoisted guard false: branch disabled
                    let disabled = f.pc();
                    f.patch_c(bind, disabled);
                    for j in disabled_jumps {
                        f.patch_b(j, disabled);
                    }
                    for g in disabled_guards {
                        f.patch_c(g, disabled);
                    }
                    self.emit_load_imm(f, dst, Value::bool(false));
                    let join = f.pc();
                    f.patch_b(j_join, join);
                    f.free_to(mark);
                    return;
                }
            }
        }

        // Regular let: call-by-need through the shared cell; save/restore
        // for reentrancy.
        let cell = self.let_cell(opdef.id);
        let cell16 = u16_of(cell as usize, "let cell");
        f.emit(Op::LetEnter, cell16, 0, 0);
        self.lower_into(f, body, dst);
        f.emit(Op::LetExit, cell16, 0, 0);
    }
}

impl Lowerer<'_> {
    /// Lower an expression as a callable zero-parameter function.
    pub fn compile(&mut self, expr: &QuintEx) -> FnId {
        self.lower_entry(expr)
    }
}
