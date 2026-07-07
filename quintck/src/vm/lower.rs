//! Lowering of Quint IR expressions to VM bytecode.
//!
//! Mirrors the closure compiler (`eval::compile`) construct for construct;
//! in particular the **emit order equals the closure engine's evaluation
//! order**, which is what keeps `ChoiceCtl` replay deterministic across
//! engines (see `choice.rs`). Do not reorder operand lowering.

use super::builtins::vm_builtin;
use super::{BuiltinCall, Chunk, FnId, Instr, MatchTable, Op, Program, Vm};
use crate::eval::CompiledExpr;
use crate::state::{Register, VarStorage, VarTable};
use crate::value::Value;
use quint_ast::{
    Declaration, LambdaParam, LookupDefinition, LookupTable, OpDef, OpQualifier, QuintEx, QuintId,
};
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::rc::Rc;

pub struct Lowerer<'t> {
    table: &'t LookupTable,
    pub storage: Rc<RefCell<VarStorage>>,
    var_index: FxHashMap<QuintId, usize>,
    vm: Rc<RefCell<Vm>>,
    param_slots: FxHashMap<QuintId, u32>,
    let_cells: FxHashMap<QuintId, u32>,
    def_refs: FxHashMap<QuintId, DefRef>,
    /// Lambda literal node id → interned lambda value + body function.
    lambda_memo: FxHashMap<QuintId, (Value, FnId)>,
    /// Top-level (zero-param) function per expression node id.
    entry_memo: FxHashMap<QuintId, FnId>,
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
        }
    }
}

fn u16_of(n: usize, what: &str) -> u16 {
    u16::try_from(n).unwrap_or_else(|_| panic!("{what} overflows the VM's 16-bit operand"))
}

impl<'t> Lowerer<'t> {
    pub fn new(table: &'t LookupTable, vars: &VarTable) -> Self {
        let storage = Rc::new(RefCell::new(VarStorage::new(vars)));
        let program = Program {
            funcs: Vec::new(),
            params: Vec::new(),
            param_names: Vec::new(),
            var_current: storage.borrow().current.clone(),
            var_next: storage.borrow().next.clone(),
            var_names: vars.names.clone(),
            errors: Vec::new(),
        };
        Lowerer {
            table,
            storage,
            var_index: vars.by_def_id.clone(),
            vm: Rc::new(RefCell::new(Vm::new(program))),
            param_slots: FxHashMap::default(),
            let_cells: FxHashMap::default(),
            def_refs: FxHashMap::default(),
            lambda_memo: FxHashMap::default(),
            entry_memo: FxHashMap::default(),
        }
    }

    pub fn vm_handle(&self) -> Rc<RefCell<Vm>> {
        self.vm.clone()
    }

    fn param_slot(&mut self, param: &LambdaParam) -> u32 {
        if let Some(&s) = self.param_slots.get(&param.id) {
            return s;
        }
        let mut vm = self.vm.borrow_mut();
        let slot = vm.program.params.len() as u32;
        vm.program.params.push(Rc::new(std::cell::Cell::new(None)));
        vm.program.param_names.push(param.name);
        drop(vm);
        self.param_slots.insert(param.id, slot);
        slot
    }

    fn let_cell(&mut self, id: QuintId) -> u32 {
        if let Some(&c) = self.let_cells.get(&id) {
            return c;
        }
        let cell = self.vm.borrow_mut().add_let_cell();
        self.let_cells.insert(id, cell);
        cell
    }

    fn error_idx(&mut self, code: &'static str, msg: String) -> u16 {
        let mut vm = self.vm.borrow_mut();
        vm.program.errors.push((code, msg));
        u16_of(vm.program.errors.len() - 1, "error table")
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
        let mut vm = self.vm.borrow_mut();
        vm.program.funcs.push(Rc::new(chunk));
        u16_of(vm.program.funcs.len() - 1, "function table") as FnId
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
        let registers: Vec<Register> = {
            let vm = self.vm.borrow();
            slots
                .iter()
                .map(|&s| vm.program.params[s as usize].clone())
                .collect()
        };
        let value = Value::lambda(registers, fnid);
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
                    DefRef::CachedVal(f, self.vm.borrow_mut().add_val_cell())
                }
                OpQualifier::PureVal => {
                    let f = self.lower_fn(&op.expr, Vec::new());
                    DefRef::CachedPure(f, self.vm.borrow_mut().add_pure_cell())
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
                    f.emit(Op::Call, dst, u16_of(fnid as usize, "fn id"), 0);
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

            "and" | "actionAll" if args.is_empty() => {
                self.emit_load_imm(f, dst, Value::bool(true));
            }
            "or" | "actionAny" if args.is_empty() => {
                self.emit_load_imm(f, dst, Value::bool(false));
            }

            "and" => {
                let mut jumps = Vec::new();
                for (i, arg) in args.iter().enumerate() {
                    self.lower_into(f, arg, dst);
                    if i + 1 < args.len() {
                        jumps.push(f.emit(Op::JumpIfFalse, dst, 0, 0));
                    }
                }
                let join = f.pc();
                for j in jumps {
                    f.patch_b(j, join);
                }
            }

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
                self.lower_into(f, &args[0], dst);
                let j_false = f.emit(Op::JumpIfFalse, dst, 0, 0);
                self.lower_into(f, &args[1], dst);
                let j_join = f.emit(Op::Jump, 0, 0, 0);
                f.patch_b(j_false, f.pc());
                self.emit_load_imm(f, dst, Value::bool(true));
                let join = f.pc();
                f.patch_b(j_join, join);
            }

            "ite" => {
                self.lower_into(f, &args[0], dst);
                let j_else = f.emit(Op::JumpIfFalse, dst, 0, 0);
                self.lower_into(f, &args[1], dst);
                let j_join = f.emit(Op::Jump, 0, 0, 0);
                f.patch_b(j_else, f.pc());
                self.lower_into(f, &args[2], dst);
                let join = f.pc();
                f.patch_b(j_join, join);
            }

            "actionAll" => {
                f.emit(Op::SnapNext, 0, 0, 0);
                let mut fails = Vec::new();
                for arg in args {
                    self.lower_into(f, arg, dst);
                    fails.push(f.emit(Op::JumpIfFalse, dst, 0, 0));
                }
                f.emit(Op::DropSnap, 0, 0, 0);
                let j_join = f.emit(Op::Jump, 0, 0, 0);
                let rollback = f.pc();
                for j in fails {
                    f.patch_b(j, rollback);
                }
                // dst is already false on this path
                f.emit(Op::RestoreSnapPop, 0, 0, 0);
                let join = f.pc();
                f.patch_b(j_join, join);
            }

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
                    f.emit(Op::CallValue, dst, r_lam, r_payload);
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
                        f.emit(Op::Call, dst, u16_of(fnid as usize, "fn id"), ab);
                    }
                    None => {
                        let r_lam = f.alloc(1);
                        self.lower_def_value(f, &def, r_lam);
                        f.emit(Op::CallValue, dst, r_lam, ab);
                    }
                }
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
                    let bind = f.emit(Op::OneOfBind, cell16, r_set, 0);
                    self.lower_into(f, body, dst);
                    f.emit(Op::LetExit, cell16, 0, 0);
                    let j_join = f.emit(Op::Jump, 0, 0, 0);
                    // empty set: this branch is disabled
                    f.patch_c(bind, f.pc());
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
    pub fn compile(&mut self, expr: &QuintEx) -> CompiledExpr {
        let fnid = self.lower_entry(expr);
        CompiledExpr {
            vm: self.vm.clone(),
            fnid,
        }
    }

    pub fn param_register(&mut self, param: &LambdaParam) -> Register {
        let slot = self.param_slot(param);
        self.vm.borrow().program.params[slot as usize].clone()
    }

    pub fn storage(&self) -> Rc<RefCell<VarStorage>> {
        self.storage.clone()
    }
}
