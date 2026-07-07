//! The bytecode VM: expressions are lowered once into flat 8-byte
//! instructions ([`lower::Lowerer`]) and executed by a register machine.
//!
//! Compared to the closure engine this removes the per-node error-wrapper
//! closure, the per-application argument `Vec`, and all `Rc<dyn Fn>`
//! indirect calls; short-circuit operators become conditional jumps and
//! record/variant access is resolved through `Symbol` tables.
//!
//! Design notes:
//! - **Calls are native recursion** (`exec` calls itself), exactly like the
//!   closure engine's nested `execute` calls — same stack-depth profile,
//!   no frame machinery. The register file is one shared `Vec<Value>`
//!   window-allocated per call.
//! - **Parameters are shared cells** (`Register`), the same representation
//!   the closure engine uses, so `BoundExpr`/temporal atoms work unchanged.
//!   Calls save/restore the callee's cells (a few `u32` moves).
//! - **Error unwinding**: an error aborts the whole run; `run` restores
//!   let-cells, parameter cells and `next_mode` from the unwind stacks
//!   (the caller abandons the run's state, so nothing else needs repair).
//! - **Choice replay**: `env.choose` is called at `OneOfBind` / `AnyBegin`
//!   in lowering order = the closure engine's evaluation order, so the
//!   `ChoiceCtl` trail semantics are preserved instruction-for-instruction.

pub mod builtins;
pub mod lower;

pub use lower::Lowerer;

use crate::error::{unsupported, QuintError};
use crate::eval::Env;
use crate::state::Register;
use crate::value::{EvalResult, U64Buf, Value};
use builtins::VmBuiltin;
use quint_ast::{QuintId, Symbol};
use std::rc::Rc;

pub type FnId = u32;

/// One instruction: fieldless op + three 16-bit operands (8 bytes).
/// Register operands are frame-relative; `LoadImm` packs a 32-bit value id
/// into b/c.
#[derive(Clone, Copy, Debug)]
pub struct Instr {
    pub op: Op,
    pub a: u16,
    pub b: u16,
    pub c: u16,
}

const _: () = assert!(std::mem::size_of::<Instr>() == 8);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Op {
    /// regs[a] = value(b | c << 16)
    LoadImm,
    /// regs[a] = regs[b]
    Move,
    /// regs[a] = (next_mode ? next : current)[b]  (QNT502 if unset)
    LoadVar,
    /// next[b] = normalize(regs[c]); regs[a] = true
    Assign,
    /// regs[a] = params[b]  (QNT500 if unset)
    LoadParam,
    /// pc = b
    Jump,
    /// if !regs[a] { pc = b }
    JumpIfFalse,
    /// if regs[a] { pc = b }
    JumpIfTrue,
    /// regs[a] = call fn b with args at regs[c..] (binds callee param cells)
    Call,
    /// regs[a] = apply lambda regs[b] to args at regs[c..]
    CallValue,
    /// return regs[a]
    Ret,
    /// regs[a] = builtin call site b (argbase/nargs in the side table)
    Builtin,
    /// regs[a] = let-cell c (call fn b on miss; caches Err too)
    CallCached,
    /// regs[a] = per-state val cache c (fn b; bypassed under next_mode)
    CallCachedVal,
    /// regs[a] = persistent pureval cache c (fn b)
    CallCachedPure,
    /// push let-cell a onto the unwind stack and clear it
    LetEnter,
    /// restore let-cell a from the unwind stack
    LetExit,
    /// nondet pick into let-cell a from set regs[b]; empty → pc = c
    OneOfBind,
    /// regs[a] = pick from set regs[b] (QNT509 on empty)
    OneOfPick,
    /// choice over branch table b; result reg a, join pc c
    AnyBegin,
    /// close an actionAny branch (result in regs[a])
    AnyEnd,
    /// storage.shift() (then/reps commit)
    Shift,
    /// enter next(x) evaluation (QNT501 unless next_allowed)
    NextEnter,
    NextExit,
    /// save param cell a and set it from regs[b] (matchVariant payload)
    ParamEnter,
    /// restore param cell a
    ParamExit,
    /// dispatch on variant regs[b] via match table c; payload → regs[a]
    MatchVariant,
    /// push a snapshot of the next-state registers
    SnapNext,
    /// pop the snapshot and restore the next-state registers
    RestoreSnapPop,
    /// pop the snapshot without restoring
    DropSnap,
    /// regs[a] = run lambda regs[c] regs[b] times, shifting in between
    Reps,
    /// fail with program error a
    Fail,
}

pub struct MatchTable {
    /// (case label, case pc), in source order; `_` matches anything.
    pub cases: Box<[(Symbol, u16)]>,
}

pub struct BuiltinCall {
    pub func: VmBuiltin,
    pub argbase: u16,
    pub nargs: u16,
}

pub struct Chunk {
    pub code: Vec<Instr>,
    /// IR node per instruction, for error annotation (same length as code).
    pub node_ids: Vec<QuintId>,
    pub nregs: u16,
    /// Parameter slots this function binds on entry (in argument order).
    pub param_slots: Box<[u32]>,
    pub branch_tables: Vec<Box<[u16]>>,
    pub match_tables: Vec<MatchTable>,
    pub builtin_calls: Vec<BuiltinCall>,
}

pub struct Program {
    pub funcs: Vec<Rc<Chunk>>,
    /// Parameter cells by slot (shared with `BoundExpr` bindings).
    pub params: Vec<Register>,
    pub param_names: Vec<Symbol>,
    pub var_current: Vec<Register>,
    pub var_next: Vec<Register>,
    pub var_names: Vec<Symbol>,
    /// Static error table for `Fail`.
    pub errors: Vec<(&'static str, String)>,
}

struct AnyRec {
    start: u64,
    tried: usize,
    table: u16,
    join_pc: u16,
    dst: u16,
}

pub struct Vm {
    pub program: Program,
    /// Register file, high-water-mark managed: `reg_top` is the logical
    /// stack cursor; slots above it keep stale values (lowering guarantees
    /// every register is written before it is read), so call frames never
    /// memset.
    regs: Vec<Value>,
    reg_top: usize,
    let_cells: Vec<Option<EvalResult>>,
    val_cells: Vec<(u64, Value)>,
    pure_cells: Vec<Option<Value>>,
    // unwind stacks (restored on error abort)
    cell_stack: Vec<(u32, Option<EvalResult>)>,
    param_stack: Vec<(u32, Option<Value>)>,
    snapshots: Vec<Vec<Option<Value>>>,
    any_stack: Vec<AnyRec>,
    next_stack: Vec<bool>,
    /// Retired snapshot buffers, reused by SnapNext/AnyBegin.
    snap_pool: Vec<Vec<Option<Value>>>,
}

/// Callee parameter counts are bounded so call frames can save the old
/// cell values on the stack.
pub const MAX_PARAMS: usize = 16;

struct Marks {
    cells: usize,
    params: usize,
    snaps: usize,
    anys: usize,
    nexts: usize,
    regs: usize,
}

impl Vm {
    pub fn new(program: Program) -> Self {
        Vm {
            program,
            regs: Vec::new(),
            reg_top: 0,
            let_cells: Vec::new(),
            val_cells: Vec::new(),
            pure_cells: Vec::new(),
            cell_stack: Vec::new(),
            param_stack: Vec::new(),
            snapshots: Vec::new(),
            any_stack: Vec::new(),
            next_stack: Vec::new(),
            snap_pool: Vec::new(),
        }
    }

    /// Push a pooled snapshot of the next-state registers.
    fn push_snapshot(&mut self, env: &Env) {
        let mut buf = self.snap_pool.pop().unwrap_or_default();
        env.storage.borrow().snapshot_next_into(&mut buf);
        self.snapshots.push(buf);
    }

    fn pop_snapshot(&mut self) {
        let buf = self.snapshots.pop().expect("unbalanced snapshot pop");
        self.snap_pool.push(buf);
    }

    pub(crate) fn add_let_cell(&mut self) -> u32 {
        self.let_cells.push(None);
        self.let_cells.len() as u32 - 1
    }

    pub(crate) fn add_val_cell(&mut self) -> u32 {
        // generation 0 never matches (VarStorage starts at 1)
        self.val_cells.push((0, Value::bool(false)));
        self.val_cells.len() as u32 - 1
    }

    pub(crate) fn add_pure_cell(&mut self) -> u32 {
        self.pure_cells.push(None);
        self.pure_cells.len() as u32 - 1
    }

    /// Top-level entry: execute function `entry` and clean up the unwind
    /// stacks if it aborts with an error.
    pub fn run(&mut self, env: &mut Env, entry: FnId) -> EvalResult {
        let marks = Marks {
            cells: self.cell_stack.len(),
            params: self.param_stack.len(),
            snaps: self.snapshots.len(),
            anys: self.any_stack.len(),
            nexts: self.next_stack.len(),
            regs: self.reg_top,
        };
        let result = self.exec(env, entry);
        if result.is_err() {
            self.unwind(env, &marks);
        } else {
            debug_assert_eq!(self.cell_stack.len(), marks.cells);
            debug_assert_eq!(self.param_stack.len(), marks.params);
            debug_assert_eq!(self.snapshots.len(), marks.snaps);
            debug_assert_eq!(self.any_stack.len(), marks.anys);
            debug_assert_eq!(self.next_stack.len(), marks.nexts);
        }
        result
    }

    fn unwind(&mut self, env: &mut Env, m: &Marks) {
        while self.cell_stack.len() > m.cells {
            let (cell, old) = self.cell_stack.pop().unwrap();
            self.let_cells[cell as usize] = old;
        }
        while self.param_stack.len() > m.params {
            let (slot, old) = self.param_stack.pop().unwrap();
            self.program.params[slot as usize].set(old);
        }
        while self.snapshots.len() > m.snaps {
            self.pop_snapshot();
        }
        self.any_stack.truncate(m.anys);
        while self.next_stack.len() > m.nexts {
            env.next_mode = self.next_stack.pop().unwrap();
        }
        self.reg_top = m.regs;
    }

    /// Execute one function in a fresh register window (native recursion).
    /// The window is not cleared: the file only grows to the high-water
    /// mark, and stale values are fine because lowering writes every
    /// register before reading it.
    fn exec(&mut self, env: &mut Env, fnid: FnId) -> EvalResult {
        let chunk = self.program.funcs[fnid as usize].clone();
        let base = self.reg_top;
        let top = base + chunk.nregs as usize;
        if self.regs.len() < top {
            self.regs.resize(top, Value::bool(false));
        }
        self.reg_top = top;
        let result = self.exec_in(env, &chunk, base);
        self.reg_top = base;
        result
    }

    /// Apply a lambda value (used by higher-order builtins and `Reps`).
    pub fn call_lambda(&mut self, env: &mut Env, lam: Value, args: &[Value]) -> EvalResult {
        let lamv = lam.as_lambda();
        let fnid = lamv.fnid;
        debug_assert_eq!(lamv.registers.len(), args.len());
        debug_assert!(args.len() <= MAX_PARAMS);
        let mut saved = [None; MAX_PARAMS];
        for (i, (cell, arg)) in lamv.registers.iter().zip(args).enumerate() {
            saved[i] = cell.replace(Some(*arg));
        }
        let result = self.exec(env, fnid);
        for (i, cell) in lamv.registers.iter().enumerate() {
            cell.set(saved[i]);
        }
        result
    }

    fn exec_in(&mut self, env: &mut Env, chunk: &Chunk, base: usize) -> EvalResult {
        macro_rules! reg {
            ($i:expr) => {
                self.regs[base + $i as usize]
            };
        }
        // Annotate an error with the faulting instruction's IR node
        // (innermost only, matching the closure engine).
        fn annotate(e: QuintError, chunk: &Chunk, idx: usize) -> QuintError {
            if e.trace.is_empty() {
                e.with_id(chunk.node_ids[idx])
            } else {
                e
            }
        }
        macro_rules! fail {
            ($pc:expr, $e:expr) => {
                return Err(annotate($e, chunk, $pc - 1))
            };
        }
        macro_rules! try_at {
            ($pc:expr, $r:expr) => {
                match $r {
                    Ok(v) => v,
                    Err(e) => fail!($pc, e),
                }
            };
        }

        let mut pc: usize = 0;
        loop {
            let ins = chunk.code[pc];
            pc += 1;
            match ins.op {
                Op::LoadImm => {
                    reg!(ins.a) = Value::from_bits(ins.b as u32 | (ins.c as u32) << 16);
                }
                Op::Move => reg!(ins.a) = reg!(ins.b),
                Op::LoadVar => {
                    let bank = if env.next_mode {
                        &self.program.var_next
                    } else {
                        &self.program.var_current
                    };
                    match bank[ins.b as usize].get() {
                        Some(v) => reg!(ins.a) = v,
                        None => fail!(
                            pc,
                            QuintError::new(
                                "QNT502",
                                format!(
                                    "Variable {} not set",
                                    self.program.var_names[ins.b as usize]
                                ),
                            )
                        ),
                    }
                }
                Op::Assign => {
                    let value = try_at!(pc, reg!(ins.c).normalize());
                    self.program.var_next[ins.b as usize].set(Some(value));
                    reg!(ins.a) = Value::bool(true);
                }
                Op::LoadParam => match self.program.params[ins.b as usize].get() {
                    Some(v) => reg!(ins.a) = v,
                    None => fail!(
                        pc,
                        QuintError::new(
                            "QNT500",
                            format!("Param {} not set", self.program.param_names[ins.b as usize]),
                        )
                    ),
                },
                Op::Jump => pc = ins.b as usize,
                Op::JumpIfFalse => {
                    if !reg!(ins.a).as_bool() {
                        pc = ins.b as usize;
                    }
                }
                Op::JumpIfTrue => {
                    if reg!(ins.a).as_bool() {
                        pc = ins.b as usize;
                    }
                }
                Op::Call => {
                    let callee = self.program.funcs[ins.b as usize].clone();
                    let ab = base + ins.c as usize;
                    debug_assert!(callee.param_slots.len() <= MAX_PARAMS);
                    let mut saved = [None; MAX_PARAMS];
                    for (i, &slot) in callee.param_slots.iter().enumerate() {
                        saved[i] = self.program.params[slot as usize]
                            .replace(Some(self.regs[ab + i]));
                    }
                    let result = self.exec(env, ins.b as u32);
                    for (i, &slot) in callee.param_slots.iter().enumerate() {
                        self.program.params[slot as usize].set(saved[i]);
                    }
                    reg!(ins.a) = try_at!(pc, result);
                }
                Op::CallValue => {
                    let lam = reg!(ins.b);
                    let ab = base + ins.c as usize;
                    let lamv = lam.as_lambda();
                    let n = lamv.registers.len();
                    debug_assert!(n <= MAX_PARAMS);
                    let fnid = lamv.fnid;
                    let mut saved = [None; MAX_PARAMS];
                    for (i, cell) in lamv.registers.iter().enumerate() {
                        saved[i] = cell.replace(Some(self.regs[ab + i]));
                    }
                    let result = self.exec(env, fnid);
                    for (i, cell) in lamv.registers.iter().enumerate() {
                        cell.set(saved[i]);
                    }
                    reg!(ins.a) = try_at!(pc, result);
                }
                Op::Ret => return Ok(reg!(ins.a)),
                Op::Builtin => {
                    let call = &chunk.builtin_calls[ins.b as usize];
                    let ab = base + call.argbase as usize;
                    let n = call.nargs as usize;
                    let result = match call.func {
                        VmBuiltin::Simple(f) => f(env, &self.regs[ab..ab + n]),
                        VmBuiltin::Ho(f) => {
                            debug_assert!(n <= 4);
                            let mut buf = [Value::bool(false); 4];
                            buf[..n].copy_from_slice(&self.regs[ab..ab + n]);
                            f(self, env, &buf[..n])
                        }
                    };
                    reg!(ins.a) = try_at!(pc, result);
                }
                Op::CallCached => {
                    let cell = ins.c as usize;
                    match &self.let_cells[cell] {
                        Some(r) => reg!(ins.a) = try_at!(pc, r.clone()),
                        None => {
                            let result = self.exec(env, ins.b as u32);
                            self.let_cells[cell] = Some(result.clone());
                            reg!(ins.a) = try_at!(pc, result);
                        }
                    }
                }
                Op::CallCachedVal => {
                    // The per-state cache keys on the *current* state; under
                    // next_mode the val reads the next bank, so bypass the
                    // cache entirely (read and write).
                    if env.next_mode {
                        let result = self.exec(env, ins.b as u32);
                        reg!(ins.a) = try_at!(pc, result);
                    } else {
                        let gen = env.storage.borrow().generation.get();
                        let cell = ins.c as usize;
                        if self.val_cells[cell].0 == gen {
                            reg!(ins.a) = self.val_cells[cell].1;
                        } else {
                            let v = try_at!(pc, self.exec(env, ins.b as u32));
                            self.val_cells[cell] = (gen, v);
                            reg!(ins.a) = v;
                        }
                    }
                }
                Op::CallCachedPure => {
                    let cell = ins.c as usize;
                    match self.pure_cells[cell] {
                        Some(v) => reg!(ins.a) = v,
                        None => {
                            let v = try_at!(pc, self.exec(env, ins.b as u32));
                            self.pure_cells[cell] = Some(v);
                            reg!(ins.a) = v;
                        }
                    }
                }
                Op::LetEnter => {
                    let cell = ins.a as usize;
                    let old = self.let_cells[cell].take();
                    self.cell_stack.push((ins.a as u32, old));
                }
                Op::LetExit => {
                    let (cell, old) = self.cell_stack.pop().expect("unbalanced LetExit");
                    debug_assert_eq!(cell, ins.a as u32);
                    self.let_cells[cell as usize] = old;
                }
                Op::OneOfBind => {
                    let set = reg!(ins.b);
                    let bounds = try_at!(pc, set.bounds());
                    let mut indices = U64Buf::new();
                    let mut empty = false;
                    for &bound in bounds.iter() {
                        match try_at!(pc, env.choose(bound)) {
                            Some(i) => indices.push(i),
                            None => {
                                // empty set: this branch is disabled
                                empty = true;
                                break;
                            }
                        }
                    }
                    if empty {
                        pc = ins.c as usize;
                    } else {
                        let picked = try_at!(
                            pc,
                            set.pick(&mut indices.iter().copied()).and_then(|v| v.normalize())
                        );
                        let cell = ins.a as usize;
                        let old = self.let_cells[cell].replace(Ok(picked));
                        self.cell_stack.push((ins.a as u32, old));
                    }
                }
                Op::OneOfPick => {
                    let set = reg!(ins.b);
                    let bounds = try_at!(pc, set.bounds());
                    let mut indices = U64Buf::new();
                    for &bound in bounds.iter() {
                        match try_at!(pc, env.choose(bound)) {
                            Some(i) => indices.push(i),
                            None => fail!(
                                pc,
                                QuintError::new("QNT509", "Applied oneOf on an empty set")
                            ),
                        }
                    }
                    reg!(ins.a) = try_at!(pc, set.pick(&mut indices.iter().copied()));
                }
                Op::AnyBegin => {
                    let table = &chunk.branch_tables[ins.b as usize];
                    let n = table.len();
                    match try_at!(pc, env.choose(n as u64)) {
                        None => {
                            reg!(ins.a) = Value::bool(false);
                            pc = ins.c as usize;
                        }
                        Some(start) => {
                            self.push_snapshot(env);
                            self.any_stack.push(AnyRec {
                                start,
                                tried: 0,
                                table: ins.b,
                                join_pc: ins.c,
                                dst: ins.a,
                            });
                            pc = table[start as usize] as usize;
                        }
                    }
                }
                Op::AnyEnd => {
                    let idx = self.any_stack.len() - 1;
                    let (start, tried, table_idx, join_pc, dst) = {
                        let r = &self.any_stack[idx];
                        (r.start, r.tried, r.table, r.join_pc, r.dst)
                    };
                    debug_assert_eq!(dst, ins.a);
                    let table = &chunk.branch_tables[table_idx as usize];
                    let n = table.len();
                    if reg!(ins.a).as_bool() {
                        self.pop_snapshot();
                        self.any_stack.pop();
                        pc = join_pc as usize;
                    } else {
                        let tried = tried + 1;
                        let tries = if env.any_fallthrough { n } else { 1 };
                        env.storage
                            .borrow()
                            .restore_next(self.snapshots.last().unwrap());
                        if tried < tries {
                            self.any_stack[idx].tried = tried;
                            let branch = (start as usize + tried) % n;
                            pc = table[branch] as usize;
                        } else {
                            self.pop_snapshot();
                            self.any_stack.pop();
                            reg!(ins.a) = Value::bool(false);
                            pc = join_pc as usize;
                        }
                    }
                }
                Op::Shift => {
                    env.storage.borrow().shift();
                }
                Op::NextEnter => {
                    if !env.next_allowed {
                        fail!(pc, unsupported("next() outside a temporal property"));
                    }
                    self.next_stack.push(env.next_mode);
                    env.next_mode = true;
                }
                Op::NextExit => {
                    env.next_mode = self.next_stack.pop().expect("unbalanced NextExit");
                }
                Op::ParamEnter => {
                    let old = self.program.params[ins.a as usize].replace(Some(reg!(ins.b)));
                    self.param_stack.push((ins.a as u32, old));
                }
                Op::ParamExit => {
                    let (slot, old) = self.param_stack.pop().expect("unbalanced ParamExit");
                    debug_assert_eq!(slot, ins.a as u32);
                    self.program.params[slot as usize].set(old);
                }
                Op::MatchVariant => {
                    let (label, payload) = reg!(ins.b).as_variant();
                    let table = &chunk.match_tables[ins.c as usize];
                    let mut target = None;
                    for (sym, case_pc) in table.cases.iter() {
                        if *sym == label || sym.as_str() == "_" {
                            target = Some(*case_pc);
                            break;
                        }
                    }
                    match target {
                        Some(t) => {
                            reg!(ins.a) = payload;
                            pc = t as usize;
                        }
                        None => fail!(
                            pc,
                            QuintError::new("QNT505", format!("No match for variant {label}"))
                        ),
                    }
                }
                Op::SnapNext => {
                    self.push_snapshot(env);
                }
                Op::RestoreSnapPop => {
                    let snap = self.snapshots.pop().expect("unbalanced RestoreSnapPop");
                    env.storage.borrow().restore_next(&snap);
                    self.snap_pool.push(snap);
                }
                Op::DropSnap => {
                    self.pop_snapshot();
                }
                Op::Reps => {
                    let reps = reg!(ins.b).as_int();
                    let lam = reg!(ins.c);
                    let mut result = Value::bool(true);
                    for i in 0..reps {
                        result = try_at!(pc, self.call_lambda(env, lam, &[Value::int(i)]));
                        if !result.as_bool() {
                            fail!(
                                pc,
                                QuintError::new(
                                    "QNT513",
                                    format!(
                                        "Reps loop could not continue after iteration #{} \
                                         evaluated to false",
                                        i + 1
                                    ),
                                )
                            );
                        }
                        if i < reps - 1 {
                            env.storage.borrow().shift();
                        }
                    }
                    reg!(ins.a) = result;
                }
                Op::Fail => {
                    let (code, msg) = &self.program.errors[ins.a as usize];
                    fail!(pc, QuintError::new(code, msg.clone()));
                }
            }
        }
    }
}
