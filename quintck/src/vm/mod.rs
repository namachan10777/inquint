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
//! - **Parameters and state variables are plain per-Vm banks** indexed by
//!   slot; calls save/restore the callee's slots (a few `u32` moves). The
//!   `Program` is immutable after lowering, so a `Vm` per worker thread
//!   shares it by reference with zero synchronization.
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
use crate::value::{value_eq, EvalResult, U64Buf, Value};
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
    // Fused superinstructions (one dispatch instead of builtin+branch):
    /// if !(regs[a] == regs[b]) { pc = c }  (value_eq semantics)
    GuardEq,
    /// if !(regs[a] != regs[b]) { pc = c }
    GuardNeq,
    /// if !(regs[a] < regs[b]) { pc = c }  (ints)
    GuardLt,
    GuardLte,
    GuardGt,
    GuardGte,
    /// regs[a] = regs[b] == regs[c]  (value_eq semantics)
    Eq,
    Neq,
    /// regs[a] = !regs[b]
    Not,
    /// regs[a] = record field of regs[b] named field_syms[c]
    RecField,
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

/// A statically resolved call site: the callee and its parameter slots,
/// so the handler binds arguments without touching the function table.
pub struct CallSite {
    pub fnid: FnId,
    pub slots: Box<[u32]>,
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
    pub call_sites: Vec<CallSite>,
    /// Field names for `RecField` (c operand indexes here).
    pub field_syms: Vec<Symbol>,
}

/// The compiled program: immutable after lowering, freely shared across
/// worker threads by reference. All mutable evaluation state (parameter
/// and variable banks, caches, stacks) lives in a per-worker [`Vm`].
pub struct Program {
    pub funcs: Vec<Chunk>,
    pub param_names: Vec<Symbol>,
    pub var_names: Vec<Symbol>,
    pub n_let_cells: usize,
    pub n_val_cells: usize,
    pub n_pure_cells: usize,
    /// Static error table for `Fail`.
    pub errors: Vec<(&'static str, String)>,
}

/// A let cell's content: 16 bytes (the error boxed behind an `Rc`), so
/// LetEnter/LetExit save/restore are cheap moves — `Option<EvalResult>`
/// would be ~72 bytes per push/pop.
type CellVal = Option<Result<Value, Rc<QuintError>>>;

struct AnyRec {
    start: u64,
    tried: usize,
    table: u16,
    join_pc: u16,
    dst: u16,
}

/// Per-worker evaluation state. Borrows the immutable [`Program`];
/// everything else is owned and thread-private.
pub struct Vm<'p> {
    pub program: &'p Program,
    /// Register file, high-water-mark managed: `reg_top` is the logical
    /// stack cursor; slots above it keep stale values (lowering guarantees
    /// every register is written before it is read), so call frames never
    /// memset.
    regs: Vec<Value>,
    reg_top: usize,
    /// Parameter bank (one slot per lambda parameter, save/restored
    /// around calls).
    params: Vec<Option<Value>>,
    /// Current/next state-variable banks (the exploration loop loads
    /// states here; `assign` writes the next bank).
    vars_cur: Vec<Option<Value>>,
    vars_next: Vec<Option<Value>>,
    /// Bumped whenever the current state changes; keys the `val` caches.
    generation: u64,
    let_cells: Vec<CellVal>,
    val_cells: Vec<(u64, Value)>,
    pure_cells: Vec<Option<Value>>,
    // unwind stacks (restored on error abort)
    cell_stack: Vec<(u32, CellVal)>,
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

/// CallValue's c operand packs (argbase << 5 | nargs); see the lowerer.
pub(crate) fn callvalue_unpack(c: u16) -> (usize, usize) {
    ((c >> 5) as usize, (c & 31) as usize)
}

struct Marks {
    cells: usize,
    params: usize,
    snaps: usize,
    anys: usize,
    nexts: usize,
    regs: usize,
}

impl<'p> Vm<'p> {
    pub fn new(program: &'p Program) -> Self {
        Vm {
            program,
            regs: Vec::new(),
            reg_top: 0,
            params: vec![None; program.param_names.len()],
            vars_cur: vec![None; program.var_names.len()],
            vars_next: vec![None; program.var_names.len()],
            generation: 1,
            let_cells: vec![None; program.n_let_cells],
            val_cells: vec![(0, Value::bool(false)); program.n_val_cells],
            pure_cells: vec![None; program.n_pure_cells],
            cell_stack: Vec::new(),
            param_stack: Vec::new(),
            snapshots: Vec::new(),
            any_stack: Vec::new(),
            next_stack: Vec::new(),
            snap_pool: Vec::new(),
        }
    }

    // -----------------------------------------------------------------
    // State-variable bank management (the exploration loop's interface)
    // -----------------------------------------------------------------

    /// Make `state` the current state.
    pub fn load(&mut self, state: &[Value]) {
        debug_assert_eq!(state.len(), self.vars_cur.len());
        for (slot, v) in self.vars_cur.iter_mut().zip(state) {
            *slot = Some(*v);
        }
        self.generation += 1;
    }

    /// Unset the current state (initial-state enumeration: reading an
    /// unassigned variable is then a QNT502 error).
    pub fn clear_current(&mut self) {
        self.vars_cur.fill(None);
        self.generation += 1;
    }

    pub fn reset_next(&mut self) {
        self.vars_next.fill(None);
    }

    /// Commit the next state: assigned next-vars move into current,
    /// unassigned keep their current value (run-test `then`/`reps`).
    pub fn commit_shift(&mut self) {
        for (cur, next) in self.vars_cur.iter_mut().zip(self.vars_next.iter_mut()) {
            if let Some(v) = next.take() {
                *cur = Some(v);
            }
        }
        self.generation += 1;
    }

    /// Load `state` into the next bank (temporal edge-atom evaluation).
    pub fn load_next(&mut self, state: &[Value]) {
        debug_assert_eq!(state.len(), self.vars_next.len());
        for (slot, v) in self.vars_next.iter_mut().zip(state) {
            *slot = Some(*v);
        }
    }

    /// Raw next-bank contents after an action run (None = unconstrained).
    pub fn take_partial(&self) -> Vec<Option<Value>> {
        self.vars_next.clone()
    }

    /// The successor state after a successful action run; every variable
    /// must have been assigned.
    pub fn take_next_state(&self) -> Result<crate::state::State, QuintError> {
        let values = self
            .vars_next
            .iter()
            .zip(self.program.var_names.iter())
            .map(|(slot, name)| {
                slot.ok_or_else(|| {
                    QuintError::new(
                        "QNT502",
                        format!("action succeeded but did not assign variable {name}"),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(std::rc::Rc::from(values.into_boxed_slice()))
    }

    /// Bind a parameter slot, returning the previous value (BoundExpr).
    pub fn param_replace(&mut self, slot: u32, v: Option<Value>) -> Option<Value> {
        std::mem::replace(&mut self.params[slot as usize], v)
    }

    /// Push a pooled snapshot of the next-state bank.
    fn push_snapshot(&mut self) {
        let mut buf = self.snap_pool.pop().unwrap_or_default();
        buf.clear();
        buf.extend(self.vars_next.iter().copied());
        self.snapshots.push(buf);
    }

    fn pop_snapshot(&mut self) {
        let buf = self.snapshots.pop().expect("unbalanced snapshot pop");
        self.snap_pool.push(buf);
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
            self.params[slot as usize] = old;
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
    ///
    fn exec(&mut self, env: &mut Env, fnid: FnId) -> EvalResult {
        // 'p outlives &mut self: no Rc, no clone
        let chunk: &'p Chunk = &self.program.funcs[fnid as usize];
        let base = self.reg_top;
        let top = base + chunk.nregs as usize;
        if self.regs.len() < top {
            self.regs.resize(top, Value::bool(false));
        }
        self.reg_top = top;
        let result = self.exec_in(env, chunk, base);
        self.reg_top = base;
        result
    }

    /// Apply a lambda value (used by higher-order builtins and `Reps`).
    pub fn call_lambda(&mut self, env: &mut Env, lam: Value, args: &[Value]) -> EvalResult {
        let lamv = lam.as_lambda();
        let fnid = lamv.fnid;
        debug_assert_eq!(lamv.slots.len(), args.len());
        debug_assert!(args.len() <= MAX_PARAMS);
        let mut saved = [None; MAX_PARAMS];
        for (i, (&slot, arg)) in lamv.slots.iter().zip(args).enumerate() {
            saved[i] = self.params[slot as usize].replace(*arg);
        }
        let result = self.exec(env, fnid);
        for (i, &slot) in lamv.slots.iter().enumerate() {
            self.params[slot as usize] = saved[i];
        }
        result
    }

    // -----------------------------------------------------------------
    // Shared operation bodies (interpreter handlers and JIT helpers)
    // -----------------------------------------------------------------

    /// Dynamic call of a lambda value with arguments at `regs[ab..]`.
    pub(crate) fn call_value_abs(&mut self, env: &mut Env, lam: Value, ab: usize) -> EvalResult {
        let lamv = lam.as_lambda();
        let n = lamv.slots.len();
        debug_assert!(n <= MAX_PARAMS);
        let fnid = lamv.fnid;
        let mut saved = [None; MAX_PARAMS];
        for (i, &slot) in lamv.slots.iter().enumerate() {
            saved[i] = self.params[slot as usize].replace(self.regs[ab + i]);
        }
        let result = self.exec(env, fnid);
        for (i, &slot) in lamv.slots.iter().enumerate() {
            self.params[slot as usize] = saved[i];
        }
        result
    }

    /// Let-scoped call-by-need (caches `Err` too, like the interpreter).
    pub(crate) fn call_cached_let(&mut self, env: &mut Env, fnid: FnId, cell: usize) -> EvalResult {
        match &self.let_cells[cell] {
            Some(Ok(v)) => Ok(*v),
            Some(Err(e)) => Err((**e).clone()),
            None => {
                let result = self.exec(env, fnid);
                self.let_cells[cell] = Some(match &result {
                    Ok(v) => Ok(*v),
                    Err(e) => Err(Rc::new(e.clone())),
                });
                result
            }
        }
    }

    /// Per-state `val` cache: keyed on the storage generation, bypassed
    /// entirely under next_mode; only `Ok` is cached.
    pub(crate) fn call_cached_val(&mut self, env: &mut Env, fnid: FnId, cell: usize) -> EvalResult {
        if env.next_mode {
            return self.exec(env, fnid);
        }
        let gen = self.generation;
        if self.val_cells[cell].0 == gen {
            return Ok(self.val_cells[cell].1);
        }
        let v = self.exec(env, fnid)?;
        self.val_cells[cell] = (gen, v);
        Ok(v)
    }

    pub(crate) fn call_cached_pure(&mut self, env: &mut Env, fnid: FnId, cell: usize) -> EvalResult {
        if let Some(v) = self.pure_cells[cell] {
            return Ok(v);
        }
        let v = self.exec(env, fnid)?;
        self.pure_cells[cell] = Some(v);
        Ok(v)
    }

    pub(crate) fn let_enter(&mut self, cell: usize) {
        let old = self.let_cells[cell].take();
        self.cell_stack.push((cell as u32, old));
    }

    pub(crate) fn let_exit(&mut self, cell: usize) {
        let (c, old) = self.cell_stack.pop().expect("unbalanced LetExit");
        debug_assert_eq!(c as usize, cell);
        self.let_cells[c as usize] = old;
    }

    pub(crate) fn param_enter(&mut self, slot: usize, v: Value) {
        let old = self.params[slot].replace(v);
        self.param_stack.push((slot as u32, old));
    }

    pub(crate) fn param_exit(&mut self, slot: usize) {
        let (s, old) = self.param_stack.pop().expect("unbalanced ParamExit");
        debug_assert_eq!(s as usize, slot);
        self.params[s as usize] = old;
    }

    /// Nondet binding: choose an element of `set` into the let cell
    /// (pushing the old cell value). Returns Ok(false) when the set is
    /// empty (branch disabled; cell untouched).
    pub(crate) fn one_of_bind(
        &mut self,
        env: &mut Env,
        cell: usize,
        set: Value,
    ) -> Result<bool, QuintError> {
        let bounds = set.bounds()?;
        let mut indices = U64Buf::new();
        for &bound in bounds.iter() {
            match env.choose(bound)? {
                Some(i) => indices.push(i),
                None => return Ok(false),
            }
        }
        let picked = set.pick(&mut indices.iter().copied())?.normalize()?;
        let old = self.let_cells[cell].replace(Ok(picked));
        self.cell_stack.push((cell as u32, old));
        Ok(true)
    }

    /// Restore the next bank from the top snapshot without popping it
    /// (actionAny retry).
    fn restore_last_snapshot(&mut self) {
        let snap = self.snapshots.last().expect("no snapshot");
        self.vars_next.copy_from_slice(snap);
    }

    fn restore_pop(&mut self) {
        let snap = self.snapshots.pop().expect("unbalanced snapshot pop");
        self.vars_next.copy_from_slice(&snap);
        self.snap_pool.push(snap);
    }

    pub(crate) fn next_enter(&mut self, env: &mut Env) {
        self.next_stack.push(env.next_mode);
        env.next_mode = true;
    }

    pub(crate) fn next_exit(&mut self, env: &mut Env) {
        env.next_mode = self.next_stack.pop().expect("unbalanced NextExit");
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
                        &self.vars_next
                    } else {
                        &self.vars_cur
                    };
                    match bank[ins.b as usize] {
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
                    self.vars_next[ins.b as usize] = Some(value);
                    reg!(ins.a) = Value::bool(true);
                }
                Op::LoadParam => match self.params[ins.b as usize] {
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
                    let site = &chunk.call_sites[ins.b as usize];
                    let ab = base + ins.c as usize;
                    debug_assert!(site.slots.len() <= MAX_PARAMS);
                    let mut saved = [None; MAX_PARAMS];
                    for (i, &slot) in site.slots.iter().enumerate() {
                        saved[i] = self.params[slot as usize].replace(self.regs[ab + i]);
                    }
                    let result = self.exec(env, site.fnid);
                    for (i, &slot) in site.slots.iter().enumerate() {
                        self.params[slot as usize] = saved[i];
                    }
                    reg!(ins.a) = try_at!(pc, result);
                }
                Op::CallValue => {
                    let lam = reg!(ins.b);
                    let (rel, _nargs) = callvalue_unpack(ins.c);
                    let result = self.call_value_abs(env, lam, base + rel);
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
                    let result = self.call_cached_let(env, ins.b as u32, ins.c as usize);
                    reg!(ins.a) = try_at!(pc, result);
                }
                Op::CallCachedVal => {
                    let result = self.call_cached_val(env, ins.b as u32, ins.c as usize);
                    reg!(ins.a) = try_at!(pc, result);
                }
                Op::CallCachedPure => {
                    let result = self.call_cached_pure(env, ins.b as u32, ins.c as usize);
                    reg!(ins.a) = try_at!(pc, result);
                }
                Op::LetEnter => self.let_enter(ins.a as usize),
                Op::LetExit => self.let_exit(ins.a as usize),
                Op::OneOfBind => {
                    let set = reg!(ins.b);
                    let bound = try_at!(pc, self.one_of_bind(env, ins.a as usize, set));
                    if !bound {
                        // empty set: this branch is disabled
                        pc = ins.c as usize;
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
                            self.push_snapshot();
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
                        self.restore_last_snapshot();
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
                Op::Shift => self.commit_shift(),
                Op::NextEnter => {
                    if !env.next_allowed {
                        fail!(pc, unsupported("next() outside a temporal property"));
                    }
                    self.next_enter(env);
                }
                Op::NextExit => self.next_exit(env),
                Op::ParamEnter => {
                    let v = reg!(ins.b);
                    self.param_enter(ins.a as usize, v);
                }
                Op::ParamExit => self.param_exit(ins.a as usize),
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
                    self.push_snapshot();
                }
                Op::RestoreSnapPop => self.restore_pop(),
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
                            self.commit_shift();
                        }
                    }
                    reg!(ins.a) = result;
                }
                Op::Fail => {
                    let (code, msg) = &self.program.errors[ins.a as usize];
                    fail!(pc, QuintError::new(code, msg.clone()));
                }
                Op::GuardEq | Op::GuardNeq => {
                    let (x, y) = (reg!(ins.a), reg!(ins.b));
                    let eq = if x == y {
                        true
                    } else if !x.is_symbolic() && !y.is_symbolic() {
                        false
                    } else {
                        try_at!(pc, value_eq(x, y))
                    };
                    if eq != (ins.op == Op::GuardEq) {
                        pc = ins.c as usize;
                    }
                }
                Op::GuardLt | Op::GuardLte | Op::GuardGt | Op::GuardGte => {
                    let (x, y) = (reg!(ins.a).as_int(), reg!(ins.b).as_int());
                    let pass = match ins.op {
                        Op::GuardLt => x < y,
                        Op::GuardLte => x <= y,
                        Op::GuardGt => x > y,
                        _ => x >= y,
                    };
                    if !pass {
                        pc = ins.c as usize;
                    }
                }
                Op::Eq | Op::Neq => {
                    let (x, y) = (reg!(ins.b), reg!(ins.c));
                    let eq = if x == y {
                        true
                    } else if !x.is_symbolic() && !y.is_symbolic() {
                        false
                    } else {
                        try_at!(pc, value_eq(x, y))
                    };
                    reg!(ins.a) = Value::bool(eq == (ins.op == Op::Eq));
                }
                Op::Not => {
                    let v = !reg!(ins.b).as_bool();
                    reg!(ins.a) = Value::bool(v);
                }
                Op::RecField => {
                    let sym = chunk.field_syms[ins.c as usize];
                    reg!(ins.a) = reg!(ins.b)
                        .record_field(sym)
                        .expect("field: no such field (type checker bug?)");
                }
            }
        }
    }
}
