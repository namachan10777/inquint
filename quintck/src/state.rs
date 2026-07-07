//! State variables: registers for the current and next state.

use crate::error::QuintError;
use crate::value::Value;
use quint_ast::{Declaration, QuintId, QuintModule, QuintName};
use rustc_hash::FxHashMap;
use std::cell::Cell;
use std::rc::Rc;

/// A state: one canonical value per variable, in declaration order.
/// Values are interned ids, so `Rc<[Value]>` hashes/compares as a flat
/// id sequence. Used for transient successor sets and traces; explored
/// states live flat in a [`StateArena`].
pub type State = Rc<[Value]>;

/// Flat storage of explored states: state `i` is
/// `data[i * n_vars .. (i + 1) * n_vars]` — no per-state allocation.
pub struct StateArena {
    n_vars: usize,
    data: Vec<Value>,
    parent: Vec<u32>, // u32::MAX = no parent (initial state)
    depth: Vec<u32>,
}

const NO_PARENT: u32 = u32::MAX;

impl StateArena {
    pub fn new(n_vars: usize) -> Self {
        assert!(n_vars > 0);
        StateArena {
            n_vars,
            data: Vec::new(),
            parent: Vec::new(),
            depth: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.parent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    pub fn get(&self, id: u32) -> &[Value] {
        let base = id as usize * self.n_vars;
        &self.data[base..base + self.n_vars]
    }

    pub fn depth(&self, id: u32) -> u32 {
        self.depth[id as usize]
    }

    pub fn push(&mut self, state: &[Value], parent: Option<u32>, depth: u32) -> u32 {
        debug_assert_eq!(state.len(), self.n_vars);
        let id = self.len() as u32;
        self.data.extend_from_slice(state);
        self.parent.push(parent.unwrap_or(NO_PARENT));
        self.depth.push(depth);
        id
    }

    /// The path from an initial state to `id`, following parent links.
    pub fn trace(&self, mut id: u32) -> Vec<State> {
        let mut trace: Vec<State> = Vec::new();
        loop {
            trace.push(Rc::from(self.get(id)));
            match self.parent[id as usize] {
                NO_PARENT => break,
                p => id = p,
            }
        }
        trace.reverse();
        trace
    }
}

/// Seen-set over arena states: stores only the state id; hashing and
/// comparison read the flat arena slices. States are id sequences, so a
/// hash is a flat `u32` hash and equality a memcmp.
#[derive(Default)]
pub struct SeenSet {
    table: hashbrown::HashTable<u32>,
}

impl SeenSet {
    fn hash_state(s: &[Value]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = rustc_hash::FxHasher::default();
        s.hash(&mut h);
        h.finish()
    }

    /// Look up `s`; if unseen, push it into the arena. Returns the state id
    /// and whether it was fresh.
    pub fn insert_or_get(
        &mut self,
        arena: &mut StateArena,
        s: &[Value],
        parent: Option<u32>,
        depth: u32,
    ) -> (u32, bool) {
        let hash = Self::hash_state(s);
        if let Some(&id) = self.table.find(hash, |&id| arena.get(id) == s) {
            return (id, false);
        }
        let id = arena.push(s, parent, depth);
        self.table
            .insert_unique(hash, id, |&id| Self::hash_state(arena.get(id)));
        (id, true)
    }
}

/// Variables of the (single, flattened) module, in declaration order.
#[derive(Debug)]
pub struct VarTable {
    pub names: Vec<QuintName>,
    pub by_def_id: FxHashMap<QuintId, usize>,
}

impl VarTable {
    pub fn from_module(module: &QuintModule) -> Self {
        let mut names = Vec::new();
        let mut by_def_id = FxHashMap::default();
        for decl in &module.declarations {
            if let Declaration::Var { id, name } = decl {
                by_def_id.insert(*id, names.len());
                names.push(*name);
            }
        }
        VarTable { names, by_def_id }
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

pub type Register = Rc<Cell<Option<Value>>>;

/// Registers shared into compiled closures: `current` is read by variable
/// references, `next` is written by `assign`. Per-state `val` caches are
/// registered here so they can be cleared when the current state changes.
pub struct VarStorage {
    pub names: Vec<QuintName>,
    pub current: Vec<Register>,
    pub next: Vec<Register>,
    /// Bumped whenever the current state changes (load / clear / shift);
    /// the VM's per-state `val` caches key on it.
    pub generation: Cell<u64>,
}

impl VarStorage {
    pub fn new(vars: &VarTable) -> Self {
        let mk = |_| Rc::new(Cell::new(None));
        VarStorage {
            names: vars.names.clone(),
            current: (0..vars.len()).map(mk).collect(),
            next: (0..vars.len()).map(mk).collect(),
            generation: Cell::new(1),
        }
    }

    fn clear_caches(&self) {
        self.generation.set(self.generation.get() + 1);
    }

    /// Make `state` the current state.
    pub fn load(&self, state: &[Value]) {
        debug_assert_eq!(state.len(), self.current.len());
        for (reg, value) in self.current.iter().zip(state.iter()) {
            reg.set(Some(*value));
        }
        self.clear_caches();
    }

    /// Unset the current state (used when enumerating initial states:
    /// reading an unassigned variable is then a QNT502 error).
    pub fn clear_current(&self) {
        for reg in &self.current {
            reg.set(None);
        }
        self.clear_caches();
    }

    pub fn reset_next(&self) {
        for reg in &self.next {
            reg.set(None);
        }
    }

    /// Snapshot of the next-state registers (ids: a flat copy).
    pub fn snapshot_next(&self) -> Vec<Option<Value>> {
        self.next.iter().map(|reg| reg.get()).collect()
    }

    pub fn restore_next(&self, snapshot: &[Option<Value>]) {
        for (reg, value) in self.next.iter().zip(snapshot.iter()) {
            reg.set(*value);
        }
    }

    /// Commit the next state: move assigned next-registers into current
    /// (unassigned variables keep their current value) and clear the
    /// per-state caches. Used by the run-test operators (`then`, `reps`).
    pub fn shift(&self) {
        for (cur, next) in self.current.iter().zip(self.next.iter()) {
            if let Some(v) = next.take() {
                cur.set(Some(v));
            }
        }
        self.clear_caches();
    }

    /// The raw next-register contents after an action run: `None` for
    /// variables the action did not assign (unconstrained in TLA terms).
    pub fn take_partial(&self) -> Vec<Option<Value>> {
        self.next.iter().map(|reg| reg.get()).collect()
    }

    /// Load `state` into the next-state registers (for evaluating
    /// next()-predicates over a concrete edge (current, next)).
    pub fn load_next(&self, state: &[Value]) {
        debug_assert_eq!(state.len(), self.next.len());
        for (reg, value) in self.next.iter().zip(state.iter()) {
            reg.set(Some(*value));
        }
    }

    /// Collect the next state after a successful action run. Every variable
    /// must have been assigned.
    pub fn take_next_state(&self) -> Result<State, QuintError> {
        let values = self
            .next
            .iter()
            .zip(self.names.iter())
            .map(|(reg, name)| {
                reg.get().ok_or_else(|| {
                    QuintError::new(
                        "QNT502",
                        format!("action succeeded but did not assign variable {name}"),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Rc::from(values.into_boxed_slice()))
    }
}
