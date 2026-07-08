//! State variables: registers for the current and next state.

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

/// TLC-style seen set: 64-bit fingerprints only, no state bodies.
/// Probabilistically sound — a fingerprint collision would silently prune
/// an unexplored state (with SipHash-quality 64-bit fingerprints the
/// probability is ~n²/2⁶⁵; the same trade-off TLC makes by default).
#[derive(Default)]
pub struct FpSet {
    table: hashbrown::HashTable<u64>,
}

impl FpSet {
    /// Fingerprint of a state (its value-id sequence). foldhash with a
    /// fixed seed — deterministic across runs and processes, with real
    /// 64-bit avalanche (FxHash would not qualify).
    pub fn fingerprint(s: &[Value]) -> u64 {
        use std::hash::BuildHasher;
        const FP_SEED: u64 = 0x5155_494e_5443_4b21; // "QUINTCK!"
        foldhash::fast::FixedState::with_seed(FP_SEED).hash_one(s)
    }

    pub fn contains(&self, fp: u64) -> bool {
        self.table.find(fp, |&e| e == fp).is_some()
    }

    /// Insert a fingerprint; returns true if it was fresh. The fingerprint
    /// is its own hash (already uniform).
    pub fn insert(&mut self, fp: u64) -> bool {
        if self.table.find(fp, |&e| e == fp).is_some() {
            return false;
        }
        self.table.insert_unique(fp, fp, |&e| e);
        true
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
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

