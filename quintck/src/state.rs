//! State variables: registers for the current and next state.

use crate::error::QuintError;
use crate::value::{EvalResult, Value};
use quint_ast::{Declaration, QuintId, QuintModule, QuintName};
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::rc::Rc;

/// A state: one canonical value per variable, in declaration order.
/// `Rc<[Value]>` hashes/compares by content, so it can key the seen-set.
pub type State = Rc<[Value]>;

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
                names.push(name.clone());
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

type Register = Rc<RefCell<Option<Value>>>;

/// Registers shared into compiled closures: `current` is read by variable
/// references, `next` is written by `assign`. Per-state `val` caches are
/// registered here so they can be cleared when the current state changes.
pub struct VarStorage {
    pub names: Vec<QuintName>,
    pub current: Vec<Register>,
    pub next: Vec<Register>,
    pub caches_to_clear: Vec<Rc<RefCell<Option<EvalResult>>>>,
}

impl VarStorage {
    pub fn new(vars: &VarTable) -> Self {
        let mk = |_| Rc::new(RefCell::new(None));
        VarStorage {
            names: vars.names.clone(),
            current: (0..vars.len()).map(mk).collect(),
            next: (0..vars.len()).map(mk).collect(),
            caches_to_clear: Vec::new(),
        }
    }

    fn clear_caches(&self) {
        for cache in &self.caches_to_clear {
            *cache.borrow_mut() = None;
        }
    }

    /// Make `state` the current state.
    pub fn load(&self, state: &State) {
        debug_assert_eq!(state.len(), self.current.len());
        for (reg, value) in self.current.iter().zip(state.iter()) {
            *reg.borrow_mut() = Some(value.clone());
        }
        self.clear_caches();
    }

    /// Unset the current state (used when enumerating initial states:
    /// reading an unassigned variable is then a QNT502 error).
    pub fn clear_current(&self) {
        for reg in &self.current {
            *reg.borrow_mut() = None;
        }
        self.clear_caches();
    }

    pub fn reset_next(&self) {
        for reg in &self.next {
            *reg.borrow_mut() = None;
        }
    }

    /// Cheap snapshot of the next-state registers (Rc clones).
    pub fn snapshot_next(&self) -> Vec<Option<Value>> {
        self.next.iter().map(|reg| reg.borrow().clone()).collect()
    }

    pub fn restore_next(&self, snapshot: &[Option<Value>]) {
        for (reg, value) in self.next.iter().zip(snapshot.iter()) {
            *reg.borrow_mut() = value.clone();
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
                reg.borrow().clone().ok_or_else(|| {
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
