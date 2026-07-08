//! Explicit-state (TLC-like) model checker core for Quint.
//!
//! Consumes the flattened JSON IR from `quint compile --target=json`
//! (parsed by the `quint-ast` crate), enumerates all successors of every
//! reachable state breadth-first, checks invariants, detects deadlocks, and
//! reconstructs counterexample traces in ITF format.

pub mod choice;
pub mod error;
pub mod eval;
pub mod explorer;
pub mod itf_out;
pub mod runner;
pub mod spec;
pub mod state;
pub mod successor;
pub mod temporal;
pub mod value;
pub mod vm;
