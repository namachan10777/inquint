//! Deliberately simple reference evaluator used to audit inquint's bytecode
//! VM. This is the last pre-VM evaluator, retained with its BTree-based value
//! representation and closure compiler so that it does not share lowering,
//! interning, canonical-container, or register-machine bugs with production.
//!
//! It is not part of the model checker's runtime or public CLI. Verification
//! tests compare its complete initial/successor sets with the production VM.

// Preserve the historical representation: symbols are now Copy, while this
// evaluator deliberately uses ordered containers around closure-bearing
// values with interior mutability.
#![allow(clippy::clone_on_copy, clippy::mutable_key_type)]

pub mod choice;
pub mod error;
pub mod eval;
pub mod state;
pub mod successor;
pub mod value;

pub use eval::Compiler;
pub use state::{State, VarStorage, VarTable};
pub use successor::enumerate;
pub use value::Value;
