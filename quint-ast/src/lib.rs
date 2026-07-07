//! Serde types for the flattened JSON IR emitted by `quint compile --target=json`.
//!
//! The input is a single flattened module: instances and constants have
//! already been lowered by the quint compiler, and every name reference
//! resolves through the [`LookupTable`] by the reference node's id. Builtin
//! operators are *not* in the table — they are dispatched by opcode string.

pub mod slab;
pub mod ir;
pub mod output;
pub mod symbol;
pub mod table;

pub use ir::{
    Declaration, LambdaParam, OpDef, OpQualifier, QuintEx, QuintId, QuintModule, QuintName,
};
pub use output::{CompiledOutput, LoadError};
pub use symbol::Symbol;
pub use table::{LookupDefinition, LookupTable};
