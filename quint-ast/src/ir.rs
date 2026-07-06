//! IR node types, mirroring the shapes in quint's `src/ir/quintIr.ts`.

use serde::Deserialize;
use std::sync::Arc;

pub type QuintId = u64;
/// Interned-ish name: cheap to clone, hashable.
pub type QuintName = Arc<str>;

#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "kind")]
pub enum QuintEx {
    #[serde(rename = "name")]
    Name { id: QuintId, name: QuintName },

    #[serde(rename = "bool")]
    Bool { id: QuintId, value: bool },

    /// Integer literal. quint uses bigints; v1 restricts to i64 and fails
    /// loudly at load time on anything larger.
    #[serde(rename = "int")]
    Int { id: QuintId, value: i64 },

    #[serde(rename = "str")]
    Str { id: QuintId, value: QuintName },

    #[serde(rename = "app")]
    App {
        id: QuintId,
        opcode: QuintName,
        args: Vec<QuintEx>,
    },

    #[serde(rename = "lambda")]
    Lambda {
        id: QuintId,
        params: Vec<LambdaParam>,
        expr: Box<QuintEx>,
    },

    #[serde(rename = "let")]
    Let {
        id: QuintId,
        opdef: Box<OpDef>,
        expr: Box<QuintEx>,
    },
}

impl QuintEx {
    pub fn id(&self) -> QuintId {
        match self {
            Self::Name { id, .. }
            | Self::Bool { id, .. }
            | Self::Int { id, .. }
            | Self::Str { id, .. }
            | Self::App { id, .. }
            | Self::Lambda { id, .. }
            | Self::Let { id, .. } => *id,
        }
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct LambdaParam {
    pub id: QuintId,
    pub name: QuintName,
}

#[derive(Deserialize, Debug, Clone)]
pub struct OpDef {
    pub id: QuintId,
    pub name: QuintName,
    pub qualifier: OpQualifier,
    pub expr: QuintEx,
    pub depth: Option<u64>,
}

#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpQualifier {
    #[serde(rename = "puredef")]
    PureDef,
    #[serde(rename = "pureval")]
    PureVal,
    #[serde(rename = "def")]
    Def,
    #[serde(rename = "val")]
    Val,
    #[serde(rename = "nondet")]
    Nondet,
    #[serde(rename = "action")]
    Action,
    #[serde(rename = "run")]
    Run,
    #[serde(rename = "temporal")]
    Temporal,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "kind")]
pub enum Declaration {
    #[serde(rename = "def")]
    OpDef(OpDef),

    #[serde(rename = "var")]
    Var { id: QuintId, name: QuintName },

    #[serde(rename = "assume")]
    Assume { id: QuintId, name: QuintName },

    #[serde(rename = "typedef")]
    TypeDef { id: QuintId },

    /// Constants should have been lowered by flattening; an error is raised
    /// only if one is actually referenced during compilation.
    #[serde(rename = "const")]
    Const { id: QuintId, name: QuintName },

    #[serde(rename = "import")]
    Import {},
    #[serde(rename = "instance")]
    Instance {},
    #[serde(rename = "export")]
    Export {},
}

#[derive(Deserialize, Debug)]
pub struct QuintModule {
    pub name: QuintName,
    pub declarations: Vec<Declaration>,
}
