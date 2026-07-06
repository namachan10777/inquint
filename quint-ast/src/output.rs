//! The top-level object printed by `quint compile --target=json`.

use crate::ir::{QuintModule, QuintName};
use crate::table::LookupTable;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct CompiledOutput {
    #[serde(default)]
    pub stage: Option<String>,
    pub modules: Vec<QuintModule>,
    pub table: LookupTable,
    pub main: QuintName,
    #[serde(default)]
    pub errors: Vec<serde_json::Value>,
}

#[derive(thiserror::Error, Debug)]
pub enum LoadError {
    #[error("failed to parse quint compile output: {0}")]
    Json(#[from] serde_json::Error),
    #[error("quint compile reported errors: {0}")]
    CompileErrors(String),
    #[error("expected exactly one flattened module, found {0} (was --flatten disabled?)")]
    NotFlattened(usize),
}

impl CompiledOutput {
    pub fn load(json: &str) -> Result<Self, LoadError> {
        let out: CompiledOutput = serde_json::from_str(json)?;
        if !out.errors.is_empty() {
            return Err(LoadError::CompileErrors(
                serde_json::to_string(&out.errors).unwrap_or_default(),
            ));
        }
        if out.modules.len() != 1 {
            return Err(LoadError::NotFlattened(out.modules.len()));
        }
        Ok(out)
    }

    /// The single flattened module.
    pub fn module(&self) -> &QuintModule {
        &self.modules[0]
    }
}
