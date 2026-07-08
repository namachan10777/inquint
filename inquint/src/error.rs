//! Runtime errors, aligned with quint's error codes where they exist.

use quint_ast::QuintId;

#[derive(thiserror::Error, Debug, Clone, PartialEq)]
#[error("[{code}] {message}")]
pub struct QuintError {
    pub code: &'static str,
    pub message: String,
    /// Expression ids from innermost to outermost, for diagnostics.
    pub trace: Vec<QuintId>,
}

impl QuintError {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        QuintError {
            code,
            message: message.into(),
            trace: Vec::new(),
        }
    }

    pub fn with_id(mut self, id: QuintId) -> Self {
        self.trace.push(id);
        self
    }
}

/// QNT501: construct not supported by this runtime.
pub fn unsupported(what: impl std::fmt::Display) -> QuintError {
    QuintError::new("QNT501", format!("{what} is not supported by inquint"))
}

/// QNT502: reading a state variable that has no value yet (during init).
pub fn undefined_var(name: &str) -> QuintError {
    QuintError::new("QNT502", format!("variable {name} is not set"))
}

/// QNT601: i64 overflow in checked arithmetic (v1 limitation).
pub fn overflow(op: &str) -> QuintError {
    QuintError::new(
        "QNT601",
        format!("integer overflow in {op} (inquint v1 uses 64-bit integers)"),
    )
}
