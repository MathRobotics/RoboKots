//! Errors from native validation, independent of Python and NumPy.
use std::fmt;

/// Validation failure in a model, motion, output request, or computation state.
/// All current native validation failures map to ValueError at the Python boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    message: String,
}

impl Error {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

pub type CoreResult<T> = std::result::Result<T, Error>;
