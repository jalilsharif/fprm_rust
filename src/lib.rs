//! FPRM: phase-space reconstruction + Gaussian process imputation (MATLAB port).
//!
//! See the crate README for MATLAB → Rust file mapping and parity notes.

pub mod embed;
pub mod fprm;
pub mod gpr;
pub mod io;
pub mod metrics;
pub mod systems;

pub use embed::{phase_space_reconstruct, EmbedError};
pub use fprm::{
    impute_from_matrix, impute_matrix_columns, impute_pair, impute_pair_reps, FprmAggregate,
    FprmConfig, FprmError, FprmResult,
};
pub use gpr::{GprError, GprModel, KernelParams, ZScore};
pub use io::{load_csv, load_table, load_xlsx, write_csv, IoError};
pub use metrics::{compute_metrics, ImputeMetrics};
