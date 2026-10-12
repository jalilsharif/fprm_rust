//! FPRM imputation pipeline — MATLAB `FPRM mian code.mlx`.

use ndarray::{Array1, Array2, Axis};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::Serialize;
use thiserror::Error;

use crate::embed::{phase_space_reconstruct, EmbedError};
use crate::gpr::{GprError, GprModel, KernelParams, ZScore};
use crate::metrics::{compute_metrics, ImputeMetrics};

#[derive(Debug, Error)]
pub enum FprmError {
    #[error(transparent)]
    Embed(#[from] EmbedError),
    #[error(transparent)]
    Gpr(#[from] GprError),
    #[error("donor and target embeddings have different lengths ({donor} vs {target})")]
    LengthMismatch { donor: usize, target: usize },
    #[error("holdout fraction must be in (0, 1) (got {0})")]
    BadHoldout(f64),
    #[error("column index {idx} out of range for matrix with {ncols} columns (1-based MATLAB style)")]
    BadColumn { idx: usize, ncols: usize },
    #[error("need at least 2 train and 1 test points after holdout")]
    BadSplit,
}

#[derive(Debug, Clone)]
pub struct FprmConfig {
    pub tau: usize,
    pub m: usize,
    /// Fraction held out as "missing" / test (MATLAB `Holdout`).
    pub holdout: f64,
    pub seed: u64,
    /// Relative-error denominator guard.
    pub rel_eps: f64,
    /// Optional fixed GPR kernel (skip hyperparameter search).
    pub kernel_params: Option<KernelParams>,
}

impl Default for FprmConfig {
    fn default() -> Self {
        Self {
            tau: 3,
            m: 5,
            holdout: 0.5,
            seed: 42,
            rel_eps: 1e-12,
            kernel_params: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct FprmResult {
    pub metrics: ImputeMetrics,
    pub y_test: Vec<f64>,
    pub y_fit: Vec<f64>,
    pub n_train: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct FprmAggregate {
    pub mean: ImputeMetrics,
    pub std: ImputeMetrics,
    pub reps: usize,
    pub per_rep: Vec<ImputeMetrics>,
}

fn matlab_col(mat: &Array2<f64>, one_based: usize) -> Result<Vec<f64>, FprmError> {
    if one_based == 0 || one_based > mat.ncols() {
        return Err(FprmError::BadColumn {
            idx: one_based,
            ncols: mat.ncols(),
        });
    }
    Ok(mat.column(one_based - 1).to_vec())
}

/// Holdout split like MATLAB `cvpartition(...,'Holdout',p)`: approximately `p` fraction test.
fn holdout_indices(n: usize, holdout: f64, rng: &mut ChaCha8Rng) -> Result<(Vec<usize>, Vec<usize>), FprmError> {
    if !(holdout > 0.0 && holdout < 1.0) {
        return Err(FprmError::BadHoldout(holdout));
    }
    let n_test = ((n as f64) * holdout).round() as usize;
    let n_test = n_test.clamp(1, n.saturating_sub(1));
    let mut idx: Vec<usize> = (0..n).collect();
    idx.shuffle(rng);
    let test = idx[..n_test].to_vec();
    let train = idx[n_test..].to_vec();
    if train.len() < 2 || test.is_empty() {
        return Err(FprmError::BadSplit);
    }
    Ok((train, test))
}

fn select_rows(x: &Array2<f64>, idx: &[usize]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((idx.len(), x.ncols()));
    for (r, &i) in idx.iter().enumerate() {
        out.row_mut(r).assign(&x.row(i));
    }
    out
}

fn select_vals(y: &[f64], idx: &[usize]) -> Array1<f64> {
    Array1::from_iter(idx.iter().map(|&i| y[i]))
}

/// Single FPRM run: donor series predicts target series via delay embedding + GPR.
pub fn impute_pair(
    donor: &[f64],
    target: &[f64],
    cfg: &FprmConfig,
) -> Result<FprmResult, FprmError> {
    let x = phase_space_reconstruct(donor, cfg.tau, cfg.m)?;
    let y_emb = phase_space_reconstruct(target, cfg.tau, cfg.m)?;
    if x.nrows() != y_emb.nrows() {
        return Err(FprmError::LengthMismatch {
            donor: x.nrows(),
            target: y_emb.nrows(),
        });
    }
    // MATLAB uses only first embedding coordinate of the target: `Y(:,1)`.
    let y: Vec<f64> = y_emb.column(0).to_vec();
    let n = y.len();

    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let (train_idx, test_idx) = holdout_indices(n, cfg.holdout, &mut rng)?;

    let x_train = select_rows(&x, &train_idx);
    let y_train = select_vals(&y, &train_idx);
    let x_test = select_rows(&x, &test_idx);
    let y_test = select_vals(&y, &test_idx);

    let zs = ZScore::fit(&x_train);
    let z_train = zs.transform(&x_train);
    let z_test = zs.transform(&x_test);

    let model = GprModel::fit_with_params(z_train, y_train, cfg.kernel_params)?;
    let y_fit = model.predict_par(&z_test)?;

    let metrics = compute_metrics(
        y_test.as_slice().unwrap(),
        y_fit.as_slice().unwrap(),
        cfg.rel_eps,
    );

    Ok(FprmResult {
        metrics,
        y_test: y_test.to_vec(),
        y_fit: y_fit.to_vec(),
        n_train: train_idx.len(),
    })
}

/// Monte-Carlo holdout: `reps` independent seeds starting at `cfg.seed`.
pub fn impute_pair_reps(
    donor: &[f64],
    target: &[f64],
    cfg: &FprmConfig,
    reps: usize,
) -> Result<FprmAggregate, FprmError> {
    let reps = reps.max(1);
    let results: Result<Vec<_>, _> = (0..reps)
        .into_par_iter()
        .map(|r| {
            let mut c = cfg.clone();
            c.seed = cfg.seed.wrapping_add(r as u64);
            impute_pair(donor, target, &c).map(|res| res.metrics)
        })
        .collect();
    let per_rep = results?;
    Ok(aggregate_metrics(&per_rep))
}

fn aggregate_metrics(xs: &[ImputeMetrics]) -> FprmAggregate {
    let n = xs.len() as f64;
    let mean_of = |f: fn(&ImputeMetrics) -> f64| xs.iter().map(f).sum::<f64>() / n;
    let std_of = |f: fn(&ImputeMetrics) -> f64| {
        let m = mean_of(f);
        if xs.len() < 2 {
            return 0.0;
        }
        let v = xs.iter().map(|x| (f(x) - m).powi(2)).sum::<f64>() / (n - 1.0);
        v.sqrt()
    };
    FprmAggregate {
        mean: ImputeMetrics {
            mse: mean_of(|m| m.mse),
            nrmse: mean_of(|m| m.nrmse),
            corr: mean_of(|m| m.corr),
            mean_rel_error: mean_of(|m| m.mean_rel_error),
            mae: mean_of(|m| m.mae),
            n_test: xs.first().map(|m| m.n_test).unwrap_or(0),
        },
        std: ImputeMetrics {
            mse: std_of(|m| m.mse),
            nrmse: std_of(|m| m.nrmse),
            corr: std_of(|m| m.corr),
            mean_rel_error: std_of(|m| m.mean_rel_error),
            mae: std_of(|m| m.mae),
            n_test: 0,
        },
        reps: xs.len(),
        per_rep: xs.to_vec(),
    }
}

/// Impute many target columns (1-based) against one donor column, in parallel.
pub fn impute_matrix_columns(
    data: &Array2<f64>,
    donor_col: usize,
    target_cols: &[usize],
    cfg: &FprmConfig,
    reps: usize,
) -> Result<Vec<(usize, FprmAggregate)>, FprmError> {
    let donor = matlab_col(data, donor_col)?;
    let out: Result<Vec<_>, _> = target_cols
        .par_iter()
        .map(|&tc| {
            let target = matlab_col(data, tc)?;
            let agg = impute_pair_reps(&donor, &target, cfg, reps)?;
            Ok((tc, agg))
        })
        .collect();
    out
}

/// Convenience: columns of an `Array2` as time series (rows = time).
pub fn column_series(data: &Array2<f64>, one_based: usize) -> Result<Vec<f64>, FprmError> {
    matlab_col(data, one_based)
}

/// Random holdout mask utility (for callers that need indices).
pub fn random_holdout_mask(n: usize, holdout: f64, seed: u64) -> Result<Vec<bool>, FprmError> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let (train, test) = holdout_indices(n, holdout, &mut rng)?;
    let mut mask = vec![false; n]; // false = train, true = test/missing
    for i in test {
        mask[i] = true;
    }
    let _ = train;
    Ok(mask)
}

/// Helper used by CLI: pick donor/target from matrix.
pub fn impute_from_matrix(
    data: &Array2<f64>,
    donor_col: usize,
    target_col: usize,
    cfg: &FprmConfig,
    reps: usize,
) -> Result<FprmAggregate, FprmError> {
    let donor = column_series(data, donor_col)?;
    let target = column_series(data, target_col)?;
    impute_pair_reps(&donor, &target, cfg, reps)
}

/// Mean along axis helper for tests.
pub fn mean_axis0(a: &Array2<f64>) -> Array1<f64> {
    a.mean_axis(Axis(0)).unwrap()
}

/// Seeded RNG draw (exposed for tests).
pub fn seeded_u64(seed: u64) -> u64 {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.gen()
}
