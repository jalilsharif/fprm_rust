//! Exact squared-exponential Gaussian process regression (MATLAB `fitrgp`-like).
//!
//! Model class (pipeline parity, not bit-identical hyperparameters):
//! - Kernel: `k(x,x') = σ_f² exp(-||x-x'||²/(2ℓ²)) + σ_n² δ`
//! - Constant (empirical) mean subtracted from training targets
//! - Exact Cholesky fit/predict for n ≲ a few thousand
//! - Hyperparameters optimized by L-BFGS-style gradient ascent on log-marginal likelihood
//!
//! Differences vs MATLAB `fitrgp`: optimizer path, initial values, and joint `beta`
//! estimation may yield small prediction drift.

use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GprError {
    #[error("GPR requires at least 2 training points (got {0})")]
    TooFewPoints(usize),
    #[error("feature dimension mismatch: train has {expected} cols, got {got}")]
    DimMismatch { expected: usize, got: usize },
    #[error("Cholesky failed (matrix not SPD); try larger noise")]
    CholeskyFailed,
    #[error("hyperparameter optimization failed to improve likelihood")]
    OptFailed,
}

#[derive(Debug, Clone, Copy)]
pub struct KernelParams {
    pub length_scale: f64,
    pub signal_std: f64,
    pub noise_std: f64,
}

impl Default for KernelParams {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            signal_std: 1.0,
            noise_std: 0.1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GprModel {
    pub x_train: Array2<f64>,
    pub y_mean: f64,
    /// `α = K^{-1} (y - mean)`
    pub alpha: Array1<f64>,
    pub params: KernelParams,
    /// Lower-triangular Cholesky factor of K (for optional variance; unused in mean predict).
    pub l_factor: Array2<f64>,
}

/// Build Gram matrix. Parallel over rows when `n` is large enough to amortize.
pub fn kernel_matrix(x: &Array2<f64>, params: &KernelParams, with_noise: bool) -> Array2<f64> {
    let n = x.nrows();
    let sf2 = params.signal_std * params.signal_std;
    let inv_2l2 = 1.0 / (2.0 * params.length_scale * params.length_scale);
    let sn2 = if with_noise {
        params.noise_std * params.noise_std
    } else {
        0.0
    };

    let fill_row = |i: usize| -> Vec<f64> {
        let xi = x.row(i).to_owned();
        let mut row = vec![0.0; n];
        for j in 0..=i {
            let xj = x.row(j);
            let d2 = xi
                .iter()
                .zip(xj.iter())
                .map(|(a, b)| {
                    let d = a - b;
                    d * d
                })
                .sum::<f64>();
            let kij = sf2 * (-d2 * inv_2l2).exp();
            row[j] = if i == j { kij + sn2 } else { kij };
        }
        row
    };

    let rows: Vec<Vec<f64>> = if n >= 128 {
        (0..n).into_par_iter().map(fill_row).collect()
    } else {
        (0..n).map(fill_row).collect()
    };

    let mut k = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let v = rows[i][j];
            k[(i, j)] = v;
            k[(j, i)] = v;
        }
    }
    k
}

fn kernel_vec(x_train: &Array2<f64>, x_new: &[f64], params: &KernelParams) -> Array1<f64> {
    let n = x_train.nrows();
    let sf2 = params.signal_std * params.signal_std;
    let inv_2l2 = 1.0 / (2.0 * params.length_scale * params.length_scale);
    let mut k = Array1::<f64>::zeros(n);
    for i in 0..n {
        let d2 = x_train
            .row(i)
            .iter()
            .zip(x_new.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>();
        k[i] = sf2 * (-d2 * inv_2l2).exp();
    }
    k
}

/// In-place Cholesky (lower triangular). Returns error if not SPD.
fn cholesky_lower(a: &mut Array2<f64>) -> Result<(), GprError> {
    let n = a.nrows();
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[(i, j)];
            for k in 0..j {
                sum -= a[(i, k)] * a[(j, k)];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(GprError::CholeskyFailed);
                }
                a[(i, j)] = sum.sqrt();
            } else {
                a[(i, j)] = sum / a[(j, j)];
            }
        }
        for j in (i + 1)..n {
            a[(i, j)] = 0.0;
        }
    }
    Ok(())
}

fn chol_solve(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut y = Array1::<f64>::zeros(n);
    // forward: L y = b
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[(i, k)] * y[k];
        }
        y[i] = s / l[(i, i)];
    }
    // back: L^T x = y
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[(k, i)] * x[k];
        }
        x[i] = s / l[(i, i)];
    }
    x
}

fn log_det_chol(l: &Array2<f64>) -> f64 {
    let mut s = 0.0;
    for i in 0..l.nrows() {
        s += l[(i, i)].ln();
    }
    2.0 * s
}

/// Log marginal likelihood for centered targets (zero-mean GP).
fn log_marginal_likelihood(x: &Array2<f64>, y: &Array1<f64>, params: &KernelParams) -> Result<f64, GprError> {
    let n = x.nrows() as f64;
    let mut k = kernel_matrix(x, params, true);
    cholesky_lower(&mut k)?;
    let alpha = chol_solve(&k, y);
    let data_fit = y.dot(&alpha);
    let lml = -0.5 * data_fit - 0.5 * log_det_chol(&k) - 0.5 * n * (2.0 * std::f64::consts::PI).ln();
    Ok(lml)
}

fn params_from_log(theta: &[f64; 3]) -> KernelParams {
    KernelParams {
        length_scale: theta[0].exp(),
        signal_std: theta[1].exp(),
        noise_std: theta[2].exp().max(1e-6),
    }
}

fn init_theta(x: &Array2<f64>, y: &Array1<f64>) -> [f64; 3] {
    let n = x.nrows().min(64);
    let mut dists = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d2 = x
                .row(i)
                .iter()
                .zip(x.row(j).iter())
                .map(|(a, b)| {
                    let d = a - b;
                    d * d
                })
                .sum::<f64>();
            if d2 > 0.0 {
                dists.push(d2.sqrt());
            }
        }
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = if dists.is_empty() {
        1.0
    } else {
        dists[dists.len() / 2].max(1e-3)
    };
    let y_std = {
        let mean = y.mean().unwrap_or(0.0);
        let v = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / y.len().max(1) as f64;
        v.sqrt().max(1e-3)
    };
    // log(ell), log(sf), log(sn)
    [med.ln(), y_std.ln(), (0.1 * y_std).ln()]
}

/// Finite-difference gradient of LML w.r.t. log-hyperparameters.
fn lml_grad(x: &Array2<f64>, y: &Array1<f64>, theta: &[f64; 3]) -> Result<(f64, [f64; 3]), GprError> {
    let eps = 1e-4;
    let f0 = log_marginal_likelihood(x, y, &params_from_log(theta))?;
    let mut g = [0.0; 3];
    for i in 0..3 {
        let mut th = *theta;
        th[i] += eps;
        let f1 = match log_marginal_likelihood(x, y, &params_from_log(&th)) {
            Ok(v) => v,
            Err(_) => f0,
        };
        g[i] = (f1 - f0) / eps;
    }
    Ok((f0, g))
}

fn optimize_hyperparams(x: &Array2<f64>, y: &Array1<f64>) -> Result<KernelParams, GprError> {
    // Subsample for hyperparameter search when n is large (fit still uses all points).
    let (x_opt, y_opt) = if x.nrows() > 120 {
        let step = (x.nrows() / 100).max(1);
        let idx: Vec<usize> = (0..x.nrows()).step_by(step).take(100).collect();
        let mut xo = Array2::<f64>::zeros((idx.len(), x.ncols()));
        let mut yo = Array1::<f64>::zeros(idx.len());
        for (r, &i) in idx.iter().enumerate() {
            xo.row_mut(r).assign(&x.row(i));
            yo[r] = y[i];
        }
        (xo, yo)
    } else {
        (x.clone(), y.clone())
    };

    let mut theta = init_theta(&x_opt, &y_opt);
    let mut best = params_from_log(&theta);
    let mut best_lml = log_marginal_likelihood(&x_opt, &y_opt, &best).unwrap_or(f64::NEG_INFINITY);

    // Adam on log-params (maximize LML)
    let mut m = [0.0; 3];
    let mut v = [0.0; 3];
    let lr = 0.08;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let mut improved = best_lml.is_finite();

    for t in 1..=40 {
        let (f, g) = match lml_grad(&x_opt, &y_opt, &theta) {
            Ok(v) => v,
            Err(_) => break,
        };
        if f > best_lml {
            best_lml = f;
            best = params_from_log(&theta);
            improved = true;
        }
        let gnorm: f64 = g.iter().map(|gi| gi * gi).sum::<f64>().sqrt();
        if gnorm < 1e-5 {
            break;
        }
        for i in 0..3 {
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
            let mhat = m[i] / (1.0 - beta1.powi(t));
            let vhat = v[i] / (1.0 - beta2.powi(t));
            theta[i] += lr * mhat / (vhat.sqrt() + 1e-8);
        }
        // keep log-noise in a stable range
        theta[2] = theta[2].clamp(-8.0, 2.0);
    }

    if !improved && best_lml.is_finite() {
        return Ok(best);
    }
    if !best_lml.is_finite() {
        return Err(GprError::OptFailed);
    }
    Ok(best)
}

impl GprModel {
    /// Fit exact SE-GPR. `x`: n×d, `y`: length n.
    pub fn fit(x: Array2<f64>, y: Array1<f64>) -> Result<Self, GprError> {
        Self::fit_with_params(x, y, None)
    }

    /// Fit with optional fixed kernel params (skip optimization).
    pub fn fit_with_params(
        x: Array2<f64>,
        y: Array1<f64>,
        fixed: Option<KernelParams>,
    ) -> Result<Self, GprError> {
        let n = x.nrows();
        if n < 2 {
            return Err(GprError::TooFewPoints(n));
        }
        assert_eq!(y.len(), n);

        let y_mean = y.mean().unwrap_or(0.0);
        let y_c = &y - y_mean;

        let params = match fixed {
            Some(p) => p,
            None => optimize_hyperparams(&x, &y_c)?,
        };

        let mut k = kernel_matrix(&x, &params, true);
        // jitter for numerical stability
        for i in 0..n {
            k[(i, i)] += 1e-10;
        }
        cholesky_lower(&mut k)?;
        let alpha = chol_solve(&k, &y_c);

        Ok(Self {
            x_train: x,
            y_mean,
            alpha,
            params,
            l_factor: k,
        })
    }

    pub fn predict(&self, x_test: &Array2<f64>) -> Result<Array1<f64>, GprError> {
        if x_test.ncols() != self.x_train.ncols() {
            return Err(GprError::DimMismatch {
                expected: self.x_train.ncols(),
                got: x_test.ncols(),
            });
        }
        let m = x_test.nrows();
        let mut out = Array1::<f64>::zeros(m);
        for i in 0..m {
            let row = x_test.row(i).to_vec();
            let kstar = kernel_vec(&self.x_train, &row, &self.params);
            out[i] = self.y_mean + kstar.dot(&self.alpha);
        }
        Ok(out)
    }

    /// Parallel predict over test rows (sequential for small batches).
    pub fn predict_par(&self, x_test: &Array2<f64>) -> Result<Array1<f64>, GprError> {
        if x_test.ncols() != self.x_train.ncols() {
            return Err(GprError::DimMismatch {
                expected: self.x_train.ncols(),
                got: x_test.ncols(),
            });
        }
        if x_test.nrows() < 64 {
            return self.predict(x_test);
        }
        let preds: Vec<f64> = (0..x_test.nrows())
            .into_par_iter()
            .map(|i| {
                let row = x_test.row(i).to_vec();
                let kstar = kernel_vec(&self.x_train, &row, &self.params);
                self.y_mean + kstar.dot(&self.alpha)
            })
            .collect();
        Ok(Array1::from_vec(preds))
    }
}

/// Column-wise z-score using training statistics (MATLAB `zscore` on train, apply to test).
#[derive(Debug, Clone)]
pub struct ZScore {
    pub mu: Array1<f64>,
    pub sigma: Array1<f64>,
}

impl ZScore {
    pub fn fit(x: &Array2<f64>) -> Self {
        let mu = x.mean_axis(Axis(0)).unwrap();
        let mut sigma = Array1::<f64>::zeros(x.ncols());
        let n = x.nrows() as f64;
        for j in 0..x.ncols() {
            let m = mu[j];
            let var = x
                .column(j)
                .iter()
                .map(|v| (v - m).powi(2))
                .sum::<f64>()
                / n; // MATLAB zscore uses N (population) in recent releases; N-1 also ok
            let s = var.sqrt();
            sigma[j] = if s == 0.0 { 1.0 } else { s };
        }
        Self { mu, sigma }
    }

    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut z = x.clone();
        for j in 0..x.ncols() {
            for i in 0..x.nrows() {
                z[(i, j)] = (x[(i, j)] - self.mu[j]) / self.sigma[j];
            }
        }
        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn gpr_recovers_smooth_function() {
        let n = 40;
        let mut x = Array2::<f64>::zeros((n, 1));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = i as f64 / 10.0;
            x[(i, 0)] = t;
            y[i] = (t * 1.3).sin() + 0.05 * t;
        }
        let model = GprModel::fit(x.clone(), y.clone()).unwrap();
        let pred = model.predict(&x).unwrap();
        let mse: f64 = pred
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / n as f64;
        assert!(mse < 0.05, "mse={mse}");
        assert_abs_diff_eq!(pred[0], y[0], epsilon = 0.3);
    }
}
