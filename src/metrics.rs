//! Evaluation metrics matching the FPRM MATLAB live script.

use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ImputeMetrics {
    /// Mean squared error (MATLAB `loss` / MSE).
    pub mse: f64,
    /// `sqrt(MSE) / std(Ytest)` — MATLAB `LL`.
    pub nrmse: f64,
    /// Pearson correlation of `[YFit, Ytest]` — MATLAB `CO(1,2)`.
    pub corr: f64,
    /// Mean absolute relative error `| (y-ŷ)/ŷ |` with epsilon guard — MATLAB `Re`.
    pub mean_rel_error: f64,
    /// Mean absolute error — MATLAB `MAE`.
    pub mae: f64,
    pub n_test: usize,
}

/// Population (MATLAB `std` with default N-1 for sample) — use sample std (N-1).
pub fn sample_std(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return 0.0;
    }
    let mean = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    var.sqrt()
}

pub fn pearson_corr(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    if n < 2 {
        return f64::NAN;
    }
    let ma = a.iter().sum::<f64>() / n as f64;
    let mb = b.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..n {
        let xa = a[i] - ma;
        let xb = b[i] - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    if da == 0.0 || db == 0.0 {
        return f64::NAN;
    }
    num / (da.sqrt() * db.sqrt())
}

/// Compute metrics. Relative error skips points with `|y_hat| < eps`.
pub fn compute_metrics(y_true: &[f64], y_hat: &[f64], rel_eps: f64) -> ImputeMetrics {
    assert_eq!(y_true.len(), y_hat.len());
    let n = y_true.len();
    let mut mse = 0.0;
    let mut mae = 0.0;
    let mut rel_sum = 0.0;
    let mut rel_n = 0usize;
    for i in 0..n {
        let e = y_true[i] - y_hat[i];
        mse += e * e;
        mae += e.abs();
        if y_hat[i].abs() >= rel_eps {
            rel_sum += (e / y_hat[i]).abs();
            rel_n += 1;
        }
    }
    mse /= n as f64;
    mae /= n as f64;
    let std_y = sample_std(y_true);
    let nrmse = if std_y > 0.0 {
        mse.sqrt() / std_y
    } else {
        f64::NAN
    };
    let mean_rel_error = if rel_n > 0 {
        rel_sum / rel_n as f64
    } else {
        f64::NAN
    };
    ImputeMetrics {
        mse,
        nrmse,
        corr: pearson_corr(y_true, y_hat),
        mean_rel_error,
        mae,
        n_test: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn metrics_known_vector() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let yh = [1.1, 1.9, 3.2, 3.8];
        let m = compute_metrics(&y, &yh, 1e-12);
        assert_abs_diff_eq!(m.mae, 0.15, epsilon = 1e-12);
        assert_abs_diff_eq!(m.mse, 0.025, epsilon = 1e-12);
        assert!(m.corr > 0.99);
    }
}
