//! 3-variable discrete ecological map — MATLAB `3D ecology system.mlx`.

use ndarray::Array2;

/// Iterate the 3-species map for `steps` iterations (plus initial), then drop `burnin` rows.
///
/// MATLAB:
/// ```text
/// y1(t+1) = y1*(4 - 4*y1 - 2*y2 - 0.4*y3)
/// y2(t+1) = y2*(3.1 - 0.31*y1 - 3.1*y2 - 0.93*y3)
/// y3(t+1) = y3*(2.12 + 0.636*y1 + 0.636*y2 - 2.12*y3)
/// ```
/// with `y1=0.4, y2=0.5, y3=0.6`, `t=1:600`, then `x = x(101:end,:)`.
pub fn simulate(steps: usize, burnin: usize) -> Array2<f64> {
    let mut y1 = vec![0.4];
    let mut y2 = vec![0.5];
    let mut y3 = vec![0.6];
    for _ in 0..steps {
        let a = *y1.last().unwrap();
        let b = *y2.last().unwrap();
        let c = *y3.last().unwrap();
        y1.push(a * (4.0 - 4.0 * a - 2.0 * b - 0.4 * c));
        y2.push(b * (3.1 - 0.31 * a - 3.1 * b - 0.93 * c));
        y3.push(c * (2.12 + 0.636 * a + 0.636 * b - 2.12 * c));
    }
    // length = steps+1 (indices 0..=steps). MATLAB keeps 101:end → burnin=100 drops first 100.
    let total = y1.len();
    let start = burnin.min(total);
    let n = total - start;
    let mut out = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        out[(i, 0)] = y1[start + i];
        out[(i, 1)] = y2[start + i];
        out[(i, 2)] = y3[start + i];
    }
    out
}

/// Default MATLAB settings: 600 steps, burn-in 100 → 501 rows.
pub fn simulate_default() -> Array2<f64> {
    simulate(600, 100)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn ecology_checkpoint() {
        // Hand-iterate one step from (0.4, 0.5, 0.6)
        let y1 = 0.4 * (4.0 - 4.0 * 0.4 - 2.0 * 0.5 - 0.4 * 0.6);
        let y2 = 0.5 * (3.1 - 0.31 * 0.4 - 3.1 * 0.5 - 0.93 * 0.6);
        let y3 = 0.6 * (2.12 + 0.636 * 0.4 + 0.636 * 0.5 - 2.12 * 0.6);
        let x = simulate(1, 0);
        assert_eq!(x.nrows(), 2);
        assert_abs_diff_eq!(x[(1, 0)], y1, epsilon = 1e-12);
        assert_abs_diff_eq!(x[(1, 1)], y2, epsilon = 1e-12);
        assert_abs_diff_eq!(x[(1, 2)], y3, epsilon = 1e-12);
    }

    #[test]
    fn default_shape() {
        let x = simulate_default();
        assert_eq!(x.nrows(), 501);
        assert_eq!(x.ncols(), 3);
    }
}
