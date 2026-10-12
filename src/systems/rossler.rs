//! Multi-coupled Rössler systems.
//!
//! The MATLAB `Coupled Rossler system.mlx` is a stub that copies Lorenz parameters and
//! calls `multiCoupledLorenz`. This module implements a real diffusively coupled Rössler
//! network:
//!   dx = -y - z + ε Σ_{j≠i}(x_j - x_i)
//!   dy = x + a y
//!   dz = b + z (x - c)

use ndarray::Array2;

#[derive(Debug, Clone, Copy)]
pub struct RosslerParams {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub epsilon: f64,
    pub n: usize,
}

impl Default for RosslerParams {
    fn default() -> Self {
        Self {
            a: 0.2,
            b: 0.2,
            c: 5.7,
            epsilon: 0.1,
            n: 40,
        }
    }
}

pub fn default_initial(n: usize) -> Vec<f64> {
    let dim = 3 * n;
    (0..dim).map(|i| 0.3 * (i as f64 + 1.0)).collect()
}

pub fn rhs(x: &[f64], p: &RosslerParams) -> Vec<f64> {
    let n = p.n;
    debug_assert_eq!(x.len(), 3 * n);
    let mut dx = vec![0.0; 3 * n];
    let sum_x: f64 = (0..n).map(|i| x[3 * i]).sum();

    for i in 0..n {
        let xi = x[3 * i];
        let yi = x[3 * i + 1];
        let zi = x[3 * i + 2];
        let mut dxi = -yi - zi;
        let dyi = xi + p.a * yi;
        let dzi = p.b + zi * (xi - p.c);
        if n > 1 {
            dxi += p.epsilon * (sum_x - (n as f64) * xi);
        }
        dx[3 * i] = dxi;
        dx[3 * i + 1] = dyi;
        dx[3 * i + 2] = dzi;
    }
    dx
}

fn rk4_step(x: &[f64], dt: f64, p: &RosslerParams) -> Vec<f64> {
    let k1 = rhs(x, p);
    let x2: Vec<f64> = x.iter().zip(k1.iter()).map(|(xi, k)| xi + 0.5 * dt * k).collect();
    let k2 = rhs(&x2, p);
    let x3: Vec<f64> = x.iter().zip(k2.iter()).map(|(xi, k)| xi + 0.5 * dt * k).collect();
    let k3 = rhs(&x3, p);
    let x4: Vec<f64> = x.iter().zip(k3.iter()).map(|(xi, k)| xi + dt * k).collect();
    let k4 = rhs(&x4, p);
    let mut out = x.to_vec();
    for i in 0..x.len() {
        out[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    out
}

pub fn integrate(
    p: &RosslerParams,
    x0: &[f64],
    t0: f64,
    t1: f64,
    dt: f64,
    burnin_samples: usize,
) -> Array2<f64> {
    assert_eq!(x0.len(), 3 * p.n);
    let mut t = t0;
    let mut x = x0.to_vec();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    rows.push(x.clone());
    while t < t1 - 0.5 * dt {
        let step = dt.min(t1 - t);
        x = rk4_step(&x, step, p);
        t += step;
        rows.push(x.clone());
    }
    let start = burnin_samples.min(rows.len());
    let n = rows.len() - start;
    let dim = 3 * p.n;
    let mut out = Array2::<f64>::zeros((n, dim));
    for (i, row) in rows[start..].iter().enumerate() {
        for j in 0..dim {
            out[(i, j)] = row[j];
        }
    }
    out
}

pub fn simulate_default() -> Array2<f64> {
    let p = RosslerParams::default();
    let x0 = default_initial(p.n);
    integrate(&p, &x0, 10.0, 40.0, 0.01, 100)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn rhs_single() {
        let p = RosslerParams {
            n: 1,
            epsilon: 0.0,
            ..Default::default()
        };
        let x = vec![1.0, 2.0, 3.0];
        let d = rhs(&x, &p);
        assert_abs_diff_eq!(d[0], -2.0 - 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d[1], 1.0 + 0.2 * 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d[2], 0.2 + 3.0 * (1.0 - 5.7), epsilon = 1e-12);
    }
}
