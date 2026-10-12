//! Network of diffusively coupled Lorenz systems — MATLAB `multiCoupledLorenz2` + mlx.

use ndarray::Array2;

#[derive(Debug, Clone, Copy)]
pub struct LorenzParams {
    pub sigma: f64,
    pub rho: f64,
    pub beta: f64,
    pub epsilon: f64,
    pub n: usize,
}

impl Default for LorenzParams {
    fn default() -> Self {
        Self {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
            epsilon: 0.1,
            n: 40,
        }
    }
}

/// MATLAB-style initial condition: `0.3:0.3:(0.3+0.3*(3*N-1))`.
pub fn default_initial(n: usize) -> Vec<f64> {
    let dim = 3 * n;
    (0..dim).map(|i| 0.3 * (i as f64 + 1.0)).collect()
}

/// RHS of multi-coupled Lorenz. State layout `[x1,y1,z1,...,xN,yN,zN]`.
///
/// Coupling (vectorized, identical to double loop): for each i,
/// `dx += epsilon * (sum_j x_j - N * x_i)` when N > 1.
pub fn rhs(x: &[f64], p: &LorenzParams) -> Vec<f64> {
    let n = p.n;
    debug_assert_eq!(x.len(), 3 * n);
    let mut dx = vec![0.0; 3 * n];

    let sum_x: f64 = (0..n).map(|i| x[3 * i]).sum();

    for i in 0..n {
        let xi = x[3 * i];
        let yi = x[3 * i + 1];
        let zi = x[3 * i + 2];
        let mut dxi = p.sigma * (yi - xi);
        let dyi = xi * (p.rho - zi) - yi;
        let dzi = xi * yi - p.beta * zi;
        if n > 1 {
            // sum_{j!=i}(x_j - x_i) = sum_j x_j - N*x_i
            dxi += p.epsilon * (sum_x - (n as f64) * xi);
        }
        dx[3 * i] = dxi;
        dx[3 * i + 1] = dyi;
        dx[3 * i + 2] = dzi;
    }
    dx
}

fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    for i in 0..x.len() {
        y[i] += a * x[i];
    }
}

fn scaled_add(y: &[f64], scales: &[(f64, &[f64])]) -> Vec<f64> {
    let mut out = y.to_vec();
    for &(a, x) in scales {
        axpy(a, x, &mut out);
    }
    out
}

/// Classic RK4 step.
fn rk4_step(x: &[f64], dt: f64, p: &LorenzParams) -> Vec<f64> {
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
    let _ = scaled_add;
    out
}

/// Integrate with fixed-step RK4 from `t0` to `t1`, recording every step.
///
/// Not bit-identical to MATLAB `ode45` (adaptive). Drop the first `burnin_samples`
/// recorded rows to mirror `X=X(101:end,:)`.
pub fn integrate(
    p: &LorenzParams,
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

/// MATLAB mlx defaults: tspan [10,40], burn-in 100 samples, dt=0.01.
pub fn simulate_default() -> Array2<f64> {
    let p = LorenzParams::default();
    let x0 = default_initial(p.n);
    integrate(&p, &x0, 10.0, 40.0, 0.01, 100)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn rhs_single_uncoupled() {
        let p = LorenzParams {
            n: 1,
            epsilon: 0.0,
            ..Default::default()
        };
        let x = vec![1.0, 2.0, 3.0];
        let d = rhs(&x, &p);
        assert_abs_diff_eq!(d[0], 10.0 * (2.0 - 1.0), epsilon = 1e-12);
        assert_abs_diff_eq!(d[1], 1.0 * (28.0 - 3.0) - 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d[2], 1.0 * 2.0 - (8.0 / 3.0) * 3.0, epsilon = 1e-12);
    }

    #[test]
    fn coupling_matches_double_loop() {
        let p = LorenzParams {
            n: 3,
            epsilon: 0.2,
            ..Default::default()
        };
        let x = vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0];
        let d = rhs(&x, &p);
        // For i=0: sum_{j!=0}(xj-x0) = (2-1)+(4-1)=4
        let expected_coup0 = 0.2 * 4.0;
        let base_dx0 = 10.0 * (0.0 - 1.0);
        assert_abs_diff_eq!(d[0], base_dx0 + expected_coup0, epsilon = 1e-12);
    }
}
