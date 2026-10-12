use approx::assert_abs_diff_eq;
use fprm_rust::systems::lorenz::{rhs, LorenzParams};

#[test]
fn matches_matlab_double_loop_coupling() {
    let p = LorenzParams {
        sigma: 10.0,
        rho: 28.0,
        beta: 8.0 / 3.0,
        epsilon: 0.1,
        n: 4,
    };
    let x: Vec<f64> = (0..12).map(|i| 0.3 * (i as f64 + 1.0)).collect();

    // Reference: naive double loop as in multiCoupledLorenz2.m
    let mut ref_dx = vec![0.0; 12];
    for i in 0..p.n {
        let xi = x[3 * i];
        let yi = x[3 * i + 1];
        let zi = x[3 * i + 2];
        let mut dx = p.sigma * (yi - xi);
        let dy = xi * (p.rho - zi) - yi;
        let dz = xi * yi - p.beta * zi;
        let mut coupling = 0.0;
        for j in 0..p.n {
            if j != i {
                coupling += x[3 * j] - xi;
            }
        }
        dx += p.epsilon * coupling;
        ref_dx[3 * i] = dx;
        ref_dx[3 * i + 1] = dy;
        ref_dx[3 * i + 2] = dz;
    }

    let d = rhs(&x, &p);
    for i in 0..12 {
        assert_abs_diff_eq!(d[i], ref_dx[i], epsilon = 1e-12);
    }
}
