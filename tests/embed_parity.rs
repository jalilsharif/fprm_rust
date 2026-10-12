use approx::assert_abs_diff_eq;
use fprm_rust::phase_space_reconstruct;

#[test]
fn matlab_phasparecon_transpose_example() {
    // Fixed series matching a hand-computed MATLAB PhaSpaRecon then transpose.
    let s: Vec<f64> = (0..20).map(|i| (i as f64) * 0.5 - 1.0).collect();
    let tau = 3;
    let m = 5;
    let xn = phase_space_reconstruct(&s, tau, m).unwrap();
    // n_states = 20 - 4*3 = 8
    assert_eq!(xn.shape(), &[8, 5]);
    // First state: s[0], s[3], s[6], s[9], s[12]
    for (j, &idx) in [0usize, 3, 6, 9, 12].iter().enumerate() {
        assert_abs_diff_eq!(xn[(0, j)], s[idx], epsilon = 1e-15);
    }
    // Last state starts at t=7: s[7], s[10], s[13], s[16], s[19]
    for (j, &idx) in [7usize, 10, 13, 16, 19].iter().enumerate() {
        assert_abs_diff_eq!(xn[(7, j)], s[idx], epsilon = 1e-15);
    }
}
