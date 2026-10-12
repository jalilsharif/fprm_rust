use approx::assert_abs_diff_eq;
use fprm_rust::systems::ecology;

#[test]
fn ecology_default_shape_and_finite() {
    let x = ecology::simulate_default();
    assert_eq!(x.nrows(), 501);
    assert_eq!(x.ncols(), 3);
    assert!(x.iter().all(|v| v.is_finite()));
}

#[test]
fn ecology_first_post_burnin_deterministic() {
    // Full trajectory without burn-in, then compare index 100 to burnin=100 first row.
    let full = ecology::simulate(600, 0);
    let trimmed = ecology::simulate(600, 100);
    for j in 0..3 {
        assert_abs_diff_eq!(trimmed[(0, j)], full[(100, j)], epsilon = 1e-14);
    }
}
