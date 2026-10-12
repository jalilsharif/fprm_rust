use fprm_rust::systems::ecology;
use fprm_rust::{impute_pair, impute_pair_reps, FprmConfig};

#[test]
fn fprm_on_ecology_finite_metrics() {
    // Modest length keeps exact-GPR smoke tests fast in debug builds.
    let x = ecology::simulate(180, 40);
    let donor: Vec<f64> = x.column(1).to_vec();
    let target: Vec<f64> = x.column(0).to_vec();
    let cfg = FprmConfig {
        tau: 3,
        m: 5,
        holdout: 0.5,
        seed: 7,
        ..Default::default()
    };
    let res = impute_pair(&donor, &target, &cfg).expect("impute");
    assert!(res.metrics.mse.is_finite());
    assert!(res.metrics.mae.is_finite());
    assert!(res.metrics.nrmse.is_finite());
    assert_eq!(res.y_fit.len(), res.y_test.len());
    assert!(res.n_train >= 2);
}

#[test]
fn fprm_reps_aggregate() {
    let x = ecology::simulate(200, 50);
    let donor: Vec<f64> = x.column(2).to_vec();
    let target: Vec<f64> = x.column(0).to_vec();
    let cfg = FprmConfig {
        tau: 2,
        m: 3,
        holdout: 0.4,
        seed: 1,
        ..Default::default()
    };
    let agg = impute_pair_reps(&donor, &target, &cfg, 4).expect("reps");
    assert_eq!(agg.reps, 4);
    assert!(agg.mean.mse.is_finite());
    assert!(agg.std.mse.is_finite() || agg.std.mse == 0.0);
}
