use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use fprm_rust::systems::{ecology, lorenz, rossler};
use fprm_rust::{
    impute_from_matrix, impute_matrix_columns, load_table, write_csv, FprmConfig,
};

#[derive(Parser, Debug)]
#[command(name = "fprm", about = "FPRM phase-space GP imputation (MATLAB port)")]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Impute a target column from a donor column via FPRM
    Impute {
        /// Input CSV or XLSX (rows=time, cols=series)
        #[arg(long)]
        data: PathBuf,
        /// Donor column (1-based, MATLAB style)
        #[arg(long, default_value_t = 2)]
        donor: usize,
        /// Target column (1-based). Ignored if --targets is set.
        #[arg(long, default_value_t = 1)]
        target: usize,
        /// Comma-separated target columns for multi-column parallel impute
        #[arg(long)]
        targets: Option<String>,
        #[arg(long, default_value_t = 3)]
        tau: usize,
        #[arg(long, default_value_t = 5)]
        m: usize,
        #[arg(long, default_value_t = 0.5)]
        holdout: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Monte-Carlo holdout replicates
        #[arg(long, default_value_t = 1)]
        reps: usize,
        /// Optional JSON metrics output path
        #[arg(long)]
        out_json: Option<PathBuf>,
    },
    /// Simulate the 3D ecology map
    Ecology {
        #[arg(long, default_value_t = 600)]
        steps: usize,
        #[arg(long, default_value_t = 100)]
        burnin: usize,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Simulate coupled Lorenz network
    Lorenz {
        #[arg(long, default_value_t = 40)]
        n: usize,
        #[arg(long, default_value_t = 0.1)]
        epsilon: f64,
        #[arg(long, default_value_t = 10.0)]
        t0: f64,
        #[arg(long, default_value_t = 40.0)]
        t1: f64,
        #[arg(long, default_value_t = 0.01)]
        dt: f64,
        #[arg(long, default_value_t = 100)]
        burnin: usize,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Simulate coupled Rössler network (real Rössler; MATLAB mlx was a Lorenz stub)
    Rossler {
        #[arg(long, default_value_t = 40)]
        n: usize,
        #[arg(long, default_value_t = 0.1)]
        epsilon: f64,
        #[arg(long, default_value_t = 10.0)]
        t0: f64,
        #[arg(long, default_value_t = 40.0)]
        t1: f64,
        #[arg(long, default_value_t = 0.01)]
        dt: f64,
        #[arg(long, default_value_t = 100)]
        burnin: usize,
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Impute {
            data,
            donor,
            target,
            targets,
            tau,
            m,
            holdout,
            seed,
            reps,
            out_json,
        } => {
            let mat = load_table(&data).with_context(|| format!("load {}", data.display()))?;
            let cfg = FprmConfig {
                tau,
                m,
                holdout,
                seed,
                ..Default::default()
            };
            if let Some(list) = targets {
                let cols: Result<Vec<usize>, _> = list
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect();
                let cols = cols.context("parse --targets")?;
                if cols.is_empty() {
                    bail!("--targets is empty");
                }
                let results = impute_matrix_columns(&mat, donor, &cols, &cfg, reps)?;
                for (tc, agg) in &results {
                    println!(
                        "target={tc} reps={} MSE={:.6}±{:.6} NRMSE={:.6}±{:.6} corr={:.4}±{:.4} MAE={:.6}±{:.6} RelErr={:.6}±{:.6}",
                        agg.reps,
                        agg.mean.mse,
                        agg.std.mse,
                        agg.mean.nrmse,
                        agg.std.nrmse,
                        agg.mean.corr,
                        agg.std.corr,
                        agg.mean.mae,
                        agg.std.mae,
                        agg.mean.mean_rel_error,
                        agg.std.mean_rel_error
                    );
                }
                if let Some(path) = out_json {
                    let json = serde_json::to_string_pretty(&results)?;
                    std::fs::write(&path, json)?;
                    eprintln!("wrote {}", path.display());
                }
            } else {
                let agg = impute_from_matrix(&mat, donor, target, &cfg, reps)?;
                println!(
                    "donor={donor} target={target} reps={} n_test≈{}",
                    agg.reps, agg.mean.n_test
                );
                println!(
                    "MSE={:.6}±{:.6}\nNRMSE(LL)={:.6}±{:.6}\ncorr={:.6}±{:.6}\nMAE={:.6}±{:.6}\nRelErr={:.6}±{:.6}",
                    agg.mean.mse,
                    agg.std.mse,
                    agg.mean.nrmse,
                    agg.std.nrmse,
                    agg.mean.corr,
                    agg.std.corr,
                    agg.mean.mae,
                    agg.std.mae,
                    agg.mean.mean_rel_error,
                    agg.std.mean_rel_error
                );
                if let Some(path) = out_json {
                    let json = serde_json::to_string_pretty(&agg)?;
                    std::fs::write(&path, json)?;
                    eprintln!("wrote {}", path.display());
                }
            }
        }
        Command::Ecology { steps, burnin, out } => {
            let x = ecology::simulate(steps, burnin);
            println!("ecology shape={}x{}", x.nrows(), x.ncols());
            if let Some(path) = out {
                write_csv(&path, &x)?;
                eprintln!("wrote {}", path.display());
            }
        }
        Command::Lorenz {
            n,
            epsilon,
            t0,
            t1,
            dt,
            burnin,
            out,
        } => {
            let p = lorenz::LorenzParams {
                n,
                epsilon,
                ..Default::default()
            };
            let x0 = lorenz::default_initial(n);
            let x = lorenz::integrate(&p, &x0, t0, t1, dt, burnin);
            println!("lorenz shape={}x{}", x.nrows(), x.ncols());
            if let Some(path) = out {
                write_csv(&path, &x)?;
                eprintln!("wrote {}", path.display());
            }
        }
        Command::Rossler {
            n,
            epsilon,
            t0,
            t1,
            dt,
            burnin,
            out,
        } => {
            let p = rossler::RosslerParams {
                n,
                epsilon,
                ..Default::default()
            };
            let x0 = rossler::default_initial(n);
            let x = rossler::integrate(&p, &x0, t0, t1, dt, burnin);
            println!("rossler shape={}x{}", x.nrows(), x.ncols());
            if let Some(path) = out {
                write_csv(&path, &x)?;
                eprintln!("wrote {}", path.display());
            }
        }
    }
    Ok(())
}
