# fprm_rust

Rust port of the MATLAB **FPRM** suite under `FPRM_Code_and_data/`: Takens phase-space reconstruction plus Gaussian-process imputation of a “missing” series from a donor embedding.

## MATLAB → Rust map

| MATLAB | Rust |
|--------|------|
| `code/PhaSpaRecon.m` | `src/embed.rs` (`phase_space_reconstruct`) |
| `code/FPRM mian code.mlx` | `src/fprm.rs` + `src/gpr.rs` + `src/metrics.rs` |
| `code/multiCoupledLorenz2.m` + Lorenz mlx | `src/systems/lorenz.rs` |
| `code/3D ecology system.mlx` | `src/systems/ecology.rs` |
| `code/Coupled Rossler system.mlx` (Lorenz stub) | `src/systems/rossler.rs` (**real** coupled Rössler) |
| `Data/*.csv` / `*.xlsx` | `src/io.rs` |
| Live-script plots | CLI metrics / JSON (no GUI plots in v1) |

## Pipeline (parity)

1. Embed donor series → predictor matrix `X` (`n_states × m`).
2. Embed target series → response `y` = first embedding coordinate (MATLAB `Y(:,1)`).
3. Random holdout (default 50%) marks “missing” test points.
4. Z-score predictors from **training** stats (`σ=0 → 1`).
5. Exact squared-exponential GPR (`fitrgp`-like) → predict test.
6. Metrics: MSE, NRMSE (`√MSE / std(Ytest)`), Pearson corr, mean relative error, MAE.

Defaults match the live script: `tau=3`, `m=5`, `holdout=0.5`. Use `--seed` for reproducibility (MATLAB used `rng('shuffle')`).

## Build & test

```bash
cd FPRM_Code_and_data/fprm_rust
cargo test
cargo build --release
```

## CLI

```bash
# Impute Wind col 1 from col 2 (1-based), 8 Monte-Carlo holdouts
cargo run --release -- impute \
  --data ../Data/Wind\ speed.csv \
  --donor 2 --target 1 --tau 3 --m 5 --holdout 0.5 --seed 42 --reps 8

# Multi-column parallel impute
cargo run --release -- impute --data ../Data/Wind\ speed.csv \
  --donor 2 --targets 1,3,4 --reps 4

# Synthetics
cargo run --release -- ecology --out /tmp/ecology.csv
cargo run --release -- lorenz --n 10 --out /tmp/lorenz.csv
cargo run --release -- rossler --n 10 --out /tmp/rossler.csv
```

## Intentional differences vs MATLAB

| Topic | MATLAB | This port |
|-------|--------|-----------|
| RNG | `rng('shuffle')` each run | ChaCha8 `--seed` (+ `--reps` for MC) |
| GPR | `fitrgp` internal L-BFGS / init | Exact SE kernel + Adam on log-marginal likelihood; empirical mean |
| ODE | `ode45` adaptive mesh | Fixed-step RK4 (`dt=0.01` default); **RHS** matches Lorenz helper |
| Rössler mlx | Copy of Lorenz | Proper multi-coupled Rössler (`a,b,c`) |
| Relative error | `(y-ŷ)/ŷ` always | Skips `|ŷ| < 1e-12` |
| Plots | Live-script `plot`/`patch` | Metrics / JSON only |

**Parity bar:** goldens for embedding, ecology map, Lorenz/Rössler RHS, and metric formulas. GPR predictions are the same *model class* as default `fitrgp` but are **not** guaranteed bit-identical. For a tighter bake-off, fit with exported MATLAB kernel params via `GprModel::fit_with_params(..., Some(params))`.

## Parallelism

- Monte-Carlo holdout replicates (`--reps`) via Rayon
- Multi-column `--targets` via Rayon
- Kernel Gram matrix rows and GPR predict rows via Rayon
- Lorenz/Rössler coupling evaluated in `O(N)` using `sum(x)` instead of an `O(N²)` double loop
