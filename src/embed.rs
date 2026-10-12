//! Phase-space (delay) reconstruction — MATLAB `PhaSpaRecon`.
//!
//! MATLAB returns `m × n_states` (each column a state vector). This port returns
//! `n_states × m` (observations × embedding dims), matching the live script's `YX = ss'`.

use ndarray::Array2;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("delay time or embedding dimension is too large for series length {len} (tau={tau}, m={m})")]
    TooLarge { len: usize, tau: usize, m: usize },
    #[error("tau and m must be >= 1 (got tau={tau}, m={m})")]
    InvalidParams { tau: usize, m: usize },
    #[error("input series is empty")]
    EmptySeries,
}

/// Takens delay embedding of a univariate series.
///
/// For embedding dimension `m` and delay `tau`, observation `t` is
/// `[s[t], s[t+tau], …, s[t+(m-1)*tau]]`, with `t = 0 .. n_states-1` where
/// `n_states = len - (m-1)*tau`.
pub fn phase_space_reconstruct(s: &[f64], tau: usize, m: usize) -> Result<Array2<f64>, EmbedError> {
    if s.is_empty() {
        return Err(EmbedError::EmptySeries);
    }
    if tau < 1 || m < 1 {
        return Err(EmbedError::InvalidParams { tau, m });
    }
    let len = s.len();
    let n_states = len.saturating_sub((m - 1) * tau);
    if n_states < 1 {
        return Err(EmbedError::TooLarge { len, tau, m });
    }

    // MATLAB: xn(i,:) = s(1+(i-1)*tau : len-(m-i)*tau); then we transpose.
    let mut out = Array2::<f64>::zeros((n_states, m));
    for i in 0..m {
        let start = i * tau;
        for t in 0..n_states {
            out[(t, i)] = s[start + t];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn embeds_like_matlab_transpose() {
        // s = 1..10, tau=2, m=3 → n_states = 10 - 4 = 6
        let s: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let xn = phase_space_reconstruct(&s, 2, 3).unwrap();
        assert_eq!(xn.nrows(), 6);
        assert_eq!(xn.ncols(), 3);
        // row 0: [s0, s2, s4] = [1,3,5]
        assert_abs_diff_eq!(xn[(0, 0)], 1.0);
        assert_abs_diff_eq!(xn[(0, 1)], 3.0);
        assert_abs_diff_eq!(xn[(0, 2)], 5.0);
        // last row: [s5, s7, s9] = [6,8,10]
        assert_abs_diff_eq!(xn[(5, 0)], 6.0);
        assert_abs_diff_eq!(xn[(5, 1)], 8.0);
        assert_abs_diff_eq!(xn[(5, 2)], 10.0);
    }

    #[test]
    fn rejects_too_large() {
        let s = vec![1.0, 2.0, 3.0];
        assert!(phase_space_reconstruct(&s, 5, 3).is_err());
    }
}
