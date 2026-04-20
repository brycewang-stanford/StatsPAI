//! PyO3 bindings for StatsPAI's HDFE inner kernel.
//!
//! Phase 1 scaffold — this crate currently exposes a single function,
//! `group_demean`, which takes a group-code array and an outcome
//! array and subtracts the per-group mean in-place. Phase 2 will add
//! a block version that demeans multiple columns in parallel via
//! Rayon.
//!
//! Target wheel matrix (see docs/superpowers/specs/
//! 2026-04-20-v095-rust-hdfe-spike.md):
//! - macOS arm64, macOS x86_64
//! - manylinux_2_17 x86_64, aarch64
//! - musllinux_1_2 x86_64
//! - Windows x86_64
//!
//! All wheels are optional — the Python side falls back to the Numba
//! kernel gracefully when the compiled extension is missing.

use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

/// In-place group demean: `y[i] -= mean(y[codes == codes[i]])`.
///
/// Parameters
/// ----------
/// codes : np.ndarray[int64]
///     Group code per observation, in [0, counts.len()).
/// y : np.ndarray[float64]
///     Observation vector, demeaned in place.
/// sums : np.ndarray[float64]
///     Scratch buffer of length `counts.len()`. Caller must provide
///     it zeroed (the caller owns its allocation; no reallocs happen
///     inside this function on the hot path).
/// counts : np.ndarray[int64]
///     Count of observations per group. Must all be positive.
#[pyfunction]
fn group_demean(
    codes: PyReadonlyArray1<i64>,
    mut y: PyReadwriteArray1<f64>,
    mut sums: PyReadwriteArray1<f64>,
    counts: PyReadonlyArray1<i64>,
) -> PyResult<()> {
    let codes = codes.as_slice()?;
    let y_slice = y.as_slice_mut()?;
    let sums_slice = sums.as_slice_mut()?;
    let counts_slice = counts.as_slice()?;

    // Phase 1 implementation: sequential. Rayon parallelisation comes
    // in Phase 3 (see spec). We still want a correct single-threaded
    // baseline to validate the FFI boundary first.

    // Zero the scratch buffer (caller supplies it but we take no
    // chances; branch-free zeroing is fine on any modern CPU).
    for s in sums_slice.iter_mut() {
        *s = 0.0;
    }

    // Pass 1: accumulate group sums.
    for i in 0..y_slice.len() {
        let g = codes[i] as usize;
        sums_slice[g] += y_slice[i];
    }

    // Pass 2: convert sums into means.
    for g in 0..sums_slice.len() {
        let c = counts_slice[g];
        if c > 0 {
            sums_slice[g] /= c as f64;
        }
    }

    // Pass 3: subtract group mean in-place.
    for i in 0..y_slice.len() {
        let g = codes[i] as usize;
        y_slice[i] -= sums_slice[g];
    }

    Ok(())
}

/// Python module definition.
#[pymodule]
fn statspai_hdfe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(group_demean, m)?)?;
    m.add("__version__", "0.1.0-alpha.1")?;
    Ok(())
}
