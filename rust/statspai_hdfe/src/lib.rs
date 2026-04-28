//! PyO3 bindings for StatsPAI's HDFE inner kernel.
//!
//! Phase 1 contents:
//! - ``group_demean``      — single-FE, in-place column demean (legacy entry point).
//! - ``demean_2d``         — K-way alternating-projection demean with Irons-Tuck
//!                           acceleration; in-place on a Fortran-order matrix,
//!                           parallel over columns via Rayon.
//! - ``singleton_mask``    — iterative K-way singleton-row detection; returns
//!                           a boolean keep-mask.
//!
//! Phase A additions (v0.3.0, weighted variants for IRLS-internal demean):
//! - ``demean_2d_weighted`` — same as ``demean_2d`` but takes per-observation
//!                            weights and a caller-precomputed wsum
//!                            (``Σ_{i ∈ g} weights[i]``); used by the IRLS
//!                            inner loop in ``sp.fast.fepois``.
//!
//! All functions take pre-factorised int64 codes and float64 counts. The
//! Python wrapper at ``statspai.fast.demean`` packs DataFrames / mixed
//! dtypes via ``pd.factorize`` before calling here.
//!
//! All wheels are optional — the Python side falls back to the NumPy /
//! Numba kernel gracefully when the compiled extension is missing.

mod demean;
mod singletons;
mod sort_perm;

use numpy::{
    PyArray1, PyReadonlyArray1, PyReadwriteArray1, PyReadwriteArray2, PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::wrap_pyfunction;

/// In-place group demean: `y[i] -= mean(y[codes == codes[i]])`.
///
/// Legacy single-FE entry point. Kept for backward compatibility with
/// ``statspai.panel.hdfe_rust.group_demean_rust``; new callers should
/// prefer the more general ``demean_2d``.
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

    for s in sums_slice.iter_mut() {
        *s = 0.0;
    }
    for i in 0..y_slice.len() {
        let g = codes[i] as usize;
        sums_slice[g] += y_slice[i];
    }
    for g in 0..sums_slice.len() {
        let c = counts_slice[g];
        if c > 0 {
            sums_slice[g] /= c as f64;
        }
    }
    for i in 0..y_slice.len() {
        let g = codes[i] as usize;
        y_slice[i] -= sums_slice[g];
    }
    Ok(())
}

/// Helper: extract a list of int64 array views from a PyList.
fn py_list_to_i64_views<'py>(
    list: &Bound<'py, PyList>,
) -> PyResult<Vec<PyReadonlyArray1<'py, i64>>> {
    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let arr: PyReadonlyArray1<i64> = item.extract()?;
        out.push(arr);
    }
    Ok(out)
}

/// Helper: extract a list of float64 array views from a PyList.
fn py_list_to_f64_views<'py>(
    list: &Bound<'py, PyList>,
) -> PyResult<Vec<PyReadonlyArray1<'py, f64>>> {
    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let arr: PyReadonlyArray1<f64> = item.extract()?;
        out.push(arr);
    }
    Ok(out)
}

/// K-way alternating-projection demean of a Fortran-order (n, p) matrix in
/// place. Returns a list of dicts (one per column) with iter / converged /
/// max_dx fields.
///
/// Parameters
/// ----------
/// x : 2-D float64 ndarray, shape (n, p), Fortran-contiguous
///     The matrix to residualise. Each column is a contiguous slice we can
///     split for parallel processing. Pass ``np.asfortranarray(X)`` on the
///     Python side to materialise.
/// fe_codes : list[ndarray[int64, shape (n,)]]
///     One code array per FE dimension (K total).
/// counts : list[ndarray[float64, shape (G_k,)]]
///     Per-group sizes for each FE dimension. Float so weighted variants
///     can drop in later.
/// max_iter : int
///     Cap on AP iterations per column.
/// tol_abs, tol_rel : float
///     Stop when ``max|dx| <= tol_abs + tol_rel * base_scale``.
/// accelerate : bool
/// accel_period : int
#[pyfunction]
#[pyo3(signature = (x, fe_codes, counts, max_iter, tol_abs, tol_rel, accelerate, accel_period))]
#[allow(clippy::too_many_arguments)]
fn demean_2d<'py>(
    py: Python<'py>,
    mut x: PyReadwriteArray2<'py, f64>,
    fe_codes: &Bound<'py, PyList>,
    counts: &Bound<'py, PyList>,
    max_iter: u32,
    tol_abs: f64,
    tol_rel: f64,
    accelerate: bool,
    accel_period: u32,
) -> PyResult<Bound<'py, PyList>> {
    if fe_codes.len() != counts.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "len(fe_codes)={} but len(counts)={}",
            fe_codes.len(),
            counts.len()
        )));
    }

    let code_views = py_list_to_i64_views(fe_codes)?;
    let count_views = py_list_to_f64_views(counts)?;

    let arr = x.as_array();
    let shape = arr.shape();
    if shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("x must be 2-D"));
    }
    let n = shape[0];
    let p = shape[1];

    for v in &code_views {
        if v.as_slice()?.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "fe_codes entry has length {} but n={}",
                v.as_slice()?.len(),
                n
            )));
        }
    }

    if !x.is_fortran_contiguous() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x must be Fortran-contiguous; pass np.asfortranarray(X)",
        ));
    }
    let mat = x.as_slice_mut()?;
    let codes_slices: Vec<&[i64]> =
        code_views.iter().map(|v| v.as_slice().unwrap()).collect();
    let counts_slices: Vec<&[f64]> =
        count_views.iter().map(|v| v.as_slice().unwrap()).collect();
    let counts_lens: Vec<usize> = counts_slices.iter().map(|s| s.len()).collect();

    let infos = py.allow_threads(|| {
        demean::demean_matrix_fortran_inplace(
            mat,
            n,
            p,
            &codes_slices,
            &counts_slices,
            &counts_lens,
            max_iter,
            tol_abs,
            tol_rel,
            accelerate,
            accel_period,
        )
    });

    let out = PyList::empty_bound(py);
    for info in &infos {
        let d = PyDict::new_bound(py);
        d.set_item("iters", info.iters)?;
        d.set_item("converged", info.converged)?;
        d.set_item("max_dx", info.max_dx)?;
        out.append(d)?;
    }
    Ok(out)
}

/// K-way **weighted** alternating-projection demean of a Fortran-order
/// (n, p) matrix in place. Returns a list of dicts (one per column)
/// with ``iters`` / ``converged`` / ``max_dx`` fields, mirroring
/// ``demean_2d``.
///
/// Parameters
/// ----------
/// x : 2-D float64 ndarray, shape (n, p), Fortran-contiguous
///     The matrix to residualise (in place).
/// fe_codes : list[ndarray[int64, shape (n,)]]
///     One code array per FE dimension (K total).
/// wsum : list[ndarray[float64, shape (G_k,)]]
///     Per-group **weighted** sum ``Σ_{i ∈ g} weights[i]``. Caller
///     precomputes via ``np.bincount(codes, weights=weights, minlength=G)``.
/// weights : ndarray[float64, shape (n,)]
///     Per-observation weights. Caller is responsible for non-negativity
///     and finiteness — no re-validation here on the hot path.
/// max_iter, tol_abs, tol_rel, accelerate, accel_period
///     Same semantics as ``demean_2d``.
#[pyfunction]
#[pyo3(signature = (x, fe_codes, wsum, weights, max_iter, tol_abs, tol_rel, accelerate, accel_period))]
#[allow(clippy::too_many_arguments)]
fn demean_2d_weighted<'py>(
    py: Python<'py>,
    mut x: PyReadwriteArray2<'py, f64>,
    fe_codes: &Bound<'py, PyList>,
    wsum: &Bound<'py, PyList>,
    weights: PyReadonlyArray1<'py, f64>,
    max_iter: u32,
    tol_abs: f64,
    tol_rel: f64,
    accelerate: bool,
    accel_period: u32,
) -> PyResult<Bound<'py, PyList>> {
    if fe_codes.len() != wsum.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "len(fe_codes)={} but len(wsum)={}",
            fe_codes.len(),
            wsum.len()
        )));
    }

    let code_views = py_list_to_i64_views(fe_codes)?;
    let wsum_views = py_list_to_f64_views(wsum)?;
    let weights_view = weights.as_slice()?;

    let arr = x.as_array();
    let shape = arr.shape();
    if shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("x must be 2-D"));
    }
    let n = shape[0];
    let p = shape[1];

    if weights_view.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "weights length {} but n={}",
            weights_view.len(),
            n
        )));
    }

    for v in &code_views {
        if v.as_slice()?.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "fe_codes entry has length {} but n={}",
                v.as_slice()?.len(),
                n
            )));
        }
    }

    if !x.is_fortran_contiguous() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x must be Fortran-contiguous; pass np.asfortranarray(X)",
        ));
    }
    let mat = x.as_slice_mut()?;
    let codes_slices: Vec<&[i64]> =
        code_views.iter().map(|v| v.as_slice().unwrap()).collect();
    let wsum_slices: Vec<&[f64]> =
        wsum_views.iter().map(|v| v.as_slice().unwrap()).collect();
    let wsum_lens: Vec<usize> = wsum_slices.iter().map(|s| s.len()).collect();

    let infos = py.allow_threads(|| {
        demean::weighted_demean_matrix_fortran_inplace(
            mat,
            n,
            p,
            &codes_slices,
            weights_view,
            &wsum_slices,
            &wsum_lens,
            max_iter,
            tol_abs,
            tol_rel,
            accelerate,
            accel_period,
        )
    });

    let out = PyList::empty_bound(py);
    for info in &infos {
        let d = PyDict::new_bound(py);
        d.set_item("iters", info.iters)?;
        d.set_item("converged", info.converged)?;
        d.set_item("max_dx", info.max_dx)?;
        out.append(d)?;
    }
    Ok(out)
}

/// Sort-aware weighted demean of a Fortran-order (n, p) matrix in place.
/// Caller has applied the primary-FE sort permutation π to ``x`` (rows),
/// ``weights``, and the secondary FE codes; this function does NOT
/// permute. Result is in π-order; caller applies π⁻¹ on return.
///
/// Parameters
/// ----------
/// x : 2-D float64 ndarray, shape (n, p), Fortran-contiguous, in π order.
/// primary_starts : ndarray[int64, shape (G1+1,)]
///     Group-start offsets for the primary FE (caller computes once via
///     ``primary_fe_sort_perm`` + cumulative count).
/// primary_wsum : ndarray[float64, shape (G1,)]
///     Weighted group sums for the primary FE, computed in π order
///     (i.e., ``np.bincount(codes_perm, weights=weights_perm)``).
/// secondary_codes : list[ndarray[int64, shape (n,)]]
///     K-1 arrays, one per non-primary FE; codes are under π.
/// secondary_wsum : list[ndarray[float64, shape (G_k,)]]
///     Weighted group sums for non-primary FEs.
/// weights_sorted : ndarray[float64, shape (n,)]
///     Per-obs weights in π order.
/// max_iter, tol_abs, tol_rel, accelerate, accel_period :
///     Same semantics as ``demean_2d_weighted``.
#[pyfunction]
#[pyo3(signature = (x, primary_starts, primary_wsum, secondary_codes, secondary_wsum, weights_sorted, max_iter, tol_abs, tol_rel, accelerate, accel_period))]
#[allow(clippy::too_many_arguments)]
fn demean_2d_weighted_sorted<'py>(
    py: Python<'py>,
    mut x: PyReadwriteArray2<'py, f64>,
    primary_starts: PyReadonlyArray1<'py, i64>,
    primary_wsum: PyReadonlyArray1<'py, f64>,
    secondary_codes: &Bound<'py, PyList>,
    secondary_wsum: &Bound<'py, PyList>,
    weights_sorted: PyReadonlyArray1<'py, f64>,
    max_iter: u32,
    tol_abs: f64,
    tol_rel: f64,
    accelerate: bool,
    accel_period: u32,
) -> PyResult<Bound<'py, PyList>> {
    if secondary_codes.len() != secondary_wsum.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "len(secondary_codes)={} but len(secondary_wsum)={}",
            secondary_codes.len(),
            secondary_wsum.len()
        )));
    }

    let primary_starts_view = primary_starts.as_slice()?;
    let primary_wsum_view = primary_wsum.as_slice()?;
    let weights_view = weights_sorted.as_slice()?;
    let sec_code_views = py_list_to_i64_views(secondary_codes)?;
    let sec_wsum_views = py_list_to_f64_views(secondary_wsum)?;

    let arr = x.as_array();
    let shape = arr.shape();
    if shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("x must be 2-D"));
    }
    let n = shape[0];
    let p = shape[1];

    if weights_view.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "weights_sorted length {} but n={}",
            weights_view.len(),
            n
        )));
    }
    if primary_starts_view.len() != primary_wsum_view.len() + 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "primary_starts length {} but expected {} (= primary_wsum.len + 1)",
            primary_starts_view.len(),
            primary_wsum_view.len() + 1
        )));
    }
    for v in &sec_code_views {
        if v.as_slice()?.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "secondary_codes entry has length {} but n={}",
                v.as_slice()?.len(),
                n
            )));
        }
    }

    if !x.is_fortran_contiguous() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x must be Fortran-contiguous; pass np.asfortranarray(X)",
        ));
    }
    let mat = x.as_slice_mut()?;

    // Convert primary_starts (passed as i64 from Python) to usize.
    let primary_starts_usize: Vec<usize> =
        primary_starts_view.iter().map(|&v| v as usize).collect();

    let sec_codes_slices: Vec<&[i64]> =
        sec_code_views.iter().map(|v| v.as_slice().unwrap()).collect();
    let sec_wsum_slices: Vec<&[f64]> =
        sec_wsum_views.iter().map(|v| v.as_slice().unwrap()).collect();
    let sec_lens: Vec<usize> = sec_wsum_slices.iter().map(|s| s.len()).collect();

    let infos = py.allow_threads(|| {
        demean::weighted_demean_matrix_fortran_inplace_sorted(
            mat,
            n,
            p,
            &primary_starts_usize,
            primary_wsum_view,
            &sec_codes_slices,
            &sec_wsum_slices,
            &sec_lens,
            weights_view,
            max_iter,
            tol_abs,
            tol_rel,
            accelerate,
            accel_period,
        )
    });

    let out = PyList::empty_bound(py);
    for info in &infos {
        let d = PyDict::new_bound(py);
        d.set_item("iters", info.iters)?;
        d.set_item("converged", info.converged)?;
        d.set_item("max_dx", info.max_dx)?;
        out.append(d)?;
    }
    Ok(out)
}

/// Iterative K-way singleton detection. Returns a uint8 keep-mask
/// (1 = keep, 0 = drop).
#[pyfunction]
fn singleton_mask<'py>(
    py: Python<'py>,
    fe_codes: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let code_views = py_list_to_i64_views(fe_codes)?;
    if code_views.is_empty() {
        return Ok(PyArray1::<u8>::zeros_bound(py, 0, false));
    }
    let codes_slices: Vec<&[i64]> =
        code_views.iter().map(|v| v.as_slice().unwrap()).collect();

    let keep = py.allow_threads(|| singletons::detect_singletons(&codes_slices));
    let as_u8: Vec<u8> = keep.into_iter().map(|b| if b { 1 } else { 0 }).collect();
    Ok(PyArray1::from_vec_bound(py, as_u8))
}

/// Python module definition.
#[pymodule]
fn statspai_hdfe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(group_demean, m)?)?;
    m.add_function(wrap_pyfunction!(demean_2d, m)?)?;
    m.add_function(wrap_pyfunction!(demean_2d_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(demean_2d_weighted_sorted, m)?)?;  // NEW B0.4
    m.add_function(wrap_pyfunction!(singleton_mask, m)?)?;
    m.add("__version__", "0.4.0")?;  // BUMPED 0.3.0 → 0.4.0
    Ok(())
}
