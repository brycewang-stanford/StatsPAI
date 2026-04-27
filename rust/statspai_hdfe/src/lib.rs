//! PyO3 bindings for StatsPAI's HDFE inner kernel.
//!
//! Phase 1 contents:
//! - ``group_demean``  — single-FE, in-place column demean (legacy entry point).
//! - ``demean_2d``     — K-way alternating-projection demean with Irons-Tuck
//!                       acceleration; in-place on a Fortran-order matrix,
//!                       parallel over columns via Rayon.
//! - ``singleton_mask``— iterative K-way singleton-row detection; returns
//!                       a boolean keep-mask.
//!
//! All functions take pre-factorised int64 codes and float64 counts. The
//! Python wrapper at ``statspai.fast.demean`` packs DataFrames / mixed
//! dtypes via ``pd.factorize`` before calling here.
//!
//! All wheels are optional — the Python side falls back to the NumPy /
//! Numba kernel gracefully when the compiled extension is missing.

mod demean;
mod singletons;

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
    m.add_function(wrap_pyfunction!(singleton_mask, m)?)?;
    m.add("__version__", "0.2.0")?;
    Ok(())
}
