//! PyO3 bindings for StatsPAI's HDFE kernels.
//!
//! - ``demean_2d`` / ``demean_2d_weighted`` — K-way FE residualisation of a
//!   Fortran-order matrix in place, by the coefficient-space solver in
//!   ``coef.rs`` (sparse cross-tabulation of the two largest FEs,
//!   Irons-Tuck acceleration), parallel over columns.
//! - ``demean_2d_weighted_sorted`` — the earlier data-space weighted kernel
//!   on input pre-sorted by the largest FE (kept for callers built
//!   against API 1).
//! - ``fepois_irls`` — the whole Poisson IRLS loop of ``sp.fast.fepois``.
//! - ``singleton_mask`` / ``separation_mask`` — iterative row-drop
//!   pre-passes, returning a uint8 keep-mask.
//! - ``cluster_meat`` — cluster-robust sandwich meat, parallel over fixed
//!   blocks of clusters (reproducible bit for bit at any thread count).
//! - ``group_demean`` — legacy single-FE in-place demean.
//! - ``set_num_threads`` / ``get_num_threads`` — size of the crate's pool.
//!
//! Every entry point validates its inputs (lengths, contiguity, code
//! ranges) before entering Rust code and raises ``ValueError`` instead of
//! indexing out of bounds. The crate is built with ``panic = "unwind"``,
//! so an unexpected panic surfaces as ``pyo3_runtime.PanicException``
//! rather than aborting the interpreter.
//!
//! The Python side (``statspai._rust_backend``) checks ``API_VERSION``
//! before using the module and falls back to NumPy / Numba otherwise.

mod cholesky;
mod cluster;
mod coef;
mod demean;
mod irls;
mod par;
mod pool;
mod separation;
mod singletons;

use numpy::ndarray::Array2;
use numpy::{
    PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::wrap_pyfunction;

/// Interface level of this build. The Python loader requires a minimum
/// level; bump it whenever an entry point changes meaning.
///
/// 1 = 0.1-0.7 (data-space AP kernels). 2 = 0.8 (coefficient-space
/// solver, validated inputs, ``fepois_irls`` reports ``singular`` and
/// ``fe_converged``, thread-pool control).
const API_VERSION: u32 = 3;

fn err<T>(msg: String) -> PyResult<T> {
    Err(PyValueError::new_err(msg))
}

fn slice1<'a, T: numpy::Element>(
    arr: &'a PyReadonlyArray1<'_, T>,
    name: &str,
) -> PyResult<&'a [T]> {
    arr.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a contiguous 1-D array")))
}

/// Every code lies in ``[0, g)``.
fn check_codes(codes: &[i64], g: usize, name: &str) -> PyResult<()> {
    let mut lo = i64::MAX;
    let mut hi = i64::MIN;
    for &c in codes {
        lo = lo.min(c);
        hi = hi.max(c);
    }
    if !codes.is_empty() && (lo < 0 || hi as u64 >= g as u64) {
        return err(format!(
            "{name}: codes must lie in [0, {g}); found range [{lo}, {hi}]"
        ));
    }
    Ok(())
}

fn check_weights(w: &[f64], name: &str) -> PyResult<()> {
    for &v in w {
        if !(v >= 0.0) || !v.is_finite() {
            return err(format!("{name} must be finite and non-negative; found {v}"));
        }
    }
    Ok(())
}

fn i64_views<'py>(list: &Bound<'py, PyList>) -> PyResult<Vec<PyReadonlyArray1<'py, i64>>> {
    list.iter().map(|item| item.extract()).collect()
}

fn f64_views<'py>(list: &Bound<'py, PyList>) -> PyResult<Vec<PyReadonlyArray1<'py, f64>>> {
    list.iter().map(|item| item.extract()).collect()
}

/// Code slices of length ``n``, each validated against its level count.
fn code_slices<'a>(
    views: &'a [PyReadonlyArray1<'_, i64>],
    n_groups: &[usize],
    n: usize,
    name: &str,
) -> PyResult<Vec<&'a [i64]>> {
    let mut out = Vec::with_capacity(views.len());
    for (k, v) in views.iter().enumerate() {
        let s = slice1(v, name)?;
        if s.len() != n {
            return err(format!("{name}[{k}] has length {} but n={n}", s.len()));
        }
        check_codes(s, n_groups[k], &format!("{name}[{k}]"))?;
        out.push(s);
    }
    Ok(out)
}

fn infos_to_list<'py>(
    py: Python<'py>,
    infos: &[demean::DemeanInfo],
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty_bound(py);
    for info in infos {
        let d = PyDict::new_bound(py);
        d.set_item("iters", info.iters)?;
        d.set_item("converged", info.converged)?;
        d.set_item("max_dx", info.max_dx)?;
        out.append(d)?;
    }
    Ok(out)
}

/// ``(n, p)`` of a Fortran-contiguous matrix, or an error.
fn fortran_shape(x: &PyReadwriteArray2<'_, f64>) -> PyResult<(usize, usize)> {
    let shape = x.shape();
    let (n, p) = (shape[0], shape[1]);
    if !x.is_fortran_contiguous() {
        return err("x must be Fortran-contiguous; pass np.asfortranarray(X)".into());
    }
    if n > u32::MAX as usize {
        return err(format!("n={n} exceeds the supported 2^32 - 1 rows"));
    }
    Ok((n, p))
}

/// In-place group demean: `y[i] -= mean(y[codes == codes[i]])`.
///
/// Legacy single-FE entry point; new callers use ``demean_2d``.
#[pyfunction]
fn group_demean(
    py: Python<'_>,
    codes: PyReadonlyArray1<i64>,
    mut y: PyReadwriteArray1<f64>,
    mut sums: PyReadwriteArray1<f64>,
    counts: PyReadonlyArray1<i64>,
) -> PyResult<()> {
    let codes = slice1(&codes, "codes")?;
    let counts = slice1(&counts, "counts")?;
    let y = y
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("y must be a contiguous 1-D array"))?;
    let sums = sums
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("sums must be a contiguous 1-D array"))?;
    if codes.len() != y.len() {
        return err(format!(
            "codes has length {} but y has {}",
            codes.len(),
            y.len()
        ));
    }
    if counts.len() != sums.len() {
        return err(format!(
            "counts has length {} but sums has {}",
            counts.len(),
            sums.len()
        ));
    }
    check_codes(codes, sums.len(), "codes")?;
    py.allow_threads(|| {
        for s in sums.iter_mut() {
            *s = 0.0;
        }
        for i in 0..y.len() {
            sums[codes[i] as usize] += y[i];
        }
        for g in 0..sums.len() {
            if counts[g] > 0 {
                sums[g] /= counts[g] as f64;
            }
        }
        for i in 0..y.len() {
            y[i] -= sums[codes[i] as usize];
        }
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn demean_coef<'py>(
    py: Python<'py>,
    mut x: PyReadwriteArray2<'py, f64>,
    fe_codes: &Bound<'py, PyList>,
    group_lists: &Bound<'py, PyList>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    max_iter: u32,
    tol_abs: f64,
    tol_rel: f64,
    accelerate: bool,
    accel_period: u32,
    lists_name: &str,
) -> PyResult<Bound<'py, PyList>> {
    if fe_codes.len() != group_lists.len() {
        return err(format!(
            "len(fe_codes)={} but len({lists_name})={}",
            fe_codes.len(),
            group_lists.len()
        ));
    }
    if accel_period == 0 {
        return err("accel_period must be >= 1".into());
    }
    if max_iter == 0 {
        return err("max_iter must be >= 1".into());
    }
    if !(tol_abs >= 0.0) || !(tol_rel >= 0.0) {
        return err("tol_abs and tol_rel must be non-negative".into());
    }
    let (n, p) = fortran_shape(&x)?;
    let code_views = i64_views(fe_codes)?;
    let group_views = f64_views(group_lists)?;
    let mut n_groups = Vec::with_capacity(group_views.len());
    for v in &group_views {
        n_groups.push(slice1(v, lists_name)?.len());
    }
    let codes = code_slices(&code_views, &n_groups, n, "fe_codes")?;
    let w: Option<Vec<f64>> = match &weights {
        None => None,
        Some(wv) => {
            let s = slice1(wv, "weights")?;
            if s.len() != n {
                return err(format!("weights length {} but n={n}", s.len()));
            }
            check_weights(s, "weights")?;
            Some(s.to_vec())
        }
    };
    let mat = x
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("x must be Fortran-contiguous"))?;

    let done = demean::DemeanInfo {
        iters: 0,
        converged: true,
        max_dx: 0.0,
    };
    if codes.is_empty() || n == 0 || p == 0 {
        return infos_to_list(py, &vec![done; p]);
    }
    let infos = py.allow_threads(|| {
        pool::install(|| {
            let mut problem = coef::Problem::new(codes, n_groups, w);
            problem.accelerate = accelerate;
            coef::demean_matrix(&problem, mat, n, p, max_iter, tol_abs, tol_rel)
        })
    });
    infos_to_list(py, &infos)
}

/// K-way FE residualisation of a Fortran-order ``(n, p)`` matrix in
/// place. Returns a list of dicts (one per column) with ``iters`` /
/// ``converged`` / ``max_dx``.
///
/// Parameters
/// ----------
/// x : 2-D float64 ndarray, shape (n, p), Fortran-contiguous
/// fe_codes : list[ndarray[int64, shape (n,)]]
///     One code array per FE, codes in ``[0, G_k)``.
/// counts : list[ndarray[float64, shape (G_k,)]]
///     One array per FE; only its length (the level count) is used.
/// max_iter : int
///     Cap on sweeps per column.
/// tol_abs, tol_rel : float
///     Stop when the bound on the change of the residual between two
///     sweeps is at most ``tol_abs + tol_rel * max|x|``.
/// accelerate : bool
///     Irons-Tuck acceleration (on by default upstream).
/// accel_period : int
///     Accepted for compatibility with API 1; must be >= 1.
#[pyfunction]
#[pyo3(signature = (x, fe_codes, counts, max_iter, tol_abs, tol_rel, accelerate, accel_period))]
#[allow(clippy::too_many_arguments)]
fn demean_2d<'py>(
    py: Python<'py>,
    x: PyReadwriteArray2<'py, f64>,
    fe_codes: &Bound<'py, PyList>,
    counts: &Bound<'py, PyList>,
    max_iter: u32,
    tol_abs: f64,
    tol_rel: f64,
    accelerate: bool,
    accel_period: u32,
) -> PyResult<Bound<'py, PyList>> {
    demean_coef(
        py,
        x,
        fe_codes,
        counts,
        None,
        max_iter,
        tol_abs,
        tol_rel,
        accelerate,
        accel_period,
        "counts",
    )
}

/// Weighted K-way FE residualisation of a Fortran-order ``(n, p)`` matrix
/// in place: the projection minimises ``sum_i w_i (x_i - fe_i)^2``.
///
/// Parameters
/// ----------
/// x : 2-D float64 ndarray, shape (n, p), Fortran-contiguous
/// fe_codes : list[ndarray[int64, shape (n,)]]
/// wsum : list[ndarray[float64, shape (G_k,)]]
///     One array per FE; only its length (the level count) is used, the
///     group weights are recomputed from ``weights``.
/// weights : ndarray[float64, shape (n,)]
///     Finite, non-negative observation weights.
/// max_iter, tol_abs, tol_rel, accelerate, accel_period
///     As in ``demean_2d``.
#[pyfunction]
#[pyo3(signature = (x, fe_codes, wsum, weights, max_iter, tol_abs, tol_rel, accelerate, accel_period))]
#[allow(clippy::too_many_arguments)]
fn demean_2d_weighted<'py>(
    py: Python<'py>,
    x: PyReadwriteArray2<'py, f64>,
    fe_codes: &Bound<'py, PyList>,
    wsum: &Bound<'py, PyList>,
    weights: PyReadonlyArray1<'py, f64>,
    max_iter: u32,
    tol_abs: f64,
    tol_rel: f64,
    accelerate: bool,
    accel_period: u32,
) -> PyResult<Bound<'py, PyList>> {
    demean_coef(
        py,
        x,
        fe_codes,
        wsum,
        Some(weights),
        max_iter,
        tol_abs,
        tol_rel,
        accelerate,
        accel_period,
        "wsum",
    )
}

/// Data-space weighted demean of a Fortran-order ``(n, p)`` matrix whose
/// rows are pre-sorted by the primary FE (API 1 entry point). The result
/// stays in the sorted order.
///
/// Parameters
/// ----------
/// x : 2-D float64 ndarray, shape (n, p), Fortran-contiguous, sorted.
/// primary_starts : ndarray[int64, shape (G1+1,)]
///     Non-decreasing group offsets with ``primary_starts[0] == 0`` and
///     ``primary_starts[G1] == n``.
/// primary_wsum : ndarray[float64, shape (G1,)]
/// secondary_codes : list[ndarray[int64, shape (n,)]]
/// secondary_wsum : list[ndarray[float64, shape (G_k,)]]
/// weights_sorted : ndarray[float64, shape (n,)]
/// max_iter, tol_abs, tol_rel, accelerate, accel_period
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
        return err(format!(
            "len(secondary_codes)={} but len(secondary_wsum)={}",
            secondary_codes.len(),
            secondary_wsum.len()
        ));
    }
    if accel_period == 0 {
        return err("accel_period must be >= 1".into());
    }
    let (n, p) = fortran_shape(&x)?;
    let starts = slice1(&primary_starts, "primary_starts")?;
    let pw = slice1(&primary_wsum, "primary_wsum")?;
    let wts = slice1(&weights_sorted, "weights_sorted")?;
    if wts.len() != n {
        return err(format!("weights_sorted length {} but n={n}", wts.len()));
    }
    check_weights(wts, "weights_sorted")?;
    if starts.len() != pw.len() + 1 {
        return err(format!(
            "primary_starts length {} but expected {} (= primary_wsum.len + 1)",
            starts.len(),
            pw.len() + 1
        ));
    }
    if starts[0] != 0
        || starts[starts.len() - 1] != n as i64
        || starts.windows(2).any(|w| w[1] < w[0])
    {
        return err(format!(
            "primary_starts must be non-decreasing from 0 to n={n}"
        ));
    }
    let sec_views = i64_views(secondary_codes)?;
    let sec_w_views = f64_views(secondary_wsum)?;
    let mut sec_lens = Vec::with_capacity(sec_w_views.len());
    let mut sec_w: Vec<&[f64]> = Vec::with_capacity(sec_w_views.len());
    for v in &sec_w_views {
        let s = slice1(v, "secondary_wsum")?;
        sec_lens.push(s.len());
        sec_w.push(s);
    }
    let sec_codes = code_slices(&sec_views, &sec_lens, n, "secondary_codes")?;
    let mat = x
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("x must be Fortran-contiguous"))?;
    let done = demean::DemeanInfo {
        iters: 0,
        converged: true,
        max_dx: 0.0,
    };
    if n == 0 || p == 0 || pw.is_empty() {
        return infos_to_list(py, &vec![done; p]);
    }
    let starts_usize: Vec<usize> = starts.iter().map(|&v| v as usize).collect();
    let infos = py.allow_threads(|| {
        pool::install(|| {
            demean::weighted_demean_matrix_fortran_inplace_sorted(
                mat,
                n,
                p,
                &starts_usize,
                pw,
                &sec_codes,
                &sec_w,
                &sec_lens,
                wts,
                max_iter,
                tol_abs,
                tol_rel,
                accelerate,
                accel_period,
            )
        })
    });
    infos_to_list(py, &infos)
}

/// Native Poisson IRLS for ``sp.fast.fepois``.
///
/// Parameters
/// ----------
/// y : ndarray[float64, shape (n,)]
/// x : ndarray[float64, shape (n, p)], Fortran-contiguous (not mutated)
/// fe_codes : list[ndarray[int64, shape (n,)]]
///     K >= 1 code arrays, codes in ``[0, g_per_fe[k])``.
/// g_per_fe : ndarray[int64, shape (K,)]
/// obs_weights : ndarray[float64, shape (n,)]
/// config : dict
///     maxiter / tol / fe_tol / fe_maxiter / eta_clip / accel_period /
///     max_halvings; missing keys take the defaults.
///
/// Returns
/// -------
/// dict with ``beta``, ``x_tilde_flat`` (F-order data of the residualised
/// X, reshape with ``.reshape(p, n).T``), ``x_tilde_n``, ``x_tilde_p``,
/// ``w``, ``eta``, ``mu``, ``deviance``, ``log_likelihood``, ``iters``,
/// ``converged``, ``n_halvings``, ``max_inner_dx``, ``fe_converged`` and
/// ``singular`` (the weighted normal equations could not be solved: a
/// regressor is collinear with the FEs or with the other regressors).
#[pyfunction]
#[pyo3(signature = (y, x, fe_codes, g_per_fe, obs_weights, config))]
fn fepois_irls<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    fe_codes: &Bound<'py, PyList>,
    g_per_fe: PyReadonlyArray1<'py, i64>,
    obs_weights: PyReadonlyArray1<'py, f64>,
    config: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    glm_irls(
        py,
        irls::Family::Poisson,
        y,
        x,
        fe_codes,
        g_per_fe,
        obs_weights,
        config,
    )
}

/// Native IRLS for ``sp.feglm`` with absorbed fixed effects (API 3).
///
/// Same arguments and returned dict as ``fepois_irls``, plus ``family``:
/// ``"poisson"``, ``"logit"`` or ``"probit"``. For the binomial families
/// ``y`` must lie in ``[0, 1]``; the working weights are Fisher scoring's
/// ``obs_weights * dmu^2 / V(mu)`` and the loop stops on fixest's relative
/// deviance change ``|dev - dev_old| / (0.1 + |dev|) < tol``.
#[pyfunction]
#[pyo3(signature = (y, x, fe_codes, g_per_fe, obs_weights, config, family))]
#[allow(clippy::too_many_arguments)]
fn feglm_irls<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    fe_codes: &Bound<'py, PyList>,
    g_per_fe: PyReadonlyArray1<'py, i64>,
    obs_weights: PyReadonlyArray1<'py, f64>,
    config: &Bound<'py, PyDict>,
    family: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let fam = irls::Family::parse(family).ok_or_else(|| {
        PyValueError::new_err(format!(
            "family must be 'poisson', 'logit' or 'probit', got {family:?}"
        ))
    })?;
    glm_irls(py, fam, y, x, fe_codes, g_per_fe, obs_weights, config)
}

#[allow(clippy::too_many_arguments)]
fn glm_irls<'py>(
    py: Python<'py>,
    fam: irls::Family,
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    fe_codes: &Bound<'py, PyList>,
    g_per_fe: PyReadonlyArray1<'py, i64>,
    obs_weights: PyReadonlyArray1<'py, f64>,
    config: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let y_view = slice1(&y, "y")?;
    let obs_w_view = slice1(&obs_weights, "obs_weights")?;
    let g_view = slice1(&g_per_fe, "g_per_fe")?;
    let shape = x.shape();
    let (n, p) = (shape[0], shape[1]);
    if !x.is_fortran_contiguous() {
        return err("x must be Fortran-contiguous; pass np.asfortranarray(X)".into());
    }
    if y_view.len() != n {
        return err(format!("y length {} but x has {n} rows", y_view.len()));
    }
    if obs_w_view.len() != n {
        return err(format!(
            "obs_weights length {} but x has {n} rows",
            obs_w_view.len()
        ));
    }
    check_weights(obs_w_view, "obs_weights")?;
    if fe_codes.is_empty() {
        return err("the IRLS kernel needs at least one FE".into());
    }
    if g_view.len() != fe_codes.len() {
        return err(format!(
            "len(g_per_fe)={} but len(fe_codes)={}",
            g_view.len(),
            fe_codes.len()
        ));
    }
    if g_view.iter().any(|&g| g < 0) {
        return err("g_per_fe must be non-negative".into());
    }
    if n == 0 || p == 0 {
        return err(format!(
            "the IRLS kernel needs n > 0 and p > 0 (got n={n}, p={p})"
        ));
    }
    if n > u32::MAX as usize {
        return err(format!("n={n} exceeds the supported 2^32 - 1 rows"));
    }
    let g_per_fe_vec: Vec<usize> = g_view.iter().map(|&g| g as usize).collect();
    let code_views = i64_views(fe_codes)?;
    let codes = code_slices(&code_views, &g_per_fe_vec, n, "fe_codes")?;
    let x_view = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("x must be Fortran-contiguous"))?;

    let mut cfg = irls::FePoisIRLSConfig::default();
    if let Some(v) = config.get_item("maxiter")? {
        cfg.maxiter = v.extract::<u32>()?;
    }
    if let Some(v) = config.get_item("tol")? {
        cfg.tol = v.extract::<f64>()?;
    }
    if let Some(v) = config.get_item("fe_tol")? {
        cfg.fe_tol = v.extract::<f64>()?;
    }
    if let Some(v) = config.get_item("fe_maxiter")? {
        cfg.fe_maxiter = v.extract::<u32>()?;
    }
    if let Some(v) = config.get_item("eta_clip")? {
        cfg.eta_clip = v.extract::<f64>()?;
    }
    if let Some(v) = config.get_item("accel_period")? {
        cfg.accel_period = v.extract::<u32>()?;
    }
    if let Some(v) = config.get_item("max_halvings")? {
        cfg.max_halvings = v.extract::<u32>()?;
    }
    if cfg.maxiter == 0 || cfg.fe_maxiter == 0 {
        return err("maxiter and fe_maxiter must be >= 1".into());
    }
    if !(cfg.tol >= 0.0) || !(cfg.fe_tol >= 0.0) || !(cfg.eta_clip > 0.0) {
        return err("tol and fe_tol must be non-negative and eta_clip positive".into());
    }
    match fam {
        irls::Family::Poisson => {
            if y_view.iter().any(|&v| !(v >= 0.0) || !v.is_finite()) {
                return err("y must be finite and non-negative for a Poisson model".into());
            }
        }
        irls::Family::Logit | irls::Family::Probit => {
            if y_view.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
                return err("y must lie in [0, 1] for a binomial model".into());
            }
        }
    }

    let result = py.allow_threads(|| {
        pool::install(|| {
            irls::glm_loop(fam, y_view, x_view, obs_w_view, &codes, &g_per_fe_vec, &cfg)
        })
    });

    let out = PyDict::new_bound(py);
    out.set_item("beta", PyArray1::from_vec_bound(py, result.beta))?;
    out.set_item("x_tilde_flat", PyArray1::from_vec_bound(py, result.x_tilde))?;
    out.set_item("x_tilde_n", n)?;
    out.set_item("x_tilde_p", p)?;
    out.set_item("w", PyArray1::from_vec_bound(py, result.w))?;
    out.set_item("eta", PyArray1::from_vec_bound(py, result.eta))?;
    out.set_item("mu", PyArray1::from_vec_bound(py, result.mu))?;
    out.set_item("deviance", result.deviance)?;
    out.set_item("log_likelihood", result.log_likelihood)?;
    out.set_item("iters", result.iters)?;
    out.set_item("converged", result.converged)?;
    out.set_item("n_halvings", result.n_halvings)?;
    out.set_item("max_inner_dx", result.max_inner_dx)?;
    out.set_item("fe_converged", result.fe_converged)?;
    out.set_item("singular", result.singular)?;
    Ok(out)
}

/// Iterative K-way singleton detection. Returns a uint8 keep-mask
/// (1 = keep, 0 = drop). Codes must be non-negative.
#[pyfunction]
fn singleton_mask<'py>(
    py: Python<'py>,
    fe_codes: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let code_views = i64_views(fe_codes)?;
    if code_views.is_empty() {
        return Ok(PyArray1::<u8>::zeros_bound(py, 0, false));
    }
    let mut codes: Vec<&[i64]> = Vec::with_capacity(code_views.len());
    for (k, v) in code_views.iter().enumerate() {
        let s = slice1(v, "fe_codes")?;
        if let Some(&first) = codes.first().map(|c: &&[i64]| c) {
            if s.len() != first.len() {
                return err(format!(
                    "fe_codes[{k}] has length {} but fe_codes[0] has {}",
                    s.len(),
                    first.len()
                ));
            }
        }
        if let Some(&m) = s.iter().min() {
            if m < 0 {
                return err(format!(
                    "fe_codes[{k}]: codes must be non-negative; found {m}"
                ));
            }
        }
        codes.push(s);
    }
    let keep = py.allow_threads(|| singletons::detect_singletons(&codes));
    let as_u8: Vec<u8> = keep.into_iter().map(u8::from).collect();
    Ok(PyArray1::from_vec_bound(py, as_u8))
}

/// Iterative Poisson-separation detection. Returns a uint8 keep-mask
/// (1 = keep, 0 = drop). Drops rows whose FE group has a zero y-sum;
/// iterates until fixed point.
#[pyfunction]
fn separation_mask<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    fe_codes: &Bound<'py, PyList>,
    g_per_fe: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let y_view = slice1(&y, "y")?;
    let g_view = slice1(&g_per_fe, "g_per_fe")?;
    let code_views = i64_views(fe_codes)?;
    let n = y_view.len();
    if code_views.is_empty() {
        return Ok(PyArray1::from_vec_bound(py, vec![1u8; n]));
    }
    if g_view.len() != code_views.len() {
        return err(format!(
            "len(g_per_fe)={} but len(fe_codes)={}",
            g_view.len(),
            code_views.len()
        ));
    }
    if g_view.iter().any(|&g| g < 0) {
        return err("g_per_fe must be non-negative".into());
    }
    let g_vec: Vec<usize> = g_view.iter().map(|&g| g as usize).collect();
    let codes = code_slices(&code_views, &g_vec, n, "fe_codes")?;
    let keep = py.allow_threads(|| separation::separation_mask(y_view, &codes, &g_vec));
    let as_u8: Vec<u8> = keep.into_iter().map(u8::from).collect();
    Ok(PyArray1::from_vec_bound(py, as_u8))
}

/// Cluster-robust sandwich "meat" matrix ``sum_g (X_g' r_g)(X_g' r_g)'``.
///
/// ``x`` and ``residuals`` must be pre-sorted by cluster;
/// ``cluster_starts[g] .. cluster_ends[g]`` is the half-open row range of
/// cluster ``g``. Fixed blocks of clusters are summed in parallel and the
/// block sums are added in order, so the result is the same at any
/// thread count.
///
/// Parameters
/// ----------
/// x : 2-D float64 ndarray, shape (n, k), C-contiguous
/// residuals : 1-D float64 ndarray, shape (n,)
/// cluster_starts, cluster_ends : 1-D int64 ndarray, shape (G,)
///
/// Returns
/// -------
/// (k, k) float64 ndarray (fully symmetric).
#[pyfunction]
fn cluster_meat<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    residuals: PyReadonlyArray1<'py, f64>,
    cluster_starts: PyReadonlyArray1<'py, i64>,
    cluster_ends: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if !x.is_c_contiguous() {
        return err("x must be C-contiguous; pass np.ascontiguousarray(X)".into());
    }
    let arr = x.as_array();
    let (n, k) = (arr.shape()[0], arr.shape()[1]);
    let r = slice1(&residuals, "residuals")?;
    if r.len() != n {
        return err(format!("len(residuals)={} but n={n}", r.len()));
    }
    let starts_i = slice1(&cluster_starts, "cluster_starts")?;
    let ends_i = slice1(&cluster_ends, "cluster_ends")?;
    if starts_i.len() != ends_i.len() {
        return err(format!(
            "len(cluster_starts)={} but len(cluster_ends)={}",
            starts_i.len(),
            ends_i.len()
        ));
    }
    for (g, (&s, &e)) in starts_i.iter().zip(ends_i.iter()).enumerate() {
        if s < 0 || e < s || e as u64 > n as u64 {
            return err(format!("cluster {g}: invalid range [{s},{e}) with n={n}"));
        }
    }
    let starts: Vec<usize> = starts_i.iter().map(|&v| v as usize).collect();
    let ends: Vec<usize> = ends_i.iter().map(|&v| v as usize).collect();
    let x_slice = arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("x must be C-contiguous"))?;

    let meat_flat = py.allow_threads(|| {
        pool::install(|| cluster::cluster_meat_sorted(x_slice, n, k, r, &starts, &ends))
    });
    let arr2 = Array2::from_shape_vec((k, k), meat_flat)
        .map_err(|e| PyValueError::new_err(format!("from_shape_vec: {e}")))?;
    Ok(PyArray2::from_owned_array_bound(py, arr2))
}

/// Set the number of threads the parallel kernels use (0 = default: one
/// per logical CPU, or ``RAYON_NUM_THREADS``). The environment variable
/// ``STATSPAI_NUM_THREADS`` sets the initial size.
#[pyfunction]
fn set_num_threads(n: usize) -> PyResult<()> {
    pool::set_num_threads(n).map_err(PyValueError::new_err)
}

/// Number of threads the parallel kernels use.
#[pyfunction]
fn get_num_threads() -> usize {
    pool::num_threads()
}

#[pymodule]
fn statspai_hdfe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(group_demean, m)?)?;
    m.add_function(wrap_pyfunction!(demean_2d, m)?)?;
    m.add_function(wrap_pyfunction!(demean_2d_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(demean_2d_weighted_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(fepois_irls, m)?)?;
    m.add_function(wrap_pyfunction!(feglm_irls, m)?)?;
    m.add_function(wrap_pyfunction!(singleton_mask, m)?)?;
    m.add_function(wrap_pyfunction!(separation_mask, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_meat, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("API_VERSION", API_VERSION)?;
    Ok(())
}
