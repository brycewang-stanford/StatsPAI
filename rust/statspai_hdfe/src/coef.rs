//! Coefficient-space HDFE projection.
//!
//! The data-space kernel in ``demean.rs`` sweeps every n-vector once per
//! FE per iteration. This module iterates on the FE *coefficients*
//! instead, which is what ``fixest`` does (Berge 2018, ``paper.bib`` key
//! ``berge2018efficient``):
//!
//! * **Two FEs.** The Gauss-Seidel sweep ``alpha_a <- mean_a(x - alpha_b)``,
//!   ``alpha_b <- mean_b(x - alpha_a)`` only needs the group sums of ``x``
//!   and the sparse cross-tabulation ``C[g, h] = sum_{a_i = g, b_i = h} w_i``.
//!   One sweep then costs ``O(nnz(C))`` instead of ``O(n)``; on panels
//!   where a unit meets few levels of the other FE (worker x firm,
//!   firm x year) ``nnz`` is a small fraction of ``n``. The state of the
//!   fixed point is the coefficient vector of the smaller FE, and the
//!   iteration is accelerated with Irons-Tuck at every step.
//! * **Three or more FEs.** The two FEs with the most groups form a
//!   block that is solved exactly (by the two-FE routine, warm-started)
//!   inside an outer Gauss-Seidel over the remaining FEs; the outer map
//!   is accelerated with Irons-Tuck on the remaining FEs' coefficients.
//!   The slowest-mixing pair is usually the two largest FEs, so the outer
//!   loop typically converges in a handful of steps.
//!
//! ## Stopping rule
//!
//! The data-space kernel stops when the largest change of the residual
//! between two consecutive sweeps is at most ``tol_abs + tol_rel *
//! max|x|``. Here the residual is ``x - sum_k alpha_k[c_k]``, so the
//! change between two consecutive sweeps is bounded by the sum over FEs of
//! the largest coefficient change; that bound is the reported ``max_dx``
//! and the same threshold applies. The projection itself is unique, so
//! the two kernels agree to within the tolerance.
//!
//! ## Returned residual
//!
//! On return the residual has exactly zero weighted means within the
//! groups of the FE updated last (up to rounding) and means bounded by
//! the threshold in every other FE.

use crate::demean::DemeanInfo;
use crate::par;

/// Sparse cross-tabulation of two FE dimensions, stored both row-wise
/// (by ``a``) and column-wise (by ``b``). Cell weights live in
/// [`CrossValues`] so the structure can be reused when only the weights
/// change (IRLS).
pub struct CrossTab {
    pub ga: usize,
    pub gb: usize,
    /// Row pointers by ``a`` (length ``ga + 1``) into ``a_idx``.
    pub a_ptr: Vec<usize>,
    /// Column (``b`` level) of each cell, in row order.
    pub a_idx: Vec<u32>,
    /// Column pointers by ``b`` (length ``gb + 1``) into ``b_idx`` / ``b_pos``.
    pub b_ptr: Vec<usize>,
    /// Row (``a`` level) of each cell, in column order.
    pub b_idx: Vec<u32>,
    /// Position in row order of each column-order cell.
    pub b_pos: Vec<u32>,
    /// Row-order cell of each observation.
    pub cell_of_row: Vec<u32>,
}

/// Cell weights and the implied group weights of a [`CrossTab`].
pub struct CrossValues {
    pub a_val: Vec<f64>,
    pub b_val: Vec<f64>,
    pub wa: Vec<f64>,
    pub wb: Vec<f64>,
}

impl CrossTab {
    /// Build the cross-tabulation of codes ``a`` (``ga`` levels) and ``b``
    /// (``gb`` levels). Codes must already be validated to lie in range,
    /// and ``a.len()`` must fit in ``u32``.
    pub fn build(a: &[i64], b: &[i64], ga: usize, gb: usize) -> CrossTab {
        let n = a.len();
        debug_assert_eq!(b.len(), n);
        debug_assert!(n <= u32::MAX as usize);

        // Counting sort of the rows by ``a``.
        let mut start = vec![0usize; ga + 1];
        for &g in a {
            start[g as usize + 1] += 1;
        }
        for g in 0..ga {
            start[g + 1] += start[g];
        }
        let mut fill = start.clone();
        let mut order = vec![0u32; n];
        for (i, &g) in a.iter().enumerate() {
            let g = g as usize;
            order[fill[g]] = i as u32;
            fill[g] += 1;
        }
        drop(fill);

        // Distinct ``b`` levels within each ``a`` group, in order of first
        // appearance. ``mark[h] == g`` means level ``h`` already has a
        // cell in row ``g``; ``slot[h]`` is that cell.
        let mut mark = vec![usize::MAX; gb];
        let mut slot = vec![0u32; gb];
        let mut a_ptr = Vec::with_capacity(ga + 1);
        a_ptr.push(0usize);
        let mut a_idx: Vec<u32> = Vec::new();
        let mut cell_of_row = vec![0u32; n];
        for g in 0..ga {
            for &i in &order[start[g]..start[g + 1]] {
                let h = b[i as usize] as usize;
                if mark[h] != g {
                    mark[h] = g;
                    slot[h] = a_idx.len() as u32;
                    a_idx.push(h as u32);
                }
                cell_of_row[i as usize] = slot[h];
            }
            a_ptr.push(a_idx.len());
        }
        drop(order);
        drop(mark);
        drop(slot);

        // Column-wise copy of the structure.
        let nnz = a_idx.len();
        let mut b_ptr = vec![0usize; gb + 1];
        for &h in &a_idx {
            b_ptr[h as usize + 1] += 1;
        }
        for h in 0..gb {
            b_ptr[h + 1] += b_ptr[h];
        }
        let mut fillb = b_ptr.clone();
        let mut b_idx = vec![0u32; nnz];
        let mut b_pos = vec![0u32; nnz];
        for g in 0..ga {
            for e in a_ptr[g]..a_ptr[g + 1] {
                let h = a_idx[e] as usize;
                let p = fillb[h];
                b_idx[p] = g as u32;
                b_pos[p] = e as u32;
                fillb[h] += 1;
            }
        }

        CrossTab {
            ga,
            gb,
            a_ptr,
            a_idx,
            b_ptr,
            b_idx,
            b_pos,
            cell_of_row,
        }
    }

    pub fn nnz(&self) -> usize {
        self.a_idx.len()
    }

    /// Cell and group weights for observation weights ``w`` (``None`` =
    /// unit weights).
    pub fn values(&self, w: Option<&[f64]>) -> CrossValues {
        let nnz = self.nnz();
        let n = self.cell_of_row.len();
        let cells = &self.cell_of_row;
        let mut a_val = vec![0.0f64; nnz];
        match w {
            None => par::scatter_add(n, &mut a_val, |r, acc| {
                for i in r {
                    acc[cells[i] as usize] += 1.0;
                }
            }),
            Some(w) => par::scatter_add(n, &mut a_val, |r, acc| {
                for i in r {
                    acc[cells[i] as usize] += w[i];
                }
            }),
        }
        let mut b_val = vec![0.0f64; nnz];
        for (p, &e) in self.b_pos.iter().enumerate() {
            b_val[p] = a_val[e as usize];
        }
        let mut wa = vec![0.0f64; self.ga];
        for g in 0..self.ga {
            let mut s = 0.0;
            for e in self.a_ptr[g]..self.a_ptr[g + 1] {
                s += a_val[e];
            }
            wa[g] = s;
        }
        let mut wb = vec![0.0f64; self.gb];
        for h in 0..self.gb {
            let mut s = 0.0;
            for p in self.b_ptr[h]..self.b_ptr[h + 1] {
                s += b_val[p];
            }
            wb[h] = s;
        }
        CrossValues {
            a_val,
            b_val,
            wa,
            wb,
        }
    }
}

/// The exactly-solved pair of a [`Problem`] with two or more FEs.
pub struct Pair {
    /// FE with at least as many levels as ``b``.
    pub a: usize,
    pub b: usize,
    pub cross: CrossTab,
    pub vals: CrossValues,
}

/// Problem description shared by every column of one call.
pub struct Problem<'a> {
    /// FE codes, one slice per FE (all validated).
    pub codes: Vec<&'a [i64]>,
    /// Number of levels per FE.
    pub n_groups: Vec<usize>,
    /// Observation weights (``None`` = unit weights).
    pub weights: Option<Vec<f64>>,
    /// The two FEs with the most levels; ``None`` with a single FE.
    pub pair: Option<Pair>,
    /// The FEs outside the pair, in the caller's order (with a single FE,
    /// that FE).
    pub rest: Vec<usize>,
    /// Group weights of each FE in ``rest`` (aligned with ``rest``).
    pub rest_w: Vec<Vec<f64>>,
    /// Irons-Tuck acceleration (on by default).
    pub accelerate: bool,
}

fn group_weights(codes: &[i64], g: usize, weights: Option<&[f64]>, out: &mut Vec<f64>) {
    out.clear();
    out.resize(g, 0.0);
    match weights {
        None => {
            for &c in codes {
                out[c as usize] += 1.0;
            }
        }
        Some(w) => {
            for (i, &c) in codes.iter().enumerate() {
                out[c as usize] += w[i];
            }
        }
    }
}

impl<'a> Problem<'a> {
    /// ``codes`` must be non-empty and validated.
    pub fn new(
        codes: Vec<&'a [i64]>,
        n_groups: Vec<usize>,
        weights: Option<Vec<f64>>,
    ) -> Problem<'a> {
        let k = codes.len();
        debug_assert!(k >= 1);
        let (pair, rest) = if k >= 2 {
            // The two FEs with the most levels (ties: the earlier one).
            let mut idx: Vec<usize> = (0..k).collect();
            idx.sort_by(|&i, &j| n_groups[j].cmp(&n_groups[i]).then(i.cmp(&j)));
            let (a, b) = (idx[0], idx[1]);
            let cross = CrossTab::build(codes[a], codes[b], n_groups[a], n_groups[b]);
            let vals = cross.values(weights.as_deref());
            let rest: Vec<usize> = (0..k).filter(|&q| q != a && q != b).collect();
            (Some(Pair { a, b, cross, vals }), rest)
        } else {
            (None, vec![0usize])
        };
        let rest_w: Vec<Vec<f64>> = rest
            .iter()
            .map(|&q| {
                let mut s = Vec::new();
                group_weights(codes[q], n_groups[q], weights.as_deref(), &mut s);
                s
            })
            .collect();
        Problem {
            codes,
            n_groups,
            weights,
            pair,
            rest,
            rest_w,
            accelerate: true,
        }
    }

    /// Replace the observation weights, keeping the structure (IRLS).
    pub fn reweight(&mut self, weights: &[f64]) {
        match &mut self.weights {
            Some(w) => w.copy_from_slice(weights),
            None => self.weights = Some(weights.to_vec()),
        }
        let w = self.weights.as_deref();
        if let Some(pair) = &mut self.pair {
            pair.vals = pair.cross.values(w);
        }
        for (r, &q) in self.rest.iter().enumerate() {
            group_weights(self.codes[q], self.n_groups[q], w, &mut self.rest_w[r]);
        }
    }

    #[inline]
    fn w(&self, i: usize) -> f64 {
        match &self.weights {
            None => 1.0,
            Some(w) => w[i],
        }
    }
}

/// Per-column scratch. Holding it across calls keeps the coefficients
/// for warm starts (IRLS demeans the same columns with new weights).
pub struct Workspace {
    // pair solve
    ba: Vec<f64>,
    bb: Vec<f64>,
    x: Vec<f64>,
    gx: Vec<f64>,
    ggx: Vec<f64>,
    a1: Vec<f64>,
    a2: Vec<f64>,
    /// Current coefficients of the pair (warm start for the next solve).
    pub alpha_a: Vec<f64>,
    pub alpha_b: Vec<f64>,
    // outer loop over the remaining FEs (concatenated coefficients)
    rx: Vec<f64>,
    rgx: Vec<f64>,
    rggx: Vec<f64>,
    rest_off: Vec<usize>,
    resid: Vec<f64>,
    resid2: Vec<f64>,
    sums: Vec<f64>,
    pa_prev: Vec<f64>,
    pb_prev: Vec<f64>,
}

impl Workspace {
    pub fn new(p: &Problem, n: usize) -> Workspace {
        let (ga, gb) = match &p.pair {
            Some(pair) => (pair.cross.ga, pair.cross.gb),
            None => (0, 0),
        };
        let mut rest_off = Vec::with_capacity(p.rest.len() + 1);
        rest_off.push(0usize);
        for &q in &p.rest {
            let last = *rest_off.last().unwrap();
            rest_off.push(last + p.n_groups[q]);
        }
        let r_len = *rest_off.last().unwrap();
        let max_rest = p.rest.iter().map(|&q| p.n_groups[q]).max().unwrap_or(0);
        let has_rest = p.pair.is_some() && !p.rest.is_empty();
        Workspace {
            ba: vec![0.0; ga],
            bb: vec![0.0; gb],
            x: vec![0.0; gb],
            gx: vec![0.0; gb],
            ggx: vec![0.0; gb],
            a1: vec![0.0; ga],
            a2: vec![0.0; ga],
            alpha_a: vec![0.0; ga],
            alpha_b: vec![0.0; gb],
            rx: vec![0.0; r_len],
            rgx: vec![0.0; r_len],
            rggx: vec![0.0; r_len],
            rest_off,
            resid: if has_rest { vec![0.0; n] } else { Vec::new() },
            resid2: if has_rest { vec![0.0; n] } else { Vec::new() },
            sums: vec![0.0; max_rest],
            pa_prev: if has_rest { vec![0.0; ga] } else { Vec::new() },
            pb_prev: if has_rest { vec![0.0; gb] } else { Vec::new() },
        }
    }

    /// Forget the coefficients (cold start).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn reset(&mut self) {
        for v in self.alpha_a.iter_mut() {
            *v = 0.0;
        }
        for v in self.alpha_b.iter_mut() {
            *v = 0.0;
        }
        for v in self.rx.iter_mut() {
            *v = 0.0;
        }
    }
}

#[inline]
fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    let mut m = 0.0f64;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > m || d.is_nan() {
            m = if d.is_nan() { f64::INFINITY } else { d };
        }
    }
    m
}

/// One Gauss-Seidel sweep of the pair, as a map on ``alpha_b``:
/// ``alpha_a_out = A(alpha_b_in)``, ``alpha_b_out = B(alpha_a_out)``.
#[inline]
fn pair_map(
    ct: &CrossTab,
    v: &CrossValues,
    ba: &[f64],
    bb: &[f64],
    alpha_b_in: &[f64],
    alpha_a_out: &mut [f64],
    alpha_b_out: &mut [f64],
) {
    for g in 0..ct.ga {
        let w = v.wa[g];
        if w > 0.0 {
            let mut s = ba[g];
            for e in ct.a_ptr[g]..ct.a_ptr[g + 1] {
                s -= v.a_val[e] * alpha_b_in[ct.a_idx[e] as usize];
            }
            alpha_a_out[g] = s / w;
        } else {
            alpha_a_out[g] = 0.0;
        }
    }
    for h in 0..ct.gb {
        let w = v.wb[h];
        if w > 0.0 {
            let mut s = bb[h];
            for p in ct.b_ptr[h]..ct.b_ptr[h + 1] {
                s -= v.b_val[p] * alpha_a_out[ct.b_idx[p] as usize];
            }
            alpha_b_out[h] = s / w;
        } else {
            alpha_b_out[h] = 0.0;
        }
    }
}

/// Irons-Tuck extrapolation ``X <- GGX - c (GGX - GX)`` with
/// ``c = <dGX, d2X> / <d2X, d2X>``, ``dGX = GGX - GX``,
/// ``d2X = dGX - GX + X``. Falls back to ``X <- GGX`` when the
/// denominator vanishes or the jump is not finite.
#[inline]
fn irons_tuck(x: &mut [f64], gx: &[f64], ggx: &[f64]) {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..x.len() {
        let dgx = ggx[i] - gx[i];
        let d2x = dgx - gx[i] + x[i];
        num += dgx * d2x;
        den += d2x * d2x;
    }
    let c = num / den;
    if den > 0.0 && c.is_finite() {
        for i in 0..x.len() {
            x[i] = ggx[i] - c * (ggx[i] - gx[i]);
        }
    } else {
        x.copy_from_slice(ggx);
    }
}

/// Solve the pair for group sums ``ws.ba`` / ``ws.bb``, warm-started from
/// ``ws.alpha_b``. On return ``ws.alpha_a`` / ``ws.alpha_b`` hold the
/// solution (``alpha_b = B(alpha_a)`` exactly).
fn solve_pair(
    pair: &Pair,
    ws: &mut Workspace,
    stop: f64,
    max_iter: u32,
    accelerate: bool,
) -> DemeanInfo {
    let ct = &pair.cross;
    let v = &pair.vals;
    ws.x.copy_from_slice(&ws.alpha_b);
    let mut iters: u32 = 0;
    let mut max_dx = f64::INFINITY;
    let mut converged = false;
    while iters < max_iter {
        pair_map(ct, v, &ws.ba, &ws.bb, &ws.x, &mut ws.a1, &mut ws.gx);
        iters += 1;
        let d0 = max_abs_diff(&ws.gx, &ws.x);
        if d0 == 0.0 {
            // Exact fixed point: alpha_b = x, alpha_a = A(x).
            ws.alpha_a.copy_from_slice(&ws.a1);
            ws.alpha_b.copy_from_slice(&ws.gx);
            return DemeanInfo {
                iters,
                converged: true,
                max_dx: 0.0,
            };
        }
        if iters >= max_iter {
            ws.alpha_a.copy_from_slice(&ws.a1);
            ws.alpha_b.copy_from_slice(&ws.gx);
            max_dx = d0;
            break;
        }
        pair_map(ct, v, &ws.ba, &ws.bb, &ws.gx, &mut ws.a2, &mut ws.ggx);
        iters += 1;
        max_dx = max_abs_diff(&ws.a2, &ws.a1) + max_abs_diff(&ws.ggx, &ws.gx);
        // State after the second sweep: alpha_a = A(gx), alpha_b = B(alpha_a).
        ws.alpha_a.copy_from_slice(&ws.a2);
        ws.alpha_b.copy_from_slice(&ws.ggx);
        if max_dx <= stop {
            converged = true;
            break;
        }
        if accelerate {
            irons_tuck(&mut ws.x, &ws.gx, &ws.ggx);
        } else {
            ws.x.copy_from_slice(&ws.ggx);
        }
    }
    DemeanInfo {
        iters,
        converged,
        max_dx,
    }
}

#[inline]
fn base_scale(x: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for &v in x {
        let a = v.abs();
        if a > s {
            s = a;
        }
    }
    s + 1e-30
}

/// Residualise one column in place.
pub fn demean_column(
    p: &Problem,
    ws: &mut Workspace,
    x: &mut [f64],
    max_iter: u32,
    tol_abs: f64,
    tol_rel: f64,
) -> DemeanInfo {
    // A non-finite value makes every group sum it touches non-finite and
    // the iteration could never converge: report it instead of iterating.
    // One pass finds both that and the scale of the stopping rule.
    let xs: &[f64] = x;
    let scale = par::max(xs.len(), |r| {
        let mut m = 0.0f64;
        for i in r {
            let a = xs[i].abs();
            if !(a <= f64::MAX) {
                return f64::INFINITY;
            }
            if a > m {
                m = a;
            }
        }
        m
    });
    if max_iter == 0 || !scale.is_finite() {
        if max_iter > 0 {
            for v in x.iter_mut() {
                *v = f64::NAN;
            }
        }
        return DemeanInfo {
            iters: 0,
            converged: false,
            max_dx: f64::INFINITY,
        };
    }
    let stop = tol_abs + tol_rel * (scale + 1e-30);
    match &p.pair {
        None => demean_column_single(p, ws, x),
        Some(pair) if p.rest.is_empty() => demean_column_pair(p, pair, ws, x, stop, max_iter),
        Some(pair) => demean_column_block(p, pair, ws, x, stop, max_iter),
    }
}

/// One FE: the projection is a single weighted group demean.
fn demean_column_single(p: &Problem, ws: &mut Workspace, x: &mut [f64]) -> DemeanInfo {
    let codes = p.codes[0];
    let g = p.n_groups[0];
    let wq = &p.rest_w[0];
    let s = &mut ws.sums[..g];
    {
        let xs: &[f64] = x;
        match p.weights.as_deref() {
            None => par::scatter_add(xs.len(), s, |r, acc| {
                for i in r {
                    acc[codes[i] as usize] += xs[i];
                }
            }),
            Some(w) => par::scatter_add(xs.len(), s, |r, acc| {
                for i in r {
                    acc[codes[i] as usize] += w[i] * xs[i];
                }
            }),
        }
    }
    for h in 0..g {
        s[h] = if wq[h] > 0.0 { s[h] / wq[h] } else { 0.0 };
    }
    let s: &[f64] = s;
    par::for_each_mut(x, |i, v| *v -= s[codes[i] as usize]);
    DemeanInfo {
        iters: 1,
        converged: true,
        max_dx: 0.0,
    }
}

/// Two FEs.
fn demean_column_pair(
    p: &Problem,
    pair: &Pair,
    ws: &mut Workspace,
    x: &mut [f64],
    stop: f64,
    max_iter: u32,
) -> DemeanInfo {
    let ca = p.codes[pair.a];
    let cb = p.codes[pair.b];
    let ga = ws.ba.len();
    // Both sets of group sums in one pass over the rows.
    let mut sums = vec![0.0f64; ga + ws.bb.len()];
    {
        let xs: &[f64] = x;
        match p.weights.as_deref() {
            None => par::scatter_add(xs.len(), &mut sums, |r, acc| {
                let (sa, sb) = acc.split_at_mut(ga);
                for i in r {
                    sa[ca[i] as usize] += xs[i];
                    sb[cb[i] as usize] += xs[i];
                }
            }),
            Some(w) => par::scatter_add(xs.len(), &mut sums, |r, acc| {
                let (sa, sb) = acc.split_at_mut(ga);
                for i in r {
                    let wx = w[i] * xs[i];
                    sa[ca[i] as usize] += wx;
                    sb[cb[i] as usize] += wx;
                }
            }),
        }
    }
    ws.ba.copy_from_slice(&sums[..ga]);
    ws.bb.copy_from_slice(&sums[ga..]);
    let info = solve_pair(pair, ws, stop, max_iter, p.accelerate);
    let (aa, ab) = (&ws.alpha_a, &ws.alpha_b);
    par::for_each_mut(x, |i, v| *v -= aa[ca[i] as usize] + ab[cb[i] as usize]);
    info
}

/// One application of the outer map at the remaining-FE coefficients
/// ``rest_in`` (concatenated): solve the pair given them, then update each
/// remaining FE once (Gauss-Seidel). Writes the updated coefficients to
/// ``rest_out`` and the matching residual to ``resid``. Returns the inner
/// solve's info.
#[allow(clippy::too_many_arguments)]
fn outer_map(
    p: &Problem,
    pair: &Pair,
    ws_pair: &mut Workspace,
    x: &[f64],
    rest_in: &[f64],
    rest_out: &mut [f64],
    resid: &mut [f64],
    sums: &mut [f64],
    rest_off: &[usize],
    stop_inner: f64,
    max_iter: u32,
) -> DemeanInfo {
    let n = x.len();
    let ca = p.codes[pair.a];
    let cb = p.codes[pair.b];
    for v in ws_pair.ba.iter_mut() {
        *v = 0.0;
    }
    for v in ws_pair.bb.iter_mut() {
        *v = 0.0;
    }
    for i in 0..n {
        let mut z = x[i];
        for (r, &q) in p.rest.iter().enumerate() {
            z -= rest_in[rest_off[r] + p.codes[q][i] as usize];
        }
        resid[i] = z;
        let wz = p.w(i) * z;
        ws_pair.ba[ca[i] as usize] += wz;
        ws_pair.bb[cb[i] as usize] += wz;
    }
    let info = solve_pair(pair, ws_pair, stop_inner, max_iter, p.accelerate);
    for i in 0..n {
        resid[i] -= ws_pair.alpha_a[ca[i] as usize] + ws_pair.alpha_b[cb[i] as usize];
    }
    rest_out.copy_from_slice(rest_in);
    for (r, &q) in p.rest.iter().enumerate() {
        let g = p.n_groups[q];
        let codes = p.codes[q];
        let s = &mut sums[..g];
        for v in s.iter_mut() {
            *v = 0.0;
        }
        for i in 0..n {
            s[codes[i] as usize] += p.w(i) * resid[i];
        }
        let wq = &p.rest_w[r];
        for h in 0..g {
            s[h] = if wq[h] > 0.0 { s[h] / wq[h] } else { 0.0 };
        }
        for i in 0..n {
            resid[i] -= s[codes[i] as usize];
        }
        let off = rest_off[r];
        for h in 0..g {
            rest_out[off + h] += s[h];
        }
    }
    info
}

/// Total pair sweeps allowed across all inner solves of one column, as a
/// multiple of ``max_iter``.
const INNER_BUDGET: u64 = 10;

/// Three or more FEs: exact pair inside an accelerated outer loop.
fn demean_column_block(
    p: &Problem,
    pair: &Pair,
    ws: &mut Workspace,
    x: &mut [f64],
    stop: f64,
    max_iter: u32,
) -> DemeanInfo {
    // At convergence the inner solve must be tighter than the outer
    // threshold, or its own slack would keep the outer change above it.
    // Early outer steps only need a rough inner solve: the inner threshold
    // starts loose and follows the outer change down (inexact block
    // Gauss-Seidel), and convergence is only declared once it is tight.
    let scale = base_scale(x);
    let stop_inner = (0.1 * stop).max(64.0 * f64::EPSILON * scale);
    let mut tol_now = (1e-3 * scale).max(stop_inner);
    // Inner sweeps share one budget, so a stalled problem costs at most
    // INNER_BUDGET * max_iter pair sweeps rather than max_iter squared.
    let mut inner_left: u64 = INNER_BUDGET * max_iter as u64;

    // Move the outer buffers out of the workspace so the pair part of it
    // can be borrowed mutably by ``outer_map``.
    let mut rx = std::mem::take(&mut ws.rx);
    let mut rgx = std::mem::take(&mut ws.rgx);
    let mut rggx = std::mem::take(&mut ws.rggx);
    let mut resid = std::mem::take(&mut ws.resid);
    let mut resid2 = std::mem::take(&mut ws.resid2);
    let mut sums = std::mem::take(&mut ws.sums);
    let mut pa_prev = std::mem::take(&mut ws.pa_prev);
    let mut pb_prev = std::mem::take(&mut ws.pb_prev);
    let rest_off = ws.rest_off.clone();

    let mut iters: u32 = 0;
    let mut converged = false;
    let mut max_dx = f64::INFINITY;
    let mut inner_ok = true;
    let mut final_in_resid2 = false;

    while iters < max_iter {
        let cap = inner_left.min(max_iter as u64) as u32;
        let i1 = outer_map(
            p, pair, ws, x, &rx, &mut rgx, &mut resid, &mut sums, &rest_off, tol_now, cap,
        );
        iters += 1;
        inner_left = inner_left.saturating_sub(i1.iters as u64);
        inner_ok = i1.converged;
        final_in_resid2 = false;
        pa_prev.copy_from_slice(&ws.alpha_a);
        pb_prev.copy_from_slice(&ws.alpha_b);
        if iters >= max_iter || inner_left == 0 {
            max_dx = max_abs_diff(&rgx, &rx);
            rx.copy_from_slice(&rgx);
            inner_ok = false;
            break;
        }
        let cap = inner_left.min(max_iter as u64) as u32;
        let i2 = outer_map(
            p,
            pair,
            ws,
            x,
            &rgx,
            &mut rggx,
            &mut resid2,
            &mut sums,
            &rest_off,
            tol_now,
            cap,
        );
        iters += 1;
        inner_left = inner_left.saturating_sub(i2.iters as u64);
        inner_ok = i2.converged;
        final_in_resid2 = true;
        max_dx = max_abs_diff(&ws.alpha_a, &pa_prev) + max_abs_diff(&ws.alpha_b, &pb_prev);
        for r in 0..p.rest.len() {
            let (lo, hi) = (rest_off[r], rest_off[r + 1]);
            max_dx += max_abs_diff(&rggx[lo..hi], &rgx[lo..hi]);
        }
        let tight = tol_now <= stop_inner;
        if max_dx <= stop && tight {
            converged = true;
            rx.copy_from_slice(&rggx);
            break;
        }
        tol_now = (0.01 * max_dx).min(tol_now).max(stop_inner);
        if p.accelerate {
            irons_tuck(&mut rx, &rgx, &rggx);
        } else {
            rx.copy_from_slice(&rggx);
        }
    }

    if final_in_resid2 {
        x.copy_from_slice(&resid2);
    } else {
        x.copy_from_slice(&resid);
    }

    ws.rx = rx;
    ws.rgx = rgx;
    ws.rggx = rggx;
    ws.resid = resid;
    ws.resid2 = resid2;
    ws.sums = sums;
    ws.pa_prev = pa_prev;
    ws.pb_prev = pb_prev;

    DemeanInfo {
        iters,
        converged: converged && inner_ok,
        max_dx,
    }
}

/// Residualise every column of a column-major ``(n, ncol)`` matrix in
/// place, in parallel over columns.
pub fn demean_matrix(
    p: &Problem,
    mat: &mut [f64],
    n: usize,
    ncol: usize,
    max_iter: u32,
    tol_abs: f64,
    tol_rel: f64,
) -> Vec<DemeanInfo> {
    use rayon::prelude::*;
    debug_assert_eq!(mat.len(), n * ncol);
    if n == 0 {
        return vec![
            DemeanInfo {
                iters: 0,
                converged: true,
                max_dx: 0.0,
            };
            ncol
        ];
    }
    mat.par_chunks_mut(n)
        .map(|col| {
            let mut ws = Workspace::new(p, n);
            demean_column(p, &mut ws, col, max_iter, tol_abs, tol_rel)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demean::demean_column_inplace;

    /// Deterministic LCG so the tests need no extra crate.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 11
        }
        fn unif(&mut self) -> f64 {
            (self.next() as f64) / ((1u64 << 53) as f64)
        }
        fn int(&mut self, m: usize) -> i64 {
            (self.next() % m as u64) as i64
        }
    }

    fn reference(x: &[f64], codes: &[&[i64]], groups: &[usize]) -> Vec<f64> {
        let counts: Vec<Vec<f64>> = codes
            .iter()
            .zip(groups)
            .map(|(c, &g)| {
                let mut s = vec![0.0; g];
                for &v in c.iter() {
                    s[v as usize] += 1.0;
                }
                s
            })
            .collect();
        let counts_ref: Vec<&[f64]> = counts.iter().map(|v| v.as_slice()).collect();
        let mut scratch: Vec<Vec<f64>> = groups.iter().map(|&g| vec![0.0; g]).collect();
        let mut out = x.to_vec();
        let info = demean_column_inplace(
            &mut out,
            codes,
            &counts_ref,
            &mut scratch,
            100_000,
            0.0,
            1e-14,
            true,
            5,
        );
        assert!(info.converged);
        out
    }

    fn group_mean_max(r: &[f64], codes: &[i64], g: usize, w: Option<&[f64]>) -> f64 {
        let mut s = vec![0.0; g];
        let mut c = vec![0.0; g];
        for i in 0..r.len() {
            let wi = w.map_or(1.0, |w| w[i]);
            s[codes[i] as usize] += wi * r[i];
            c[codes[i] as usize] += wi;
        }
        (0..g)
            .filter(|&h| c[h] > 0.0)
            .map(|h| (s[h] / c[h]).abs())
            .fold(0.0, f64::max)
    }

    fn check(n: usize, groups: &[usize], weighted: bool, seed: u64) {
        let mut rng = Lcg(seed);
        let codes: Vec<Vec<i64>> = groups
            .iter()
            .map(|&g| (0..n).map(|_| rng.int(g)).collect())
            .collect();
        let x: Vec<f64> = (0..n)
            .map(|i| rng.unif() * 3.0 + (codes[0][i] as f64) * 0.1)
            .collect();
        let w: Vec<f64> = (0..n).map(|_| 0.2 + rng.unif()).collect();
        let code_refs: Vec<&[i64]> = codes.iter().map(|v| v.as_slice()).collect();
        let wopt = if weighted { Some(w.as_slice()) } else { None };
        let p = Problem::new(code_refs.clone(), groups.to_vec(), wopt.map(|v| v.to_vec()));
        let mut ws = Workspace::new(&p, n);
        let mut out = x.clone();
        let info = demean_column(&p, &mut ws, &mut out, 10_000, 0.0, 1e-13);
        assert!(info.converged, "not converged: {:?}", info);
        for (k, c) in codes.iter().enumerate() {
            let m = group_mean_max(&out, c, groups[k], wopt);
            assert!(m < 1e-9, "fe {k}: max group mean {m}");
        }
        if !weighted {
            let r = reference(&x, &code_refs, groups);
            let d = max_abs_diff(&out, &r);
            assert!(d < 1e-9, "differs from AP reference by {d}");
        }
    }

    /// Above ``par::PAR_MIN_ROWS`` the group sums and residuals run over
    /// row chunks: still the exact projection, and bit-identical for any
    /// thread count.
    #[test]
    fn row_parallel_paths_are_exact_and_thread_independent() {
        for (groups, weighted) in [
            (vec![300usize, 12], false),
            (vec![300, 12], true),
            (vec![500], true),
        ] {
            let n = 120_000;
            let mut rng = Lcg(17);
            let codes: Vec<Vec<i64>> = groups
                .iter()
                .map(|&g| (0..n).map(|_| rng.int(g)).collect())
                .collect();
            let x: Vec<f64> = (0..n)
                .map(|i| rng.unif() + (codes[0][i] as f64) * 0.01)
                .collect();
            let w: Vec<f64> = (0..n).map(|_| 0.2 + rng.unif()).collect();
            let refs: Vec<&[i64]> = codes.iter().map(|v| v.as_slice()).collect();
            let wopt = if weighted { Some(w.clone()) } else { None };
            let run = |threads: usize| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                pool.install(|| {
                    let p = Problem::new(refs.clone(), groups.clone(), wopt.clone());
                    let mut ws = Workspace::new(&p, n);
                    let mut out = x.clone();
                    let info = demean_column(&p, &mut ws, &mut out, 10_000, 0.0, 1e-13);
                    assert!(info.converged);
                    out
                })
            };
            let one = run(1);
            let eight = run(8);
            assert!(one
                .iter()
                .zip(&eight)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
            for (k, c) in codes.iter().enumerate() {
                let m = group_mean_max(&one, c, groups[k], wopt.as_deref());
                assert!(m < 1e-9, "fe {k}: max group mean {m}");
            }
            if !weighted {
                let d = max_abs_diff(&one, &reference(&x, &refs, &groups));
                assert!(d < 1e-9, "differs from AP reference by {d}");
            }
        }
    }

    #[test]
    fn two_fe_matches_reference() {
        check(2_000, &[60, 9], false, 1);
        check(2_000, &[9, 60], false, 2);
        check(500, &[200, 200], false, 3);
    }

    #[test]
    fn three_and_four_fe_match_reference() {
        check(3_000, &[300, 40, 7], false, 4);
        check(3_000, &[7, 40, 300], false, 5);
        check(3_000, &[100, 80, 60, 5], false, 6);
    }

    #[test]
    fn weighted_group_means_vanish() {
        check(2_000, &[60, 9], true, 7);
        check(3_000, &[300, 40, 7], true, 8);
    }

    #[test]
    fn nested_fe_converges_immediately() {
        // b is a coarsening of a: the projection is the one-way demean by a.
        let n = 1000;
        let a: Vec<i64> = (0..n).map(|i| (i % 50) as i64).collect();
        let b: Vec<i64> = a.iter().map(|&g| g / 10).collect();
        let x: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let p = Problem::new(vec![&a, &b], vec![50, 5], None);
        let mut ws = Workspace::new(&p, n);
        let mut out = x.clone();
        let info = demean_column(&p, &mut ws, &mut out, 100, 0.0, 1e-13);
        assert!(info.converged);
        assert!(group_mean_max(&out, &a, 50, None) < 1e-12);
    }

    #[test]
    fn empty_levels_are_ignored() {
        // Level 3 of the first FE and level 1 of the second never occur.
        let a: Vec<i64> = vec![0, 0, 1, 1, 2, 2, 4, 4];
        let b: Vec<i64> = vec![0, 2, 0, 2, 0, 2, 0, 2];
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0];
        let p = Problem::new(vec![&a, &b], vec![5, 3], None);
        let mut ws = Workspace::new(&p, 8);
        let mut out = x.clone();
        let info = demean_column(&p, &mut ws, &mut out, 1000, 0.0, 1e-14);
        assert!(info.converged);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(group_mean_max(&out, &a, 5, None) < 1e-12);
        assert!(group_mean_max(&out, &b, 3, None) < 1e-12);
    }

    #[test]
    fn crosstab_counts_cells() {
        let a: Vec<i64> = vec![0, 0, 1, 1, 1, 2];
        let b: Vec<i64> = vec![0, 0, 0, 1, 1, 1];
        let ct = CrossTab::build(&a, &b, 3, 2);
        assert_eq!(ct.nnz(), 4);
        let v = ct.values(None);
        assert_eq!(v.wa, vec![2.0, 3.0, 1.0]);
        assert_eq!(v.wb, vec![3.0, 3.0]);
        let mut total = 0.0;
        for x in &v.a_val {
            total += x;
        }
        assert_eq!(total, 6.0);
    }

    #[test]
    fn single_fe_is_one_sweep() {
        let a: Vec<i64> = vec![0, 0, 1, 1, 1, 2];
        let x: Vec<f64> = vec![1.0, 3.0, 2.0, 4.0, 6.0, 5.0];
        let w: Vec<f64> = vec![1.0, 3.0, 1.0, 1.0, 2.0, 1.0];
        let p = Problem::new(vec![&a], vec![3], Some(w.clone()));
        let mut ws = Workspace::new(&p, 6);
        let mut out = x.clone();
        let info = demean_column(&p, &mut ws, &mut out, 10, 0.0, 1e-14);
        assert!(info.converged);
        // weighted means: g0 = (1 + 9) / 4 = 2.5, g1 = (2 + 4 + 12) / 4 = 4.5, g2 = 5
        let expect = [-1.5, 0.5, -2.5, -0.5, 1.5, 0.0];
        for i in 0..6 {
            assert!((out[i] - expect[i]).abs() < 1e-14);
        }
    }

    #[test]
    fn reweight_matches_fresh_problem_and_warm_start() {
        let mut rng = Lcg(11);
        let n = 3_000;
        let groups = [200usize, 30, 6];
        let codes: Vec<Vec<i64>> = groups
            .iter()
            .map(|&g| (0..n).map(|_| rng.int(g)).collect())
            .collect();
        let refs: Vec<&[i64]> = codes.iter().map(|v| v.as_slice()).collect();
        let x: Vec<f64> = (0..n).map(|_| rng.unif()).collect();
        let w1: Vec<f64> = (0..n).map(|_| 0.5 + rng.unif()).collect();
        let w2: Vec<f64> = (0..n).map(|_| 0.5 + rng.unif()).collect();

        let mut p = Problem::new(refs.clone(), groups.to_vec(), Some(w1));
        let mut ws = Workspace::new(&p, n);
        let mut out1 = x.clone();
        assert!(demean_column(&p, &mut ws, &mut out1, 10_000, 0.0, 1e-13).converged);
        p.reweight(&w2);
        let mut warm = x.clone();
        let iw = demean_column(&p, &mut ws, &mut warm, 10_000, 0.0, 1e-13);
        assert!(iw.converged);

        let fresh = Problem::new(refs, groups.to_vec(), Some(w2));
        let mut ws2 = Workspace::new(&fresh, n);
        let mut cold = x.clone();
        assert!(demean_column(&fresh, &mut ws2, &mut cold, 10_000, 0.0, 1e-13).converged);
        assert!(max_abs_diff(&warm, &cold) < 1e-10);
    }

    #[test]
    fn matrix_entry_handles_columns_independently() {
        let mut rng = Lcg(12);
        let n = 1_000;
        let a: Vec<i64> = (0..n).map(|_| rng.int(40)).collect();
        let b: Vec<i64> = (0..n).map(|_| rng.int(7)).collect();
        let p = Problem::new(vec![&a, &b], vec![40, 7], None);
        let mut mat: Vec<f64> = (0..3 * n).map(|_| rng.unif()).collect();
        let cols: Vec<Vec<f64>> = (0..3).map(|j| mat[j * n..(j + 1) * n].to_vec()).collect();
        let infos = demean_matrix(&p, &mut mat, n, 3, 1000, 0.0, 1e-13);
        assert_eq!(infos.len(), 3);
        for j in 0..3 {
            let mut ws = Workspace::new(&p, n);
            let mut c = cols[j].clone();
            demean_column(&p, &mut ws, &mut c, 1000, 0.0, 1e-13);
            assert_eq!(&mat[j * n..(j + 1) * n], c.as_slice());
        }
    }

    #[test]
    fn non_finite_input_is_reported_not_iterated() {
        let a: Vec<i64> = (0..300).map(|i| (i % 30) as i64).collect();
        let b: Vec<i64> = (0..300).map(|i| (i % 7) as i64).collect();
        let c: Vec<i64> = (0..300).map(|i| (i % 4) as i64).collect();
        let p = Problem::new(vec![&a, &b, &c], vec![30, 7, 4], None);
        let mut ws = Workspace::new(&p, 300);
        let mut x: Vec<f64> = (0..300).map(|i| i as f64).collect();
        x[17] = f64::NAN;
        let info = demean_column(&p, &mut ws, &mut x, 10_000, 0.0, 1e-12);
        assert!(!info.converged);
        assert_eq!(info.iters, 0);
        assert!(x.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn stalled_block_solve_respects_the_budget() {
        // A threshold of zero can never be met; the solve must stop.
        let mut rng = Lcg(21);
        let n = 2_000;
        let codes: Vec<Vec<i64>> = [150usize, 40, 6]
            .iter()
            .map(|&g| (0..n).map(|_| rng.int(g)).collect())
            .collect();
        let refs: Vec<&[i64]> = codes.iter().map(|v| v.as_slice()).collect();
        let p = Problem::new(refs, vec![150, 40, 6], None);
        let mut ws = Workspace::new(&p, n);
        let mut x: Vec<f64> = (0..n).map(|_| rng.unif()).collect();
        let info = demean_column(&p, &mut ws, &mut x, 50, 0.0, 0.0);
        assert!(!info.converged);
        assert!(info.iters <= 50);
        assert!(x.iter().all(|v| v.is_finite()));
    }
}
