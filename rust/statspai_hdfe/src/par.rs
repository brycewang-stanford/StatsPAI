//! Row-parallel loops whose results do not depend on the thread count.
//!
//! A sum over rows is split into a fixed number of contiguous chunks that
//! depends only on the problem size (never on how many threads run), each
//! chunk is reduced in row order, and the chunk results are added in chunk
//! order. The answer is therefore the same with one thread or sixteen; it
//! differs from a single left-to-right pass only in how the additions are
//! grouped (rounding at the level of 1e-16 relative).

use rayon::prelude::*;
use std::ops::Range;

/// Below this many rows every loop stays serial: spawning costs more than
/// it saves.
pub const PAR_MIN_ROWS: usize = 1 << 15;

/// Rows per chunk of an elementwise loop or of a reduction into a few
/// scalars.
pub const CHUNK_ROWS: usize = 1 << 14;

/// Most chunks a scatter-add into per-group sums is split into.
const MAX_SCATTER_CHUNKS: usize = 64;

/// Chunk boundaries for ``n`` rows: ``k`` contiguous ranges of nearly equal
/// length.
fn ranges(n: usize, k: usize) -> Vec<Range<usize>> {
    let k = k.max(1).min(n.max(1));
    let base = n / k;
    let extra = n % k;
    let mut out = Vec::with_capacity(k);
    let mut start = 0;
    for c in 0..k {
        let len = base + usize::from(c < extra);
        out.push(start..start + len);
        start += len;
    }
    out
}

/// How many chunks a scatter-add over ``n`` rows into ``groups`` sums uses:
/// enough rows per chunk that merging the per-chunk sums (``groups`` adds
/// each) costs a small fraction of the scatter itself.
pub fn scatter_chunks(n: usize, groups: usize) -> usize {
    if n < PAR_MIN_ROWS {
        return 1;
    }
    (n / (8 * groups.max(1024))).clamp(1, MAX_SCATTER_CHUNKS)
}

/// ``out[g] = sum over rows of the values ``body`` adds for group ``g``,
/// over rows ``0..n`` in fixed chunks. ``body(range, acc)`` adds the
/// contribution of the rows in ``range`` to ``acc`` (length ``out.len()``,
/// zeroed); chunk results are added to ``out`` in chunk order.
pub fn scatter_add<F>(n: usize, out: &mut [f64], body: F)
where
    F: Fn(Range<usize>, &mut [f64]) + Sync,
{
    let k = scatter_chunks(n, out.len());
    for v in out.iter_mut() {
        *v = 0.0;
    }
    if k == 1 {
        body(0..n, out);
        return;
    }
    let len = out.len();
    let parts: Vec<Vec<f64>> = ranges(n, k)
        .into_par_iter()
        .map(|r| {
            let mut acc = vec![0.0f64; len];
            body(r, &mut acc);
            acc
        })
        .collect();
    for part in &parts {
        for (o, v) in out.iter_mut().zip(part) {
            *o += v;
        }
    }
}

/// Apply ``f(i, &mut x[i])`` to every element, in parallel over chunks
/// when the slice is long.
pub fn for_each_mut<F>(x: &mut [f64], f: F)
where
    F: Fn(usize, &mut f64) + Sync,
{
    if x.len() < PAR_MIN_ROWS {
        for (i, v) in x.iter_mut().enumerate() {
            f(i, v);
        }
        return;
    }
    x.par_chunks_mut(CHUNK_ROWS)
        .enumerate()
        .for_each(|(c, chunk)| {
            let off = c * CHUNK_ROWS;
            for (j, v) in chunk.iter_mut().enumerate() {
                f(off + j, v);
            }
        });
}

/// Sum of ``f(range)`` over fixed chunks of ``0..n`` (chunk results added
/// in order). ``f`` returns a vector of ``m`` partial sums.
pub fn sum_vec<F>(n: usize, m: usize, f: F) -> Vec<f64>
where
    F: Fn(Range<usize>) -> Vec<f64> + Sync,
{
    if n < PAR_MIN_ROWS {
        return f(0..n);
    }
    let k = n.div_ceil(CHUNK_ROWS);
    let parts: Vec<Vec<f64>> = ranges(n, k).into_par_iter().map(&f).collect();
    let mut out = vec![0.0f64; m];
    for part in &parts {
        for (o, v) in out.iter_mut().zip(part) {
            *o += v;
        }
    }
    out
}

/// Scalar version of [`sum_vec`].
pub fn sum<F>(n: usize, f: F) -> f64
where
    F: Fn(Range<usize>) -> f64 + Sync,
{
    sum_vec(n, 1, |r| vec![f(r)])[0]
}

/// Largest ``f(range)`` over fixed chunks (order-independent).
pub fn max<F>(n: usize, f: F) -> f64
where
    F: Fn(Range<usize>) -> f64 + Sync,
{
    if n < PAR_MIN_ROWS {
        return f(0..n);
    }
    let k = n.div_ceil(CHUNK_ROWS);
    ranges(n, k)
        .into_par_iter()
        .map(&f)
        .reduce(|| f64::NEG_INFINITY, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_cover_and_partition() {
        for (n, k) in [(10, 3), (7, 7), (5, 9), (0, 4), (100_000, 64)] {
            let r = ranges(n, k);
            assert_eq!(r.first().map(|x| x.start).unwrap_or(0), 0);
            assert_eq!(r.last().map(|x| x.end).unwrap_or(0), n);
            for w in r.windows(2) {
                assert_eq!(w[0].end, w[1].start);
            }
        }
    }

    #[test]
    fn results_do_not_depend_on_thread_count() {
        let n = 300_000;
        let x: Vec<f64> = (0..n)
            .map(|i| ((i * 7919) % 1000) as f64 * 1e-3 + 1e-9 * i as f64)
            .collect();
        let codes: Vec<usize> = (0..n).map(|i| (i * 31) % 97).collect();
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut g = vec![0.0; 97];
                scatter_add(n, &mut g, |r, acc| {
                    for i in r {
                        acc[codes[i]] += x[i];
                    }
                });
                let s = sum(n, |r| r.map(|i| x[i] * x[i]).sum());
                (g, s)
            })
        };
        let (g1, s1) = run(1);
        let (g8, s8) = run(8);
        assert_eq!(g1, g8);
        assert_eq!(s1.to_bits(), s8.to_bits());
        let serial: f64 = x.iter().map(|v| v * v).sum();
        assert!((s1 - serial).abs() <= 1e-12 * serial);
    }
}
