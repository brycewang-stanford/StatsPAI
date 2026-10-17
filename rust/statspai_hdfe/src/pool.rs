//! The thread pool every parallel kernel runs on.
//!
//! The crate never touches Rayon's global pool. Its own pool is built on
//! first use with ``STATSPAI_NUM_THREADS`` threads if that variable is set
//! (otherwise Rayon's default: ``RAYON_NUM_THREADS`` or one per logical
//! CPU), and [`set_num_threads`] replaces it. The pool remembers the
//! process id it was built in: a forked child, whose copy of the pool has
//! no worker threads, gets a fresh pool instead of hanging.

use std::sync::{Arc, RwLock};

struct State {
    pool: Option<Arc<rayon::ThreadPool>>,
    pid: u32,
    /// Requested size; 0 = default.
    requested: usize,
    env_read: bool,
}

static STATE: RwLock<State> = RwLock::new(State {
    pool: None,
    pid: 0,
    requested: 0,
    env_read: false,
});

fn build(n: usize) -> Result<rayon::ThreadPool, String> {
    let mut b = rayon::ThreadPoolBuilder::new().thread_name(|i| format!("statspai-hdfe-{i}"));
    if n > 0 {
        b = b.num_threads(n);
    }
    b.build().map_err(|e| e.to_string())
}

fn env_threads() -> usize {
    std::env::var("STATSPAI_NUM_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(0)
}

fn pool() -> Arc<rayon::ThreadPool> {
    let pid = std::process::id();
    {
        let st = STATE.read().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = &st.pool {
            if st.pid == pid {
                return Arc::clone(p);
            }
        }
    }
    let mut st = STATE.write().unwrap_or_else(|e| e.into_inner());
    if let Some(p) = &st.pool {
        if st.pid == pid {
            return Arc::clone(p);
        }
        // Inherited across fork: the worker threads do not exist here.
        // Dropping the pool would try to signal them, so leak it.
        if let Some(old) = st.pool.take() {
            std::mem::forget(old);
        }
    }
    if !st.env_read {
        st.requested = env_threads();
        st.env_read = true;
    }
    let p = Arc::new(build(st.requested).unwrap_or_else(|_| {
        build(1).expect("cannot build a one-thread pool")
    }));
    st.pool = Some(Arc::clone(&p));
    st.pid = pid;
    p
}

/// Run ``f`` on the crate's pool.
pub fn install<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    pool().install(f)
}

/// Rebuild the pool with ``n`` threads (0 = Rayon's default).
pub fn set_num_threads(n: usize) -> Result<(), String> {
    let p = build(n)?;
    let mut st = STATE.write().unwrap_or_else(|e| e.into_inner());
    if st.pid != std::process::id() {
        // A pool inherited across fork has no worker threads: leak it
        // rather than signal threads that do not exist.
        if let Some(old) = st.pool.take() {
            std::mem::forget(old);
        }
    }
    st.requested = n;
    st.env_read = true;
    st.pool = Some(Arc::new(p));
    st.pid = std::process::id();
    Ok(())
}

/// Number of threads the pool runs.
pub fn num_threads() -> usize {
    pool().current_num_threads()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_and_query() {
        set_num_threads(2).unwrap();
        assert_eq!(num_threads(), 2);
        let s: usize = install(|| {
            use rayon::prelude::*;
            (0..100usize).into_par_iter().sum()
        });
        assert_eq!(s, 4950);
        set_num_threads(1).unwrap();
        assert_eq!(num_threads(), 1);
    }
}
