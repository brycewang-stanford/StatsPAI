"""Reference parity: causal-discovery family on known linear SEMs.

Estimators
----------
``sp.pc_algorithm`` (constraint-based CPDAG learning), ``sp.lingam``
(DirectLiNGAM, linear non-Gaussian acyclic model) and ``sp.notears``
(continuous-optimisation DAG learning).  Each previously had only a smoke
test; this file is their first **structure-recovery** guarantee.

Why structure metrics, not scalar tolerances
---------------------------------------------
These estimators return graphs, not a single ATE, so the right ground
truth is the *known DAG* of a hand-built linear SEM.  We assert
precision/recall of the recovered edge sets, orientation correctness, and
seed-stability — facts that a finiteness/`is not None` check could never
catch, and that a biased graph would violate.

DGPs (every draw seeded via ``np.random.default_rng``)
------------------------------------------------------
- **Chain** ``X1 -> X2 -> X3 -> X4`` with strong coefficient 1.5 and low
  noise.  True skeleton (undirected) = ``{X1-X2, X2-X3, X3-X4}``; true
  directed edges = ``{X1->X2, X2->X3, X3->X4}``.  Crucially the marginal
  correlation ``|corr(X1,X4)|`` is large (~0.92) yet ``X1 ⟂ X4 | X3``, so
  a correlation-threshold method would wrongly link X1-X4 while PC must
  not.  A Gaussian-noise variant is used for PC/skeleton facts; a
  **non-Gaussian** variant (cubed-uniform disturbances) is used for
  LiNGAM, whose identifiability *requires* non-Gaussianity.
- **Pure collider** ``X0 -> X2 <- X1`` with ``X0 ⟂ X1``.  The
  v-structure direction is identifiable from observational data, so PC
  must orient *both* arrows into X2 (Spirtes-Glymour-Scheines v-structure
  rule).
- **Fork+collider** ``X0 -> X1``, ``X0 -> X2``, ``X1 -> X3``,
  ``X2 -> X3``.  Used for NOTEARS skeleton recovery (NOTEARS is known to
  be unreliable for *edge orientation* on standardised Gaussian data —
  the varsortability issue — but recovers the undirected skeleton
  perfectly and returns a genuine DAG, h(W)=0).

Anchors
-------
A. **PC skeleton precision = recall = 1** (structure recovery).  On the
   Gaussian chain the recovered undirected skeleton equals the true
   skeleton exactly across multiple seeds (probed: exact on 8/8 seeds).
   Pins the learned edge SET to truth, not its cardinality.
B. **PC naive-correlation contrast** (naive_bias).  ``|corr(X1,X4)|``
   ~0.92 — a naive correlation-threshold edge detector would declare an
   X1-X4 edge — yet PC, by conditioning on X3, leaves X1-X4 ABSENT from
   the skeleton.  We assert BOTH the naive marginal dependence is strong
   AND PC drops the edge (de-confounding of a spurious marginal link).
C. **PC v-structure orientation** (orientation).  On the pure collider
   PC orients exactly ``X0->X2`` and ``X1->X2`` (both into the collider)
   and leaves no spurious X0-X1 edge.
D. **LiNGAM directed-edge recovery precision = recall = 1** (structure
   recovery).  On the non-Gaussian chain LiNGAM recovers the causal order
   ``[X1,X2,X3,X4]`` and the directed edge set ``{X1->X2,X2->X3,X3->X4}``
   exactly; the recovered coefficients match the true 1.5s within a tight
   MC band (probed B means 1.50 / 1.50 / 1.56, SD ~0.01-0.02).
E. **NOTEARS skeleton recovery + valid DAG** (structure recovery).  On
   the fork+collider NOTEARS recovers the undirected skeleton exactly
   (precision = recall = 1) and returns h(W) = 0 (a genuine acyclic
   graph), across multiple seeds.
F. **Seed stability** (consistency).  PC skeleton, LiNGAM order, and
   NOTEARS skeleton are identical across >=3 independent seeds — the
   recovered structure is a property of the DGP, not of the RNG.

Implementation facts the anchors rely on (cited file:line)
----------------------------------------------------------
- ``pc.py:258`` builds ``skeleton`` as a symmetric 0/1 DataFrame; the
  CPDAG (``pc.py:259``) has ``cpdag[i,j]=1`` meaning ``i -> j`` and both
  directions set for an undirected edge.  ``edges`` (``pc.py:247``) are
  directed ``(parent, child)`` tuples, ``undirected_edges``
  (``pc.py:249``) are ``(node1, node2)`` pairs.
- ``pc.py:366-386`` orients v-structures ``i -> j <- k`` when ``i,k`` are
  non-adjacent and ``j`` is not in their separating set — the fact
  anchor C relies on.
- ``lingam.py:122-130`` ``LiNGAMResult.edges(threshold)`` returns
  ``(src, dst, weight)`` with ``src -> dst`` for every ``|B[dst,src]|``
  above the threshold; ``adjacency[i,j]`` is the direct effect of ``j``
  on ``i`` (``lingam.py:115``).  ``order`` (``lingam.py:114``) is the
  recovered causal order, most exogenous first.
- ``notears.py:201-211`` returns ``adjacency`` (W, ``W_ij`` means
  ``i -> j``), ``dag`` (binary), ``edges`` ``(parent, child, weight)``,
  and ``h_value`` (acyclicity constraint; ``~0`` for a valid DAG).

References
----------
- PC algorithm: Spirtes, Glymour & Scheines (2000), *Causation,
  Prediction, and Search* (2nd ed.), MIT Press. [@spirtes2000causation]
- NOTEARS: Zheng, Aragam, Ravikumar & Xing (2018), "DAGs with NO TEARS:
  Continuous Optimization for Structure Learning", NeurIPS 31.
  [@zheng2018dags]
- DirectLiNGAM: Shimizu, Inazumi, Sogawa, Hyvärinen, Kawahara, Washio,
  Hoyer & Bollen, "DirectLiNGAM: A Direct Method for Learning a Linear
  Non-Gaussian Structural Equation Model", JMLR 12 (2011), 1225-1248.
  (No bib key in paper.bib — method named, not cited via a fabricated
  key; see CLAUDE.md §10.)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# ---------------------------------------------------------------------------
# Deterministic DGP builders.  True graphs are stated as frozensets of
# frozensets (skeleton) / sets of ordered tuples (directed edges) right
# next to each builder, so the ground truth is hand-set, not read back
# from any estimator.
# ---------------------------------------------------------------------------

# Chain X1 -> X2 -> X3 -> X4.  Hand-set structural coefficient.
CHAIN_COEF = 1.5
CHAIN_VARS = ["X1", "X2", "X3", "X4"]
CHAIN_TRUE_SKELETON = {
    frozenset(["X1", "X2"]),
    frozenset(["X2", "X3"]),
    frozenset(["X3", "X4"]),
}
CHAIN_TRUE_DIRECTED = {("X1", "X2"), ("X2", "X3"), ("X3", "X4")}

# Pure collider X0 -> X2 <- X1 (X0 _||_ X1).
COLLIDER_VARS = ["X0", "X1", "X2"]
COLLIDER_TRUE_DIRECTED = {("X0", "X2"), ("X1", "X2")}

# Fork + collider:  X0 -> X1, X0 -> X2, X1 -> X3, X2 -> X3.
FORKCOL_VARS = ["X0", "X1", "X2", "X3"]
FORKCOL_TRUE_SKELETON = {
    frozenset(["X0", "X1"]),
    frozenset(["X0", "X2"]),
    frozenset(["X1", "X3"]),
    frozenset(["X2", "X3"]),
}


def _make_chain(seed, n=3000, noise=0.5, gaussian=True):
    """Linear chain X1->X2->X3->X4, coef = CHAIN_COEF.

    ``gaussian=True`` for PC/skeleton facts (orientation of a 2-node link
    is unidentifiable under Gaussian noise but the SKELETON is not).
    ``gaussian=False`` uses cubed-uniform (non-Gaussian, zero-mean)
    disturbances so DirectLiNGAM's identifiability assumption holds.
    """
    rng = np.random.default_rng(seed)

    def disturb():
        if gaussian:
            return rng.normal(scale=noise, size=n)
        return (rng.uniform(-1.0, 1.0, size=n) ** 3) * noise

    x1 = rng.normal(size=n) if gaussian else rng.uniform(-1.0, 1.0, size=n) ** 3
    x2 = CHAIN_COEF * x1 + disturb()
    x3 = CHAIN_COEF * x2 + disturb()
    x4 = CHAIN_COEF * x3 + disturb()
    return pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "X4": x4})


def _make_collider(seed, n=3000, noise=0.4):
    """Pure collider X0 -> X2 <- X1 with X0 _||_ X1.

    The v-structure direction IS identifiable from observational data
    (X0, X1 marginally independent but dependent given X2), so PC must
    orient both arrows into X2.
    """
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    x2 = 1.6 * x0 + 1.6 * x1 + rng.normal(scale=noise, size=n)
    return pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})


def _make_forkcol(seed, n=4000, noise=0.3):
    """Fork+collider: X0->X1, X0->X2, X1->X3, X2->X3."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = 2.0 * x0 + rng.normal(scale=noise, size=n)
    x2 = -1.8 * x0 + rng.normal(scale=noise, size=n)
    x3 = 1.5 * x1 + 1.5 * x2 + rng.normal(scale=noise, size=n)
    return pd.DataFrame({"X0": x0, "X1": x1, "X2": x2, "X3": x3})


# ---------------------------------------------------------------------------
# Structure-recovery helpers (graph -> comparable edge SETS).
# ---------------------------------------------------------------------------


def _pc_skeleton_set(result):
    """Undirected edge set of a PC result.

    Combines directed (``edges``) and undirected (``undirected_edges``)
    output into one set of frozenset pairs — the recovered SKELETON
    irrespective of orientation.
    """
    skel = {frozenset(pair) for pair in result["undirected_edges"]}
    skel |= {frozenset([p, c]) for p, c in result["edges"]}
    return skel


def _notears_skeleton_set(result):
    """Undirected edge set from NOTEARS thresholded edges."""
    return {frozenset([p, c]) for p, c, _w in result["edges"]}


def _lingam_directed_set(result, threshold):
    """Directed edge set {(src, dst)} above ``threshold``."""
    return {(src, dst) for src, dst, _w in result.edges(threshold=threshold)}


def _prec_recall(pred_set, true_set):
    if not pred_set:
        return 0.0, 0.0
    tp = len(pred_set & true_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    return precision, recall


def _pc(df, variables):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.pc_algorithm(df, variables=variables, alpha=0.05)


def _notears(df, variables):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # w_threshold 0.3 (default-ish) prunes float-noise edges; lambda1
        # 0.05 keeps the four strong structural edges. Not under test.
        return sp.notears(df, variables=variables, w_threshold=0.3, lambda1=0.05)


# ---------------------------------------------------------------------------
# A. PC skeleton precision = recall = 1.
# ---------------------------------------------------------------------------


class TestPCSkeletonRecovery:
    """PC recovers the exact undirected skeleton of the linear chain."""

    def test_skeleton_precision_recall_unity(self):
        df = _make_chain(20260614, gaussian=True)
        r = _pc(df, CHAIN_VARS)
        pred = _pc_skeleton_set(r)
        precision, recall = _prec_recall(pred, CHAIN_TRUE_SKELETON)
        # Exact set identity: the recovered skeleton must EQUAL the true
        # one, not merely overlap. A spurious or missing edge drops
        # precision or recall below 1.0. (A 20% structural bias that
        # perturbs an extra/fewer edge breaks this anchor.)
        assert precision == 1.0 and recall == 1.0, (
            f"PC skeleton precision={precision:.2f} recall={recall:.2f}; "
            f"recovered {sorted(tuple(sorted(s)) for s in pred)} vs truth "
            f"{sorted(tuple(sorted(s)) for s in CHAIN_TRUE_SKELETON)}"
        )

    def test_skeleton_symmetric_binary(self):
        """skeleton DataFrame is a symmetric 0/1 adjacency (pc.py:258).

        Guards the output contract the precision/recall anchor reads
        from: a symmetric matrix with a zero diagonal and only 0/1
        entries.  Not a finiteness check — it pins the exact reciprocal
        structure of an undirected adjacency.
        """
        df = _make_chain(7, gaussian=True)
        r = _pc(df, CHAIN_VARS)
        sk = r["skeleton"].values
        assert np.array_equal(sk, sk.T), "skeleton not symmetric"
        assert np.array_equal(np.diag(sk), np.zeros(len(CHAIN_VARS)))
        assert set(np.unique(sk)).issubset({0, 1})


# ---------------------------------------------------------------------------
# B. PC naive-correlation contrast (spurious marginal link is dropped).
# ---------------------------------------------------------------------------


class TestPCNaiveCorrelationContrast:
    """A strong marginal corr(X1,X4) must NOT survive as a PC edge."""

    def test_marginal_dependence_dropped_by_conditioning(self):
        df = _make_chain(101, gaussian=True)
        # Naive correlation-threshold detector: |corr(X1,X4)| is large in
        # a chain (each link multiplies through), so a marginal method
        # would wrongly link X1-X4. Hand-rolled numpy, no statspai.
        naive_corr = abs(np.corrcoef(df["X1"].values, df["X4"].values)[0, 1])
        # The contrast is only meaningful if the spurious marginal link
        # really is strong (probed ~0.92).
        assert naive_corr > 0.6, (
            f"|corr(X1,X4)|={naive_corr:.3f} not strong enough for the "
            f"naive-edge contrast — the chain lost its through-correlation."
        )

        r = _pc(df, CHAIN_VARS)
        skel = _pc_skeleton_set(r)
        # PC conditions on X3 (the mediator) and removes X1-X4: the edge
        # is ABSENT despite the strong marginal dependence.  Both halves
        # asserted -> non-tautological de-confounding of a spurious link.
        assert frozenset(["X1", "X4"]) not in skel, (
            f"PC kept the spurious X1-X4 edge despite "
            f"X1 ⟂ X4 | X3; skeleton={sorted(tuple(sorted(s)) for s in skel)}"
        )
        # And the genuine adjacent links it should keep are present.
        assert frozenset(["X1", "X2"]) in skel
        assert frozenset(["X3", "X4"]) in skel


# ---------------------------------------------------------------------------
# C. PC v-structure orientation.
# ---------------------------------------------------------------------------


class TestPCVStructureOrientation:
    """PC orients the collider X0 -> X2 <- X1 (both arrows into X2)."""

    def test_collider_oriented_into_X2(self):
        df = _make_collider(2024)
        r = _pc(df, COLLIDER_VARS)
        directed = set(r["edges"])
        # Both true arrows must be present and directed INTO the collider;
        # no spurious X0-X1 edge (the parents are marginally independent).
        assert directed == COLLIDER_TRUE_DIRECTED, (
            f"PC v-structure mis-oriented: directed={sorted(directed)} vs "
            f"truth {sorted(COLLIDER_TRUE_DIRECTED)}; "
            f"undirected={r['undirected_edges']}"
        )
        # X0 and X1 are non-adjacent (collider parents): no skeleton edge.
        skel = _pc_skeleton_set(r)
        assert frozenset(["X0", "X1"]) not in skel, (
            "PC added a spurious edge between the independent collider "
            "parents X0, X1."
        )


# ---------------------------------------------------------------------------
# D. LiNGAM directed-edge recovery (precision = recall = 1) + coefficients.
# ---------------------------------------------------------------------------


class TestLiNGAMDirectedRecovery:
    """DirectLiNGAM recovers the exact directed chain (non-Gaussian)."""

    # |B| > 0.5 cleanly separates the true ~1.5 effects from float noise
    # near zero; the smallest true coefficient is CHAIN_COEF = 1.5.
    EDGE_THRESHOLD = 0.5

    def test_directed_edges_precision_recall_unity(self):
        df = _make_chain(55, gaussian=False)
        res = sp.lingam(df)
        pred = _lingam_directed_set(res, self.EDGE_THRESHOLD)
        precision, recall = _prec_recall(pred, CHAIN_TRUE_DIRECTED)
        assert precision == 1.0 and recall == 1.0, (
            f"LiNGAM directed precision={precision:.2f} recall={recall:.2f}; "
            f"recovered {sorted(pred)} vs truth {sorted(CHAIN_TRUE_DIRECTED)}"
        )

    def test_causal_order_recovered(self):
        df = _make_chain(56, gaussian=False)
        res = sp.lingam(df)
        order_names = [res.names[i] for i in res.order]
        # The unique causal order of a chain is X1, X2, X3, X4 (most
        # exogenous first). A flipped order would mis-orient every edge.
        assert order_names == CHAIN_VARS, (
            f"LiNGAM causal order {order_names} != true chain order " f"{CHAIN_VARS}"
        )

    def test_coefficients_recover_truth(self):
        """40-rep MC of each B coefficient vs the hand-set CHAIN_COEF.

        ``adjacency[i,j]`` is the direct effect of j on i, so the three
        chain coefficients are B[X2,X1], B[X3,X2], B[X4,X3], all = 1.5 by
        construction. Averaging cancels per-draw noise (probed per-coef
        SD ~0.01-0.03); the MC mean must sit within 4*SD/sqrt(R) of 1.5.
        A 20% bias (-> 1.8) lands many band-widths out.
        """
        reps = 40
        names = ["X2", "X3", "X4"]
        parents = ["X1", "X2", "X3"]
        cols = {(c, p): [] for c, p in zip(names, parents)}
        for s in range(reps):
            res = sp.lingam(_make_chain(3000 + s, n=1500, gaussian=False))
            B = res.to_frame()
            for c, p in zip(names, parents):
                cols[(c, p)].append(B.loc[c, p])
        for (c, p), vals in cols.items():
            arr = np.asarray(vals)
            mc_mean = float(arr.mean())
            mc_sd = float(arr.std(ddof=1))
            band = 4.0 * mc_sd / np.sqrt(reps)
            assert abs(mc_mean - CHAIN_COEF) <= band, (
                f"B[{c},{p}] MC mean {mc_mean:.4f} drifted from truth "
                f"{CHAIN_COEF} (band {band:.4f}, SD {mc_sd:.4f}) — "
                f"systematic LiNGAM coefficient bias."
            )


# ---------------------------------------------------------------------------
# E. NOTEARS skeleton recovery + valid acyclic DAG.
# ---------------------------------------------------------------------------


class TestNOTEARSSkeletonRecovery:
    """NOTEARS recovers the fork+collider skeleton and returns a DAG.

    NOTEARS on standardised Gaussian data is unreliable for *edge
    orientation* (varsortability, Reisach et al. 2021), so we anchor the
    robust facts it DOES deliver: the exact undirected skeleton and a
    genuinely acyclic solution (h(W) = 0).
    """

    def test_skeleton_precision_recall_unity(self):
        df = _make_forkcol(31)
        r = _notears(df, FORKCOL_VARS)
        pred = _notears_skeleton_set(r)
        precision, recall = _prec_recall(pred, FORKCOL_TRUE_SKELETON)
        assert precision == 1.0 and recall == 1.0, (
            f"NOTEARS skeleton precision={precision:.2f} recall={recall:.2f};"
            f" recovered {sorted(tuple(sorted(s)) for s in pred)} vs truth "
            f"{sorted(tuple(sorted(s)) for s in FORKCOL_TRUE_SKELETON)}"
        )

    def test_solution_is_acyclic(self):
        """h(W) = tr(e^{W∘W}) - d must be ~0 (a real DAG, no cycles).

        Tolerance 1e-6: the augmented-Lagrangian targets h_tol=1e-8 and
        thresholds tiny weights to 0, so the reported h on the pruned W
        is exactly 0 in practice (probed 0.0). 1e-6 leaves headroom while
        still rejecting any non-trivial cycle (a 2-cycle a<->b with
        weights ~0.5 gives h ~ 0.25, far above 1e-6).
        """
        df = _make_forkcol(31)
        r = _notears(df, FORKCOL_VARS)
        assert (
            abs(r["h_value"]) < 1e-6
        ), f"NOTEARS returned a non-acyclic W: h={r['h_value']:.3e}"
        # The binary dag adjacency must encode exactly the recovered edges
        # (output-contract guard, notears.py:210-211).
        assert int(r["dag"].values.sum()) == r["n_edges"]


# ---------------------------------------------------------------------------
# F. Seed stability — recovered structure is a property of the DGP.
# ---------------------------------------------------------------------------


class TestSeedStability:
    """PC skeleton, LiNGAM order, NOTEARS skeleton stable across seeds."""

    SEEDS = [0, 1, 2]

    def test_pc_skeleton_stable(self):
        skels = [
            _pc_skeleton_set(_pc(_make_chain(s, gaussian=True), CHAIN_VARS))
            for s in self.SEEDS
        ]
        # Every seed yields the SAME skeleton == the true one.
        for s, sk in zip(self.SEEDS, skels):
            assert sk == CHAIN_TRUE_SKELETON, (
                f"PC skeleton drifted on seed {s}: "
                f"{sorted(tuple(sorted(e)) for e in sk)}"
            )
        assert all(sk == skels[0] for sk in skels)

    def test_lingam_order_stable(self):
        orders = []
        for s in self.SEEDS:
            res = sp.lingam(_make_chain(s, gaussian=False))
            orders.append([res.names[i] for i in res.order])
        for s, o in zip(self.SEEDS, orders):
            assert o == CHAIN_VARS, f"LiNGAM order drifted on seed {s}: {o}"
        assert all(o == orders[0] for o in orders)

    def test_notears_skeleton_stable(self):
        skels = [
            _notears_skeleton_set(_notears(_make_forkcol(s), FORKCOL_VARS))
            for s in self.SEEDS
        ]
        for s, sk in zip(self.SEEDS, skels):
            assert sk == FORKCOL_TRUE_SKELETON, (
                f"NOTEARS skeleton drifted on seed {s}: "
                f"{sorted(tuple(sorted(e)) for e in sk)}"
            )
        assert all(sk == skels[0] for sk in skels)
