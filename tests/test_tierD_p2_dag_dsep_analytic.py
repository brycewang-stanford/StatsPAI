"""Tier D P2 known-truth upgrades — d-separation & do-calculus on canonical DAGs.

Part of the P1/P2 "Tier D analytic special-cases" campaign (see
``.tierd_campaign/CAMPAIGN.md``). The Tier D probe surfaced a HIGH-severity
correctness bug: ``_d_separated`` moralised the ancestral graph by marrying
*siblings* (children of a common parent) instead of *co-parents* (parents of a
common child), so it was wrong on every fork and collider — see
``.tierd_campaign/BUG_d_separation_moralization.md``. Fixed per CLAUDE.md §12
(CHANGELOG + MIGRATION, ``⚠️ Correctness fix``).

These tests are the regression guard: the three canonical d-separation
structures (chain / fork / collider) plus the back-door adjustment set and
Rule 2 of Pearl's do-calculus.

Purely additive test file (the estimator fix is logged separately).
"""

import statspai as sp


# ---------------------------------------------------------------------------
# d-separation: chain / fork / collider canonical truths
# ---------------------------------------------------------------------------
class TestDSeparationAnalytic:

    def test_chain(self):
        # A -> B -> C: open unconditionally, blocked by the mediator B.
        g = sp.dag("A -> B; B -> C")
        assert g.d_separated("A", "C") is False
        assert g.d_separated("A", "C", {"B"}) is True

    def test_fork(self):
        # A <- M -> C: open unconditionally, blocked by the common cause M.
        g = sp.dag("M -> A; M -> C")
        assert g.d_separated("A", "C") is False
        assert g.d_separated("A", "C", {"M"}) is True

    def test_collider(self):
        # A -> K <- C: blocked unconditionally, OPENED by conditioning on the
        # collider K (this is the case the pre-fix moralisation got backwards).
        g = sp.dag("A -> K; C -> K")
        assert g.d_separated("A", "C") is True
        assert g.d_separated("A", "C", {"K"}) is False

    def test_collider_descendant_also_opens(self):
        # Conditioning on a descendant of a collider also opens the path.
        g = sp.dag("A -> K; C -> K; K -> D")
        assert g.d_separated("A", "C") is True
        assert g.d_separated("A", "C", {"D"}) is False


# ---------------------------------------------------------------------------
# back-door adjustment set + do-calculus Rule 2
# ---------------------------------------------------------------------------
class TestBackdoorAndDoCalculus:

    @staticmethod
    def _confounded():
        # W confounds X and Y; X -> Y is the effect of interest.
        return sp.dag("W -> X; W -> Y; X -> Y")

    def test_backdoor_adjustment_set_is_the_confounder(self):
        g = self._confounded()
        sets = g.adjustment_sets("X", "Y")
        assert {"W"} in sets
        assert g.backdoor_paths("X", "Y")  # the X <- W -> Y path exists

    def test_rule2_swap_requires_adjustment(self):
        # do(X) can be swapped for observing X iff the back-door path is
        # blocked: applicable given {W}, not applicable with no adjustment.
        g = self._confounded()
        assert sp.do_rule2(g, Y="Y", X=[], Z="X", W=["W"]).applicable is True
        assert sp.do_rule2(g, Y="Y", X=[], Z="X", W=[]).applicable is False

    def test_unconfounded_needs_no_adjustment(self):
        # With no common cause, the empty set already identifies X -> Y.
        g = sp.dag("X -> Y")
        assert g.d_separated("X", "Y") is False  # X causes Y
        assert set() in g.adjustment_sets("X", "Y")
