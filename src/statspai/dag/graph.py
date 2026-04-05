"""
Causal DAG declaration and analysis.

Provides the :class:`DAG` class for declaring causal structures and
computing identification-relevant quantities:

- **Backdoor / frontdoor adjustment sets** (Pearl's criteria)
- **Path enumeration** — all paths, backdoor paths, causal paths
- **Path classification** — open / closed given a conditioning set
- **Variable role detection** — confounder, mediator, collider, bad control
- **Collider detection** and **d-separation** queries
- **Interventional ``do()`` graphs**
- **Ancestors / descendants / paths** traversal
- **Classic textbook DAGs** (Cunningham *Mixtape* ch. 3)
- **ASCII and matplotlib visualisation**

Modelled after R's ``dagitty`` — no external graph library required.

References
----------
Pearl, J. (2009). *Causality*. Cambridge University Press.
Greenland, S., Pearl, J. and Robins, J.M. (1999).
"Causal Diagrams for Epidemiologic Research." *Epidemiology*, 10(1), 37-48.
Cunningham, S. (2021). *Causal Inference: The Mixtape*. Yale University Press.
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union


class DAG:
    """
    A directed acyclic graph for causal reasoning.

    Parameters
    ----------
    spec : str
        Edge specification.  Supported formats:

        - ``"X -> Y; Z -> X; Z -> Y"``  (semicolon-separated)
        - ``"X -> Y\\n Z -> X\\n Z -> Y"``  (newline-separated)
        - ``"X -> Y, Z -> X, Z -> Y"``  (comma-separated)
        - Bidirected (latent common cause): ``"X <-> Y"`` adds a latent
          node ``_L_X_Y`` with edges to both X and Y.

    Examples
    --------
    >>> g = DAG('Z -> X; Z -> Y; X -> Y')
    >>> g.adjustment_sets('X', 'Y')
    [{'Z'}]

    >>> g = DAG('X -> M -> Y; X -> Y; U <-> Y')
    >>> g.nodes
    {'X', 'M', 'Y', '_L_U_Y'}
    """

    def __init__(self, spec: str = ""):
        self._edges: Dict[str, Set[str]] = {}  # parent -> set of children
        self._nodes: Set[str] = set()

        if spec:
            self._parse(spec)

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    def add_node(self, name: str) -> "DAG":
        self._nodes.add(name)
        return self

    def add_edge(self, parent: str, child: str) -> "DAG":
        self._nodes.update([parent, child])
        self._edges.setdefault(parent, set()).add(child)
        return self

    def add_bidirected(self, a: str, b: str) -> "DAG":
        """Add a latent common cause of *a* and *b*."""
        latent = f"_L_{a}_{b}"
        self.add_edge(latent, a)
        self.add_edge(latent, b)
        return self

    @property
    def nodes(self) -> Set[str]:
        return set(self._nodes)

    @property
    def observed_nodes(self) -> Set[str]:
        """Nodes that are not latent (latents start with ``_L_``)."""
        return {n for n in self._nodes if not n.startswith("_L_")}

    @property
    def edges(self) -> List[Tuple[str, str]]:
        return [(p, c) for p, children in self._edges.items() for c in children]

    # ------------------------------------------------------------------ #
    #  Graph queries
    # ------------------------------------------------------------------ #

    def parents(self, node: str) -> Set[str]:
        return {p for p, children in self._edges.items() if node in children}

    def children(self, node: str) -> Set[str]:
        return set(self._edges.get(node, set()))

    def ancestors(self, node: str) -> Set[str]:
        """All ancestors of *node* (not including *node* itself)."""
        visited: Set[str] = set()
        stack = list(self.parents(node))
        while stack:
            n = stack.pop()
            if n not in visited:
                visited.add(n)
                stack.extend(self.parents(n))
        return visited

    def descendants(self, node: str) -> Set[str]:
        """All descendants of *node*."""
        visited: Set[str] = set()
        stack = list(self.children(node))
        while stack:
            n = stack.pop()
            if n not in visited:
                visited.add(n)
                stack.extend(self.children(n))
        return visited

    def is_ancestor(self, node: str, of: str) -> bool:
        return node in self.ancestors(of)

    def is_collider(self, node: str, path: List[str]) -> bool:
        """
        Check if *node* is a collider on *path*.

        A node is a collider on a path if both its neighbours on the
        path are parents of it (arrows point into it: ``→ node ←``).
        """
        if node not in path:
            return False
        idx = path.index(node)
        if idx == 0 or idx == len(path) - 1:
            return False
        prev_node = path[idx - 1]
        next_node = path[idx + 1]
        return node in self.children(prev_node) and node in self.children(next_node)

    # ------------------------------------------------------------------ #
    #  Path enumeration
    # ------------------------------------------------------------------ #

    def all_paths(
        self, x: str, y: str, *, directed_only: bool = False,
    ) -> List[List[str]]:
        """
        Enumerate all simple paths between *x* and *y*.

        Parameters
        ----------
        x, y : str
            Start and end nodes.
        directed_only : bool
            If True, only follow directed edges parent→child.
            If False (default), traverse edges in either direction
            (needed for finding backdoor paths).

        Returns
        -------
        list of list of str
            Each inner list is an ordered path from *x* to *y*.
        """
        results: List[List[str]] = []
        self._find_paths(x, y, [x], set(), results, directed_only)
        return results

    def _find_paths(
        self,
        current: str,
        target: str,
        path: List[str],
        visited: Set[str],
        results: List[List[str]],
        directed_only: bool,
    ) -> None:
        if current == target and len(path) > 1:
            results.append(list(path))
            return
        visited = visited | {current}
        neighbours: Set[str]
        if directed_only:
            neighbours = self.children(current)
        else:
            neighbours = self.children(current) | self.parents(current)
        for nb in neighbours:
            if nb not in visited:
                path.append(nb)
                self._find_paths(nb, target, path, visited, results, directed_only)
                path.pop()

    def causal_paths(self, exposure: str, outcome: str) -> List[List[str]]:
        """
        All directed (causal) paths from *exposure* to *outcome*.

        These are the paths through which the treatment *actually causes*
        changes in the outcome.
        """
        return self.all_paths(exposure, outcome, directed_only=True)

    def backdoor_paths(self, exposure: str, outcome: str) -> List[List[str]]:
        """
        All backdoor (non-causal) paths from *exposure* to *outcome*.

        A backdoor path is any path that starts with an arrow *into*
        the exposure (← exposure), creating spurious association.
        """
        all_p = self.all_paths(exposure, outcome)
        causal_p = set(tuple(p) for p in self.causal_paths(exposure, outcome))
        return [p for p in all_p if tuple(p) not in causal_p]

    def is_path_open(
        self, path: List[str], conditioned: Optional[Set[str]] = None,
    ) -> bool:
        """
        Check if a path is open (active) given a conditioning set.

        Rules (Pearl 2009):
        - A non-collider on the path blocks if conditioned on.
        - A collider on the path blocks *unless* it or a descendant is
          conditioned on.
        """
        conditioned = conditioned or set()
        cond_ancestors: Set[str] = set()
        for c in conditioned:
            cond_ancestors |= self.ancestors(c)
            cond_ancestors.add(c)

        for i in range(1, len(path) - 1):
            node = path[i]
            prev_node = path[i - 1]
            next_node = path[i + 1]
            is_coll = (
                node in self.children(prev_node) and node in self.children(next_node)
            )
            if is_coll:
                # Collider: path blocked unless node or descendant is conditioned
                if node not in cond_ancestors:
                    return False
            else:
                # Non-collider: path blocked if conditioned on
                if node in conditioned:
                    return False
        return True

    def path_status(
        self,
        exposure: str,
        outcome: str,
        conditioned: Optional[Set[str]] = None,
    ) -> List[dict]:
        """
        Classify every path between *exposure* and *outcome*.

        Returns
        -------
        list of dict
            Each dict has keys ``'path'``, ``'type'`` (``'causal'`` or
            ``'backdoor'``), and ``'open'`` (bool given conditioning set).

        Examples
        --------
        >>> g = sp.dag('Z -> X; Z -> Y; X -> Y')
        >>> g.path_status('X', 'Y')
        [{'path': ['X', 'Y'], 'type': 'causal', 'open': True},
         {'path': ['X', 'Z', 'Y'], 'type': 'backdoor', 'open': True}]
        >>> g.path_status('X', 'Y', conditioned={'Z'})
        [{'path': ['X', 'Y'], 'type': 'causal', 'open': True},
         {'path': ['X', 'Z', 'Y'], 'type': 'backdoor', 'open': False}]
        """
        conditioned = conditioned or set()
        causal_set = set(tuple(p) for p in self.causal_paths(exposure, outcome))
        all_p = self.all_paths(exposure, outcome)
        result = []
        for p in all_p:
            ptype = "causal" if tuple(p) in causal_set else "backdoor"
            result.append({
                "path": p,
                "type": ptype,
                "open": self.is_path_open(p, conditioned),
            })
        return result

    # ------------------------------------------------------------------ #
    #  Variable role classification
    # ------------------------------------------------------------------ #

    def classify_variable(
        self, node: str, exposure: str, outcome: str,
    ) -> Set[str]:
        """
        Classify the role(s) of *node* relative to *exposure* → *outcome*.

        Returns a set that may include:
        ``'confounder'``, ``'mediator'``, ``'collider'``,
        ``'instrument'``, ``'ancestor_of_treatment'``,
        ``'ancestor_of_outcome'``.

        Examples
        --------
        >>> g = sp.dag('Z -> X; Z -> Y; X -> Y')
        >>> g.classify_variable('Z', 'X', 'Y')
        {'confounder', 'ancestor_of_treatment', 'ancestor_of_outcome'}
        """
        roles: Set[str] = set()
        if node == exposure or node == outcome:
            return roles

        # Ancestor relationships
        if node in self.ancestors(exposure):
            roles.add("ancestor_of_treatment")
        if node in self.ancestors(outcome):
            roles.add("ancestor_of_outcome")

        # On a causal path? → mediator
        for p in self.causal_paths(exposure, outcome):
            if node in p and node != exposure and node != outcome:
                roles.add("mediator")

        # On a backdoor path as non-collider? → confounder
        for p in self.backdoor_paths(exposure, outcome):
            if node in p:
                idx = p.index(node)
                if 0 < idx < len(p) - 1:
                    prev_n, next_n = p[idx - 1], p[idx + 1]
                    is_coll = (
                        node in self.children(prev_n)
                        and node in self.children(next_n)
                    )
                    if is_coll:
                        roles.add("collider")
                    else:
                        roles.add("confounder")

        # Instrument: causes exposure, no direct/backdoor path to outcome
        # except through exposure
        if node in self.ancestors(exposure) and node not in self.ancestors(outcome):
            # Check: removing exposure, is node d-separated from outcome?
            modified = DAG()
            for n in self._nodes:
                modified.add_node(n)
            for parent, children in self._edges.items():
                for child in children:
                    if parent != exposure:
                        modified.add_edge(parent, child)
            if modified.d_separated(node, outcome):
                roles.add("instrument")

        return roles

    def bad_controls(self, exposure: str, outcome: str) -> dict:
        """
        Identify variables that should **not** be conditioned on.

        Returns a dict mapping variable names to the reason they are bad
        controls.  Based on Cinelli, Forney & Pearl (2022) and the "bad
        controls" discussion in Cunningham (2021, ch. 3).

        Categories of bad controls:

        - **descendant_of_treatment**: conditioning on a descendant of
          exposure blocks part of the causal effect (over-control bias).
        - **collider**: conditioning opens a previously closed backdoor
          path (collider bias / selection bias).
        - **mediator**: conditioning on a mediator blocks the indirect
          causal effect (over-control / mediation bias).
        - **M-bias**: conditioning on a pre-treatment variable that is
          a collider on a backdoor path, opening a non-causal path.

        Examples
        --------
        >>> g = sp.dag('D -> O -> Y; A -> O; A -> Y; D -> Y')
        >>> g.bad_controls('D', 'Y')
        {'O': ['collider — conditioning opens D→O←A→Y']}
        """
        warnings: dict = {}
        descendants_x = self.descendants(exposure)
        observed = self.observed_nodes - {exposure, outcome}

        for v in observed:
            reasons = []

            # 1. Descendant of treatment (not mediator)
            if v in descendants_x:
                on_causal = any(
                    v in p for p in self.causal_paths(exposure, outcome)
                )
                if on_causal:
                    reasons.append(
                        "mediator — conditioning blocks indirect causal effect"
                    )
                else:
                    reasons.append(
                        "descendant_of_treatment — biases the causal effect estimate"
                    )

            # 2. Collider on a backdoor path
            for p in self.backdoor_paths(exposure, outcome):
                if v in p:
                    idx = p.index(v)
                    if 0 < idx < len(p) - 1:
                        prev_n, next_n = p[idx - 1], p[idx + 1]
                        is_coll = (
                            v in self.children(prev_n)
                            and v in self.children(next_n)
                        )
                        if is_coll:
                            reasons.append(
                                f"collider — conditioning opens "
                                f"{'→'.join(p)}"
                            )
                            break

            if reasons:
                warnings[v] = reasons

        return warnings

    # ------------------------------------------------------------------ #
    #  Interventional graph (do-operator)
    # ------------------------------------------------------------------ #

    def do(self, intervention: Union[str, Set[str]]) -> "DAG":
        """
        Return the **interventional graph** G_{\\overline{X}}: the graph
        with all incoming edges to the intervention node(s) removed.

        This implements Pearl's *do*-operator at the graphical level.

        Parameters
        ----------
        intervention : str or set of str
            The node(s) being intervened on.

        Returns
        -------
        DAG
            A new DAG with incoming edges to intervention removed.

        Examples
        --------
        >>> g = sp.dag('Z -> X -> Y; Z -> Y')
        >>> g_do = g.do('X')
        >>> g_do.edges  # Z -> X edge is removed
        [('X', 'Y'), ('Z', 'Y')]
        """
        if isinstance(intervention, str):
            intervention = {intervention}

        modified = DAG()
        for n in self._nodes:
            modified.add_node(n)
        for parent, children in self._edges.items():
            for child in children:
                if child not in intervention:
                    modified.add_edge(parent, child)
        return modified

    # ------------------------------------------------------------------ #
    #  Frontdoor criterion
    # ------------------------------------------------------------------ #

    def frontdoor_sets(
        self, exposure: str, outcome: str,
    ) -> List[Set[str]]:
        """
        Find sets satisfying Pearl's **frontdoor criterion**.

        A set M satisfies the frontdoor criterion relative to (X, Y) if:

        1. M intercepts all directed paths from X to Y.
        2. There is no unblocked backdoor path from X to M.
        3. All backdoor paths from M to Y are blocked by X.

        Returns
        -------
        list of set
            Valid frontdoor adjustment sets (possibly empty).

        Examples
        --------
        >>> g = sp.dag('U <-> X; U <-> Y; X -> M -> Y')
        >>> g.frontdoor_sets('X', 'Y')
        [{'M'}]
        """
        # Candidates: observed, not exposure, not outcome
        candidates = self.observed_nodes - {exposure, outcome}
        valid: List[Set[str]] = []

        max_size = min(len(candidates), 6)
        candidate_list = sorted(candidates)

        for size in range(1, max_size + 1):
            for combo in combinations(candidate_list, size):
                m_set = set(combo)
                if self._is_valid_frontdoor(exposure, outcome, m_set):
                    valid.append(m_set)
            if valid:
                break  # minimal sets found

        return valid

    def _is_valid_frontdoor(
        self, exposure: str, outcome: str, m_set: Set[str],
    ) -> bool:
        """Check the three frontdoor conditions."""
        # 1. M intercepts all directed (causal) paths from X to Y
        for path in self.causal_paths(exposure, outcome):
            if not any(n in m_set for n in path[1:-1]):
                return False

        # 2. No unblocked backdoor path from X to any node in M
        for m in m_set:
            bd_paths = self.backdoor_paths(exposure, m)
            for p in bd_paths:
                if self.is_path_open(p):
                    return False

        # 3. All backdoor paths from M to Y are blocked by {X}
        for m in m_set:
            bd_paths = self.backdoor_paths(m, outcome)
            for p in bd_paths:
                if self.is_path_open(p, conditioned={exposure}):
                    return False

        return True

    # ------------------------------------------------------------------ #
    #  d-separation
    # ------------------------------------------------------------------ #

    def d_separated(
        self, x: str, y: str, conditioned: Optional[Set[str]] = None,
    ) -> bool:
        """
        Test if *x* and *y* are d-separated given *conditioned*.

        Uses the Bayes-Ball algorithm (Shachter 1998).
        """
        conditioned = conditioned or set()
        # Use reachability via active paths
        reachable = self._active_reachable(x, conditioned)
        return y not in reachable

    def _active_reachable(self, source: str, conditioned: Set[str]) -> Set[str]:
        """Nodes reachable from *source* via active paths given *conditioned*."""
        # Ancestors of conditioned set (needed for collider activation)
        cond_ancestors: Set[str] = set()
        for c in conditioned:
            cond_ancestors |= self.ancestors(c)
            cond_ancestors.add(c)

        visited: Set[Tuple[str, str]] = set()  # (node, direction)
        queue: List[Tuple[str, str]] = [(source, "up"), (source, "down")]
        reachable: Set[str] = set()

        while queue:
            node, direction = queue.pop()
            if (node, direction) in visited:
                continue
            visited.add((node, direction))

            if node != source:
                reachable.add(node)

            # Traverse based on direction
            if direction == "up" and node not in conditioned:
                # Going up through a non-conditioned node
                for parent in self.parents(node):
                    queue.append((parent, "up"))
                for child in self.children(node):
                    queue.append((child, "down"))

            elif direction == "down":
                if node not in conditioned:
                    # Non-collider, not conditioned — pass through
                    for child in self.children(node):
                        queue.append((child, "down"))
                if node in cond_ancestors:
                    # Collider (or descendant of collider) that is conditioned on
                    for parent in self.parents(node):
                        queue.append((parent, "up"))

        return reachable

    # ------------------------------------------------------------------ #
    #  Adjustment sets
    # ------------------------------------------------------------------ #

    def adjustment_sets(
        self,
        exposure: str,
        outcome: str,
        method: str = "backdoor",
        minimal: bool = True,
    ) -> List[Set[str]]:
        """
        Find valid adjustment sets for estimating the causal effect of
        *exposure* on *outcome*.

        Parameters
        ----------
        exposure, outcome : str
            Treatment and outcome nodes.
        method : str
            ``'backdoor'`` — Pearl's backdoor criterion (default).
        minimal : bool
            If True, return only minimal sufficient adjustment sets.

        Returns
        -------
        list of set
            Each set is a valid adjustment set (possibly empty).
        """
        if method != "backdoor":
            raise ValueError(f"Only 'backdoor' method is currently supported, got '{method}'")

        # Candidate nodes: observed, not exposure, not outcome, not descendants of exposure
        descendants_x = self.descendants(exposure)
        candidates = self.observed_nodes - {exposure, outcome} - descendants_x

        valid_sets: List[Set[str]] = []

        # Check all subsets (up to reasonable size)
        max_size = min(len(candidates), 6)  # cap for combinatorial explosion
        candidate_list = sorted(candidates)

        for size in range(0, max_size + 1):
            for combo in combinations(candidate_list, size):
                s = set(combo)
                if self._is_valid_adjustment(exposure, outcome, s):
                    valid_sets.append(s)
            if minimal and valid_sets:
                # Found valid sets at this size — return only these
                break

        return valid_sets

    def _is_valid_adjustment(
        self, exposure: str, outcome: str, adj_set: Set[str],
    ) -> bool:
        """Check if adj_set satisfies the backdoor criterion."""
        # Backdoor criterion:
        # 1. No node in adj_set is a descendant of exposure
        # 2. adj_set blocks all backdoor paths from exposure to outcome
        descendants_x = self.descendants(exposure)
        if adj_set & descendants_x:
            return False

        # Check d-separation of X and Y in the manipulated graph
        # (remove all edges out of X, then check d-sep given adj_set)
        return self._d_sep_manipulated(exposure, outcome, adj_set)

    def _d_sep_manipulated(
        self, exposure: str, outcome: str, conditioned: Set[str],
    ) -> bool:
        """d-separation in the graph with outgoing edges of exposure removed."""
        # Create a modified DAG without edges from exposure
        modified = DAG()
        for n in self._nodes:
            modified.add_node(n)
        for parent, children in self._edges.items():
            for child in children:
                if parent != exposure:
                    modified.add_edge(parent, child)

        return modified.d_separated(exposure, outcome, conditioned)

    # ------------------------------------------------------------------ #
    #  Visualization
    # ------------------------------------------------------------------ #

    def to_ascii(self) -> str:
        """Simple text representation of the DAG."""
        lines = ["DAG:"]
        seen_latent: Set[str] = set()
        for parent in sorted(self._edges.keys()):
            if parent.startswith("_L_"):
                if parent not in seen_latent:
                    seen_latent.add(parent)
                    other_children = sorted(self._edges[parent])
                    if len(other_children) == 2:
                        a, b = other_children
                        lines.append(f"  {a} <-> {b}  (latent: {parent})")
            else:
                for child in sorted(self._edges[parent]):
                    lines.append(f"  {parent} -> {child}")
        return "\n".join(lines)

    def plot(
        self,
        exposure: Optional[str] = None,
        outcome: Optional[str] = None,
        conditioned: Optional[Set[str]] = None,
        positions: Optional[Dict[str, Tuple[float, float]]] = None,
        figsize: tuple = (8, 6),
        seed: int = 42,
        title: Optional[str] = None,
        style: str = "ggdag",
        node_size: float = 0.22,
        font_size: int = 12,
        ax=None,
    ):
        """
        Plot the DAG with publication-quality styling.

        When *exposure* and *outcome* are provided, nodes are colour-coded
        by causal role (like R's ``ggdag``):

        - **Exposure**: green
        - **Outcome**: blue
        - **Confounder**: orange
        - **Mediator**: purple
        - **Collider / bad control**: red
        - **Unobserved (latent)**: grey dashed outline
        - **Adjusted / conditioned**: hatched fill

        Bidirected edges (latent common causes) are rendered as curved
        dashed arcs rather than routing through hidden nodes.

        Parameters
        ----------
        exposure, outcome : str, optional
            Treatment and outcome nodes for role colouring.
        conditioned : set of str, optional
            Nodes being conditioned on (shown with hatched fill).
        positions : dict, optional
            ``{node_name: (x, y)}`` for custom layout. If ``None``,
            an automatic topological layout is used.
        figsize : tuple
            Figure size (width, height) in inches.
        seed : int
            Random seed for layout jitter.
        title : str, optional
            Plot title. Auto-generated if exposure/outcome given.
        style : str
            ``'ggdag'`` (default, clean white) or ``'classic'``
            (grey background, smaller nodes).
        node_size : float
            Radius of node circles in data coordinates.
        font_size : int
            Font size for node labels.
        ax : matplotlib Axes, optional
            Axes to draw on. If ``None``, creates a new figure.

        Returns
        -------
        (fig, ax) : matplotlib figure and axes
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.patches import FancyArrowPatch
            import numpy as np
        except ImportError:
            raise ImportError(
                "matplotlib required for DAG plotting: pip install matplotlib"
            )

        conditioned = conditioned or set()

        # --- Positions ---
        if positions is None:
            pos = self._layout(seed)
        else:
            pos = dict(positions)
            # Ensure latent nodes get positions too
            for n in self._nodes:
                if n not in pos and n.startswith("_L_"):
                    children_of_l = sorted(self._edges.get(n, set()))
                    if len(children_of_l) == 2 and all(c in pos for c in children_of_l):
                        c0, c1 = children_of_l
                        mx = (pos[c0][0] + pos[c1][0]) / 2
                        my = (pos[c0][1] + pos[c1][1]) / 2 + 0.8
                        pos[n] = (mx, my)

        # --- Colour mapping ---
        role_colors = {
            "exposure": "#2ca02c",     # green
            "outcome": "#1f77b4",      # blue
            "confounder": "#ff7f0e",   # orange
            "mediator": "#9467bd",     # purple
            "collider": "#d62728",     # red
            "instrument": "#8c564b",   # brown
            "default": "#7f7f7f",      # grey
        }

        node_colors: Dict[str, str] = {}
        node_roles_map: Dict[str, Set[str]] = {}
        if exposure and outcome:
            node_colors[exposure] = role_colors["exposure"]
            node_colors[outcome] = role_colors["outcome"]
            for v in self.observed_nodes - {exposure, outcome}:
                roles = self.classify_variable(v, exposure, outcome)
                node_roles_map[v] = roles
                if "collider" in roles and "mediator" not in roles:
                    node_colors[v] = role_colors["collider"]
                elif "mediator" in roles:
                    node_colors[v] = role_colors["mediator"]
                elif "confounder" in roles:
                    node_colors[v] = role_colors["confounder"]
                elif "instrument" in roles:
                    node_colors[v] = role_colors["instrument"]
                else:
                    node_colors[v] = role_colors["default"]
        else:
            for v in self.observed_nodes:
                node_colors[v] = "#555555"

        # --- Figure ---
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        ax.set_aspect("equal")

        if style == "ggdag":
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
        else:
            fig.patch.set_facecolor("#f5f5f5")
            ax.set_facecolor("#f5f5f5")

        # --- Collect bidirected pairs ---
        bidirected_pairs: List[Tuple[str, str]] = []
        latent_nodes_to_skip: Set[str] = set()
        for n in self._nodes:
            if n.startswith("_L_"):
                children_of_l = sorted(self._edges.get(n, set()))
                if len(children_of_l) == 2:
                    bidirected_pairs.append((children_of_l[0], children_of_l[1]))
                    latent_nodes_to_skip.add(n)

        # --- Draw directed edges ---
        for parent, child in self.edges:
            if parent in latent_nodes_to_skip:
                continue  # handled as bidirected arc

            x0, y0 = pos[parent]
            x1, y1 = pos[child]

            # Shorten arrow to not overlap with node circles
            dx, dy = x1 - x0, y1 - y0
            dist = np.sqrt(dx**2 + dy**2)
            if dist < 1e-6:
                continue
            ux, uy = dx / dist, dy / dist
            shrink = node_size + 0.04
            sx0 = x0 + ux * shrink
            sy0 = y0 + uy * shrink
            sx1 = x1 - ux * shrink
            sy1 = y1 - uy * shrink

            ax.annotate(
                "",
                xy=(sx1, sy1),
                xytext=(sx0, sy0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#333333",
                    lw=1.8,
                    mutation_scale=16,
                ),
                zorder=2,
            )

        # --- Draw bidirected edges as curved dashed arcs ---
        for a, b in bidirected_pairs:
            if a not in pos or b not in pos:
                continue
            x0, y0 = pos[a]
            x1, y1 = pos[b]

            mx = (x0 + x1) / 2
            my = (y0 + y1) / 2
            dx, dy = x1 - x0, y1 - y0
            dist = np.sqrt(dx**2 + dy**2)
            # Perpendicular offset for the arc bulge
            offset = max(0.5, dist * 0.4)
            if dist > 1e-6:
                nx, ny = -dy / dist, dx / dist
            else:
                nx, ny = 0, 1
            cx = mx + nx * offset
            cy = my + ny * offset

            # Draw quadratic bezier as arc
            arrow = FancyArrowPatch(
                (x0, y0), (x1, y1),
                connectionstyle=f"arc3,rad={0.4}",
                arrowstyle="<->",
                color="#d62728",
                linestyle="--",
                lw=1.5,
                mutation_scale=14,
                shrinkA=node_size * 52,
                shrinkB=node_size * 52,
                zorder=2,
            )
            ax.add_patch(arrow)

        # --- Draw nodes ---
        for node in sorted(self.observed_nodes):
            if node not in pos:
                continue
            x, y = pos[node]
            color = node_colors.get(node, "#555555")

            # Conditioned nodes get hatched fill
            if node in conditioned:
                facecolor = "#e0e0e0"
                hatch = "///"
                edgecolor = color
                lw = 2.5
            else:
                facecolor = "white"
                hatch = None
                edgecolor = color
                lw = 2.5

            circle = plt.Circle(
                (x, y),
                node_size,
                fill=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                lw=lw,
                hatch=hatch,
                zorder=5,
            )
            ax.add_patch(circle)
            ax.text(
                x, y, node,
                ha="center", va="center",
                fontsize=font_size,
                fontweight="bold",
                color="#222222",
                zorder=6,
            )

        # --- Legend (when roles are shown) ---
        if exposure and outcome:
            legend_items = [
                mpatches.Patch(
                    facecolor="white", edgecolor=role_colors["exposure"],
                    lw=2, label="Exposure",
                ),
                mpatches.Patch(
                    facecolor="white", edgecolor=role_colors["outcome"],
                    lw=2, label="Outcome",
                ),
            ]
            # Only add roles that are present
            present_roles = set()
            for roles in node_roles_map.values():
                present_roles |= roles
            if "confounder" in present_roles:
                legend_items.append(mpatches.Patch(
                    facecolor="white", edgecolor=role_colors["confounder"],
                    lw=2, label="Confounder",
                ))
            if "mediator" in present_roles:
                legend_items.append(mpatches.Patch(
                    facecolor="white", edgecolor=role_colors["mediator"],
                    lw=2, label="Mediator",
                ))
            if "collider" in present_roles:
                legend_items.append(mpatches.Patch(
                    facecolor="white", edgecolor=role_colors["collider"],
                    lw=2, label="Collider (bad control)",
                ))
            if "instrument" in present_roles:
                legend_items.append(mpatches.Patch(
                    facecolor="white", edgecolor=role_colors["instrument"],
                    lw=2, label="Instrument",
                ))
            if conditioned:
                legend_items.append(mpatches.Patch(
                    facecolor="#e0e0e0", edgecolor="#555555",
                    lw=2, hatch="///", label="Conditioned",
                ))
            if bidirected_pairs:
                legend_items.append(mpatches.FancyArrow(
                    0, 0, 0.001, 0, color="#d62728",
                    width=0.001, label="Unobserved (latent)",
                ))
            ax.legend(
                handles=legend_items,
                loc="best",
                fontsize=9,
                framealpha=0.95,
                edgecolor="#cccccc",
            )

        # --- Title ---
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        elif exposure and outcome:
            cond_str = ""
            if conditioned:
                cond_str = f" | {', '.join(sorted(conditioned))}"
            ax.set_title(
                f"Causal DAG: {exposure} → {outcome}{cond_str}",
                fontsize=14,
                fontweight="bold",
                pad=12,
            )

        # --- Axis cleanup ---
        xs = [p[0] for n, p in pos.items() if not n.startswith("_L_")]
        ys = [p[1] for n, p in pos.items() if not n.startswith("_L_")]
        if xs and ys:
            margin = node_size + 0.6
            ax.set_xlim(min(xs) - margin, max(xs) + margin)
            ax.set_ylim(min(ys) - margin, max(ys) + margin)
        ax.axis("off")
        plt.tight_layout()
        return fig, ax

    def _layout(self, seed: int = 42) -> Dict[str, Tuple[float, float]]:
        """Topological depth-based layout with lateral spacing."""
        import numpy as np
        rng = np.random.RandomState(seed)

        depth: Dict[str, int] = {}
        for n in self._nodes:
            depth[n] = self._topo_depth(n, depth)

        # Group by depth
        layers: Dict[int, List[str]] = {}
        for n, d in depth.items():
            layers.setdefault(d, []).append(n)

        positions: Dict[str, Tuple[float, float]] = {}
        for d, nodes in layers.items():
            nodes_sorted = sorted(nodes)
            n_in_layer = len(nodes_sorted)
            for i, node in enumerate(nodes_sorted):
                x = (i - (n_in_layer - 1) / 2) * 1.8
                y = -(d * 1.8)
                x += rng.uniform(-0.08, 0.08)
                positions[node] = (x, y)

        return positions

    def _topo_depth(self, node: str, cache: Dict[str, int]) -> int:
        if node in cache:
            return cache[node]
        parents = self.parents(node)
        if not parents:
            cache[node] = 0
        else:
            cache[node] = max(self._topo_depth(p, cache) for p in parents) + 1
        return cache[node]

    # ------------------------------------------------------------------ #
    #  Parsing
    # ------------------------------------------------------------------ #

    def _parse(self, spec: str):
        """Parse edge specification string."""
        # Split by semicolons, newlines, or commas (but not commas inside parens)
        parts = re.split(r"[;\n]", spec)
        # Also split by comma if no semicolons/newlines were effective
        if len(parts) == 1 and "," in spec:
            parts = spec.split(",")

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Bidirected edge: X <-> Y
            if "<->" in part:
                nodes = [n.strip() for n in part.split("<->")]
                if len(nodes) == 2:
                    self.add_bidirected(nodes[0], nodes[1])
                continue

            # Chain: X -> M -> Y
            nodes = [n.strip() for n in re.split(r"\s*->\s*", part)]
            if len(nodes) >= 2:
                for i in range(len(nodes) - 1):
                    self.add_edge(nodes[i], nodes[i + 1])

    def summary(self, exposure: str, outcome: str) -> str:
        """
        Print a rich text summary of the DAG for identification analysis.

        Shows all paths, their status, adjustment sets, bad controls,
        and variable roles — a one-stop diagnostic like ``dagitty``'s
        ``adjustmentSets`` + ``paths``.

        Examples
        --------
        >>> g = sp.dag('Z -> X; Z -> Y; X -> Y')
        >>> print(g.summary('X', 'Y'))
        """
        lines = [f"DAG Summary: effect of {exposure} on {outcome}", "=" * 50]

        # Paths
        status = self.path_status(exposure, outcome)
        lines.append(f"\nPaths ({len(status)} total):")
        for s in status:
            arrow = " → ".join(s["path"])
            open_str = "OPEN" if s["open"] else "CLOSED"
            lines.append(f"  [{s['type']:>8}] {arrow}  ({open_str})")

        # Adjustment sets
        adj = self.adjustment_sets(exposure, outcome)
        lines.append(f"\nBackdoor adjustment sets: {len(adj)} found")
        if adj:
            for a in adj:
                lines.append(f"  {{ {', '.join(sorted(a)) if a else '∅'} }}")
        else:
            lines.append("  ⚠ No valid backdoor adjustment set exists!")

        # Frontdoor sets
        fd = self.frontdoor_sets(exposure, outcome)
        if fd:
            lines.append(f"\nFrontdoor adjustment sets: {len(fd)} found")
            for f in fd:
                lines.append(f"  {{ {', '.join(sorted(f))} }}")

        # Bad controls
        bad = self.bad_controls(exposure, outcome)
        if bad:
            lines.append(f"\n⚠ Bad controls (do NOT condition on):")
            for v, reasons in bad.items():
                for r in reasons:
                    lines.append(f"  {v}: {r}")

        # Variable roles
        lines.append(f"\nVariable roles:")
        for v in sorted(self.observed_nodes - {exposure, outcome}):
            roles = self.classify_variable(v, exposure, outcome)
            if roles:
                lines.append(f"  {v}: {', '.join(sorted(roles))}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        obs = self.observed_nodes
        return f"DAG({len(obs)} nodes, {len(self.edges)} edges)"


# ====================================================================== #
#  Convenience function
# ====================================================================== #

def dag(spec: str = "") -> DAG:
    """
    Create a causal DAG from a string specification.

    Parameters
    ----------
    spec : str
        Edge specification. Examples:

        - ``"X -> Y; Z -> X; Z -> Y"``
        - ``"X -> M -> Y; X -> Y"``
        - ``"X <-> Y; Z -> X; Z -> Y"``  (bidirected = latent common cause)

    Returns
    -------
    DAG

    Examples
    --------
    >>> g = sp.dag('Z -> X -> Y; Z -> Y')
    >>> g.adjustment_sets('X', 'Y')
    [{'Z'}]

    >>> g = sp.dag('X -> Y; X <-> Y')  # unobserved confounder
    >>> g.adjustment_sets('X', 'Y')
    []  # no valid adjustment set exists
    """
    return DAG(spec)


# ====================================================================== #
#  Classic textbook DAGs (Cunningham, Causal Inference: The Mixtape, ch.3)
# ====================================================================== #

_EXAMPLES = {
    "confounding": {
        "spec": "Z -> X; Z -> Y; X -> Y",
        "description": (
            "Simple confounding. Z is a common cause of X and Y. "
            "Condition on Z to close the backdoor path."
        ),
        "exposure": "X",
        "outcome": "Y",
    },
    "collider": {
        "spec": "X -> M; Y -> M",
        "description": (
            "Classic collider structure. X and Y are independent, but "
            "conditioning on M (the collider) opens a spurious path."
        ),
        "exposure": "X",
        "outcome": "Y",
    },
    "mediation": {
        "spec": "X -> M -> Y; X -> Y",
        "description": (
            "Mediation with direct and indirect effects. M mediates part "
            "of the effect of X on Y. Do NOT condition on M to estimate "
            "the total effect."
        ),
        "exposure": "X",
        "outcome": "Y",
    },
    "discrimination": {
        "spec": "D -> O; A -> O; A -> Y; D -> Y; O -> Y",
        "description": (
            "Gender discrimination and occupational sorting "
            "(Cunningham 2021, §3). D = discrimination, O = occupation, "
            "A = ability, Y = earnings. O is a collider on the path "
            "D→O←A→Y. Conditioning on O alone flips the sign of the "
            "discrimination effect."
        ),
        "exposure": "D",
        "outcome": "Y",
    },
    "movie_star": {
        "spec": "Beauty -> Star; Talent -> Star",
        "description": (
            "Beauty–Talent collider (Mixtape §3). In the population, "
            "Beauty and Talent are independent. But restricting to "
            "Stars (conditioning on Star) creates a spurious negative "
            "correlation between Beauty and Talent."
        ),
        "exposure": "Beauty",
        "outcome": "Talent",
    },
    "police": {
        "spec": "D -> M; U -> M; U -> Y; D -> Y; M -> Y",
        "description": (
            "Police use of force (Knox et al. 2020, Mixtape §3). "
            "D = racial discrimination, M = police stop, "
            "U = officer suspicion, Y = use of force. "
            "All administrative data conditions on M (a collider), "
            "opening the spurious path D→M←U→Y."
        ),
        "exposure": "D",
        "outcome": "Y",
    },
    "frontdoor": {
        "spec": "X <-> Y; X -> M -> Y",
        "description": (
            "Frontdoor criterion example. X and Y share an unobserved "
            "confounder (shown as bidirected arc). M fully mediates "
            "X→Y and has no unblocked backdoor from X. The frontdoor "
            "adjustment via M identifies the causal effect."
        ),
        "exposure": "X",
        "outcome": "Y",
    },
    "bad_control_earnings": {
        "spec": (
            "B -> PE; B -> D; PE -> D; PE -> I; I -> D; I -> Y; D -> Y"
        ),
        "description": (
            "College wage premium with unobserved background (Mixtape §3). "
            "B = unobserved family background, PE = parental education, "
            "I = family income, D = college education, Y = earnings. "
            "Multiple backdoor paths — I alone closes all of them, but "
            "B is unobservable."
        ),
        "exposure": "D",
        "outcome": "Y",
    },
    "m_bias": {
        "spec": "U1 -> X; U1 -> M; U2 -> Y; U2 -> M; X -> Y",
        "description": (
            "M-bias (butterfly bias). U1 and U2 are unobserved. "
            "M is a collider on the path X←U1→M←U2→Y. "
            "Conditioning on M opens this path and creates bias, "
            "even though M looks like a pre-treatment variable."
        ),
        "exposure": "X",
        "outcome": "Y",
    },
}


_EXAMPLE_POSITIONS = {
    "confounding": {"Z": (0, 0), "X": (-1, -1.5), "Y": (1, -1.5)},
    "collider": {"X": (-1.5, 0), "M": (0, -1.5), "Y": (1.5, 0)},
    "mediation": {"X": (-1.5, 0), "M": (0, -1.5), "Y": (1.5, 0)},
    "discrimination": {
        "D": (-1.8, 0), "A": (1.8, 0), "O": (0, -1.5), "Y": (0, -3),
    },
    "movie_star": {
        "Beauty": (-1.5, 0), "Talent": (1.5, 0), "Star": (0, -1.5),
    },
    "police": {
        "D": (-1.8, 0), "U": (1.8, 0), "M": (0, -1.5), "Y": (0, -3),
    },
    "frontdoor": {"X": (-1.5, 0), "M": (0, -1.5), "Y": (1.5, 0)},
    "bad_control_earnings": {
        "B": (-2, 0), "PE": (-0.5, -1.5), "I": (1, -1.5),
        "D": (-1, -3), "Y": (2, -3),
    },
    "m_bias": {
        "U1": (-1.5, 0), "U2": (1.5, 0), "M": (0, -1.5),
        "X": (-1.5, -3), "Y": (1.5, -3),
    },
}


def dag_examples() -> List[str]:
    """List available classic DAG examples."""
    return sorted(_EXAMPLES.keys())


def dag_example(name: str) -> DAG:
    """
    Load a classic textbook DAG by name.

    Parameters
    ----------
    name : str
        One of: ``'confounding'``, ``'collider'``, ``'mediation'``,
        ``'discrimination'``, ``'movie_star'``, ``'police'``,
        ``'frontdoor'``, ``'bad_control_earnings'``, ``'m_bias'``.

    Returns
    -------
    DAG
        The example DAG. Call ``.summary(exposure, outcome)`` for analysis,
        or ``.plot(exposure, outcome)`` for a ggdag-style visualisation
        with role-coloured nodes.

    Examples
    --------
    >>> g = sp.dag_example('discrimination')
    >>> print(g.summary('D', 'Y'))
    >>> g.plot('D', 'Y')

    >>> g = sp.dag_example('frontdoor')
    >>> g.plot('X', 'Y', positions=sp.dag_example_positions('frontdoor'))
    """
    if name not in _EXAMPLES:
        avail = ", ".join(sorted(_EXAMPLES.keys()))
        raise ValueError(f"Unknown example '{name}'. Available: {avail}")
    info = _EXAMPLES[name]
    g = DAG(info["spec"])
    g._example_info = info  # type: ignore[attr-defined]
    # Attach hand-tuned positions
    if name in _EXAMPLE_POSITIONS:
        g._positions = _EXAMPLE_POSITIONS[name]  # type: ignore[attr-defined]
    return g


def dag_example_positions(name: str) -> Dict[str, Tuple[float, float]]:
    """Return hand-tuned node positions for a named example DAG."""
    if name not in _EXAMPLE_POSITIONS:
        avail = ", ".join(sorted(_EXAMPLE_POSITIONS.keys()))
        raise ValueError(
            f"No hand-tuned positions for '{name}'. Available: {avail}"
        )
    return dict(_EXAMPLE_POSITIONS[name])


# ====================================================================== #
#  Simulations (Cunningham, The Mixtape, ch. 3)
# ====================================================================== #

def dag_simulate(
    name: str,
    n: int = 10000,
    seed: int = 42,
) -> "pd.DataFrame":
    """
    Run a classic DAG simulation from Cunningham (2021, ch. 3).

    Available simulations:

    - ``'discrimination'`` — Gender discrimination / occupational sorting.
      True effect of discrimination on wage is **-1**. Conditioning on
      occupation alone flips the sign (collider bias).
    - ``'movie_star'`` — Beauty–Talent collider. Beauty and Talent are
      independent in the population, but conditioning on Star status
      induces a spurious negative correlation.

    Parameters
    ----------
    name : str
        ``'discrimination'`` or ``'movie_star'``.
    n : int
        Number of observations (default 10000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Simulated dataset.

    Examples
    --------
    >>> df = sp.dag_simulate('discrimination')
    >>> import statsmodels.formula.api as smf
    >>> # Biased: wrong sign due to collider
    >>> smf.ols('wage ~ female + occupation', data=df).fit().params['female']
    >>> # Correct: includes ability
    >>> smf.ols('wage ~ female + occupation + ability', data=df).fit().params['female']
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)

    if name == "discrimination":
        female = rng.binomial(1, 0.5, size=n)
        ability = rng.standard_normal(n)
        discrimination = female.copy()
        occupation = (
            1 + 2 * ability + 0 * female - 2 * discrimination
            + rng.standard_normal(n)
        )
        wage = (
            1 - 1 * discrimination + 1 * occupation + 2 * ability
            + rng.standard_normal(n)
        )
        return pd.DataFrame({
            "female": female,
            "ability": ability,
            "discrimination": discrimination,
            "occupation": occupation,
            "wage": wage,
        })

    elif name == "movie_star":
        beauty = rng.standard_normal(n)
        talent = rng.standard_normal(n)
        score = beauty + talent
        c85 = np.percentile(score, 85)
        star = (score >= c85).astype(int)
        return pd.DataFrame({
            "beauty": beauty,
            "talent": talent,
            "score": score,
            "star": star,
        })

    else:
        avail = "'discrimination', 'movie_star'"
        raise ValueError(f"Unknown simulation '{name}'. Available: {avail}")
