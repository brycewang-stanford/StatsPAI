"""
Causal DAG declaration and analysis.

Provides the :class:`DAG` class for declaring causal structures and
computing identification-relevant quantities:

- **Backdoor adjustment sets** (Pearl's backdoor criterion)
- **Collider detection**
- **d-separation** queries
- **Ancestors / descendants / paths** traversal
- **ASCII and matplotlib visualisation**

Modelled after R's ``dagitty`` — no external graph library required.

References
----------
Pearl, J. (2009). *Causality*. Cambridge University Press.
Greenland, S., Pearl, J. and Robins, J.M. (1999).
"Causal Diagrams for Epidemiologic Research." *Epidemiology*, 10(1), 37-48.
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
        for parent in sorted(self._edges.keys()):
            for child in sorted(self._edges[parent]):
                if parent.startswith("_L_"):
                    # Show latent as bidirected
                    other_children = sorted(self._edges[parent])
                    if len(other_children) == 2:
                        a, b = other_children
                        lines.append(f"  {a} <-> {b}  (latent: {parent})")
                        break
                else:
                    lines.append(f"  {parent} -> {child}")
        return "\n".join(lines)

    def plot(self, figsize: tuple = (8, 6), seed: int = 42):
        """
        Plot the DAG using matplotlib.

        Requires matplotlib. Uses a simple force-directed layout.

        Returns
        -------
        (fig, ax) : matplotlib figure and axes
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError("matplotlib required for DAG plotting: pip install matplotlib")

        # Simple layered layout based on topological depth
        positions = self._layout(seed)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")

        # Draw edges
        for parent, child in self.edges:
            if parent.startswith("_L_"):
                # Bidirected: draw dashed
                style = "--"
                color = "red"
            else:
                style = "-"
                color = "#333333"

            x0, y0 = positions[parent]
            x1, y1 = positions[child]

            ax.annotate(
                "",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="->", color=color, linestyle=style,
                    lw=1.5, shrinkA=15, shrinkB=15,
                ),
            )

        # Draw nodes
        for node, (x, y) in positions.items():
            if node.startswith("_L_"):
                continue  # don't draw latent nodes
            circle = plt.Circle((x, y), 0.12, fill=True,
                                facecolor="white", edgecolor="#333333", lw=2, zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, node, ha="center", va="center", fontsize=11,
                    fontweight="bold", zorder=6)

        # Axis cleanup
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        margin = 0.5
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)
        ax.axis("off")
        ax.set_title("Causal DAG", fontsize=14)
        plt.tight_layout()
        return fig, ax

    def _layout(self, seed: int = 42) -> Dict[str, Tuple[float, float]]:
        """Simple topological depth-based layout."""
        import numpy as np
        rng = np.random.RandomState(seed)

        # Compute topological depth
        depth: Dict[str, int] = {}
        for n in self._nodes:
            depth[n] = self._topo_depth(n, depth)

        max_depth = max(depth.values()) if depth else 0

        # Group by depth
        layers: Dict[int, List[str]] = {}
        for n, d in depth.items():
            layers.setdefault(d, []).append(n)

        positions: Dict[str, Tuple[float, float]] = {}
        for d, nodes in layers.items():
            nodes_sorted = sorted(nodes)
            n_in_layer = len(nodes_sorted)
            for i, node in enumerate(nodes_sorted):
                x = (i - (n_in_layer - 1) / 2) * 1.5
                y = -(d * 1.5)
                # Small jitter for aesthetics
                x += rng.uniform(-0.1, 0.1)
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
