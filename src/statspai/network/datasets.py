"""Canonical reference networks for ``sp.network``.

Two public-domain networks used throughout the social-network-analysis
literature, shipped so that examples and parity tests have a fixed,
well-understood substrate:

* :func:`karate_club` — Zachary's (1977) karate club: 34 members, 78
  friendship ties, the canonical community-detection benchmark.  The
  observed factional split (Mr Hi vs. the Officer) is available as
  :data:`KARATE_FACTION`.
* :func:`florentine_families` — Padgett & Ansell's (1993) Renaissance
  Florentine marriage network: 15 connected families, 20 marriage ties
  (the isolated Pucci family is omitted, as in the standard ``statnet``
  ``flomarriage`` object).

References
----------
Zachary, W. W. (1977). "An information flow model for conflict and fission
in small groups." *Journal of Anthropological Research*, 33(4), 452-473.
[@zachary1977information]

Padgett, J. F. & Ansell, C. K. (1993). "Robust action and the rise of the
Medici, 1400-1434." *American Journal of Sociology*, 98(6), 1259-1319.
[@padgett1993robust]
"""

from __future__ import annotations

from typing import List

from ._core import Graph

__all__ = ["karate_club", "florentine_families", "KARATE_FACTION"]


# Zachary (1977), Table data as reproduced in the standard benchmark
# (undirected friendship ties; node ids 0..33, Mr Hi = 0, Officer = 33).
_KARATE_EDGES: List[tuple] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 10),
    (0, 11),
    (0, 12),
    (0, 13),
    (0, 17),
    (0, 19),
    (0, 21),
    (0, 31),
    (1, 2),
    (1, 3),
    (1, 7),
    (1, 13),
    (1, 17),
    (1, 19),
    (1, 21),
    (1, 30),
    (2, 3),
    (2, 7),
    (2, 8),
    (2, 9),
    (2, 13),
    (2, 27),
    (2, 28),
    (2, 32),
    (3, 7),
    (3, 12),
    (3, 13),
    (4, 6),
    (4, 10),
    (5, 6),
    (5, 10),
    (5, 16),
    (6, 16),
    (8, 30),
    (8, 32),
    (8, 33),
    (9, 33),
    (13, 33),
    (14, 32),
    (14, 33),
    (15, 32),
    (15, 33),
    (18, 32),
    (18, 33),
    (19, 33),
    (20, 32),
    (20, 33),
    (22, 32),
    (22, 33),
    (23, 25),
    (23, 27),
    (23, 29),
    (23, 32),
    (23, 33),
    (24, 25),
    (24, 27),
    (24, 31),
    (25, 31),
    (26, 29),
    (26, 33),
    (27, 33),
    (28, 31),
    (28, 33),
    (29, 32),
    (29, 33),
    (30, 32),
    (30, 33),
    (31, 32),
    (31, 33),
    (32, 33),
]

#: Zachary's observed factional split after the fission (0 = Mr Hi's
#: faction, 1 = the Officer's faction), ordered by node id 0..33.
KARATE_FACTION: List[int] = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    0,
    0,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    0,
    1,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
]


_FLORENTINE_NODES: List[str] = [
    "Acciaiuoli",
    "Medici",
    "Castellani",
    "Peruzzi",
    "Strozzi",
    "Barbadori",
    "Ridolfi",
    "Tornabuoni",
    "Albizzi",
    "Salviati",
    "Pazzi",
    "Bischeri",
    "Guadagni",
    "Ginori",
    "Lamberteschi",
]

_FLORENTINE_EDGES: List[tuple] = [
    ("Acciaiuoli", "Medici"),
    ("Albizzi", "Ginori"),
    ("Albizzi", "Guadagni"),
    ("Bischeri", "Guadagni"),
    ("Castellani", "Barbadori"),
    ("Castellani", "Peruzzi"),
    ("Castellani", "Strozzi"),
    ("Guadagni", "Lamberteschi"),
    ("Medici", "Albizzi"),
    ("Medici", "Barbadori"),
    ("Medici", "Ridolfi"),
    ("Medici", "Salviati"),
    ("Medici", "Tornabuoni"),
    ("Peruzzi", "Bischeri"),
    ("Peruzzi", "Strozzi"),
    ("Ridolfi", "Tornabuoni"),
    ("Salviati", "Pazzi"),
    ("Strozzi", "Bischeri"),
    ("Strozzi", "Ridolfi"),
    ("Tornabuoni", "Guadagni"),
]


def karate_club() -> Graph:
    """Zachary's karate club friendship network (undirected, 34 nodes).

    Returns
    -------
    Graph
        Binary undirected graph with 34 nodes and 78 edges.

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.karate_club()
    >>> g.n_nodes, g.n_edges
    (34, 78)

    References
    ----------
    zachary1977information
    """
    return Graph.from_edgelist(_KARATE_EDGES, directed=False, nodes=list(range(34)))


def florentine_families() -> Graph:
    """Padgett's Florentine marriage network (undirected, 15 families).

    Returns
    -------
    Graph
        Binary undirected graph; nodes are labelled by family name.  The
        Medici sit at the structural centre (highest betweenness).

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.florentine_families()
    >>> g.n_nodes, g.n_edges
    (15, 20)

    References
    ----------
    padgett1993robust
    """
    return Graph.from_edgelist(
        _FLORENTINE_EDGES, directed=False, nodes=_FLORENTINE_NODES
    )
