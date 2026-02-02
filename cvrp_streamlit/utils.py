from __future__ import annotations

import math


def compute_distance_matrix(
    coords: dict[int, tuple[float, float]]
) -> dict[tuple[int, int], float]:
    """
    Compute a full Euclidean distance matrix for the given coordinates.

    Parameters
    ----------
    coords : dict[int, (float, float)]
        Mapping from node id to (x, y) coordinates.

    Returns
    -------
    dict[(int, int), float]
        A dictionary mapping (i, j) pairs to the Euclidean distance
        between node i and node j.
    """
    dist: dict[tuple[int, int], float] = {}
    nodes = list(coords.keys())

    for i in nodes:
        xi, yi = coords[i]
        for j in nodes:
            xj, yj = coords[j]
            dist[(i, j)] = math.hypot(xi - xj, yi - yj)

    return dist


def route_cost(route: list[int], dist_matrix: dict[tuple[int, int], float]) -> float:
    """
    Compute the total cost of a single route.

    Parameters
    ----------
    route : list[int]
        Sequence of node ids, e.g. [0, 3, 5, 0].
    dist_matrix : dict[(int, int), float]
        Distance matrix as returned by compute_distance_matrix.

    Returns
    -------
    float
        Sum of distances along the route.
    """
    total = 0.0
    for a, b in zip(route, route[1:]):
        total += dist_matrix[(a, b)]
    return total


def solution_cost(routes: list[list[int]], dist_matrix: dict[tuple[int, int], float]) -> float:
    """
    Compute the total cost of all routes in the solution.

    Parameters
    ----------
    routes : list[list[int]]
        List of routes, each route is a list of node ids.
    dist_matrix : dict[(int, int), float]
        Distance matrix.

    Returns
    -------
    float
        Total cost of the solution.
    """
    return sum(route_cost(r, dist_matrix) for r in routes)



