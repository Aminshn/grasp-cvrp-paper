from math import hypot
from typing import Dict, List, Tuple, Optional
import tempfile
import os


def _edge_distance(
    a: int,
    b: int,
    coords: Dict[int, Tuple[float, float]],
    dist_matrix: Optional[Dict[Tuple[int, int], float]],
) -> Optional[float]:
    """Compute distance between nodes using dist_matrix if available, else Euclidean."""
    if dist_matrix:
        d = dist_matrix.get((a, b), dist_matrix.get((b, a), None))
        if d is not None:
            return d

    ca = coords.get(a)
    cb = coords.get(b)
    if ca is None or cb is None:
        return None

    try:
        return hypot(ca[0] - cb[0], ca[1] - cb[1])
    except Exception:
        return None


def plot_convergence(iterations, best_costs, title="Convergence Plot"):
    """Plot convergence curve using Plotly."""
    # ...existing implementation...

def render_average_convergence_plot(job_ids, engine):
    """Render average convergence plot from job results."""
    # ...existing implementation...
