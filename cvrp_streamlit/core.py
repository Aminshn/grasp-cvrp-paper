from typing import Dict, List, Tuple


class CVRP:
    """
    Minimal CVRP container used by the Streamlit app.

    Attributes:
        q: dict mapping node -> demand.
        Q: vehicle capacity.
        routes: list of routes, each a list of nodes.
        N: sorted list of customer nodes.
        N0: list of nodes including depot 0.
    """

    def __init__(self, q: Dict[int, int], Q: int, routes: List[List[int]]) -> None:
        self.q: Dict[int, int] = dict(q)
        self.Q: int = int(Q)
        self.routes: List[List[int]] = [list(r) for r in routes]

        self.N: List[int] = sorted(self.q.keys())
        self.N0: List[int] = [0] + self.N

    def build_x(self) -> Dict[Tuple[int, int], int]:
        """
        Build edge usage matrix x[(i, j)] = number of times edge (i, j) appears.

        Returns:
            dict mapping (i, j) -> usage count.
        """
        x: Dict[Tuple[int, int], int] = {
            (i, j): 0 for i in self.N0 for j in self.N0 if i != j
        }

        for route in self.routes:
            for i, j in zip(route[:-1], route[1:]):
                if i == j:
                    continue
                if (i, j) not in x:
                    x[(i, j)] = 0
                x[(i, j)] += 1

        return x
