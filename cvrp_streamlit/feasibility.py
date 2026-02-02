from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict, deque


def check_customers_served_once(
    demands: Dict[int, int],
    routes: List[List[int]],
    depot: int = 0,
) -> Dict:
    """
    Verify that each customer (non-depot node) appears exactly once across all routes.
    """
    customers = {n for n in demands.keys() if n != depot}
    visits = Counter()

    for route in routes:
        for node in route:
            if node != depot:
                visits[node] += 1

    missing = sorted(customers - visits.keys())
    extra = sorted(visits.keys() - customers)
    multi = sorted([n for n, c in visits.items() if c > 1])

    ok = not missing and not extra and not multi

    return {
        "ok": ok,
        "missing_customers": missing,
        "unrecognized_nodes": extra,
        "multi_visited_customers": multi,
        "visits": dict(visits),
    }


def check_capacity_per_route(
    demands: Dict[int, int],
    routes: List[List[int]],
    capacity: int,
    depot: int = 0,
) -> Dict:
    """
    Verify that the total demand on each route does not exceed the vehicle capacity.
    """
    route_loads = []
    violating = []

    for idx, route in enumerate(routes):
        load = sum(demands.get(n, 0) for n in route if n != depot)
        route_loads.append((idx, load))
        if load > capacity:
            violating.append(idx)

    return {
        "ok": not violating,
        "route_loads": route_loads,
        "violating_routes": violating,
    }


def check_route_structure_basic(
    routes: List[List[int]],
    depot: int = 0,
) -> Dict:
    """
    Basic structural checks on routes:
      - each route is non-empty
      - each route starts and ends at the depot
    """
    empty = [i for i, r in enumerate(routes) if not r]
    wrong_depot = [
        i for i, r in enumerate(routes)
        if r and (r[0] != depot or r[-1] != depot)
    ]

    return {
        "ok": not empty and not wrong_depot,
        "empty_routes": empty,
        "wrong_depot_routes": wrong_depot,
    }


def build_edge_usage_from_routes(
    routes: List[List[int]],
) -> Dict[Tuple[int, int], int]:
    """
    Build an edge-usage dictionary x[(i, j)] from explicit route lists.
    """
    x: Dict[Tuple[int, int], int] = {}
    for route in routes:
        for i, j in zip(route[:-1], route[1:]):
            if i == j:
                continue
            x[(i, j)] = x.get((i, j), 0) + 1
    return x


def check_structure_from_edges(
    x: Dict[Tuple[int, int], int],
    demands: Dict[int, int],
    depot: int = 0,
) -> Dict:
    """
    Graph-based structural checks using edge usage x[(i, j)]:

      - each non-depot customer has indegree <= 1 and outdegree <= 1
      - all served customers that appear in the graph are reachable
        from the depot (no disconnected subtours)
    """
    succ: Dict[int, List[int]] = defaultdict(list)
    indeg: Dict[int, int] = defaultdict(int)
    outdeg: Dict[int, int] = defaultdict(int)
    used_nodes = set()

    for (i, j), val in x.items():
        if not val:
            continue
        succ[i].append(j)
        outdeg[i] += 1
        indeg[j] += 1
        used_nodes.add(i)
        used_nodes.add(j)

    customers = {n for n in demands.keys() if n != depot and demands[n] > 0}

    bad_indeg = [n for n in customers if indeg[n] > 1]
    bad_outdeg = [n for n in customers if outdeg[n] > 1]

    # reachability from depot
    reachable = set()
    q = deque([depot])

    while q:
        u = q.popleft()
        if u in reachable:
            continue
        reachable.add(u)
        for v in succ.get(u, []):
            if v not in reachable:
                q.append(v)

    unreachable_customers = sorted(
        [n for n in customers if n in used_nodes and n not in reachable]
    )

    ok = not bad_indeg and not bad_outdeg and not unreachable_customers

    return {
        "ok": ok,
        "bad_indegree_customers": sorted(bad_indeg),
        "bad_outdegree_customers": sorted(bad_outdeg),
        "unreachable_customers": unreachable_customers,
        "reachable_from_depot": sorted(reachable),
    }


def check_cvrp_feasibility(
    demands: Dict[int, int],
    routes: List[List[int]],
    capacity: int,
    depot: int = 0,
    x: Optional[Dict[Tuple[int, int], int]] = None,
) -> Dict:
    """
    Full CVRP feasibility check combining:

      - served-once constraint
      - per-route capacity
      - basic route structure (start/end at depot, non-empty)
      - graph-based structure from edges (degrees + subtours)
    """
    served = check_customers_served_once(demands, routes, depot)
    capacity_check = check_capacity_per_route(demands, routes, capacity, depot)
    structure_basic = check_route_structure_basic(routes, depot)

    if x is None:
        x = build_edge_usage_from_routes(routes)

    structure_graph = check_structure_from_edges(x, demands, depot)

    overall_ok = (
        served["ok"]
        and capacity_check["ok"]
        and structure_basic["ok"]
        and structure_graph["ok"]
    )

    return {
        "ok": overall_ok,
        "served_once": served,
        "capacity": capacity_check,
        "structure_basic": structure_basic,
        "structure_graph": structure_graph,
    }
