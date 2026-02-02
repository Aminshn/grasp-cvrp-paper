from __future__ import annotations
import random
import math
import time
from dataclasses import dataclass
from typing import Literal, Optional, Sequence
from .utils import compute_distance_matrix, route_cost, solution_cost


# These are configs and types used in the algorithm.
# These are just type hints and data structures to help with clarity for the app.
RCLType = Literal["Threshold", "Cardinality"]
NeighborhoodName = Literal["2opt", "relocate", "swap"]
ImprovementType = Literal["first", "best"]

# this is the configuration dataclass for the GRASP algorithm. 
@dataclass
class GraspConfig:
    iterations: int = 100
    rcl_type: RCLType = "Threshold"
    rcl_param: float = 0.3
    neighborhoods: Sequence[tuple[NeighborhoodName, ImprovementType]] = (
        ("2opt", "first"),
    )
    seed: Optional[int] = None
    construction_strategy: Literal["insertion"] = "insertion"
    local_search_type: Literal[
        "sequential", "vnd_first_improvement", "vnd_best_improvement"
    ] = "sequential"
    time_limit: Optional[float] = None




# One of the common ways to build a RCL list is by threshold (alpha)
# alpha in range [0,1], where 0 is completely greedy and 1 is completely random.
# this function takes a list of candidates (custmer and the values) and alpha, and returns the RCL list.
# it calculates the min and max values and then calculates the value threshold based on alpha.
# at end, it returns the candidates that are below the threshold.

def _build_rcl_by_alpha(candidates, alpha: float):
    values = [v for _, v, *_ in candidates]
    v_min, v_max = min(values), max(values)
    if math.isclose(v_min, v_max): # is_close is used to avoid floating point issues
        return candidates
    threshold = v_min + alpha * (v_max - v_min)
    return [c for c in candidates if c[1] <= threshold]

# another common way to build a RCL list is by size (k)
# this function takes a list of candidates and then k, and returns the top k candidates based on their values.

def _build_rcl_by_size(candidates, k: int):
    if k <= 0 or k >= len(candidates):
        return candidates
    return sorted(candidates, key=lambda c: c[1])[:k]



# 2-opt is neighborhood operator that removes two edges and reconnects them in a reverse way to reduce the route cost especially removing crossings.
# for doing this, algorithm iteratively checks all possible pairs of edges in the route and see if reversing the segment between them reduces the cost.
# there are two common strategies for improvement: first improvement and best improvement.
# first improvement stops as soon as it finds a better solution, while best improvement checks all possibilities and selects the best one.
# this function implements the 2-opt algorithm for a single route and then we use it in a wrapper for all routes.

def _two_opt_route(route, dist_matrix, improvement: ImprovementType = "first"):

    route = route[:] 
    # n is the number of nodes in the route
    n = len(route)
    if n < 4: # does not make sense to do it for routes with less than 4 nodes.
        return route, route_cost(route, dist_matrix)

    d = lambda a, b: dist_matrix.get((a, b), float("inf")) # an small helper function to get distance between two nodes. using inf if not found for reliability.

    best_cost = 0.0
    # this is the initial cost of the route. for further improvement we can also take this as input to avoid recomputation. #todo
    for k in range(n - 1):
        best_cost += d(route[k], route[k + 1])

    improved = True
    eps = 1e-9

    while improved:
        improved = False
        # depending on the improvement strategy, we either do first improvement or best improvement.
        # in first improvement, we break as soon as we find an improving move.
        if improvement == "first":
            # selecting the fir
            for i in range(1, n - 2): # we do not consider the depot at start and end.
                a, b = route[i - 1], route[i]
                dab = d(a, b) # old distance between a and b

                for j in range(i + 1, n - 1):
                    if j == i + 1:
                        continue # skipping adjacent edges because reversing them does not make sense.

                    c, dnode = route[j - 1], route[j]
                    dcd = d(c, dnode) # old distance between c and dnode

                    # when we want to reverse the segment between i and j, we remove edges (a,b) and (c,dnode) and add edges (a,c) and (b,dnode).
                    # consider the change in cost is delta = new_edges - old_edges because the edge between them just get reversed and does not change.
                    new_edges = d(a, c) + d(b, dnode)
                    old_edges = dab + dcd
                    delta = new_edges - old_edges
                    # use eps to avoid floating point issues.
                    # if delta is negative, it means we have an improvement.
                    if delta < -eps:
                        route[i:j] = reversed(route[i:j])
                        best_cost += delta # calculating the new cost of the route.
                        improved = True
                        break # this is first improvement, so we break as soon as we find an improving move.

                if improved:
                    break 
        # in best improvement, the process is similar but we save the best founded move and apply it after checking all possibilities.            
        elif improvement == "best":
            best_delta = 0.0
            best_i, best_j = -1, -1

            for i in range(1, n - 2):
                a, b = route[i - 1], route[i]
                old_ab = d(a, b)

                for j in range(i + 1, n - 1):
                    if j == i + 1:
                        continue

                    c, dnode = route[j - 1], route[j]
                    old_cd = d(c, dnode)

                    delta = (d(a, c) + d(b, dnode)) - (old_ab + old_cd)

                    if delta < best_delta - eps:
                        best_delta = delta
                        best_i, best_j = i, j

            if best_i != -1: # if we found an improving move
                route[best_i:best_j] = reversed(route[best_i:best_j])
                best_cost += best_delta # updating the cost based on the best delta found.
                improved = True
                
        # just in case of wrong input.
        else:
            raise ValueError("improvement must be 'first' or 'best'")

    return route, best_cost # returning the improved route and its cost.

# this is a wrapper function that applies the 2-opt local search to all routes in the solution.
# it takes the list of routes and applies the _two_opt_route function to each route iteratively.
def _local_search_2opt_wrapper(routes, dist_matrix, improvement: ImprovementType = "first"):

    routes = [r[:] for r in routes]
    d = lambda a, b: dist_matrix.get((a, b), float("inf"))

    route_costs = []
    # calculating the initial costs of all routes.
    for r in routes:
        c = 0.0 # initial cost of the route
        for k in range(len(r) - 1): # calculating the cost of the route
            c += d(r[k], r[k + 1]) # adding the distance between consecutive nodes #todo: we can optimize this by passing the cost as input to avoid recomputation.
        route_costs.append(c)

    total_cost = sum(route_costs)
    eps = 1e-9

    improved = True
    while improved:
        improved = False # set the flag to false at the start of each iteration and if any improvement is found, set it to true.
        # iterating over all routes and applying 2-opt to each route.
        for idx, r in enumerate(routes):
            old_cost = route_costs[idx] # take the old cost of the route
            new_r, new_cost = _two_opt_route(r, dist_matrix, improvement=improvement)
            # checking if the new cost is better than the old cost. #todo: it can be also optimized by adding an improvement flag in the _two_opt_route function to avoid recomputation of cost.
            if new_cost + eps < old_cost:
                routes[idx] = new_r
                route_costs[idx] = new_cost
                total_cost += (new_cost - old_cost)
                improved = True
                break

    return routes, total_cost

# its relocate inter-route local search.
# here we try to move a customer from one node to another route to reduce the overall cost.
# similar to 2-opt, we have two improvement strategies: first improvement and best improvement.
# when we relocate a customer, we need to ensure that the capacity constraints of the routes are not violated.
def _relocate_inter_route(
    routes,
    dist_matrix,
    demands,
    capacity,
    improvement: ImprovementType = "first",
):
    routes = [r[:] for r in routes]
    d = lambda a, b: dist_matrix.get((a, b), float("inf")) # helper function to get distance between two nodes.
    # calculating the initial loads of all routes.
    loads = [sum(demands.get(c, 0) for c in r if c != 0) for r in routes]

    # initial total cost #todo: can be optimized by passing the cost as input to avoid recomputation.
    total_cost = 0.0
    for r in routes:
        for k in range(len(r) - 1):
            total_cost += d(r[k], r[k + 1])

    eps = 1e-9
    improved = True
    while improved:
        # set the flag to false at the start of each iteration and if any improvement is found, set it to true.
        improved = False
        # here we try to relocate each customer from one route to another. and check if it improves the overall cost.
        # in first improvement, we break as soon as we find an improving move.
        if improvement == "first":
            for i, ri in enumerate(routes): # iterating over all routes
                for pi in range(1, len(ri) - 1): # iterating over all customers in the route except depot at start and end.
                    cust = ri[pi] # customer to be relocated
                    dem = demands.get(cust, 0) # demand of the customer

                    for j, rj in enumerate(routes): # iterating over all routes again
                        # checking 2 conditions:
                        # first is that we cannot relocate to the same route.
                        # second is related to capacity constraint of the route. calcualte the load after relocation and check if it exceeds capacity.
                        if i == j or loads[j] + dem > capacity:
                            continue
                        # now, after checking primary conditions, we try to insert the customer at all possible positions in the new route.
                        for pj in range(1, len(rj)):
                            # delta = new - old
                            # calculating the change in cost if we relocate the customer from route i to route j at position pj.
                            # this means we remove the customer from route i and insert it in route j.
                            delta = (
                                d(ri[pi - 1], ri[pi + 1]) # distance between the nodes before and after the selected customer in origin route
                                - d(ri[pi - 1], cust) # reducing the distance from previous node to customer in origin route
                                - d(cust, ri[pi + 1]) # reducing the distance from customer to next node in origin route
                                + d(rj[pj - 1], cust) # adding the distance from previous node to customer in destination route
                                + d(cust, rj[pj]) # adding the distance from customer to next node in destination route
                                - d(rj[pj - 1], rj[pj]) # removing the distance between previous and next node in destination route
                            )
                            # if delta is negative this means we have an improvement.
                            if delta < -eps:
                                routes[i].pop(pi) # removing the customer from origin route
                                routes[j].insert(pj, cust) # inserting the customer in destination route at position pj
                                loads[i] -= dem # updating the load of origin route
                                loads[j] += dem # updating the load of destination route
                                total_cost += delta # updating the total cost
                                improved = True # because we found an improvement, we set the flag to true.
                                break # breaking because this is first improvement.
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        # in best improvement, we check all possibilities and select the best one.
        elif improvement == "best":
            best_delta = 0.0
            best_move = None
            # iterating over all routes and customers similar to first improvement.
            for i, ri in enumerate(routes):
                for pi in range(1, len(ri) - 1):
                    cust = ri[pi]
                    dem = demands.get(cust, 0)

                    for j, rj in enumerate(routes):
                        if i == j or loads[j] + dem > capacity:
                            continue

                        for pj in range(1, len(rj)):
                            delta = (
                                d(ri[pi - 1], ri[pi + 1])
                                - d(ri[pi - 1], cust)
                                - d(cust, ri[pi + 1])
                                + d(rj[pj - 1], cust)
                                + d(cust, rj[pj])
                                - d(rj[pj - 1], rj[pj])
                            )

                            if delta < best_delta - eps:
                                best_delta = delta # saving the best delta found so far, this is the difference from first improvement.
                                best_move = (i, pi, j, pj, cust, dem)

            if best_move is not None:
                i, pi, j, pj, cust, dem = best_move
                routes[i].pop(pi)
                routes[j].insert(pj, cust)
                loads[i] -= dem
                loads[j] += dem # updating the load of destination route
                total_cost += best_delta # updating the total cost
                improved = True # because we found an improvement, we set the flag to true.

        else:
            raise ValueError("improvement must be 'first' or 'best'")

    return routes, total_cost

# in this function, we implement the swap inter-route local search.
# the difference with relocate is that here we swap two customers between two different routes.
# so we need to ensure that the capacity constraints of both routes are not violated after the swap.
# the calculation of delta is almost similar to relocate.
def _swap_inter_route(
    routes,
    dist_matrix,
    demands,
    capacity,
    improvement: ImprovementType = "first",
):
    routes = [r[:] for r in routes]
    d = lambda a, b: dist_matrix.get((a, b), float("inf"))
    # calculating the initial loads of all routes.
    loads = [sum(demands.get(c, 0) for c in r if c != 0) for r in routes]

    # initial total cost once
    total_cost = 0.0
    # calculating the initial total cost of all routes. #todo: can be optimized by passing the cost as input to avoid recomputation.
    for r in routes:
        for k in range(len(r) - 1):
            total_cost += d(r[k], r[k + 1])

    eps = 1e-9
    improved = True
    while improved:
        # set the flag to false at the start of each iteration and if any improvement is found, set it to true.
        improved = False
        # here we try to swap each customer from one route to another. and check if it improves the overall cost.
        # in first improvement, we break as soon as we find an improving move.
        if improvement == "first":
            for i in range(len(routes)):
                for j in range(i + 1, len(routes)):
                    ri, rj = routes[i], routes[j]
                    # iterating over all customers in route i except depot at start and end.
                    for pi in range(1, len(ri) - 1):
                        ci = ri[pi] # selected customer in route i
                        di = demands.get(ci, 0) # demand of selected customer in route i
                        # iterating over all customers in route j except depot at start and end.
                        for pj in range(1, len(rj) - 1):
                            cj = rj[pj] # selected customer in route j
                            dj = demands.get(cj, 0) # demand of selected customer in route j
                            
                            # checking the capacity constraints of both routes after the swap.
                            if loads[i] - di + dj > capacity or loads[j] - dj + di > capacity:
                                continue

                            # delta = new - old (route i)
                            delta_i = (
                                d(ri[pi - 1], cj) + d(cj, ri[pi + 1])
                                - d(ri[pi - 1], ci) - d(ci, ri[pi + 1])
                            )
                            # delta = new - old (route j)
                            delta_j = (
                                d(rj[pj - 1], ci) + d(ci, rj[pj + 1])
                                - d(rj[pj - 1], cj) - d(cj, rj[pj + 1])
                            )

                            total_delta = delta_i + delta_j

                            if total_delta < -eps:
                                routes[i][pi] = cj
                                routes[j][pj] = ci

                                loads[i] = loads[i] - di + dj
                                loads[j] = loads[j] - dj + di

                                total_cost += total_delta
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        # in best improvement, we check all possibilities and select the best one.
        # other calculations are similar to first improvement.
        elif improvement == "best":
            best_delta = 0.0
            best_move = None

            for i in range(len(routes)):
                for j in range(i + 1, len(routes)):
                    ri, rj = routes[i], routes[j]

                    for pi in range(1, len(ri) - 1):
                        ci = ri[pi]
                        di = demands.get(ci, 0)

                        for pj in range(1, len(rj) - 1):
                            cj = rj[pj]
                            dj = demands.get(cj, 0)

                            if loads[i] - di + dj > capacity or loads[j] - dj + di > capacity:
                                continue

                            delta_i = (
                                d(ri[pi - 1], cj) + d(cj, ri[pi + 1])
                                - d(ri[pi - 1], ci) - d(ci, ri[pi + 1])
                            )
                            delta_j = (
                                d(rj[pj - 1], ci) + d(ci, rj[pj + 1])
                                - d(rj[pj - 1], cj) - d(cj, rj[pj + 1])
                            )

                            total_delta = delta_i + delta_j

                            if total_delta < best_delta - eps:
                                best_delta = total_delta
                                best_move = (i, pi, j, pj, ci, cj, di, dj)

            if best_move:
                i, pi, j, pj, ci, cj, di, dj = best_move

                routes[i][pi] = cj
                routes[j][pj] = ci

                loads[i] = loads[i] - di + dj
                loads[j] = loads[j] - dj + di

                total_cost += best_delta
                improved = True

        else:
            raise ValueError("improvement must be 'first' or 'best'")

    return routes, total_cost





# here we implement the main local search strategies used in the GRASP algorithm.
# there are three main strategies: sequential, VND with first improvement, and VND with best improvement.

# this is the simplest way, just applying operators one by ofe in the given order in neighborhoods list.
# then returning the final solution after applying all operators.
def _local_search_sequential(routes, dist_matrix, neighborhoods, demands, capacity):
    cost = None
    for op, imp in neighborhoods:

        if op == "2opt":
            routes, cost = _local_search_2opt_wrapper(routes, dist_matrix, improvement=imp)

        elif op == "relocate":
            routes, cost = _relocate_inter_route(
                routes, dist_matrix, demands, capacity, improvement=imp
            )

        elif op == "swap":
            routes, cost = _swap_inter_route(
                routes, dist_matrix, demands, capacity, improvement=imp
            )

    if cost is None:
        cost = solution_cost(routes, dist_matrix)
    return routes, cost

# in VND with first improvement, we iteratively apply each operator in the neighborhoods list.
# if any operator finds an improvement, we restart the process from the first operator.
# if no operator finds an improvement, we move to the next operator.

def _local_search_first_improvement_vnd(routes, dist_matrix, neighborhoods, demands, capacity):
    k = 0
    best_cost = solution_cost(routes, dist_matrix)

    while k < len(neighborhoods):
        op, imp = neighborhoods[k]

        if op == "2opt":
            new_routes, new_cost = _local_search_2opt_wrapper(
                routes, dist_matrix, improvement=imp
            )

        elif op == "relocate":
            new_routes, new_cost = _relocate_inter_route(
                routes, dist_matrix, demands, capacity, improvement=imp
            )

        elif op == "swap":
            new_routes, new_cost = _swap_inter_route(
                routes, dist_matrix, demands, capacity, improvement=imp
            )

        else:
            k += 1
            continue

        if new_cost + 1e-9 < best_cost:
            routes = new_routes
            best_cost = new_cost
            k = 0
        else:
            k += 1

    return routes, best_cost


# in VND with best improvement, we iteratively apply each operator in the neighborhoods list.
# but here we check all operators in each iteration and select the best improvement found among them.
# if any operator finds an improvement, we restart the process from the first operator.
# if no operator finds an improvement, the process stops.

def _local_search_best_improvement_vnd(routes, dist_matrix, neighborhoods, demands, capacity):
    best_overall_routes = [r[:] for r in routes]
    best_overall_cost = solution_cost(best_overall_routes, dist_matrix)

    improved = True
    while improved:
        improved = False
        best_iteration_routes = None
        best_iteration_cost = best_overall_cost

        for op, imp in neighborhoods:
            if op == "2opt":
                new_routes, new_cost = _local_search_2opt_wrapper(
                    best_overall_routes, dist_matrix, improvement=imp
                )
            elif op == "relocate":
                new_routes, new_cost = _relocate_inter_route(
                    best_overall_routes, dist_matrix, demands, capacity, improvement=imp
                )
            elif op == "swap":
                new_routes, new_cost = _swap_inter_route(
                    best_overall_routes, dist_matrix, demands, capacity, improvement=imp
                )
            else:
                continue

            if new_cost + 1e-9 < best_iteration_cost:
                best_iteration_cost = new_cost
                best_iteration_routes = new_routes

        if best_iteration_routes is not None:
            best_overall_routes = best_iteration_routes
            best_overall_cost = best_iteration_cost
            improved = True

    return best_overall_routes, best_overall_cost


# here we are adding customers to the routes one by one until all customers are routed.
# in the process of adding customers, we use the rcl_builder function to create an rcl list based on the candidates.
def _construct_greedy_randomized_solution(demands, capacity, dist_matrix, rcl_builder, construction_strategy):
    if construction_strategy != "insertion":
        raise ValueError("Only 'insertion' construction is implemented.")

    depot = 0
    unrouted = set(i for i in demands if i != depot)
    routes = []

    while unrouted:
        route = [depot]
        load = 0

        while True:
            candidates = []
            cycle = route + [depot]

            for c in unrouted:
                if load + demands[c] <= capacity:
                    best_delta = float("inf")
                    best_pos = -1
                    for i in range(len(cycle) - 1):
                        delta = (
                            dist_matrix.get((cycle[i], c), float("inf"))
                            + dist_matrix.get((c, cycle[i + 1]), float("inf"))
                            - dist_matrix.get((cycle[i], cycle[i + 1]), float("inf"))
                        )
                        if delta < best_delta:
                            best_delta = delta
                            best_pos = i + 1
                    if best_pos != -1 and math.isfinite(best_delta):
                        candidates.append((c, best_delta, best_pos))

            if not candidates:
                route.append(depot)
                routes.append(route)
                break

            rcl = rcl_builder(candidates)
            c, _, pos = random.choice(rcl)
            route.insert(pos, c)
            load += demands[c]
            unrouted.remove(c)

    return routes


# this is the main grasp function that mixes all components together.
def grasp_solve_cvrp(demands, capacity, coords, config: GraspConfig):
    # if seed is provided, set the seed for reproducibility.
    if config.seed is not None:
        random.seed(config.seed)
    # compute the distance matrix here.
    dist_matrix = compute_distance_matrix(coords)
    # defining variables to keep track of best solution found and also timing info and history.
    best_routes, best_cost, time_to_best = None, float("inf"), 0.0
    history = []
    # check which type of rcl to use based on input config.
    if config.rcl_type == "Threshold":
        alpha = float(config.rcl_param)
        rcl_builder = lambda c: _build_rcl_by_alpha(c, alpha)
    else:
        rcl_builder = lambda c: _build_rcl_by_size(c, int(config.rcl_param))

    start_time = time.perf_counter()
    it = 0
    # main loop and also time limit or iteration limit check.
    # in the implemented version, we can either have a time limit or iteration limit.
    while True:
        elapsed = time.perf_counter() - start_time
        if config.time_limit is not None and elapsed >= config.time_limit:
            break
        if config.time_limit is None and it >= config.iterations:
            break

        it += 1
        # timing the construction phase
        t0 = time.perf_counter()
        # constructing a greedy randomized solution.
        routes = _construct_greedy_randomized_solution(
            demands, capacity, dist_matrix, rcl_builder, config.construction_strategy
        )
        # set the time after construction
        t1 = time.perf_counter()
        
        # applying local search
        if config.local_search_type == "vnd_first_improvement":
            routes, cost = _local_search_first_improvement_vnd(
                routes, dist_matrix, config.neighborhoods, demands, capacity
            )
        elif config.local_search_type == "vnd_best_improvement":
            routes, cost = _local_search_best_improvement_vnd(
                routes, dist_matrix, config.neighborhoods, demands, capacity
            )
        elif config.local_search_type == "sequential":
            routes, cost = _local_search_sequential(
                routes, dist_matrix, config.neighborhoods, demands, capacity
            )
        else:
            raise ValueError(f"Unknown local_search_type: {config.local_search_type}")
        # set the time after local search
        t2 = time.perf_counter()
        # check if we have improved the best solution found till now.
        if cost < best_cost:
            best_cost = cost
            best_routes = routes
            time_to_best = time.perf_counter() - start_time
        # append the info about iteration to history. iteration number, 
        # best cost so far, cost, time for construction, time for local search, time to best.
        history.append((it, best_cost, cost, t1 - t0, t2 - t1, time_to_best))

    return best_routes or [], best_cost, history
