# streamlit_app/components/__init__.py

# Re-export parsing logic from shared layer
from cvrp_streamlit.inputs import (
    parse_demands,
    parse_coords,
    parse_vrp, 
    parse_sol,
)
from cvrp_streamlit.feasibility import (
    check_cvrp_feasibility,
    check_customers_served_once,
    check_capacity_per_route,
    check_route_structure_basic,
    check_structure_from_edges,
)

# UI Components (keep these local)
from .visualization import render_network
from .errors import render_exception

__all__ = [
    "parse_demands",
    "parse_coords",
    "parse_vrp",
    "parse_sol",
    "render_network",
    "render_exception",
    "check_cvrp_feasibility",
    "check_customers_served_once",
    "check_capacity_per_route",
    "check_route_structure_basic",
    "check_structure_from_edges"
]