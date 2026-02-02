from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Root directory containing benchmark sets. Adjust as needed in your runtime.
BENCHMARK_ROOT_DIR: Path = Path(__file__).resolve().parents[1] / "benchmarks"

# ---------------------------------------------------------------------------
# Helper: Robust mapping/text parser
# ---------------------------------------------------------------------------

def _parse_mapping_text(text: str, label: str) -> Dict[Any, Any]:
    """
    Robustly parse a text representation of a mapping (either Python literal or JSON).

    Args:
        text: Input text expected to represent a dict/object.
        label: Human-readable label for error messages (e.g., 'demands').

    Returns:
        A Python dict parsed from the text.

    Raises:
        ValueError: If the text cannot be parsed into a dict.
    """
    s = (text or "").strip()
    if not s:
        return {}

    # Try Python literal first (supports single quotes, tuples, etc.)
    try:
        value = ast.literal_eval(s)
        if isinstance(value, dict):
            return value
        raise ValueError(f"{label} must be a mapping/dict literal")
    except Exception:
        # Fallback to JSON
        try:
            value = json.loads(s)
            if isinstance(value, dict):
                return value
            raise ValueError(f"{label} must be a JSON object (dict)")
        except Exception as exc:
            raise ValueError(f"Could not parse {label}: {exc}") from exc


# ---------------------------------------------------------------------------
# Mapping-specific parsers
# ---------------------------------------------------------------------------

def parse_demands(demand_str: str) -> Dict[int, int]:
    """Parse demand string into a dictionary."""
    raw = _parse_mapping_text(demand_str, "demands")
    demands: Dict[int, int] = {}
    for k, v in raw.items():
        try:
            demands[int(k)] = int(v)
        except Exception as exc:
            raise ValueError(f"Invalid demand entry for key {k}: {exc}") from exc
    return demands


def parse_coords(coords_str: str) -> Dict[Tuple[int, int], float]:
    """Parse coordinate string into a distance matrix."""
    raw = _parse_mapping_text(coords_str, "coordinates")
    coords: Dict[Tuple[int, int], float] = {}

    for k, v in raw.items():
        try:
            node = int(k)

            if isinstance(v, dict):
                x = float(v.get("x"))
                y = float(v.get("y"))
            else:
                # Assume a sequence (list/tuple) or a string that can be split
                if isinstance(v, str):
                    # Accept formats like "(1,2)" or "1,2"
                    v_items = re.findall(r"-?\d+\.?\d*", v)
                    if len(v_items) >= 2:
                        x, y = float(v_items[0]), float(v_items[1])
                    else:
                        raise ValueError("String coordinate must contain two numbers")
                else:
                    x, y = v  # type: ignore
                    x = float(x)
                    y = float(y)

            coords[node] = (x, y)
        except Exception as exc:
            raise ValueError(f"Invalid coordinate for node {k}: {exc}") from exc

    return coords


# ---------------------------------------------------------------------------
# VRP (TSPLIB-like) parser
# ---------------------------------------------------------------------------

def parse_vrp(text: str) -> Dict[str, Any]:
    """
    Lightweight parser for VRP-like files (CMT-style). It extracts:
      - Q: capacity (int or None)
      - coords: dict[node] -> (x,y)
      - demands: dict[node] -> demand
      - depot: depot index (if specified)
      - remapped: whether indices were shifted to start at 0

    The parser is forgiving: non-critical parse errors are ignored so a
    partial parse can still be returned.
    """
    lines = [ln.rstrip() for ln in (text or "").splitlines()]

    section: Optional[str] = None
    coords: Dict[int, Tuple[float, float]] = {}
    demands: Dict[int, int] = {}
    capacity: Optional[int] = None
    depot: Optional[int] = None

    for ln in lines:
        if not ln:
            continue
        up = ln.strip().upper()

        # Capacity lines can be written in multiple formats
        if up.startswith("CAPACITY") or ("CAPACITY" in up and (":" in ln or "=" in ln)):
            # Try formats like: "CAPACITY: 200" or "CAPACITY 200" or "CAPACITY=200"
            try:
                if ":" in ln or "=" in ln:
                    parts = ln.replace("=", ":").split(":", 1)
                    capacity = int(parts[1].strip())
                else:
                    parts = ln.split()
                    capacity = int(parts[-1])
            except Exception:
                # Silently ignore parsing errors for capacity
                pass
            continue

        if up.startswith("NODE_COORD_SECTION"):
            section = "coords"
            continue
        if up.startswith("DEMAND_SECTION"):
            section = "demands"
            continue
        if up.startswith("DEPOT_SECTION"):
            section = "depot"
            continue
        if up.startswith("EOF"):
            break

        # Parse based on current section
        if section == "coords":
            parts = ln.split()
            if len(parts) >= 3:
                try:
                    idx = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    coords[idx] = (x, y)
                except Exception:
                    # Ignore malformed coord line
                    pass
            continue

        if section == "demands":
            parts = ln.split()
            if len(parts) >= 2:
                try:
                    idx = int(parts[0])
                    d = int(float(parts[1]))
                    demands[idx] = d
                except Exception:
                    # Ignore malformed demand line
                    pass
            continue

        if section == "depot":
            try:
                v = int(ln.split()[0])
                if v == -1:
                    section = None
                else:
                    depot = v
            except Exception:
                pass
            continue

    # If nodes are 1-based (no 0 key), remap them to 0-based indices
    remapped = False
    if coords and (0 not in coords and 0 not in demands):
        min_idx = min(coords.keys())
        if min_idx > 0:
            coords = {k - min_idx: v for k, v in coords.items()}
            demands = {k - min_idx: v for k, v in demands.items()}
            if depot is not None:
                try:
                    depot = int(depot) - min_idx
                except Exception:
                    depot = None
            remapped = True

    return {
        "Q": capacity,
        "coords": coords,
        "demands": demands,
        "depot": depot,
        "remapped": remapped,
    }


# ---------------------------------------------------------------------------
# SOL parser
# ---------------------------------------------------------------------------

def parse_sol(text: str, remap_offset: int = 0, add_depot: bool = True) -> Tuple[List[List[int]], Optional[float]]:
    """
    Parse a solver .sol file (CMT-style). Returns (routes, cost).

    Parameters:
      - remap_offset: If routes in .sol use 1-based indices, set remap_offset=1
                      to subtract that offset.
      - add_depot: If True, ensure each route starts and ends with depot (0).
    """
    routes: List[List[int]] = []
    cost: Optional[float] = None

    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue

        low = ln.lower()

        # Extract cost lines like: "Cost 1234.56"
        if low.startswith("cost"):
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", ln)
            if matches:
                try:
                    cost = float(matches[-1])
                except ValueError:
                    pass
            continue

        # Route lines usually contain a colon ("Route #1:") or start with "Route"
        if not (low.startswith("route") or ":" in ln):
            continue

        if ":" in ln:
            _, after = ln.split(":", 1)
            seq_txt = after.strip()
        else:
            seq_txt = ln

        nums = re.findall(r"\d+", seq_txt)
        if not nums:
            continue

        parsed = [int(n) for n in nums]

        if remap_offset:
            parsed = [n - remap_offset for n in parsed]

        if add_depot:
            if not parsed or parsed[0] != 0:
                parsed = [0] + parsed
            if parsed[-1] != 0:
                parsed = parsed + [0]

        routes.append(parsed)

    return routes, cost


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_instance_data(
    demands_raw: Dict[Any, Any], coords_raw: Dict[Any, Any]
) -> Tuple[Dict[int, int], Dict[int, Tuple[float, float]]]:
    """
    Normalize instance dicts:
      1. Convert keys to ints.
      2. Shift indices so the minimum becomes 0 (depot).

    Returns:
        (demands, coords) where demands: Dict[int,int], coords: Dict[int,(x,y)].

    Raises:
        ValueError: If coords are missing or normalization fails.
    """
    # Convert keys and values to proper types
    demands: Dict[int, int] = {int(k): int(v) for k, v in demands_raw.items()}
    coords: Dict[int, Tuple[float, float]] = {int(k): tuple(v) for k, v in coords_raw.items()}

    if not coords:
        raise ValueError("Invalid instance: no coordinates provided")

    min_node = min(coords.keys())

    if min_node != 0:
        # Shift keys so that the minimum becomes 0
        shift = min_node
        coords = {k - shift: v for k, v in coords.items()}

        # Shift demands: only keep demands that correspond to coordinates
        shifted_demands: Dict[int, int] = {}
        for k, v in demands.items():
            new_k = k - shift
            if new_k in coords:
                shifted_demands[new_k] = v
        demands = shifted_demands

    # Final validation
    if 0 not in coords:
        raise ValueError(f"Normalization failed: expected depot node 0 but it is missing")

    return demands, coords


# ---------------------------------------------------------------------------
# Benchmark utilities
# ---------------------------------------------------------------------------

def _extract_benchmark_set_name(instance_name: str) -> str:
    """
    Extract an alphabetical prefix used as the benchmark subfolder.

    Examples:
      - "CMT1" -> "CMT"
      - "A-n32-k5" -> "A-N32-K5" (returns whole prefix upper-cased)

    This heuristic strips trailing digits from the initial token.
    """
    if not instance_name:
        return ""

    token = re.match(r"([A-Za-z0-9\-\_]+)", instance_name)
    if not token:
        return instance_name.upper()

    candidate = token.group(1)
    # Remove trailing digits commonly used in CMT-style names
    stripped = re.sub(r"\d+$", "", candidate)
    return stripped.strip("-_").upper() if stripped else candidate.upper()


def get_benchmark_solution_cost(
    instance_name: str, remap_offset: int = 0
) -> Optional[Tuple[float, int, List[List[int]]]]:
    """
    Given an instance name (e.g. "CMT1"), locate the corresponding .sol file
    under BENCHMARK_ROOT_DIR/<SET>/<INSTANCE>.sol and return a tuple:
        (best_known_cost, vehicle_count, best_known_routes)

    If the .sol file is missing or cannot be parsed, returns None.
    """
    if not instance_name or instance_name == "Manual_Input":
        return None, None, None

    clean_name = instance_name.replace(".vrp", "").replace(".VRP", "").strip()
    set_name = _extract_benchmark_set_name(clean_name)
    sol_path = Path(BENCHMARK_ROOT_DIR) / set_name / f"{clean_name}.sol"

    if not sol_path.exists():
        # File not found
        return None, None, None

    try:
        sol_text = sol_path.read_text(encoding="utf-8", errors="ignore")
        routes, cost = parse_sol(sol_text, remap_offset=remap_offset, add_depot=True)
        vehicle_count = len(routes)
        return (cost, vehicle_count, routes)
    except Exception:
        return None, None, None
