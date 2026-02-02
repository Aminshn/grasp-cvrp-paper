import streamlit as st
from pathlib import Path
from typing import Optional, List
import re

# --- Shared Logic Imports ---
from cvrp_streamlit.inputs import (
    parse_vrp,
    parse_sol,
    parse_demands,
    parse_coords,
    _extract_benchmark_set_name,
)

# --- Default Values ---
DEFAULT_Q = 20
DEFAULT_DEMANDS = "{" + "1:4, 2:6, 3:7, 4:3, 5:5, 6:2" + "}"
DEFAULT_COORDS = (
    "{"
    + "0:(0,0), 1:(1,0), 2:(2,0), 3:(0,1), 4:(2,1), 5:(1,2), 6:(3,0)"
    + "}"
)

# Root directory containing benchmark sets
BENCHMARK_ROOT_DIR = Path(__file__).resolve().parents[1] / "benchmarks"


def render_sidebar():
    """
    Render the full sidebar for the CVRP Streamlit application.

    Responsibilities:
    - Load and list all available VRP benchmark instances across subfolders.
    - Render input controls for selecting a benchmark instance.
    - Parse VRP file if selected, pre-fill demands and coordinates.
    - Detect user modifications (Dirty Flag) and generate a proper instance name.

    Returns:
        Q (int): Vehicle capacity
        demands_text (str): Demand dictionary (as text)
        coords_text (str): Coordinates dictionary (as text)
        allow_non_depot (bool): Always true for now
        run (bool): Whether user submitted the form
        final_instance_name (str): Derived name including modification flags
    """

    # --- Initialization ---
    parsed = None
    selected_name = None
    set_name = "CMT"

    original_demands_text = DEFAULT_DEMANDS
    original_coords_text = DEFAULT_COORDS

    all_instances = []

    with st.sidebar:
        st.header("Input")

        # -------------------------------------------------
        # 1) Collect All Benchmark Instances
        # -------------------------------------------------
        if BENCHMARK_ROOT_DIR.exists():
            for subdir in BENCHMARK_ROOT_DIR.iterdir():
                if subdir.is_dir():
                    for vrp_file in subdir.glob("*.vrp"):
                        all_instances.append(vrp_file.stem)

        all_instances = sorted(set(all_instances))

        if not all_instances:
            st.warning("No benchmark files found under 'benchmarks/'.")
        else:
            default_index = all_instances.index("CMT1") if "CMT1" in all_instances else 0

            selected_name = st.selectbox(
                "Select benchmark instance", all_instances, index=default_index
            )

            # -------------------------------------------------
            # 2) Parse Selected VRP File
            # -------------------------------------------------
            if selected_name:
                set_name = _extract_benchmark_set_name(selected_name)
                selected_instance_name = selected_name

                vrp_path = BENCHMARK_ROOT_DIR / set_name / f"{selected_name}.vrp"

                try:
                    content = vrp_path.read_text(encoding="utf-8", errors="ignore")
                    parsed = parse_vrp(content)
                except Exception as e:
                    st.error(f"Failed to parse VRP file {vrp_path.name}: {e}")
                    parsed = None

        # -------------------------------------------------
        # 3) Main Input Form
        # -------------------------------------------------
        with st.form(key="input_form"):

            # Vehicle capacity (Q)
            q_val = int(parsed.get("Q")) if parsed and parsed.get("Q") else DEFAULT_Q
            Q = st.number_input("Vehicle capacity Q", min_value=1, value=q_val)

            # ----------------------
            # Demands text area
            # ----------------------
            demands_text_value = DEFAULT_DEMANDS
            if parsed and parsed.get("demands"):
                demands = parsed.get("demands")
                try:
                    demands_text_value = (
                        "{"
                        + ", ".join(
                            f"{int(k)}:{int(v)}" for k, v in sorted(demands.items())
                        )
                        + "}"
                    )
                except Exception:
                    pass

            original_demands_text = demands_text_value
            demands_text = st.text_area(
                "Demands (Python dict or JSON)",
                value=demands_text_value,
                height=120,
            )

            # ----------------------
            # Coordinates text area
            # ----------------------
            coords_text_value = DEFAULT_COORDS
            if parsed and parsed.get("coords"):
                coords = parsed.get("coords")
                try:
                    coords_text_value = (
                        "{"
                        + ", ".join(
                            f"{k}:(%s,%s)" % (coords[k][0], coords[k][1])
                            for k in sorted(coords.keys())
                        )
                        + "}"
                    )
                except Exception:
                    pass

            original_coords_text = coords_text_value
            coords_text = st.text_area(
                "Node coordinates (Python dict: node:(x,y))",
                value=coords_text_value,
                height=120,
            )

            # Route input is intentionally removed
            allow_non_depot = True
            run = st.form_submit_button("Execute")

    # -------------------------------------------------
    # 4) Dirty Flag Detection
    # -------------------------------------------------
    is_demands_modified = demands_text.strip() != original_demands_text.strip()
    is_coords_modified = coords_text.strip() != original_coords_text.strip()
    is_capacity_modified = Q != q_val

    if is_demands_modified or is_coords_modified or is_capacity_modified:
        final_instance_name = f"Manual_Modified_{selected_name}"
    else:
        final_instance_name = selected_name

    # -------------------------------------------------
    # 5) Final Return
    # -------------------------------------------------
    return Q, demands_text, coords_text, allow_non_depot, run, final_instance_name