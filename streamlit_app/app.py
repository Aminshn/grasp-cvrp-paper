# app.py
import sys
import os
import json
import random
import datetime
from pathlib import Path
from itertools import product
from typing import List

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Internal imports
from streamlit_app.components import render_network
from cvrp_streamlit.inputs import (
    parse_demands,
    parse_vrp,
    BENCHMARK_ROOT_DIR,
    _extract_benchmark_set_name,
)

# =============================================================
# CONSTANTS / HELPERS
# =============================================================

DEFAULT_IMP = {"2opt": "first", "relocate": "best", "swap": "best"}
ALL_OPS = ["2opt", "relocate", "swap"]
ALL_IMP = ["first", "best"]

LS_MAP = {
    "Sequential": "sequential",
    "VND First Improvement": "vnd_first_improvement",
    "VND Best Improvement": "vnd_best_improvement",
}

# Local-search scenarios now store (operator, improvement) pairs (JSON list-of-pairs)


def _generate_unique_seeds(count: int, used=None) -> List[int]:
    """Generate a list of unique seeds, avoiding those already used."""
    used_set = set(used or [])
    seeds: List[int] = []
    while len(seeds) < count:
        cand = random.randint(1, 2_000_000_000)
        if cand in used_set or cand in seeds:
            continue
        seeds.append(cand)
    return seeds


def _parse_k_values(k_str: str, customer_count: int) -> List[int]:
    """
    Parse K cardinality values from user input.
    Supports:
      - Absolute integers: "2" -> 2
      - Relative expressions: "0.1n" -> 0.1 * customer_count
    
    Examples:
      - "2, 0.1n, 0.2n" with customer_count=50 -> [2, 5, 10]
      - "1, 3, 5" with customer_count=100 -> [1, 3, 5]
    
    Returns list of integers, rounded from float expressions.
    """
    k_values: List[int] = []
    
    for item in k_str.split(","):
        item = item.strip()
        if not item:
            continue
        
        if item.endswith("n") or item.endswith("N"):
            # Relative expression like "0.1n"
            try:
                coeff = float(item[:-1])
                k_val = int(round(coeff * customer_count))
                if k_val > 0:
                    k_values.append(k_val)
            except ValueError:
                # Skip invalid expressions
                pass
        else:
            # Absolute integer
            try:
                k_val = int(item)
                if k_val > 0:
                    k_values.append(k_val)
            except ValueError:
                # Skip invalid expressions
                pass
    
    return k_values


def _safe_json_loads(x):
    if x is None:
        return None
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None
    return None


def normalize_ls_ops(ls_ops_value):
    """
    Normalize local_search_operators into canonical list-of-pairs:
      [["2opt","first"],["swap","best"],...]

    Accepts:
      - list of strings: ["2opt","swap"]
      - list of pairs:   [["2opt","first"],["swap","best"]]
      - stringified JSON of either
      - CSV string: "2opt, swap"
    """
    parsed = _safe_json_loads(ls_ops_value)

    # Stringified non-JSON fallback
    if parsed is None and isinstance(ls_ops_value, str):
        s = ls_ops_value.strip()
        if "," in s:
            ops = [o.strip() for o in s.split(",") if o.strip()]
            return [[op, DEFAULT_IMP.get(op, "best")] for op in ops if op in ALL_OPS]
        return []

    if parsed is None:
        return []

    if isinstance(parsed, list):
        if not parsed:
            return []
        # list-of-pairs
        if isinstance(parsed[0], (list, tuple)) and len(parsed[0]) == 2:
            out = []
            for op, imp in parsed:
                if op in ALL_OPS and imp in ALL_IMP:
                    out.append([op, imp])
            return out
        # list-of-strings
        if isinstance(parsed[0], str):
            out = []
            for op in parsed:
                if op in ALL_OPS:
                    out.append([op, DEFAULT_IMP.get(op, "best")])
            return out

    return []


def ls_ops_key(ls_ops_value) -> str:
    """Stable key for grouping/legend."""
    pairs = normalize_ls_ops(ls_ops_value)
    if not pairs:
        return "none"
    return "|".join([f"{op}:{imp}" for op, imp in pairs])


def ls_ops_pretty(ls_ops_value) -> str:
    pairs = normalize_ls_ops(ls_ops_value)
    if not pairs:
        return "None"
    return ", ".join([f"{op} ({imp})" for op, imp in pairs])


def count_customers_from_demands_json(d_text: str):
    try:
        demands = parse_demands(d_text)
        return len([k for k in demands.keys() if int(k) != 0])
    except Exception:
        return None


def termination_label_from_row(row):
    try:
        is_time = bool(row.get("is_time_based")) if hasattr(row, "get") else False
        time_limit = row.get("time_limit") if hasattr(row, "get") else None
        iterations = row.get("iterations") if hasattr(row, "get") else None

        if is_time:
            if pd.notna(time_limit):
                return f"Time {float(time_limit):.2f}s"
            return "Time-based"

        if pd.notna(iterations):
            return f"{int(iterations)} iters"

        return "Iteration-based"
    except Exception:
        return "N/A"


# =============================================================
# STREAMLIT CONFIG
# =============================================================

st.set_page_config(page_title="GRASP-CVRP Solver", layout="wide", page_icon="🚚")
st.title("🚚 GRASP-CVRP Solver & Parameter Analysis")

# =============================================================
# DATABASE & SECURITY
# =============================================================

DB_CONN = os.getenv("DB_CONNECTION_STRING")
APP_ACCESS_CODE = os.getenv("APP_ACCESS_CODE", "admin")

engine = create_engine(DB_CONN) if DB_CONN else None
if not engine:
    st.error("Database connection string is missing. Check environment variables.")
    st.stop()

# Sidebar security is global so it works in every tab
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔐 Security")
user_code = st.sidebar.text_input("Enter Access Code to Submit", type="password")

# =============================================================
# TABS
# =============================================================

tab_batch, tab_monitor, tab_analysis = st.tabs(
    ["Batch Experiments", "Job Monitor", "Sensitivity Analysis"]
)

# =============================================================
# TAB 1 - BATCH EXPERIMENTS
# =============================================================

with tab_batch:
    st.header("🚀 Batch Experiment Submission")
    st.markdown("Submit multiple jobs at once for sensitivity analysis.")

    col_b1, col_b2 = st.columns(2)
    batch_name = col_b1.text_input("Batch Name (Optional)", placeholder="e.g., Alpha/K Sensitivity")
    batch_desc = col_b2.text_input("Description (Optional)", placeholder="Testing different RCL parameters")

    st.subheader("1) Select Benchmarks")
    all_instances = []
    if BENCHMARK_ROOT_DIR.exists():
        for subdir in BENCHMARK_ROOT_DIR.iterdir():
            if subdir.is_dir():
                for vrp_file in subdir.glob("*.vrp"):
                    all_instances.append(vrp_file.stem)
    all_instances = sorted(set(all_instances))

    selected_instances = st.multiselect(
        "Choose Instances",
        all_instances,
        default=all_instances[:1] if all_instances else None,
    )

    st.subheader("2) Parameter Grid")
    batch_rcl_strat = st.multiselect(
        "RCL Strategies",
        options=["Threshold", "Cardinality"],
        default=["Threshold"],
        key="batch_rcl",
        help="Choose one or both; parameters below adapt to your selection.",
    )

    alpha_input = ""
    k_input = ""
    col_p1, col_p2 = st.columns(2)
    if "Threshold" in batch_rcl_strat:
        alpha_input = col_p1.text_input(
            "Alpha values for Threshold (comma separated)",
            value="0.1, 0.3, 0.5",
            help="Used when RCL strategy 'Threshold' is selected.",
        )
    if "Cardinality" in batch_rcl_strat:
        k_input = col_p2.text_input(
            "K values for Cardinality (comma separated)",
            value="3, 5, 10",
            help="Used when RCL strategy 'Cardinality' is selected.",
        )

    col_term1, col_term2 = st.columns(2)
    is_time_based = col_term1.checkbox(
        "Use Time-Based Termination",
        value=False,
        help="Stop each run when its scaled time limit is reached instead of by iteration count.",
    )
    base_time = None
    if is_time_based:
        base_time = col_term2.number_input(
            "Base time for smallest instance (s)",
            min_value=0.1,
            value=7.0,
            step=0.5,
            help="Time limit applied to the smallest instance; larger instances scale linearly by customer count.",
        )

    col_p3, col_p4 = st.columns(2)
    iters_input = col_p3.text_input(
        "Iterations (comma separated)",
        value="100, 500",
        disabled=is_time_based,
        help="Ignored when time-based termination is enabled.",
    )
    repetitions = col_p4.number_input("Repetitions per Config", min_value=1, value=5)

    st.subheader("3) Fixed Parameters")
    col_f2, col_f3 = st.columns(2)

    batch_ls_ops = col_f2.multiselect(
        "Local Search Operators",
        options=ALL_OPS,
        default=["2opt", "relocate", "swap"],
        key="batch_ls_ops",
        help="Pick operators (leave empty to run construction-only / no local search).",
    )

    batch_ls_strategies = col_f3.multiselect(
        "Local Search Strategies",
        options=list(LS_MAP.keys()),
        default=["Sequential"],
        key="batch_ls_strat",
    )

    # Operator-level improvement selection (multi-select per operator)
    neighborhoods_combos = [[]]
    if batch_ls_ops:
        st.markdown("**Improvement Type per Operator**")
        imp_cols = st.columns(len(batch_ls_ops))
        op_imps = []
        for i, op in enumerate(batch_ls_ops):
            with imp_cols[i]:
                imps = st.multiselect(
                    f"{op}",
                    options=ALL_IMP,
                    default=ALL_IMP,
                    key=f"batch_imp_{op}",
                    help="Select one or both improvement types; all chosen combinations will be submitted.",
                )
                if not imps:
                    imps = [DEFAULT_IMP.get(op, "best")]
                op_imps.append((op, imps))

        # Build all combinations across operators
        if op_imps:
            neighborhoods_combos = [
                [[op, imp] for op, imp in combo]
                for combo in product(*[[ (op, imp) for imp in imps ] for op, imps in op_imps])
            ]

    st.markdown("---")
    submit_batch = st.button("🚀 Submit Batch", type="primary")

    if submit_batch:
        if user_code != APP_ACCESS_CODE:
            st.error("⛔ Unauthorized! Invalid Access Code (Check Sidebar).")
        elif not selected_instances:
            st.warning("Please select at least one benchmark instance.")
        elif not batch_rcl_strat:
            st.warning("Please select at least one RCL strategy.")
        elif not batch_ls_strategies:
            st.warning("Please select at least one Local Search strategy.")
        else:
            try:
                if is_time_based:
                    if base_time is None or base_time <= 0:
                        st.error("Please provide a positive base time for time-based termination.")
                        st.stop()
                    iter_list = [None]
                else:
                    iter_list = [int(x.strip()) for x in iters_input.split(",") if x.strip()]

                alpha_list = []
                if "Threshold" in batch_rcl_strat:
                    alpha_list = [float(x.strip()) for x in alpha_input.split(",") if x.strip()]

                if "Threshold" in batch_rcl_strat and not alpha_list:
                    st.error("You selected 'Threshold' but did not provide alpha values.")
                    st.stop()

                if "Cardinality" in batch_rcl_strat and not k_input.strip():
                    st.error("You selected 'Cardinality' but did not provide K values.")
                    st.stop()

                if not iter_list and not is_time_based:
                    st.error("Please provide valid values for Iterations.")
                    st.stop()

                strategy_param_pairs_template = []
                if "Threshold" in batch_rcl_strat:
                    strategy_param_pairs_template += [("Threshold", a) for a in alpha_list]
                
                estimated_k_count = 1
                if "Cardinality" in batch_rcl_strat and selected_instances:
                    first_inst_name = selected_instances[0]
                    first_set_name = _extract_benchmark_set_name(first_inst_name)
                    first_vrp_path = BENCHMARK_ROOT_DIR / first_set_name / f"{first_inst_name}.vrp"
                    try:
                        first_content = first_vrp_path.read_text(encoding="utf-8", errors="ignore")
                        first_parsed = parse_vrp(first_content)
                        first_customer_count = len([k for k in first_parsed.get("demands", {}).keys() if int(k) != 0])
                        estimated_k_count = len(_parse_k_values(k_input, first_customer_count))
                    except Exception:
                        estimated_k_count = len(k_input.split(","))
                
                estimated_strat_pairs = len(strategy_param_pairs_template) + estimated_k_count
                total_jobs = (
                    len(selected_instances)
                    * estimated_strat_pairs
                    * len(iter_list)
                    * len(neighborhoods_combos)
                    * len(batch_ls_strategies)
                    * repetitions
                )
                st.info(f"Preparing to submit ~{total_jobs} jobs (exact count varies by instance K values)...")

                # Pre-parse instances to reuse data and compute scaled time limits
                instances_payload = []
                min_customers = None

                for inst_name in selected_instances:
                    set_name = _extract_benchmark_set_name(inst_name)
                    vrp_path = BENCHMARK_ROOT_DIR / set_name / f"{inst_name}.vrp"

                    content = vrp_path.read_text(encoding="utf-8", errors="ignore")
                    parsed = parse_vrp(content)

                    inst_capacity = parsed.get("Q", 100) or 100
                    inst_demands_dict = parsed.get("demands", {}) or {}
                    inst_coords_dict = parsed.get("coords", {}) or {}

                    customer_count = len([k for k in inst_demands_dict.keys() if int(k) != 0])
                    if min_customers is None:
                        min_customers = customer_count
                    else:
                        min_customers = min(min_customers, customer_count)

                    instances_payload.append(
                        {
                            "name": inst_name,
                            "capacity": inst_capacity,
                            "demands_json": json.dumps(inst_demands_dict),
                            "coords_json": json.dumps(inst_coords_dict),
                            "customer_count": customer_count,
                        }
                    )

                if is_time_based and (min_customers is None or min_customers <= 0):
                    st.error("Could not determine customer counts to scale time limits. Check benchmark data.")
                    st.stop()

                with engine.begin() as conn:
                    # Check if batch with this name already exists
                    existing_batch = conn.execute(
                        text("SELECT batch_id FROM Batches WHERE name = :name"),
                        {"name": batch_name},
                    ).fetchone()

                    if existing_batch:
                        batch_id = existing_batch[0]
                        st.info(f"Batch '{batch_name}' already exists. Adding jobs to existing batch #{batch_id}.")
                    else:
                        res = conn.execute(
                            text(
                                """
                                INSERT INTO Batches (name, description)
                                OUTPUT INSERTED.batch_id
                                VALUES (:name, :desc)
                                """
                            ),
                            {"name": batch_name, "desc": batch_desc},
                        )
                        batch_id = res.fetchone()[0]

                    progress_bar = st.progress(0.0)
                    processed_count = 0

                    for inst_payload in instances_payload:
                        inst_name = inst_payload["name"]
                        inst_capacity = inst_payload["capacity"]
                        inst_demands = inst_payload["demands_json"]
                        inst_coords = inst_payload["coords_json"]
                        customer_count = inst_payload.get("customer_count") or 0

                        # Determine seeds per run for this instance within the batch
                        existing_seed_rows = conn.execute(
                            text(
                                """
                                SELECT seed
                                FROM (
                                    SELECT seed, MIN(job_id) AS min_jid
                                    FROM Jobs
                                    WHERE batch_id = :bid AND instance_name = :inst AND seed IS NOT NULL
                                    GROUP BY seed
                                ) s
                                ORDER BY s.min_jid
                                """
                            ),
                            {"bid": batch_id, "inst": inst_name},
                        ).fetchall()

                        existing_seeds = [row[0] for row in existing_seed_rows]
                        seeds_by_run = existing_seeds[: repetitions]

                        if len(seeds_by_run) < repetitions:
                            needed = repetitions - len(seeds_by_run)
                            seeds_by_run.extend(
                                _generate_unique_seeds(needed, used=existing_seeds)
                            )

                        strategy_param_pairs = list(strategy_param_pairs_template)
                        if "Cardinality" in batch_rcl_strat:
                            k_list = _parse_k_values(k_input, customer_count)
                            if k_list:
                                strategy_param_pairs += [("Cardinality", k) for k in k_list]

                        for (rcl_s, rcl_param_val) in strategy_param_pairs:
                            for iters in iter_list:
                                for neighborhoods in neighborhoods_combos:
                                    for ls_strat_name in batch_ls_strategies:
                                        ls_strat_value = LS_MAP[ls_strat_name]

                                        for run_idx in range(repetitions):
                                            job_seed = seeds_by_run[run_idx]

                                            job_iterations = iters if not is_time_based else None
                                            time_limit_val = None
                                            if is_time_based and min_customers:
                                                time_limit_val = float(base_time) * (customer_count / min_customers)

                                            conn.execute(
                                                text(
                                                    """
                                                    INSERT INTO Jobs (
                                                        batch_id, instance_name, capacity, alpha, seed, iterations,
                                                        is_time_based, base_time, time_limit, customer_count, min_customer_count,
                                                        rcl_strategy, local_search_operators, construction_strategy,
                                                        local_search_strategy, demands_data, coords_data, status
                                                    ) VALUES (
                                                        :bid, :inst, :cap, :alpha, :seed, :iters,
                                                        :is_time, :base_time, :tlimit, :cust_count, :min_cust,
                                                        :strat, :ls, :constr,
                                                        :ls_strat, :demands, :coords, 'PENDING'
                                                    )
                                                    """
                                                ),
                                                {
                                                    "bid": batch_id,
                                                    "inst": inst_name,
                                                    "cap": inst_capacity,
                                                    "alpha": rcl_param_val,
                                                    "seed": job_seed,
                                                    "iters": job_iterations,
                                                    "is_time": is_time_based,
                                                    "base_time": float(base_time) if is_time_based else None,
                                                    "tlimit": time_limit_val,
                                                    "cust_count": customer_count,
                                                    "min_cust": min_customers,
                                                    "strat": rcl_s,
                                                    # store list-of-pairs JSON
                                                    "ls": json.dumps(neighborhoods),
                                                    "constr": "insertion",
                                                    "ls_strat": ls_strat_value,
                                                    "demands": inst_demands,
                                                    "coords": inst_coords,
                                                },
                                            )

                                            processed_count += 1
                                            progress_bar.progress(processed_count / total_jobs)

                st.success(f"✅ Batch #{batch_id} submitted successfully with {processed_count} jobs!")

            except Exception as e:
                st.error(f"An error occurred during batch submission: {e}")

# =============================================================
# TAB 2 - MONITOR JOBS
# =============================================================

with tab_monitor:
    try:
        with engine.connect() as conn:
            summary_query = text(
                """
                SELECT
                    j.job_id,
                    j.status,
                    j.instance_name,
                    j.alpha,
                    j.iterations,
                    j.is_time_based,
                    j.time_limit,
                    j.rcl_strategy,
                    j.local_search_operators,
                    j.local_search_strategy,
                    j.construction_strategy,
                    r.gap_to_bks,
                    r.solve_time,
                    j.error_message,
                    j.created_at
                FROM Jobs j
                LEFT JOIN Results r ON j.job_id = r.job_id
                ORDER BY j.job_id DESC
                """
            )
            df_jobs = pd.read_sql(summary_query, conn)

        st.subheader("Job Monitor")
        st.caption("Statuses are typically PENDING → PROCESSING → COMPLETED (or FAILED).")

        if st.button("🔄 Refresh Status", key="refresh_monitor"):
            st.rerun()

        if df_jobs.empty:
            st.info("No jobs found in the database yet.")
        else:
            df_display = df_jobs.copy()
            df_display["local_search_operators_pretty"] = df_display["local_search_operators"].apply(ls_ops_pretty)
            df_display["termination"] = df_display.apply(termination_label_from_row, axis=1)

            column_config_job_table = {
                "job_id": st.column_config.NumberColumn("Job ID"),
                "status": st.column_config.TextColumn("Status"),
                "instance_name": st.column_config.TextColumn("Instance"),
                "alpha": st.column_config.NumberColumn("RCL Param (Alpha/K)", format="%.2f"),
                "termination": st.column_config.TextColumn("Termination"),
                "rcl_strategy": st.column_config.TextColumn("RCL Strategy"),
                "construction_strategy": st.column_config.TextColumn("Construction"),
                "local_search_strategy": st.column_config.TextColumn("LS Strategy"),
                "local_search_operators_pretty": st.column_config.TextColumn("LS Operators"),
                "gap_to_bks": st.column_config.NumberColumn("Gap (%)", format="%.2f"),
                "solve_time": st.column_config.NumberColumn("Solve Time (s)", format="%.2f"),
                "error_message": st.column_config.TextColumn("Error Message"),
                "created_at": st.column_config.DatetimeColumn("Created At", format="YYYY-MM-DD HH:mm"),
            }

            event = st.dataframe(
                df_display[
                    [
                        "job_id",
                        "status",
                        "instance_name",
                        "alpha",
                        "termination",
                        "rcl_strategy",
                        "construction_strategy",
                        "local_search_strategy",
                        "local_search_operators_pretty",
                        "gap_to_bks",
                        "solve_time",
                        "error_message",
                        "created_at",
                    ]
                ].reset_index(drop=True),
                column_config=column_config_job_table,
                width="stretch",
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun",
                key="job_table",
            )

            selected_rows = event.selection.rows
            if not selected_rows:
                st.markdown("---")
                st.info("Select a job row above to see its details.")
            else:
                row_idx = selected_rows[0]
                selected_job_id = int(df_display.reset_index(drop=True).iloc[row_idx]["job_id"])

                st.markdown("---")
                st.markdown(f"### Selected Job: {selected_job_id}")

                with engine.connect() as conn:
                    details_query = text(
                        """
                        SELECT
                            j.job_id,
                            j.status,
                            j.instance_name,
                            j.capacity,
                            j.alpha,
                            j.seed,
                            j.iterations,
                            j.is_time_based,
                            j.time_limit,
                            j.base_time,
                            j.rcl_strategy,
                            j.local_search_operators,
                            j.local_search_strategy,
                            j.created_at,
                            j.started_at,
                            j.completed_at,
                            j.error_message,
                            r.solve_time,
                            r.objective_value,
                            r.vehicle_count,
                            r.avg_load_factor,
                            r.best_known_cost,
                            r.gap_to_bks,
                            r.best_known_vehicle_count,
                            r.gap_to_bks_vehicle_count,
                            r.routes,
                            r.bks_routes,
                            r.history,
                            j.demands_data,
                            j.coords_data
                        FROM Jobs j
                        LEFT JOIN Results r ON j.job_id = r.job_id
                        WHERE j.job_id = :job_id
                        """
                    )
                    df_detail = pd.read_sql(details_query, conn, params={"job_id": selected_job_id})

                if df_detail.empty:
                    st.warning("No detailed results found for this job yet (maybe still running?).")
                else:
                    job = df_detail.iloc[0]

                    # Overview
                    st.markdown("#### Overview")
                    with st.container(border=True):
                        c1, c2, c3 = st.columns(3)
                        c1.write(f"**Status:** {job.get('status')}")
                        c2.write(f"**Instance:** {job.get('instance_name')}")
                        if pd.notna(job.get("created_at")):
                            c3.write(f"**Created at:** {job.get('created_at')}")

                    # Parameters
                    st.markdown("#### GRASP Configuration")
                    with st.container(border=True):
                        termination_text = termination_label_from_row(job)
                        p1, p2, p3 = st.columns(3)
                        p4, p5, p6 = st.columns(3)

                        p1.metric("Termination", termination_text)
                        p2.metric("RCL Strategy", str(job.get("rcl_strategy")))
                        p3.metric("RCL Param (Alpha/K)", str(job.get("alpha")))

                        p4.metric("LS Strategy", str(job.get("local_search_strategy")))
                        p5.metric("LS Operators", ls_ops_pretty(job.get("local_search_operators")))
                        p6.metric("Seed", str(job.get("seed")))

                    # KPIs
                    st.markdown("#### Key Performance Indicators")
                    with st.container(border=True):
                        k1, k2, k3 = st.columns(3)
                        best_cost = job.get("objective_value")
                        best_known_cost = job.get("best_known_cost")
                        gap = job.get("gap_to_bks")

                        k1.metric("Best Cost", f"{best_cost:.2f}" if pd.notna(best_cost) else "N/A")
                        k2.metric("Best-Known Cost", f"{best_known_cost:.2f}" if pd.notna(best_known_cost) else "N/A")
                        k3.metric("Gap to BKS", f"{gap:.2f}%" if pd.notna(gap) else "N/A")

                        k4, k5, k6 = st.columns(3)
                        vehicles = job.get("vehicle_count")
                        avg_load = job.get("avg_load_factor")
                        solve_time = job.get("solve_time")

                        k4.metric("Vehicles Used", str(int(vehicles)) if pd.notna(vehicles) else "N/A")
                        k5.metric("Avg Load Factor", f"{avg_load*100:.1f}%" if pd.notna(avg_load) else "N/A")
                        k6.metric("Solve Time (s)", f"{solve_time:.2f}" if pd.notna(solve_time) else "N/A")

                    # Visualization
                    if pd.notna(job.get("routes")):
                        st.markdown("#### Solution Visualization")
                        with st.container(border=True):
                            demands_vis = {int(k): v for k, v in json.loads(job["demands_data"]).items()}
                            coords_vis = {int(k): tuple(v) for k, v in json.loads(job["coords_data"]).items()}
                            routes_vis = json.loads(job["routes"])

                            bks_routes = None
                            if job.get("bks_routes"):
                                try:
                                    bks_routes = json.loads(job["bks_routes"])
                                except Exception:
                                    bks_routes = None

                            render_network(
                                coords=coords_vis,
                                routes=routes_vis,
                                demands=demands_vis,
                                capacity=job["capacity"],
                                bks_routes=bks_routes,
                                bks_cost=job.get("best_known_cost"),
                                solver_cost=job.get("objective_value"),
                            )

                    # Error block
                    if str(job.get("status")) == "FAILED":
                        st.markdown("#### Error")
                        err_msg = job.get("error_message") or "(no error message)"
                        st.error("This job failed with the following error:")
                        st.code(err_msg)

    except Exception as e:
        st.error(f"Error loading jobs: {e}")

# =============================================================
# TAB 3 - SENSITIVITY ANALYSIS
# =============================================================

with tab_analysis:
    st.subheader("Sensitivity Analysis & Comparison")
    st.caption("Select multiple jobs to compare performance side-by-side.")

    try:
        with engine.connect() as conn:
            jobs_query = text(
                """
                SELECT
                    j.job_id,
                    j.instance_name,
                    j.alpha,
                    j.rcl_strategy,
                    j.iterations,
                    j.is_time_based,
                    j.time_limit,
                    j.base_time,
                    j.local_search_operators,
                    j.local_search_strategy,
                    j.construction_strategy,
                    j.demands_data,
                    r.objective_value,
                    r.solve_time,
                    r.gap_to_bks,
                    r.history,
                    j.created_at,
                    b.name as batch_name
                FROM Jobs j
                JOIN Results r ON j.job_id = r.job_id
                LEFT JOIN Batches b ON j.batch_id = b.batch_id
                WHERE j.status = 'COMPLETED'
                ORDER BY j.job_id DESC
                """
            )
            df_completed = pd.read_sql(jobs_query, conn)

        if df_completed.empty:
            st.info("No completed jobs available for analysis.")
            st.stop()

        # Derivations
        df_completed["ls_ops_key"] = df_completed["local_search_operators"].apply(ls_ops_key)
        df_completed["ls_ops_pretty"] = df_completed["local_search_operators"].apply(ls_ops_pretty)
        df_completed["termination"] = df_completed.apply(termination_label_from_row, axis=1)

        if "demands_data" in df_completed.columns:
            df_completed["customer_count"] = df_completed["demands_data"].apply(count_customers_from_demands_json)

        # Batch filter
        unique_batches = df_completed["batch_name"].dropna().unique().tolist()
        selected_batches = []
        if unique_batches:
            selected_batches = st.multiselect(
                "Filter by Batch",
                options=unique_batches,
                placeholder="Select a batch to include its jobs...",
            )
            if selected_batches:
                df_completed = df_completed[df_completed["batch_name"].isin(selected_batches)].reset_index(drop=True)

        st.markdown("---")
        if st.button("🔄 Refresh Data", key="refresh_analysis"):
            st.rerun()

        column_config_analysis = {
            "job_id": st.column_config.NumberColumn("Job ID", format="%d"),
            "batch_name": st.column_config.TextColumn("Batch"),
            "instance_name": st.column_config.TextColumn("Instance"),
            "alpha": st.column_config.NumberColumn("RCL Param (Alpha/K)", format="%.2f"),
            "rcl_strategy": st.column_config.TextColumn("RCL Strategy"),
            "termination": st.column_config.TextColumn("Termination"),
            "local_search_strategy": st.column_config.TextColumn("LS Strategy"),
            "ls_ops_pretty": st.column_config.TextColumn("LS Operators"),
            "construction_strategy": st.column_config.TextColumn("Construction"),
            "objective_value": st.column_config.NumberColumn("Cost", format="%.2f"),
            "solve_time": st.column_config.NumberColumn("Time (s)", format="%.2f"),
            "gap_to_bks": st.column_config.NumberColumn("Gap (%)", format="%.2f"),
            "created_at": st.column_config.DatetimeColumn("Created At", format="YYYY-MM-DD HH:mm"),
        }

        event = st.dataframe(
            df_completed[
                [
                    "job_id",
                    "batch_name",
                    "instance_name",
                    "alpha",
                    "rcl_strategy",
                    "termination",
                    "local_search_strategy",
                    "ls_ops_pretty",
                    "construction_strategy",
                    "objective_value",
                    "solve_time",
                    "gap_to_bks",
                    "created_at",
                ]
            ].reset_index(drop=True),
            column_config=column_config_analysis,
            use_container_width=True,
            hide_index=True,
            selection_mode="multi-row",
            on_select="rerun",
            key="analysis_table",
        )

        selected_rows = event.selection.rows
        selected_jobs = pd.DataFrame()
        if selected_rows:
            selected_jobs = df_completed.iloc[list(selected_rows)].copy()
        elif selected_batches:
            selected_jobs = df_completed.copy()

        if selected_jobs.empty:
            st.info("Select one or more rows (or filter by a Batch) to compare.")
            st.stop()

        st.markdown("---")
        st.markdown(f"### Comparing {len(selected_jobs)} Selected Jobs")

        # Summary table
        st.markdown("#### Metrics Summary")
        group_cols = [
            "instance_name",
            "alpha",
            "rcl_strategy",
            "termination",
            "local_search_strategy",
            "ls_ops_key",
            "construction_strategy",
        ]
        summary_df = (
            selected_jobs.groupby(group_cols)
            .agg(
                count=("objective_value", "count"),
                avg_cost=("objective_value", "mean"),
                min_cost=("objective_value", "min"),
                std_cost=("objective_value", "std"),
                avg_time=("solve_time", "mean"),
                avg_gap=("gap_to_bks", "mean"),
            )
            .reset_index()
        )

        # Attach pretty operator label for readability
        key_to_pretty = (
            selected_jobs[["ls_ops_key", "ls_ops_pretty"]]
            .drop_duplicates()
            .set_index("ls_ops_key")["ls_ops_pretty"]
            .to_dict()
        )
        summary_df["ls_ops_pretty"] = summary_df["ls_ops_key"].map(key_to_pretty).fillna(summary_df["ls_ops_key"])

        summary_df = summary_df[
            [
                "instance_name",
                "alpha",
                "rcl_strategy",
                "termination",
                "local_search_strategy",
                "ls_ops_pretty",
                "construction_strategy",
                "count",
                "avg_cost",
                "min_cost",
                "std_cost",
                "avg_time",
                "avg_gap",
            ]
        ].rename(
            columns={
                "instance_name": "Instance",
                "alpha": "RCL Param (Alpha/K)",
                "rcl_strategy": "RCL Strategy",
                "termination": "Termination",
                "local_search_strategy": "LS Strategy",
                "ls_ops_pretty": "LS Operators",
                "construction_strategy": "Construction",
                "count": "Count",
                "avg_cost": "Avg Cost",
                "min_cost": "Min Cost",
                "std_cost": "Std Dev Cost",
                "avg_time": "Avg Time (s)",
                "avg_gap": "Avg Gap (%)",
            }
        )

        st.dataframe(
            summary_df.style.format(
                {
                    "RCL Param (Alpha/K)": "{:.2f}",
                    "Avg Cost": "{:.2f}",
                    "Min Cost": "{:.2f}",
                    "Std Dev Cost": "{:.2f}",
                    "Avg Time (s)": "{:.2f}",
                    "Avg Gap (%)": "{:.2f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

        # Visual comparisons
        st.markdown("#### Visual Comparison")

        # Gap vs parameter (alpha/K)
        st.markdown("### Gap vs RCL Parameter")
        df_gap = selected_jobs.dropna(subset=["gap_to_bks", "alpha", "rcl_strategy"]).copy()
        if df_gap.empty:
            st.warning("No gap data available for plotting.")
        else:
            grouped = (
                df_gap.groupby(["instance_name", "alpha"])["gap_to_bks"]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values(["instance_name", "alpha"])
            )
            fig = go.Figure()
            for instance_name, sdf in grouped.groupby("instance_name"):
                fig.add_trace(
                    go.Scatter(
                        x=sdf["alpha"],
                        y=sdf["mean"],
                        mode="lines+markers",
                        name=str(instance_name),
                        error_y=dict(type="data", array=sdf["std"].fillna(0.0), visible=True),
                        hovertemplate="Instance: %{fullData.name}<br>Param: %{x}<br>Mean gap: %{y:.2f}%<extra></extra>",
                    )
                )
            fig.update_layout(
                height=520,
                xaxis_title="RCL Parameter (Alpha or K)",
                yaxis_title="Gap to BKS (%)",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Gap vs time scatter
        st.markdown("### Gap vs Solve Time")
        df_scatter = selected_jobs.dropna(subset=["solve_time", "gap_to_bks"]).copy()
        if df_scatter.empty:
            st.warning("No data available to plot Gap vs Time.")
        else:
            scat = go.Figure()
            # color by LS ops key (stable even for pair format)
            for key, sdf in df_scatter.groupby("ls_ops_key"):
                label = key_to_pretty.get(key, key)
                scat.add_trace(
                    go.Scatter(
                        x=sdf["solve_time"],
                        y=sdf["gap_to_bks"],
                        mode="markers",
                        name=label,
                        hovertemplate="Time: %{x:.2f}s<br>Gap: %{y:.2f}%<extra></extra>",
                    )
                )
            scat.update_layout(
                height=520,
                xaxis_title="Solve Time (s)",
                yaxis_title="Gap to BKS (%)",
                hovermode="closest",
            )
            st.plotly_chart(scat, use_container_width=True)

        # Execution time vs customers (if possible)
        st.markdown("### Average Execution Time vs Number of Customers")
        df_time = selected_jobs.dropna(subset=["demands_data", "solve_time"]).copy()
        df_time["customer_count"] = df_time["demands_data"].apply(count_customers_from_demands_json)
        df_time = df_time.dropna(subset=["customer_count"]).copy()

        if df_time.empty:
            st.warning("No data available to plot execution time vs customers.")
        else:
            inst_totals = (
                df_time.groupby(["instance_name", "customer_count"])["solve_time"]
                .mean()
                .reset_index()
                .rename(columns={"solve_time": "avg_solve_time"})
                .sort_values("customer_count")
            )
            exec_fig = make_subplots(specs=[[{"secondary_y": True}]])
            exec_fig.add_trace(
                go.Scatter(
                    x=inst_totals["customer_count"],
                    y=inst_totals["avg_solve_time"],
                    mode="lines+markers",
                    name="Avg solve time",
                    hovertemplate="Customers: %{x}<br>Avg time: %{y:.2f}s<extra></extra>",
                    text=inst_totals["instance_name"],
                ),
                secondary_y=False,
            )
            inst_totals["avg_time_per_customer"] = inst_totals.apply(
                lambda r: r["avg_solve_time"] / r["customer_count"] if r["customer_count"] else None,
                axis=1,
            )
            exec_fig.add_trace(
                go.Bar(
                    x=inst_totals["customer_count"],
                    y=inst_totals["avg_time_per_customer"],
                    name="Avg time per customer",
                    hovertemplate="Customers: %{x}<br>Time/customer: %{y:.4f}s<extra></extra>",
                ),
                secondary_y=True,
            )
            exec_fig.update_layout(
                height=520,
                xaxis_title="Number of Customers",
            )
            exec_fig.update_yaxes(title_text="Avg Solve Time (s)", secondary_y=False)
            exec_fig.update_yaxes(title_text="Avg Time per Customer (s)", secondary_y=True)
            st.plotly_chart(exec_fig, use_container_width=True)

        # Convergence curves
        st.markdown("### Best Cost Found Over Iterations (Convergence Curves)")
        df_hist = selected_jobs.dropna(subset=["history"]).copy()
        if df_hist.empty:
            st.warning("No history data available.")
        else:
            # Build per-scenario lists of best_cost-over-iterations
            scenario_cost_lists = {}
            scenario_time_lists = {}

            for _, row in df_hist.iterrows():
                key = row.get("ls_ops_key", "none")
                label = row.get("ls_ops_pretty", "None")
                hist = _safe_json_loads(row.get("history"))
                if not isinstance(hist, list) or not hist:
                    continue

                best_costs = []
                cons_times = []
                ls_times = []
                for h in hist:
                    # history tuples: (it, best_cost, cost, t_construct, t_ls, time_to_best)
                    if isinstance(h, (list, tuple)) and len(h) >= 5:
                        best_costs.append(float(h[1]))
                        cons_times.append(float(h[3]))
                        ls_times.append(float(h[4]))

                if best_costs:
                    scenario_cost_lists.setdefault((key, label), []).append(best_costs)
                if cons_times or ls_times:
                    scenario_time_lists.setdefault((key, label), []).append((cons_times, ls_times))

            if not scenario_cost_lists:
                st.warning("Could not parse history data from selected jobs.")
            else:
                # Plot average convergence
                conv_fig = go.Figure()
                for (key, label), lists in scenario_cost_lists.items():
                    max_len = max(len(x) for x in lists)
                    avg = []
                    std = []
                    for i in range(max_len):
                        vals = [x[i] for x in lists if i < len(x)]
                        if not vals:
                            continue
                        m = sum(vals) / len(vals)
                        v = sum((t - m) ** 2 for t in vals) / len(vals)
                        avg.append(m)
                        std.append(v ** 0.5)

                    iters = list(range(1, len(avg) + 1))
                    upper = [a + s for a, s in zip(avg, std)]
                    lower = [a - s for a, s in zip(avg, std)]

                    conv_fig.add_trace(
                        go.Scatter(
                            x=iters + iters[::-1],
                            y=upper + lower[::-1],
                            fill="toself",
                            line=dict(width=0),
                            name=f"{label} (±1σ)",
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
                    conv_fig.add_trace(
                        go.Scatter(
                            x=iters,
                            y=avg,
                            mode="lines",
                            name=label,
                            hovertemplate="Iter: %{x}<br>Best cost: %{y:.2f}<extra></extra>",
                        )
                    )

                conv_fig.update_layout(
                    height=600,
                    xaxis_title="Iteration",
                    yaxis_title="Best Cost Found",
                    hovermode="x unified",
                )
                st.plotly_chart(conv_fig, use_container_width=True)

                # Time split plot (construction vs LS) if available
                if scenario_time_lists:
                    time_fig = go.Figure()
                    for (key, label), lists in scenario_time_lists.items():
                        max_len = max(max(len(c), len(l)) for c, l in lists)
                        avg_c = []
                        avg_l = []
                        for i in range(max_len):
                            cvals = [c[i] for c, _ in lists if i < len(c)]
                            lvals = [l[i] for _, l in lists if i < len(l)]
                            avg_c.append(sum(cvals) / len(cvals) if cvals else None)
                            avg_l.append(sum(lvals) / len(lvals) if lvals else None)

                        iters = list(range(1, max_len + 1))
                        time_fig.add_trace(
                            go.Scatter(
                                x=iters,
                                y=avg_c,
                                mode="lines",
                                name=f"Construction · {label}",
                                hovertemplate="Iter: %{x}<br>Construction: %{y:.4f}s<extra></extra>",
                            )
                        )
                        time_fig.add_trace(
                            go.Scatter(
                                x=iters,
                                y=avg_l,
                                mode="lines",
                                name=f"Local Search · {label}",
                                line=dict(dash="dash"),
                                hovertemplate="Iter: %{x}<br>Local Search: %{y:.4f}s<extra></extra>",
                            )
                        )

                    time_fig.update_layout(
                        height=560,
                        xaxis_title="Iteration",
                        yaxis_title="Time per Iteration (s)",
                        hovermode="x unified",
                    )
                    st.plotly_chart(time_fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error in analysis tab: {e}")
