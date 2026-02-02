import time
import json
import sys
import os
import traceback
from pathlib import Path
from sqlalchemy import create_engine, text

# --- Ensure shared modules are in path ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- Import shared CVRP logic ---
try:
    from cvrp_streamlit.algorithms import grasp_solve_cvrp, GraspConfig
    from cvrp_streamlit.inputs import normalize_instance_data, get_benchmark_solution_cost
except ImportError:
    print("CRITICAL ERROR: Could not import shared modules. Ensure cvrp_streamlit/inputs.py and algorithms.py exist.")
    sys.exit(1)

# --- Database connection ---
DB_CONN = os.getenv("DB_CONNECTION_STRING")
if not DB_CONN:
    print("FATAL ERROR: DB_CONNECTION_STRING is not set.")
    sys.exit(1)

engine = create_engine(DB_CONN)
print("Worker initialized. Polling for PENDING jobs...")


DEFAULT_IMP = {"2opt": "first", "relocate": "best", "swap": "best"}


while True:
    try:
        with engine.connect() as conn:
            # --- Fetch one pending job and mark as PROCESSING ---
            trans = conn.begin()
            fetch_sql = text("""
                UPDATE TOP (1) Jobs
                SET status = 'PROCESSING', started_at = GETDATE()
                OUTPUT
                    inserted.job_id,
                    inserted.capacity,
                    inserted.demands_data,
                    inserted.coords_data,
                    inserted.alpha,
                    inserted.seed,
                    inserted.iterations,
                    inserted.instance_name,
                    inserted.rcl_strategy,
                    inserted.local_search_operators,
                    inserted.construction_strategy,
                    inserted.local_search_strategy,
                    inserted.is_time_based,
                    inserted.base_time,
                    inserted.time_limit,
                    inserted.customer_count,
                    inserted.min_customer_count
                WHERE status = 'PENDING'
            """)
            row = conn.execute(fetch_sql).fetchone()
            trans.commit()

            if not row:
                time.sleep(2)
                continue

            # --- Unpack job parameters ---
            job_id = row[0]
            capacity = row[1]
            demands_str = row[2]
            coords_str = row[3]
            alpha_val = row[4]
            seed_val = row[5]
            iter_val = row[6]
            inst_name = row[7]
            rcl_strat = row[8]
            ls_ops_str = row[9]
            constr_strat = row[10] if row[10] else "insertion"
            ls_strat = row[11] if row[11] else "sequential"
            is_time_based = bool(row[12]) if row[12] is not None else False
            base_time_val = row[13]
            time_limit_val = row[14]
            cust_count_val = row[15]
            min_cust_count_val = row[16]

            print(
                f"--> Processing Job {job_id} | Capacity: {capacity} | "
                f"alpha={alpha_val} | seed={seed_val} | constr={constr_strat} | ls={ls_strat} | "
                f"time_limit={time_limit_val if is_time_based else 'iterations'}"
            )

            try:
                # --- Parse input data ---
                raw_demands = json.loads(demands_str)
                raw_coords = json.loads(coords_str) if coords_str else {}

                # Normalize nodes and coordinates
                demands, coords = normalize_instance_data(raw_demands, raw_coords)

                # Get benchmark cost and vehicle count if available
                bks_cost, bks_vehicle_count, bks_routes = get_benchmark_solution_cost(inst_name)

                # --- Parse local search operators ---
                # New expected format (JSON): [["relocate","first"], ["swap","best"], ["2opt","best"]]
                # Backward compatible formats:
                #   - ["relocate","swap","2opt"]
                #   - "relocate,swap,2opt"
                neighborhoods = [("2opt", DEFAULT_IMP["2opt"])]

                if ls_ops_str:
                    s = ls_ops_str.strip()

                    if s.startswith("["):
                        try:
                            parsed = json.loads(s)

                            # Case 1: list of pairs: [["relocate","first"], ...]
                            if parsed and isinstance(parsed[0], (list, tuple)) and len(parsed[0]) == 2:
                                neighborhoods = [(op, imp) for op, imp in parsed]

                            # Case 2: list of strings: ["relocate","swap","2opt"]
                            else:
                                neighborhoods = [(op, DEFAULT_IMP.get(op, "best")) for op in parsed]

                        except json.JSONDecodeError:
                            # Fallback: treat like "[relocate, swap, 2opt]" (non-JSON)
                            ops = [op.strip() for op in s.strip("[]").split(",") if op.strip()]
                            neighborhoods = [(op, DEFAULT_IMP.get(op, "best")) for op in ops]

                    else:
                        # CSV fallback: "relocate, swap, 2opt"
                        ops = [op.strip() for op in s.split(",") if op.strip()]
                        neighborhoods = [(op, DEFAULT_IMP.get(op, "best")) for op in ops]

                # --- Configure GRASP solver ---
                config = GraspConfig(
                    iterations=iter_val if iter_val else 100,
                    rcl_param=alpha_val if alpha_val is not None else 0.3,
                    seed=seed_val,
                    neighborhoods=neighborhoods,  # list of (op, imp)
                    rcl_type=rcl_strat if rcl_strat else "Threshold",
                    construction_strategy=constr_strat,
                    local_search_type=ls_strat,
                    time_limit=time_limit_val if is_time_based else None,
                )

                start_t = time.perf_counter()
                best_routes, best_cost, history = grasp_solve_cvrp(
                    demands=demands,
                    capacity=capacity,
                    coords=coords,
                    config=config
                )
                duration = time.perf_counter() - start_t

                # --- Calculate metrics ---
                vehicle_count = len(best_routes)
                total_demand = sum(demands.values())
                total_capacity = vehicle_count * capacity
                load_factor = (total_demand / total_capacity) if total_capacity > 0 else 0.0

                gap_cost = ((best_cost - bks_cost) / bks_cost * 100) if bks_cost else None
                gap_vehicles = ((vehicle_count - bks_vehicle_count) / bks_vehicle_count * 100) if bks_vehicle_count else None

                bks_routes_json = json.dumps(bks_routes) if bks_routes else None

                # --- Insert results into DB ---
                conn.execute(text("""
                    INSERT INTO Results (
                        job_id, solve_time, objective_value,
                        routes, bks_routes, history,
                        vehicle_count, avg_load_factor,
                        best_known_cost, gap_to_bks,
                        best_known_vehicle_count, gap_to_bks_vehicle_count
                    )
                    VALUES (
                        :jid, :time, :cost,
                        :routes, :bks_routes, :hist,
                        :vc, :lf,
                        :bks, :gap_cost,
                        :bks_vc, :gap_vc
                    )
                """), {
                    "jid": job_id,
                    "time": duration,
                    "cost": best_cost,
                    "routes": json.dumps(best_routes),
                    "bks_routes": bks_routes_json,
                    "hist": json.dumps(history),
                    "vc": vehicle_count,
                    "lf": load_factor,
                    "bks": bks_cost,
                    "gap_cost": gap_cost,
                    "bks_vc": bks_vehicle_count,
                    "gap_vc": gap_vehicles
                })

                conn.execute(
                    text("UPDATE Jobs SET status='COMPLETED', completed_at=GETDATE() WHERE job_id=:jid"),
                    {"jid": job_id}
                )
                conn.commit()
                print(f"--> Job {job_id} completed. Cost: {best_cost}, Vehicles: {vehicle_count}, Time: {duration:.3f}s")

            except Exception as job_e:
                print(f"--> Job {job_id} FAILED: {job_e}")
                traceback.print_exc()
                try:
                    error_msg = f"{str(job_e)}\n{traceback.format_exc()}"
                    conn.execute(
                        text("UPDATE Jobs SET status='FAILED', error_message=:msg WHERE job_id=:jid"),
                        {"jid": job_id, "msg": error_msg}
                    )
                except Exception:
                    conn.execute(
                        text("UPDATE Jobs SET status='FAILED' WHERE job_id=:jid"),
                        {"jid": job_id}
                    )
                conn.commit()

    except Exception as db_e:
        print(f"Database Connection Error: {db_e}")
        time.sleep(5)
