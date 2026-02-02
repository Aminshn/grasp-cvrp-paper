import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objects as go
import json
# -------------------------------------------------------
# Config
# -------------------------------------------------------
# Main configuration variables
FULL_TO_KNN_THRESHOLD = 30  # if number of nodes > this, use KNN
KNN_K = 5  # neighbors per node in KNN mode

# -------------------------------------------------------
# Color Palette
# -------------------------------------------------------
def get_palette(n):
    """Return a list of n visually distinct colors from a colormap."""
    if n <= 20:
        cmap = cm.get_cmap("tab20", n) 
    else:
        cmap = cm.get_cmap("gist_rainbow", n)
    return [mcolors.to_hex(cmap(i)) for i in range(n)]

# -------------------------------------------------------
# Edge helpers (KNN and Full Connectivity Logic)
# -------------------------------------------------------
def edge_in_routes(a, b, routes):
    """Check if edge (a,b) or (b,a) exists inside any route."""
    for r in routes:
        for u, v in zip(r, r[1:]):
            if (u == a and v == b) or (u == b and v == a):
                return True
    return False

def build_full_edge_pairs(node_ids):
    """All unordered pairs (i,j) with i<j."""
    pairs = []
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            pairs.append((node_ids[i], node_ids[j]))
    return pairs

def build_knn_edge_pairs(coords, k):
    """Build undirected KNN edges based on Euclidean distances."""
    nodes = list(coords.keys())
    pts = np.array([coords[n] for n in nodes], dtype=float)

    n = len(nodes)
    if n <= 1:
        return []

    edges = set()

    for i in range(n):
        pi = pts[i]
        diff = pts - pi
        dist = np.linalg.norm(diff, axis=1) 
        dist[i] = np.inf
        nn_idx = np.argsort(dist)[: min(k, n - 1)]

        for j in nn_idx:
            a = nodes[i]
            b = nodes[j]
            if a == b:
                continue
            e = (a, b) if a < b else (b, a)
            edges.add(e)

    return list(edges)

# -------------------------------------------------------
# Deprecated smart scaling (using original coords)
# -------------------------------------------------------
def smart_scale_coords(coords):
    """Stub to keep functions compatible. Plotly uses original coords."""
    return coords, 1.0 

# -------------------------------------------------------
# Internal Figure Generator (Refactored)
# -------------------------------------------------------
def _create_network_figure(coords, demands, routes, capacity, title_suffix=""):
    # 1. Prepare Data
    node_ids = list(coords.keys())
    n_nodes = len(node_ids)
    
    # Determine mode: Full or KNN
    use_knn = n_nodes > FULL_TO_KNN_THRESHOLD
    
    edge_x = []
    edge_y = []
    
    # 1. Weak Background Edges (Potential connections)
    if use_knn:
        pairs = build_knn_edge_pairs(coords, k=KNN_K)
    else:
        pairs = build_full_edge_pairs(node_ids)
        
    for u, v in pairs:
        if not edge_in_routes(u, v, routes):
            x0, y0 = coords[u]
            x1, y1 = coords[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    traces = []
    
    # Add weak edges trace
    traces.append(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#e0e0e0'),
        hoverinfo='none',
        mode='lines',
        name='Potential Edges',
        showlegend=False,
        legendgroup="group_weak"
    ))

    # 2. Strong Colored Route Edges
    route_colors = get_palette(len(routes))
    
    for idx, route in enumerate(routes):
        route_color = route_colors[idx % len(route_colors)]
        
        r_x = []
        r_y = []
        mid_x = []
        mid_y = []
        hover_texts = []
        
        for i in range(len(route) - 1):
            u = route[i]
            v = route[i+1]
            
            x1, y1 = coords[u]
            x2, y2 = coords[v]
            
            r_x.append(x1)
            r_y.append(y1)
            
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            mid_x.append(mx)
            mid_y.append(my)
            
            dist = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
            
            hover_texts.append(
                f"<b>Route {idx+1}</b><br>" +
                f"From Node {u} to {v}<br>" +
                f"Distance: {dist:.2f}"
            )
            
        r_x.append(coords[route[-1]][0])
        r_y.append(coords[route[-1]][1])

        # Line trace
        traces.append(go.Scatter(
            x=r_x, y=r_y,
            mode='lines',
            line=dict(width=2, color=route_color),
            name=f"Route {idx+1}",
            hoverinfo='skip',
            opacity=0.8,
            legendgroup=f"group_{idx}"
        ))

        # Midpoint trace
        traces.append(go.Scatter(
            x=mid_x, y=mid_y,
            mode='markers',
            marker=dict(size=15, color=route_color, opacity=0),
            hoverinfo='text',
            hovertext=hover_texts,
            showlegend=False,
            legendgroup=f"group_{idx}"
        ))

    # 3. Nodes
    depot_x = []
    depot_y = []
    depot_text = []
    customer_x = []
    customer_y = []
    customer_text = []

    for node_id in node_ids:
        x, y = coords[node_id]
        d = demands.get(node_id, 0)

        if node_id == 0:
            depot_x.append(x)
            depot_y.append(y)
            depot_text.append(f"<b>DEPOT</b><br>Pos: ({x},{y})")
        else:
            customer_x.append(x)
            customer_y.append(y)
            customer_text.append(
                f"<b>Customer {node_id}</b><br>Demand: {d}<br>Pos: ({x},{y})"
            )

    if depot_x:
        traces.append(go.Scatter(
            x=depot_x, y=depot_y,
            mode='markers',
            marker=dict(
                symbol='square',
                size=12,
                color='red',
                line=dict(width=1, color='white')
            ),
            text=depot_text,
            hoverinfo='text',
            name='Depot',
            legendgroup="nodes"
        ))

    if customer_x:
        traces.append(go.Scatter(
            x=customer_x, y=customer_y,
            mode='markers',
            marker=dict(
                symbol='circle',
                size=8,
                color='blue',
                line=dict(width=1, color='white')
            ),
            text=customer_text,
            hoverinfo='text',
            name='Customers',
            legendgroup="nodes"
        ))

    # Layout
    fig = go.Figure(data=traces)
    
    total_traces = len(traces)
    nodes_trace_index = total_traces - 1
    route_start_index = 1
    route_end_index = nodes_trace_index - 1
    
    indices_to_toggle = list(range(route_start_index, route_end_index + 1))
    indices_to_toggle.insert(0, 0)

    fig.update_layout(
        title=f"Network Visualization {title_suffix}",
        showlegend=True,
        hovermode='closest',
        margin=dict(l=0, r=0, t=40, b=0),
        template="plotly_white",
        xaxis=dict(
            title="X Coordinate",
            showgrid=True,
            gridcolor="#E8E8E8",
            zeroline=False,
            showticklabels=True,
            showline=True,
            linecolor="#000000",
            linewidth=1.5,
        ),
        yaxis=dict(
            title="Y Coordinate",
            showgrid=True,
            gridcolor="#E8E8E8",
            zeroline=False,
            showticklabels=True,
            showline=True,
            linecolor="#000000",
            linewidth=1.5,
            scaleanchor="x",
            scaleratio=1,
        ),
        plot_bgcolor="rgba(240, 240, 245, 0.3)",
        paper_bgcolor="white",
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"visible": True}, indices_to_toggle],
                        label="Show All Routes",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": "legendonly"}, indices_to_toggle],
                        label="Hide All Routes",
                        method="restyle"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1,
                xanchor="right",
                y=1.15,
                yanchor="top"
            ),
        ]
    )
    return fig

# -------------------------------------------------------
# Main Render Function
# -------------------------------------------------------
def render_network(coords, demands, routes, capacity, bks_routes=None, bks_cost=None, solver_cost=None):
    """
    Renders the network. If bks_routes is provided, shows Tabs.
    """
    
    solver_suffix = f"(Solver - Cost: {solver_cost:.2f})" if solver_cost is not None else "(Solver)"

    if bks_routes:
        tab1, tab2 = st.tabs(["📍 Solver Solution", "🏆 Best Known Solution (BKS)"])
        
        with tab1:
            fig1 = _create_network_figure(coords, demands, routes, capacity, title_suffix=solver_suffix)
            st.plotly_chart(fig1, use_container_width=True)
            
        with tab2:
            suffix = f"(BKS - Cost: {bks_cost})" if bks_cost else "(BKS)"
            fig2 = _create_network_figure(coords, demands, bks_routes, capacity, title_suffix=suffix)
            st.plotly_chart(fig2, use_container_width=True)
            
    else:
        # Standard single view
        fig = _create_network_figure(coords, demands, routes, capacity, title_suffix=solver_suffix)
        st.plotly_chart(fig, use_container_width=True)


