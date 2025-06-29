"""
Visualization functions for the P2P Smart Grid Simulator
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List
from models.data_models import House, Connection, Trade

# Handle orjson import issues
try:
    import plotly.io as pio
    # Set renderer to avoid orjson issues
    pio.renderers.default = "plotly_mimetype"
except ImportError:
    pass


def create_network_plot(network_data: Dict, show_trades: bool = True) -> go.Figure:
    """Create interactive network visualization"""
    houses = network_data["houses"]
    connections = network_data["connections"]
    recent_trades = network_data["trades_history"]
    
    fig = go.Figure()
    
    # Draw connections
    for conn in connections.values():
        house1 = houses[conn.from_house]
        house2 = houses[conn.to_house]
        
        # Color based on utilization
        if conn.utilization > 80:
            color = "red"
            width = 4
        elif conn.utilization > 50:
            color = "orange"
            width = 3
        else:
            color = "green"
            width = 2
        
        fig.add_trace(go.Scatter(
            x=[house1.x, house2.x],
            y=[house1.y, house2.y],
            mode='lines',
            line=dict(color=color, width=width),
            hoverinfo='text',
            hovertext=f"Capacity: {conn.capacity:.1f} kW<br>Usage: {conn.utilization:.1f}%",
            showlegend=False
        ))
    
    # Draw trade paths
    if show_trades and recent_trades:
        for trade in recent_trades:
            path_x = [houses[node_id].x for node_id in trade.path]
            path_y = [houses[node_id].y for node_id in trade.path]
            
            fig.add_trace(go.Scatter(
                x=path_x,
                y=path_y,
                mode='lines',
                line=dict(color="purple", width=3, dash="dash"),
                hoverinfo='text',
                hovertext=f"Trade: {trade.amount:.1f} kW @ ${trade.price:.2f}",
                showlegend=False
            ))
    
    # Draw houses
    house_x = [h.x for h in houses.values()]
    house_y = [h.y for h in houses.values()]
    house_colors = ["green" if h.is_seller else "red" if h.is_buyer else "gray" for h in houses.values()]
    house_sizes = [max(20, abs(h.energy) * 2) for h in houses.values()]
    house_text = [f"House {h.id}<br>Energy: {h.energy:.1f} kW<br>Bid: ${h.bid_price:.2f}" for h in houses.values()]
    
    fig.add_trace(go.Scatter(
        x=house_x,
        y=house_y,
        mode='markers+text',
        marker=dict(
            size=house_sizes,
            color=house_colors,
            line=dict(width=2, color="black")
        ),
        text=[str(h.id) for h in houses.values()],
        textposition="middle center",
        textfont=dict(color="white", size=12),
        hoverinfo='text',
        hovertext=house_text,
        showlegend=False
    ))
    
    fig.update_layout(
        title="Smart Grid Network Visualization",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        height=600,
        plot_bgcolor="white"
    )
    
    return fig


def create_metrics_plots(metrics_history: List[Dict]) -> go.Figure:
    """Create metrics dashboard"""
    if not metrics_history:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Trades per Step", "Average Price", "Market Efficiency", "Network Congestion"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    steps = list(range(len(metrics_history)))
    
    # Trades per step
    fig.add_trace(
        go.Scatter(x=steps, y=[m["total_trades"] for m in metrics_history], name="Trades"),
        row=1, col=1
    )
    
    # Average price
    fig.add_trace(
        go.Scatter(x=steps, y=[m["avg_price"] for m in metrics_history], name="Avg Price"),
        row=1, col=2
    )
    
    # Efficiency
    fig.add_trace(
        go.Scatter(x=steps, y=[m["efficiency"] for m in metrics_history], name="Efficiency"),
        row=2, col=1
    )
    
    # Congestion
    fig.add_trace(
        go.Scatter(x=steps, y=[m["congestion"] for m in metrics_history], name="Congestion"),
        row=2, col=2
    )
    
    fig.update_layout(height=500, showlegend=False)
    return fig 