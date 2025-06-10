import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import heapq
from collections import defaultdict
import json

# Configure Streamlit page
st.set_page_config(
    page_title="P2P Smart Grid Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class House:
    """Represents a house in the smart grid network"""
    id: int
    x: float
    y: float
    energy: float  # Positive = surplus, Negative = deficit
    bid_price: float
    house_type: str  # 'prosumer' or 'consumer'
    capacity: float = 100.0
    
    @property
    def is_seller(self) -> bool:
        return self.energy > 0
    
    @property
    def is_buyer(self) -> bool:
        return self.energy < 0
    
    @property
    def demand(self) -> float:
        return abs(self.energy) if self.is_buyer else 0
    
    @property
    def supply(self) -> float:
        return self.energy if self.is_seller else 0

@dataclass
class Connection:
    """Represents a power line connection between houses"""
    from_house: int
    to_house: int
    capacity: float
    current_usage: float = 0.0
    
    @property
    def utilization(self) -> float:
        return (self.current_usage / self.capacity) * 100 if self.capacity > 0 else 0
    
    @property
    def is_congested(self) -> bool:
        return self.utilization > 80

@dataclass
class Trade:
    """Represents an energy trade between two houses"""
    buyer_id: int
    seller_id: int
    amount: float
    price: float
    path: List[int]
    timestamp: int
    success: bool = True

class AStarPathfinder:
    """A* algorithm implementation for finding optimal energy flow paths"""
    
    def __init__(self, graph: nx.Graph, houses: Dict[int, House], connections: Dict[str, Connection]):
        self.graph = graph
        self.houses = houses
        self.connections = connections
    
    def heuristic(self, node1: int, node2: int) -> float:
        """Euclidean distance heuristic"""
        h1, h2 = self.houses[node1], self.houses[node2]
        return np.sqrt((h1.x - h2.x)**2 + (h1.y - h2.y)**2)
    
    def find_path(self, start: int, end: int, capacity_threshold: float = 0.8) -> Optional[Dict]:
        """Find optimal path using A* algorithm"""
        if start == end:
            return {"path": [start], "cost": 0}
        
        # Priority queue: (f_score, g_score, node, path)
        open_set = [(self.heuristic(start, end), 0, start, [start])]
        closed_set = set()
        g_scores = {start: 0}
        
        while open_set:
            f_score, g_score, current, path = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
                
            if current == end:
                return {"path": path, "cost": g_score}
            
            closed_set.add(current)
            
            for neighbor in self.graph.neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # Check connection capacity
                conn_key = f"{min(current, neighbor)}_{max(current, neighbor)}"
                connection = self.connections.get(conn_key)
                
                if not connection or connection.utilization >= capacity_threshold * 100:
                    continue
                
                tentative_g = g_score + 1  # Uniform cost for simplicity
                
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    h_score = self.heuristic(neighbor, end)
                    f_score = tentative_g + h_score
                    
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor, path + [neighbor]))
        
        return None

class DoubleAuction:
    """Continuous Double Auction mechanism for energy trading"""
    
    def __init__(self, houses: Dict[int, House]):
        self.houses = houses
    
    def get_sorted_buyers(self) -> List[House]:
        """Get buyers sorted by bid price (highest first)"""
        buyers = [h for h in self.houses.values() if h.is_buyer]
        return sorted(buyers, key=lambda x: x.bid_price, reverse=True)
    
    def get_sorted_sellers(self) -> List[House]:
        """Get sellers sorted by ask price (lowest first)"""
        sellers = [h for h in self.houses.values() if h.is_seller]
        return sorted(sellers, key=lambda x: x.bid_price * 0.8)  # Sellers accept 80% of their bid
    
    def match_trades(self, pathfinder: AStarPathfinder) -> List[Trade]:
        """Execute double auction matching"""
        buyers = self.get_sorted_buyers()
        sellers = self.get_sorted_sellers()
        trades = []
        
        for buyer in buyers:
            if buyer.demand <= 0:
                continue
                
            for seller in sellers:
                if seller.supply <= 0:
                    continue
                
                # Check if trade is profitable
                seller_min_price = seller.bid_price * 0.8
                if buyer.bid_price >= seller_min_price:
                    # Find path between buyer and seller
                    path_result = pathfinder.find_path(seller.id, buyer.id)
                    
                    if path_result:
                        # Calculate trade details
                        trade_amount = min(buyer.demand, seller.supply, 30.0)  # Max 30 kW per trade
                        trade_price = (buyer.bid_price + seller_min_price) / 2
                        
                        # Create trade
                        trade = Trade(
                            buyer_id=buyer.id,
                            seller_id=seller.id,
                            amount=trade_amount,
                            price=trade_price,
                            path=path_result["path"],
                            timestamp=int(time.time())
                        )
                        
                        trades.append(trade)
                        
                        # Update house energy levels
                        buyer.energy += trade_amount  # Reduces deficit
                        seller.energy -= trade_amount  # Reduces surplus
                        
                        break  # Move to next buyer
        
        return trades

class SmartGridSimulator:
    """Main simulator class"""
    
    def __init__(self):
        self.houses: Dict[int, House] = {}
        self.connections: Dict[str, Connection] = {}
        self.graph = nx.Graph()
        self.trades_history: List[Trade] = []
        self.step_count = 0
        self.pathfinder = None
        self.auction = None
    
    def initialize_network(self, num_houses: int, network_type: str = "ring"):
        """Initialize the network topology"""
        self.houses.clear()
        self.connections.clear()
        self.graph.clear()
        self.trades_history.clear()
        self.step_count = 0
        
        # Create houses in circular arrangement
        center_x, center_y = 0, 0
        radius = 100
        
        for i in range(num_houses):
            angle = (i / num_houses) * 2 * np.pi
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            
            house = House(
                id=i,
                x=x,
                y=y,
                energy=np.random.uniform(-50, 50),  # Random energy balance
                bid_price=np.random.uniform(10, 30),  # Random bid price
                house_type=np.random.choice(['prosumer', 'consumer'], p=[0.6, 0.4])
            )
            
            self.houses[i] = house
            self.graph.add_node(i, pos=(x, y))
        
        # Create connections based on network type
        if network_type == "ring":
            self._create_ring_topology(num_houses)
        elif network_type == "mesh":
            self._create_mesh_topology(num_houses)
        elif network_type == "star":
            self._create_star_topology(num_houses)
        
        # Initialize algorithms
        self.pathfinder = AStarPathfinder(self.graph, self.houses, self.connections)
        self.auction = DoubleAuction(self.houses)
    
    def _create_ring_topology(self, num_houses: int):
        """Create ring network topology"""
        for i in range(num_houses):
            next_house = (i + 1) % num_houses
            self._add_connection(i, next_house, np.random.uniform(80, 120))
            
            # Add some diagonal connections for redundancy
            if np.random.random() > 0.4:
                next_next = (i + 2) % num_houses
                self._add_connection(i, next_next, np.random.uniform(60, 100))
    
    def _create_mesh_topology(self, num_houses: int):
        """Create mesh network topology"""
        for i in range(num_houses):
            for j in range(i + 1, num_houses):
                distance = np.sqrt((self.houses[i].x - self.houses[j].x)**2 + 
                                 (self.houses[i].y - self.houses[j].y)**2)
                
                # Connect houses within reasonable distance
                if distance < 150 and np.random.random() > 0.3:
                    capacity = max(50, 120 - distance * 0.5)
                    self._add_connection(i, j, capacity)
    
    def _create_star_topology(self, num_houses: int):
        """Create star network topology"""
        # Central hub (house 0)
        for i in range(1, num_houses):
            self._add_connection(0, i, np.random.uniform(100, 150))
        
        # Add some edge connections
        for i in range(1, num_houses):
            if np.random.random() > 0.7:
                j = np.random.randint(1, num_houses)
                if i != j:
                    self._add_connection(i, j, np.random.uniform(60, 100))
    
    def _add_connection(self, house1: int, house2: int, capacity: float):
        """Add a connection between two houses"""
        conn_key = f"{min(house1, house2)}_{max(house1, house2)}"
        if conn_key not in self.connections:
            self.connections[conn_key] = Connection(house1, house2, capacity)
            self.graph.add_edge(house1, house2, capacity=capacity)
    
    def add_house_manually(self, x: float, y: float, energy: float, bid_price: float, house_type: str):
        """Add a house manually to the network"""
        house_id = max(self.houses.keys()) + 1 if self.houses else 0
        
        house = House(
            id=house_id,
            x=x,
            y=y,
            energy=energy,
            bid_price=bid_price,
            house_type=house_type
        )
        
        self.houses[house_id] = house
        self.graph.add_node(house_id, pos=(x, y))
        
        # Connect to nearest houses
        for other_id, other_house in self.houses.items():
            if other_id != house_id:
                distance = np.sqrt((house.x - other_house.x)**2 + (house.y - other_house.y)**2)
                if distance < 100:  # Connect if within 100 units
                    capacity = max(50, 120 - distance * 0.5)
                    self._add_connection(house_id, other_id, capacity)
        
        # Reinitialize algorithms
        self.pathfinder = AStarPathfinder(self.graph, self.houses, self.connections)
        self.auction = DoubleAuction(self.houses)
        
        return house_id
    
    def simulate_step(self) -> Dict:
        """Execute one simulation step"""
        self.step_count += 1
        
        # Reset connection usage
        for conn in self.connections.values():
            conn.current_usage = 0
        
        # Run auction
        new_trades = self.auction.match_trades(self.pathfinder)
        
        # Update connection usage based on trades
        for trade in new_trades:
            for i in range(len(trade.path) - 1):
                from_id, to_id = trade.path[i], trade.path[i + 1]
                conn_key = f"{min(from_id, to_id)}_{max(from_id, to_id)}"
                if conn_key in self.connections:
                    self.connections[conn_key].current_usage += trade.amount
        
        self.trades_history.extend(new_trades)
        
        # Calculate metrics
        metrics = self._calculate_metrics(new_trades)
        
        return {
            "trades": new_trades,
            "metrics": metrics,
            "step": self.step_count
        }
    
    def _calculate_metrics(self, trades: List[Trade]) -> Dict:
        """Calculate simulation metrics"""
        if not trades:
            return {
                "total_trades": 0,
                "avg_price": 0,
                "total_volume": 0,
                "efficiency": 0,
                "congestion": 0,
                "price_std": 0
            }
        
        prices = [t.price for t in trades]
        volumes = [t.amount for t in trades]
        
        total_supply = sum(h.supply for h in self.houses.values())
        total_demand = sum(h.demand for h in self.houses.values())
        total_traded = sum(volumes)
        
        efficiency = (total_traded / max(total_supply, total_demand)) * 100 if max(total_supply, total_demand) > 0 else 0
        
        # Calculate network congestion
        utilizations = [conn.utilization for conn in self.connections.values()]
        avg_congestion = np.mean(utilizations) if utilizations else 0
        
        return {
            "total_trades": len(trades),
            "avg_price": np.mean(prices),
            "total_volume": total_traded,
            "efficiency": efficiency,
            "congestion": avg_congestion,
            "price_std": np.std(prices) if len(prices) > 1 else 0
        }
    
    def get_network_data(self) -> Dict:
        """Get current network state for visualization"""
        return {
            "houses": self.houses,
            "connections": self.connections,
            "graph": self.graph,
            "trades_history": self.trades_history[-10:],  # Last 10 trades
            "step_count": self.step_count
        }

# Visualization functions
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

# Main Streamlit Application
def main():
    # Title and description
    st.title("⚡ P2P Smart Grid Simulator")
    st.markdown("**Interactive Demo - Energy Trading with AI Algorithms**")
    
    # Initialize session state
    if 'simulator' not in st.session_state:
        st.session_state.simulator = SmartGridSimulator()
        st.session_state.metrics_history = []
        st.session_state.is_running = False
        st.session_state.auto_step = False
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Network configuration
        st.subheader("Network Configuration")
        num_houses = st.slider("Number of Houses", 4, 15, 8)
        network_type = st.selectbox("Network Topology", ["ring", "mesh", "star"])
        
        if st.button("🔄 Initialize Network"):
            st.session_state.simulator.initialize_network(num_houses, network_type)
            st.session_state.metrics_history = []
            st.rerun()
        
        st.divider()
        
        # Simulation controls
        st.subheader("Simulation Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Step"):
                result = st.session_state.simulator.simulate_step()
                st.session_state.metrics_history.append(result["metrics"])
                st.rerun()
        
        with col2:
            if st.button("⏹️ Reset"):
                st.session_state.simulator.initialize_network(num_houses, network_type)
                st.session_state.metrics_history = []
                st.rerun()
        
        # Auto-run toggle
        auto_run = st.checkbox("🔄 Auto Run", value=st.session_state.auto_step)
        if auto_run != st.session_state.auto_step:
            st.session_state.auto_step = auto_run
            st.rerun()
        
        if auto_run:
            time.sleep(2)  # Wait 2 seconds between steps
            result = st.session_state.simulator.simulate_step()
            st.session_state.metrics_history.append(result["metrics"])
            st.rerun()
        
        st.divider()
        
        # Manual house addition
        st.subheader("Add House Manually")
        with st.form("add_house_form"):
            col1, col2 = st.columns(2)
            with col1:
                x_pos = st.number_input("X Position", value=0.0)
                y_pos = st.number_input("Y Position", value=0.0)
            with col2:
                energy = st.number_input("Energy (kW)", value=0.0, help="Positive=surplus, Negative=deficit")
                bid_price = st.number_input("Bid Price ($)", value=20.0, min_value=0.1)
            
            house_type = st.selectbox("House Type", ["prosumer", "consumer"])
            
            if st.form_submit_button("➕ Add House"):
                new_id = st.session_state.simulator.add_house_manually(x_pos, y_pos, energy, bid_price, house_type)
                st.success(f"Added House {new_id}!")
                st.rerun()
    
    # Main content area
    if not st.session_state.simulator.houses:
        st.info("👆 Please initialize the network using the sidebar controls.")
        return
    
    # Get current network data
    network_data = st.session_state.simulator.get_network_data()
    
    # Create main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Network visualization
        st.subheader("🔗 Network Visualization")
        show_trade_paths = st.checkbox("Show Trade Paths", value=True)
        
        network_fig = create_network_plot(network_data, show_trade_paths)
        st.plotly_chart(network_fig, use_container_width=True)
        
        # Legend
        st.markdown("""
        **Legend:**
        - 🟢 Green Houses: Energy Surplus (Sellers)
        - 🔴 Red Houses: Energy Deficit (Buyers)  
        - 🟢 Green Lines: Low Usage (<50%)
        - 🟠 Orange Lines: Medium Usage (50-80%)
        - 🔴 Red Lines: High Usage (>80%)
        - 🟣 Purple Dashed: Active Trade Paths
        """)
    
    with col2:
        # Current metrics
        st.subheader("📊 Live Metrics")
        
        if st.session_state.metrics_history:
            latest_metrics = st.session_state.metrics_history[-1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Trades", latest_metrics["total_trades"])
                st.metric("Efficiency", f"{latest_metrics['efficiency']:.1f}%")
            
            with col2:
                st.metric("Avg Price", f"${latest_metrics['avg_price']:.2f}")
                st.metric("Congestion", f"{latest_metrics['congestion']:.1f}%")
        
        # Recent trades
        st.subheader("💰 Recent Trades")
        recent_trades = network_data["trades_history"]
        
        if recent_trades:
            for trade in recent_trades[-5:]:  # Show last 5 trades
                with st.expander(f"Trade: House {trade.seller_id} → House {trade.buyer_id}"):
                    st.write(f"**Amount:** {trade.amount:.1f} kW")
                    st.write(f"**Price:** ${trade.price:.2f}")
                    st.write(f"**Path:** {' → '.join(map(str, trade.path))}")
        else:
            st.info("No trades yet. Run a simulation step!")
        
        # House status
        st.subheader("🏠 House Status")
        houses_df = pd.DataFrame([
            {
                "ID": h.id,
                "Energy": f"{h.energy:.1f}",
                "Bid": f"${h.bid_price:.2f}",
                "Type": h.house_type,
                "Status": "Seller" if h.is_seller else "Buyer" if h.is_buyer else "Neutral"
            }
            for h in network_data["houses"].values()
        ])
        st.dataframe(houses_df, use_container_width=True)
    
    # Metrics history
    # if st.session_state.metrics_history:
    #     st.subheader("📈 Performance Metrics")
    #     metrics_fig = create_metrics_plots(st.session_state.metrics_history)
    #     st.plotly_chart(metrics_fig, use_container_width=True)
    
    # Algorithm information
    st.subheader("🧠 Algorithms in Action")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **A* Pathfinding**
        
        Finds optimal energy flow paths while considering:
        - Line capacity constraints
        - Network congestion
        - Shortest path optimization
        """)
    
    with col2:
        st.success("""
        **Double Auction**
        
        Matches energy trades by:
        - Sorting buyers by highest bid
        - Sorting sellers by lowest ask
        - Fair price discovery
        """)
    
    with col3:
        st.warning("""
        **Priority Queues**
        
        Efficiently manages:
        - Bid/ask ordering
        - Path finding priorities
        - Trade execution sequence
        """)

if __name__ == "__main__":
    main()