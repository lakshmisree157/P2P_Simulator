"""
Main Smart Grid Simulator
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional
from models.data_models import House, Connection, Trade
from algorithms.pathfinding import AStarPathfinder
from algorithms.auction import DoubleAuction


class SmartGridSimulator:
    """Main simulator class"""
    
    def __init__(self):
        self.houses: Dict[int, House] = {}
        self.connections: Dict[str, Connection] = {}
        self.graph = nx.Graph()
        self.trades_history: List[Trade] = []
        self.step_count = 0
        self.pathfinder: Optional[AStarPathfinder] = None
        self.auction: Optional[DoubleAuction] = None
    
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
        if self.auction and self.pathfinder:
            new_trades = self.auction.match_trades(self.pathfinder)
        else:
            new_trades = []
        
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