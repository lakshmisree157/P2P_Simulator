"""
A* Pathfinding Algorithm for Smart Grid Energy Flow
"""

import numpy as np
import networkx as nx
import heapq
from typing import Dict, Optional, List
from models.data_models import House, Connection


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