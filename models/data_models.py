"""
Data models for the P2P Smart Grid Simulator
"""

from dataclasses import dataclass
from typing import List


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