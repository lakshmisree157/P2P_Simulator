"""
Double Auction Algorithm for Energy Trading
"""

import time
from typing import Dict, List
from models.data_models import House, Trade
from algorithms.pathfinding import AStarPathfinder


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