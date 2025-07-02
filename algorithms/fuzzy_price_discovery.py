"""
Fuzzy Logic Price Discovery for P2P Smart Grid
Considers multiple fuzzy factors for sophisticated price determination
"""

import numpy as np
from typing import Dict, List, Tuple
from models.data_models import House


class FuzzyPriceDiscovery:
    """Fuzzy logic system for intelligent price discovery"""
    
    def __init__(self):
        # Fuzzy membership functions for different factors
        self.demand_urgency_levels = ['low', 'medium', 'high']
        self.supply_availability_levels = ['scarce', 'moderate', 'abundant']
        self.time_of_day_levels = ['off_peak', 'normal', 'peak']
        self.weather_levels = ['favorable', 'normal', 'unfavorable']
        
        # Price adjustment factors for each combination
        self.price_adjustments = self._initialize_price_adjustments()
    
    def _initialize_price_adjustments(self) -> Dict:
        """Initialize fuzzy rules for price adjustments"""
        rules = {}
        
        # Demand urgency rules
        for urgency in self.demand_urgency_levels:
            for supply in self.supply_availability_levels:
                for time in self.time_of_day_levels:
                    for weather in self.weather_levels:
                        key = (urgency, supply, time, weather)
                        
                        # Base price adjustment logic
                        adjustment = 1.0  # Base multiplier
                        
                        # Demand urgency effects
                        if urgency == 'high':
                            adjustment *= 1.3
                        elif urgency == 'medium':
                            adjustment *= 1.1
                        # low = 1.0 (no adjustment)
                        
                        # Supply availability effects
                        if supply == 'scarce':
                            adjustment *= 1.4
                        elif supply == 'moderate':
                            adjustment *= 1.1
                        # abundant = 1.0 (no adjustment)
                        
                        # Time of day effects
                        if time == 'peak':
                            adjustment *= 1.2
                        elif time == 'normal':
                            adjustment *= 1.05
                        # off_peak = 1.0 (no adjustment)
                        
                        # Weather effects
                        if weather == 'unfavorable':
                            adjustment *= 1.25
                        elif weather == 'normal':
                            adjustment *= 1.05
                        # favorable = 1.0 (no adjustment)
                        
                        rules[key] = adjustment
        
        return rules
    
    def _fuzzify_demand_urgency(self, total_demand: float, total_supply: float) -> Dict[str, float]:
        """Convert demand urgency to fuzzy membership values"""
        if total_supply == 0:
            demand_ratio = 2.0  # High urgency if no supply
        else:
            demand_ratio = total_demand / total_supply
        
        membership = {'low': 0.0, 'medium': 0.0, 'high': 0.0}
        
        if demand_ratio <= 0.5:
            membership['low'] = 1.0
        elif demand_ratio <= 1.0:
            membership['low'] = 2.0 - 2.0 * demand_ratio
            membership['medium'] = 2.0 * demand_ratio - 1.0
        elif demand_ratio <= 1.5:
            membership['medium'] = 3.0 - 2.0 * demand_ratio
            membership['high'] = 2.0 * demand_ratio - 2.0
        else:
            membership['high'] = 1.0
        
        return membership
    
    def _fuzzify_supply_availability(self, total_supply: float, total_demand: float) -> Dict[str, float]:
        """Convert supply availability to fuzzy membership values"""
        if total_demand == 0:
            supply_ratio = 2.0  # Abundant if no demand
        else:
            supply_ratio = total_supply / total_demand
        
        membership = {'scarce': 0.0, 'moderate': 0.0, 'abundant': 0.0}
        
        if supply_ratio <= 0.5:
            membership['scarce'] = 1.0
        elif supply_ratio <= 1.0:
            membership['scarce'] = 2.0 - 2.0 * supply_ratio
            membership['moderate'] = 2.0 * supply_ratio - 1.0
        elif supply_ratio <= 1.5:
            membership['moderate'] = 3.0 - 2.0 * supply_ratio
            membership['abundant'] = 2.0 * supply_ratio - 2.0
        else:
            membership['abundant'] = 1.0
        
        return membership
    
    def _fuzzify_time_of_day(self, hour: int) -> Dict[str, float]:
        """Convert time of day to fuzzy membership values"""
        membership = {'off_peak': 0.0, 'normal': 0.0, 'peak': 0.0}
        
        # Peak hours: 6-9 AM and 5-8 PM
        # Off-peak: 11 PM - 6 AM
        # Normal: rest of the day
        
        if 6 <= hour <= 9 or 17 <= hour <= 20:  # Peak hours
            membership['peak'] = 1.0
        elif 23 <= hour or hour <= 6:  # Off-peak hours
            membership['off_peak'] = 1.0
        else:  # Normal hours
            membership['normal'] = 1.0
        
        return membership
    
    def _fuzzify_weather(self, weather_factor: float) -> Dict[str, float]:
        """Convert weather conditions to fuzzy membership values"""
        # weather_factor: 0.0 (bad) to 1.0 (good)
        membership = {'unfavorable': 0.0, 'normal': 0.0, 'favorable': 0.0}
        
        if weather_factor <= 0.3:
            membership['unfavorable'] = 1.0
        elif weather_factor <= 0.7:
            membership['normal'] = 1.0
        else:
            membership['favorable'] = 1.0
        
        return membership
    
    def _defuzzify_price_adjustment(self, 
                                  demand_membership: Dict[str, float],
                                  supply_membership: Dict[str, float],
                                  time_membership: Dict[str, float],
                                  weather_membership: Dict[str, float]) -> float:
        """Combine fuzzy memberships to get final price adjustment"""
        total_weight = 0.0
        weighted_adjustment = 0.0
        
        for urgency in self.demand_urgency_levels:
            for supply in self.supply_availability_levels:
                for time in self.time_of_day_levels:
                    for weather in self.weather_levels:
                        # Calculate membership strength for this combination
                        strength = (demand_membership[urgency] * 
                                  supply_membership[supply] * 
                                  time_membership[time] * 
                                  weather_membership[weather])
                        
                        if strength > 0:
                            adjustment = self.price_adjustments[(urgency, supply, time, weather)]
                            weighted_adjustment += strength * adjustment
                            total_weight += strength
        
        if total_weight > 0:
            return weighted_adjustment / total_weight
        else:
            return 1.0  # Default no adjustment
    
    def calculate_fuzzy_price(self, 
                            houses: Dict[int, House],
                            time_of_day: int,
                            weather_factor: float = 0.7,
                            base_price: float = 20.0) -> float:
        """Calculate price using fuzzy logic considering all factors"""
        
        # Calculate total supply and demand
        total_supply = sum(h.energy for h in houses.values() if h.energy > 0)
        total_demand = sum(abs(h.energy) for h in houses.values() if h.energy < 0)
        
        # Fuzzify all inputs
        demand_membership = self._fuzzify_demand_urgency(total_demand, total_supply)
        supply_membership = self._fuzzify_supply_availability(total_supply, total_demand)
        time_membership = self._fuzzify_time_of_day(time_of_day)
        weather_membership = self._fuzzify_weather(weather_factor)
        
        # Get price adjustment factor
        adjustment_factor = self._defuzzify_price_adjustment(
            demand_membership, supply_membership, time_membership, weather_membership
        )
        
        # Apply adjustment to base price
        fuzzy_price = base_price * adjustment_factor
        
        return fuzzy_price
    
    def get_fuzzy_analysis(self, 
                          houses: Dict[int, House],
                          time_of_day: int,
                          weather_factor: float = 0.7) -> Dict:
        """Get detailed fuzzy analysis for debugging/display"""
        
        total_supply = sum(h.energy for h in houses.values() if h.energy > 0)
        total_demand = sum(abs(h.energy) for h in houses.values() if h.energy < 0)
        
        demand_membership = self._fuzzify_demand_urgency(total_demand, total_supply)
        supply_membership = self._fuzzify_supply_availability(total_supply, total_demand)
        time_membership = self._fuzzify_time_of_day(time_of_day)
        weather_membership = self._fuzzify_weather(weather_factor)
        
        return {
            'total_supply': total_supply,
            'total_demand': total_demand,
            'demand_urgency': demand_membership,
            'supply_availability': supply_membership,
            'time_of_day': time_membership,
            'weather': weather_membership,
            'demand_ratio': total_demand / total_supply if total_supply > 0 else float('inf'),
            'supply_ratio': total_supply / total_demand if total_demand > 0 else float('inf')
        } 