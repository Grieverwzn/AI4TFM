"""
Time-varying metrics computation for delay, queue, and speed.
"""

import numpy as np
from typing import Tuple, Optional, Dict


class TimeVaryingMetrics:
    """
    Compute time-varying delay, queue, and speed metrics.
    
    This class provides methods to calculate various traffic performance
    metrics that vary over time.
    """
    
    def __init__(self):
        """Initialize TimeVaryingMetrics calculator."""
        pass
    
    def compute_delay_profile(
        self,
        speeds: np.ndarray,
        free_flow_speed: float,
        length: float
    ) -> np.ndarray:
        """
        Compute time-varying delay profile from speeds.
        
        Args:
            speeds: Array of speeds over time (mph or km/h)
            free_flow_speed: Free flow speed (mph or km/h)
            length: Segment length (miles or km)
            
        Returns:
            Array of delays (hours) at each time step
        """
        free_flow_time = length / free_flow_speed
        actual_times = length / np.maximum(speeds, 0.1)  # Avoid division by zero
        delays = actual_times - free_flow_time
        return np.maximum(delays, 0)  # Delay cannot be negative
    
    def compute_queue_profile(
        self,
        flows: np.ndarray,
        capacity: float,
        time_steps: np.ndarray
    ) -> np.ndarray:
        """
        Compute time-varying queue length profile.
        
        Args:
            flows: Array of flow rates (veh/h)
            capacity: Road capacity (veh/h)
            time_steps: Time values (hours)
            
        Returns:
            Array of queue lengths (vehicles)
        """
        dt = np.diff(time_steps, prepend=0)
        
        # Cumulative arrivals and departures
        cumulative_arrivals = np.cumsum(flows * dt)
        cumulative_departures = np.minimum(
            cumulative_arrivals,
            capacity * time_steps
        )
        
        queue_lengths = cumulative_arrivals - cumulative_departures
        return queue_lengths
    
    def compute_speed_profile(
        self,
        flows: np.ndarray,
        capacity: float,
        free_flow_speed: float,
        jam_density: float = 200.0
    ) -> np.ndarray:
        """
        Compute speed profile from flow using fundamental diagram.
        
        Args:
            flows: Array of flow rates (veh/h)
            capacity: Road capacity (veh/h)
            free_flow_speed: Free flow speed (mph or km/h)
            jam_density: Jam density (veh/mile or veh/km)
            
        Returns:
            Array of speeds (mph or km/h)
        """
        # Use Greenshields model for speed-density relationship
        critical_density = capacity / free_flow_speed
        
        # Estimate density from flow
        densities = np.where(
            flows <= capacity,
            flows / free_flow_speed,  # Free flow regime
            critical_density + (flows - capacity) / (free_flow_speed * 0.5)  # Congested regime
        )
        
        # Compute speed from density
        speeds = free_flow_speed * (1 - densities / jam_density)
        return np.maximum(speeds, 0)  # Speed cannot be negative
    
    def compute_performance_measures(
        self,
        speeds: np.ndarray,
        flows: np.ndarray,
        time_steps: np.ndarray,
        length: float,
        capacity: float,
        free_flow_speed: float
    ) -> Dict[str, np.ndarray]:
        """
        Compute comprehensive time-varying performance measures.
        
        Args:
            speeds: Array of speeds (mph or km/h)
            flows: Array of flows (veh/h)
            time_steps: Time values (hours)
            length: Segment length (miles or km)
            capacity: Road capacity (veh/h)
            free_flow_speed: Free flow speed (mph or km/h)
            
        Returns:
            Dictionary containing arrays for various metrics
        """
        delays = self.compute_delay_profile(speeds, free_flow_speed, length)
        queues = self.compute_queue_profile(flows, capacity, time_steps)
        
        # Travel times
        travel_times = length / np.maximum(speeds, 0.1)
        
        # Level of Service (simplified based on speed ratio)
        speed_ratio = speeds / free_flow_speed
        los = np.select(
            [
                speed_ratio >= 0.85,
                speed_ratio >= 0.67,
                speed_ratio >= 0.50,
                speed_ratio >= 0.40,
                speed_ratio >= 0.30,
            ],
            ['A', 'B', 'C', 'D', 'E'],
            default='F'
        )
        
        # Volume-to-capacity ratio
        v_c_ratio = flows / capacity
        
        return {
            'delays': delays,
            'queues': queues,
            'travel_times': travel_times,
            'speeds': speeds,
            'flows': flows,
            'v_c_ratio': v_c_ratio,
            'level_of_service': los,
        }
    
    def aggregate_metrics(
        self,
        metrics: Dict[str, np.ndarray],
        aggregation_period: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate time-varying metrics over specified periods.
        
        Args:
            metrics: Dictionary of metric arrays
            aggregation_period: Number of time steps to aggregate over
                               (None for no aggregation)
            
        Returns:
            Dictionary of aggregated metrics
        """
        if aggregation_period is None:
            return metrics
        
        aggregated = {}
        for key, values in metrics.items():
            if key == 'level_of_service':
                # Mode for categorical data
                n = len(values)
                n_groups = n // aggregation_period
                aggregated[key] = np.array([
                    self._mode(values[i*aggregation_period:(i+1)*aggregation_period])
                    for i in range(n_groups)
                ])
            else:
                # Mean for numerical data
                n = len(values)
                n_groups = n // aggregation_period
                aggregated[key] = np.array([
                    np.mean(values[i*aggregation_period:(i+1)*aggregation_period])
                    for i in range(n_groups)
                ])
        
        return aggregated
    
    @staticmethod
    def _mode(arr):
        """Compute mode of array."""
        unique, counts = np.unique(arr, return_counts=True)
        return unique[np.argmax(counts)]
    
    def compute_total_delay(
        self,
        delays: np.ndarray,
        flows: np.ndarray,
        time_steps: np.ndarray
    ) -> float:
        """
        Compute total vehicle-hours of delay.
        
        Args:
            delays: Array of delays (hours)
            flows: Array of flows (veh/h)
            time_steps: Time values (hours)
            
        Returns:
            Total delay (vehicle-hours)
        """
        dt = np.diff(time_steps, prepend=0)
        vehicle_hours = flows * delays * dt
        return np.sum(vehicle_hours)
