"""
Traffic flow models including Polynomial Arrival Queue and Queue-based VDF.
"""

import numpy as np
from typing import Union, Tuple, Optional


class PolynomialArrivalQueue:
    """
    Polynomial Arrival Queue (PAQ) model for traffic flow analysis.
    
    This model uses polynomial functions to represent arrival patterns and
    compute queue dynamics, delays, and speeds over time.
    
    Attributes:
        capacity: Road segment capacity (vehicles per hour)
        free_flow_speed: Free flow speed (mph or km/h)
        jam_density: Jam density (vehicles per mile or km)
    """
    
    def __init__(
        self,
        capacity: float,
        free_flow_speed: float,
        jam_density: float = 200.0
    ):
        """
        Initialize the Polynomial Arrival Queue model.
        
        Args:
            capacity: Road segment capacity (veh/h)
            free_flow_speed: Free flow speed (mph or km/h)
            jam_density: Jam density (veh/mile or veh/km), default 200
        """
        self.capacity = capacity
        self.free_flow_speed = free_flow_speed
        self.jam_density = jam_density
        
    def compute_queue(
        self,
        arrivals: np.ndarray,
        time_steps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute queue length over time using the arrival queue model.
        
        Args:
            arrivals: Array of arrival rates (veh/h) at each time step
            time_steps: Array of time values (hours)
            
        Returns:
            Tuple of (queue_lengths, cumulative_departures)
        """
        dt = np.diff(time_steps, prepend=time_steps[0])
        cumulative_arrivals = np.cumsum(arrivals * dt)
        cumulative_departures = np.minimum(
            cumulative_arrivals,
            self.capacity * time_steps
        )
        
        queue_lengths = cumulative_arrivals - cumulative_departures
        return queue_lengths, cumulative_departures
    
    def compute_delay(
        self,
        arrivals: np.ndarray,
        time_steps: np.ndarray,
        length: float
    ) -> np.ndarray:
        """
        Compute time-varying delay for each vehicle.
        
        Args:
            arrivals: Array of arrival rates (veh/h)
            time_steps: Array of time values (hours)
            length: Segment length (miles or km)
            
        Returns:
            Array of delays (hours) at each time step
        """
        queue_lengths, _ = self.compute_queue(arrivals, time_steps)
        free_flow_time = length / self.free_flow_speed
        
        # Delay is proportional to queue length
        delay = (queue_lengths / self.capacity) * free_flow_time
        return delay
    
    def compute_speed(
        self,
        arrivals: np.ndarray,
        time_steps: np.ndarray,
        length: float
    ) -> np.ndarray:
        """
        Compute time-varying speed based on queue dynamics.
        
        Args:
            arrivals: Array of arrival rates (veh/h)
            time_steps: Array of time values (hours)
            length: Segment length (miles or km)
            
        Returns:
            Array of speeds (mph or km/h) at each time step
        """
        delays = self.compute_delay(arrivals, time_steps, length)
        free_flow_time = length / self.free_flow_speed
        total_time = free_flow_time + delays
        
        # Avoid division by zero
        speeds = np.where(total_time > 0, length / total_time, self.free_flow_speed)
        return speeds


class QueueBasedVDF:
    """
    Queue-based Volume-Delay Function (VDF) for static traffic assignment.
    
    This class synthesizes time-varying delay, queue, and speed using
    static traffic assignment results and queue dynamics.
    
    Attributes:
        alpha: VDF parameter (default 0.15)
        beta: VDF parameter (default 4.0)
    """
    
    def __init__(self, alpha: float = 0.15, beta: float = 4.0):
        """
        Initialize Queue-based VDF with BPR-like parameters.
        
        Args:
            alpha: Calibration parameter (default 0.15)
            beta: Congestion sensitivity parameter (default 4.0)
        """
        self.alpha = alpha
        self.beta = beta
    
    def compute_travel_time(
        self,
        flow: Union[float, np.ndarray],
        capacity: float,
        free_flow_time: float
    ) -> Union[float, np.ndarray]:
        """
        Compute travel time using the queue-based VDF.
        
        Uses the Bureau of Public Roads (BPR) function:
        t = t0 * (1 + alpha * (flow/capacity)^beta)
        
        Args:
            flow: Traffic flow (veh/h)
            capacity: Road capacity (veh/h)
            free_flow_time: Free flow travel time (hours)
            
        Returns:
            Travel time (hours)
        """
        ratio = np.maximum(flow / capacity, 0)
        travel_time = free_flow_time * (1 + self.alpha * np.power(ratio, self.beta))
        return travel_time
    
    def compute_speed(
        self,
        flow: Union[float, np.ndarray],
        capacity: float,
        free_flow_speed: float,
        length: float
    ) -> Union[float, np.ndarray]:
        """
        Compute speed based on flow using VDF.
        
        Args:
            flow: Traffic flow (veh/h)
            capacity: Road capacity (veh/h)
            free_flow_speed: Free flow speed (mph or km/h)
            length: Segment length (miles or km)
            
        Returns:
            Speed (mph or km/h)
        """
        free_flow_time = length / free_flow_speed
        travel_time = self.compute_travel_time(flow, capacity, free_flow_time)
        speed = length / travel_time
        return speed
    
    def compute_delay(
        self,
        flow: Union[float, np.ndarray],
        capacity: float,
        free_flow_time: float
    ) -> Union[float, np.ndarray]:
        """
        Compute delay using VDF.
        
        Args:
            flow: Traffic flow (veh/h)
            capacity: Road capacity (veh/h)
            free_flow_time: Free flow travel time (hours)
            
        Returns:
            Delay (hours)
        """
        travel_time = self.compute_travel_time(flow, capacity, free_flow_time)
        delay = travel_time - free_flow_time
        return delay
    
    def synthesize_time_varying(
        self,
        static_flows: np.ndarray,
        capacity: float,
        free_flow_time: float,
        time_steps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Synthesize time-varying delay, queue, and speed from static assignment.
        
        Args:
            static_flows: Static flow values for each time period
            capacity: Road capacity (veh/h)
            free_flow_time: Free flow travel time (hours)
            time_steps: Time step array (hours)
            
        Returns:
            Tuple of (delays, queues, speeds)
        """
        delays = self.compute_delay(static_flows, capacity, free_flow_time)
        
        # Approximate queue from delay
        queues = (static_flows / capacity) * delays * capacity
        
        # Compute speeds
        total_times = free_flow_time + delays
        speeds = np.where(total_times > 0, 1.0 / total_times, 1.0 / free_flow_time)
        
        return delays, queues, speeds
