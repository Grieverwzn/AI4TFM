"""
Analysis tools for bottleneck identification and corridor performance comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Bottleneck:
    """
    Represents an identified bottleneck.
    
    Attributes:
        location: Location identifier (e.g., TMC code)
        start_time: Start of congestion window (hours or datetime)
        end_time: End of congestion window (hours or datetime)
        severity: Severity score (0-1)
        avg_speed: Average speed during congestion
        speed_drop: Speed drop from free-flow
    """
    location: str
    start_time: float
    end_time: float
    severity: float
    avg_speed: float
    speed_drop: float


class BottleneckAnalyzer:
    """
    Identify peak-period bottlenecks and congestion windows.
    
    This class analyzes traffic data to identify locations and times
    where significant congestion occurs.
    """
    
    def __init__(
        self,
        speed_threshold: float = 45.0,
        min_duration: float = 0.25,
        severity_threshold: float = 0.3
    ):
        """
        Initialize the BottleneckAnalyzer.
        
        Args:
            speed_threshold: Speed below which indicates congestion (mph/km/h)
            min_duration: Minimum duration for a valid bottleneck (hours)
            severity_threshold: Minimum severity score (0-1) to report
        """
        self.speed_threshold = speed_threshold
        self.min_duration = min_duration
        self.severity_threshold = severity_threshold
    
    def identify_bottlenecks(
        self,
        speeds: np.ndarray,
        time_steps: np.ndarray,
        location: str,
        free_flow_speed: float
    ) -> List[Bottleneck]:
        """
        Identify bottlenecks from speed data.
        
        Args:
            speeds: Array of observed or modeled speeds
            time_steps: Array of time values (hours)
            location: Location identifier
            free_flow_speed: Free flow speed for reference
            
        Returns:
            List of identified Bottleneck objects
        """
        # Find congested periods
        is_congested = speeds < self.speed_threshold
        
        bottlenecks = []
        in_congestion = False
        start_idx = 0
        
        for i, congested in enumerate(is_congested):
            if congested and not in_congestion:
                # Start of congestion
                in_congestion = True
                start_idx = i
            elif not congested and in_congestion:
                # End of congestion
                duration = time_steps[i] - time_steps[start_idx]
                if duration >= self.min_duration:
                    bottleneck = self._create_bottleneck(
                        location, time_steps[start_idx:i], speeds[start_idx:i],
                        free_flow_speed, time_steps[start_idx], time_steps[i-1]
                    )
                    if bottleneck.severity >= self.severity_threshold:
                        bottlenecks.append(bottleneck)
                in_congestion = False
        
        # Handle case where congestion extends to end
        if in_congestion:
            duration = time_steps[-1] - time_steps[start_idx]
            if duration >= self.min_duration:
                bottleneck = self._create_bottleneck(
                    location, time_steps[start_idx:], speeds[start_idx:],
                    free_flow_speed, time_steps[start_idx], time_steps[-1]
                )
                if bottleneck.severity >= self.severity_threshold:
                    bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _create_bottleneck(
        self,
        location: str,
        times: np.ndarray,
        speeds: np.ndarray,
        free_flow_speed: float,
        start_time: float,
        end_time: float
    ) -> Bottleneck:
        """Create a Bottleneck object from congestion data."""
        avg_speed = np.mean(speeds)
        speed_drop = free_flow_speed - avg_speed
        
        # Severity based on speed reduction and duration
        speed_ratio = avg_speed / free_flow_speed
        duration = end_time - start_time
        severity = (1 - speed_ratio) * min(duration / 2.0, 1.0)  # Normalize to 0-1
        
        return Bottleneck(
            location=location,
            start_time=start_time,
            end_time=end_time,
            severity=severity,
            avg_speed=avg_speed,
            speed_drop=speed_drop
        )
    
    def get_congestion_windows(
        self,
        bottlenecks: List[Bottleneck]
    ) -> List[Tuple[float, float]]:
        """
        Extract congestion time windows from bottlenecks.
        
        Args:
            bottlenecks: List of Bottleneck objects
            
        Returns:
            List of (start_time, end_time) tuples
        """
        return [(b.start_time, b.end_time) for b in bottlenecks]


class CorridorComparison:
    """
    Compare corridor performance across TMCs (Traffic Message Channels) and days.
    
    This class provides tools to analyze and compare traffic performance
    across different road segments and time periods.
    """
    
    def __init__(self):
        """Initialize CorridorComparison analyzer."""
        pass
    
    def compare_tmcs(
        self,
        tmc_data: Dict[str, Dict[str, np.ndarray]]
    ) -> pd.DataFrame:
        """
        Compare performance metrics across multiple TMCs.
        
        Args:
            tmc_data: Dictionary with TMC codes as keys and dictionaries
                     containing 'speeds', 'flows', 'delays' arrays as values
                     
        Returns:
            DataFrame with comparison metrics for each TMC
        """
        results = []
        
        for tmc_code, data in tmc_data.items():
            speeds = data.get('speeds', np.array([]))
            flows = data.get('flows', np.array([]))
            delays = data.get('delays', np.array([]))
            
            metrics = {
                'tmc': tmc_code,
                'avg_speed': np.mean(speeds) if len(speeds) > 0 else np.nan,
                'min_speed': np.min(speeds) if len(speeds) > 0 else np.nan,
                'max_speed': np.max(speeds) if len(speeds) > 0 else np.nan,
                'std_speed': np.std(speeds) if len(speeds) > 0 else np.nan,
                'avg_flow': np.mean(flows) if len(flows) > 0 else np.nan,
                'total_delay': np.sum(delays) if len(delays) > 0 else np.nan,
                'avg_delay': np.mean(delays) if len(delays) > 0 else np.nan,
            }
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def compare_days(
        self,
        daily_data: Dict[str, Dict[str, np.ndarray]]
    ) -> pd.DataFrame:
        """
        Compare performance metrics across different days.
        
        Args:
            daily_data: Dictionary with day labels as keys and dictionaries
                       containing 'speeds', 'flows', 'delays' arrays as values
                       
        Returns:
            DataFrame with comparison metrics for each day
        """
        results = []
        
        for day_label, data in daily_data.items():
            speeds = data.get('speeds', np.array([]))
            flows = data.get('flows', np.array([]))
            delays = data.get('delays', np.array([]))
            
            metrics = {
                'day': day_label,
                'avg_speed': np.mean(speeds) if len(speeds) > 0 else np.nan,
                'min_speed': np.min(speeds) if len(speeds) > 0 else np.nan,
                'peak_congestion': np.min(speeds) if len(speeds) > 0 else np.nan,
                'avg_flow': np.mean(flows) if len(flows) > 0 else np.nan,
                'max_flow': np.max(flows) if len(flows) > 0 else np.nan,
                'total_delay': np.sum(delays) if len(delays) > 0 else np.nan,
                'congestion_hours': np.sum(speeds < 45.0) if len(speeds) > 0 else 0,
            }
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def compute_reliability_metrics(
        self,
        speeds: np.ndarray,
        reference_speed: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute reliability metrics for corridor performance.
        
        Args:
            speeds: Array of speed observations
            reference_speed: Reference speed (default: 95th percentile)
            
        Returns:
            Dictionary with reliability metrics
        """
        if reference_speed is None:
            reference_speed = np.percentile(speeds, 95)
        
        metrics = {
            'planning_time_index': np.percentile(speeds, 95) / np.mean(speeds),
            'buffer_time_index': (np.percentile(speeds, 95) - np.mean(speeds)) / np.mean(speeds),
            'travel_time_index': reference_speed / np.mean(speeds),
            'std_dev': np.std(speeds),
            'coefficient_of_variation': np.std(speeds) / np.mean(speeds) if np.mean(speeds) > 0 else np.nan,
        }
        
        return metrics
