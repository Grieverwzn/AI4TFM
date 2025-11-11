"""
AI4TFM: Advanced Traffic Flow Models for MPO Planning

A lightweight Python module to help MPOs analyze congestion with observed speeds
and derived flows using Polynomial Arrival Queue models and Queue-based Volume-Delay Functions.
"""

__version__ = "0.1.0"

from .models import PolynomialArrivalQueue, QueueBasedVDF
from .analysis import BottleneckAnalyzer, CorridorComparison
from .metrics import TimeVaryingMetrics
from .visualization import SpeedHeatmap

__all__ = [
    "PolynomialArrivalQueue",
    "QueueBasedVDF",
    "BottleneckAnalyzer",
    "CorridorComparison",
    "TimeVaryingMetrics",
    "SpeedHeatmap",
]
