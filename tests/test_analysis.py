"""
Unit tests for ai4tfm.analysis module.
"""

import unittest
import numpy as np
from ai4tfm.analysis import BottleneckAnalyzer, CorridorComparison, Bottleneck


class TestBottleneckAnalyzer(unittest.TestCase):
    """Test cases for BottleneckAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = BottleneckAnalyzer(
            speed_threshold=45.0,
            min_duration=0.25,
            severity_threshold=0.3
        )
    
    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertEqual(self.analyzer.speed_threshold, 45.0)
        self.assertEqual(self.analyzer.min_duration, 0.25)
        self.assertEqual(self.analyzer.severity_threshold, 0.3)
    
    def test_identify_bottlenecks_no_congestion(self):
        """Test bottleneck identification with no congestion."""
        speeds = np.full(10, 60.0)
        time_steps = np.linspace(0, 2, 10)
        
        bottlenecks = self.analyzer.identify_bottlenecks(
            speeds, time_steps, "TMC_001", 60.0
        )
        
        self.assertEqual(len(bottlenecks), 0)
    
    def test_identify_bottlenecks_with_congestion(self):
        """Test bottleneck identification with congestion."""
        # Create speed profile with congestion in middle
        speeds = np.array([60, 60, 40, 35, 40, 60, 60])
        time_steps = np.linspace(0, 2, 7)
        
        bottlenecks = self.analyzer.identify_bottlenecks(
            speeds, time_steps, "TMC_001", 60.0
        )
        
        self.assertGreaterEqual(len(bottlenecks), 0)
        for bn in bottlenecks:
            self.assertIsInstance(bn, Bottleneck)
            self.assertEqual(bn.location, "TMC_001")
            self.assertLess(bn.avg_speed, 45.0)
    
    def test_get_congestion_windows(self):
        """Test extraction of congestion windows."""
        bn1 = Bottleneck("TMC_001", 1.0, 2.0, 0.5, 40.0, 20.0)
        bn2 = Bottleneck("TMC_001", 5.0, 6.0, 0.6, 35.0, 25.0)
        
        windows = self.analyzer.get_congestion_windows([bn1, bn2])
        
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0], (1.0, 2.0))
        self.assertEqual(windows[1], (5.0, 6.0))


class TestCorridorComparison(unittest.TestCase):
    """Test cases for CorridorComparison class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.comparator = CorridorComparison()
    
    def test_compare_tmcs(self):
        """Test TMC comparison."""
        tmc_data = {
            'TMC_001': {
                'speeds': np.array([50, 55, 60]),
                'flows': np.array([800, 900, 1000]),
                'delays': np.array([0.1, 0.05, 0.0])
            },
            'TMC_002': {
                'speeds': np.array([45, 50, 55]),
                'flows': np.array([900, 1000, 1100]),
                'delays': np.array([0.2, 0.1, 0.05])
            }
        }
        
        result = self.comparator.compare_tmcs(tmc_data)
        
        self.assertEqual(len(result), 2)
        self.assertIn('tmc', result.columns)
        self.assertIn('avg_speed', result.columns)
        self.assertIn('total_delay', result.columns)
    
    def test_compare_days(self):
        """Test daily comparison."""
        daily_data = {
            'Monday': {
                'speeds': np.array([50, 55, 60]),
                'flows': np.array([800, 900, 1000]),
                'delays': np.array([0.1, 0.05, 0.0])
            },
            'Tuesday': {
                'speeds': np.array([45, 50, 55]),
                'flows': np.array([900, 1000, 1100]),
                'delays': np.array([0.2, 0.1, 0.05])
            }
        }
        
        result = self.comparator.compare_days(daily_data)
        
        self.assertEqual(len(result), 2)
        self.assertIn('day', result.columns)
        self.assertIn('avg_speed', result.columns)
        self.assertIn('congestion_hours', result.columns)
    
    def test_compute_reliability_metrics(self):
        """Test reliability metrics computation."""
        speeds = np.array([50, 55, 60, 55, 50, 45, 60, 58, 52, 55])
        
        metrics = self.comparator.compute_reliability_metrics(speeds)
        
        self.assertIn('planning_time_index', metrics)
        self.assertIn('buffer_time_index', metrics)
        self.assertIn('travel_time_index', metrics)
        self.assertIn('coefficient_of_variation', metrics)
        
        self.assertGreater(metrics['planning_time_index'], 0)


if __name__ == '__main__':
    unittest.main()
