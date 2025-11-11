"""
Unit tests for ai4tfm.metrics module.
"""

import unittest
import numpy as np
from ai4tfm.metrics import TimeVaryingMetrics


class TestTimeVaryingMetrics(unittest.TestCase):
    """Test cases for TimeVaryingMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = TimeVaryingMetrics()
        self.speeds = np.array([60, 55, 50, 45, 50, 55, 60])
        self.flows = np.array([500, 700, 900, 1000, 900, 700, 500])
        self.time_steps = np.linspace(0, 2, 7)
    
    def test_compute_delay_profile(self):
        """Test delay profile computation."""
        delays = self.metrics.compute_delay_profile(
            self.speeds, free_flow_speed=60, length=2.0
        )
        
        self.assertEqual(len(delays), len(self.speeds))
        self.assertTrue(np.all(delays >= 0))
    
    def test_compute_queue_profile(self):
        """Test queue profile computation."""
        queues = self.metrics.compute_queue_profile(
            self.flows, capacity=1000, time_steps=self.time_steps
        )
        
        self.assertEqual(len(queues), len(self.flows))
        self.assertTrue(np.all(queues >= 0))
    
    def test_compute_speed_profile(self):
        """Test speed profile computation."""
        speeds = self.metrics.compute_speed_profile(
            self.flows, capacity=1000, free_flow_speed=60, jam_density=200
        )
        
        self.assertEqual(len(speeds), len(self.flows))
        self.assertTrue(np.all(speeds >= 0))
        self.assertTrue(np.all(speeds <= 60))
    
    def test_compute_performance_measures(self):
        """Test comprehensive performance measures."""
        measures = self.metrics.compute_performance_measures(
            speeds=self.speeds,
            flows=self.flows,
            time_steps=self.time_steps,
            length=2.0,
            capacity=1000,
            free_flow_speed=60
        )
        
        self.assertIn('delays', measures)
        self.assertIn('queues', measures)
        self.assertIn('travel_times', measures)
        self.assertIn('speeds', measures)
        self.assertIn('flows', measures)
        self.assertIn('v_c_ratio', measures)
        self.assertIn('level_of_service', measures)
        
        # Check array lengths
        for key in measures:
            self.assertEqual(len(measures[key]), len(self.speeds))
    
    def test_compute_total_delay(self):
        """Test total delay computation."""
        delays = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
        flows = np.array([800, 900, 1000, 900, 800])
        time_steps = np.linspace(0, 1, 5)
        
        total_delay = self.metrics.compute_total_delay(delays, flows, time_steps)
        
        self.assertGreater(total_delay, 0)
        self.assertIsInstance(total_delay, (int, float))
    
    def test_aggregate_metrics(self):
        """Test metrics aggregation."""
        measures = {
            'speeds': np.array([60, 55, 50, 45, 50, 55]),
            'flows': np.array([500, 700, 900, 1000, 900, 700]),
            'level_of_service': np.array(['A', 'B', 'C', 'D', 'C', 'B'])
        }
        
        aggregated = self.metrics.aggregate_metrics(measures, aggregation_period=2)
        
        self.assertEqual(len(aggregated['speeds']), 3)
        self.assertEqual(len(aggregated['flows']), 3)
        self.assertEqual(len(aggregated['level_of_service']), 3)


if __name__ == '__main__':
    unittest.main()
