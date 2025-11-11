"""
Unit tests for ai4tfm.models module.
"""

import unittest
import numpy as np
from ai4tfm.models import PolynomialArrivalQueue, QueueBasedVDF


class TestPolynomialArrivalQueue(unittest.TestCase):
    """Test cases for PolynomialArrivalQueue class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.paq = PolynomialArrivalQueue(
            capacity=1000,
            free_flow_speed=60,
            jam_density=200
        )
        self.time_steps = np.linspace(0, 1, 10)
        self.arrivals = np.full(10, 500)
    
    def test_initialization(self):
        """Test PAQ initialization."""
        self.assertEqual(self.paq.capacity, 1000)
        self.assertEqual(self.paq.free_flow_speed, 60)
        self.assertEqual(self.paq.jam_density, 200)
    
    def test_compute_queue_no_congestion(self):
        """Test queue computation with arrivals below capacity."""
        queue_lengths, departures = self.paq.compute_queue(
            self.arrivals, self.time_steps
        )
        self.assertEqual(len(queue_lengths), len(self.time_steps))
        self.assertTrue(np.all(queue_lengths >= 0))
    
    def test_compute_delay(self):
        """Test delay computation."""
        delays = self.paq.compute_delay(self.arrivals, self.time_steps, length=2.0)
        self.assertEqual(len(delays), len(self.time_steps))
        self.assertTrue(np.all(delays >= 0))
    
    def test_compute_speed(self):
        """Test speed computation."""
        speeds = self.paq.compute_speed(self.arrivals, self.time_steps, length=2.0)
        self.assertEqual(len(speeds), len(self.time_steps))
        self.assertTrue(np.all(speeds > 0))
        self.assertTrue(np.all(speeds <= self.paq.free_flow_speed))


class TestQueueBasedVDF(unittest.TestCase):
    """Test cases for QueueBasedVDF class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vdf = QueueBasedVDF(alpha=0.15, beta=4.0)
        self.capacity = 1000
        self.free_flow_time = 0.05
    
    def test_initialization(self):
        """Test VDF initialization."""
        self.assertEqual(self.vdf.alpha, 0.15)
        self.assertEqual(self.vdf.beta, 4.0)
    
    def test_compute_travel_time_low_flow(self):
        """Test travel time with flow below capacity."""
        flow = 500
        travel_time = self.vdf.compute_travel_time(
            flow, self.capacity, self.free_flow_time
        )
        self.assertGreater(travel_time, self.free_flow_time)
        self.assertLess(travel_time, self.free_flow_time * 2)
    
    def test_compute_travel_time_at_capacity(self):
        """Test travel time at capacity."""
        flow = self.capacity
        travel_time = self.vdf.compute_travel_time(
            flow, self.capacity, self.free_flow_time
        )
        expected = self.free_flow_time * (1 + self.vdf.alpha * 1**self.vdf.beta)
        self.assertAlmostEqual(travel_time, expected, places=5)
    
    def test_compute_speed(self):
        """Test speed computation."""
        flow = 800
        speed = self.vdf.compute_speed(flow, self.capacity, 60, 2.0)
        self.assertGreater(speed, 0)
        self.assertLessEqual(speed, 60)
    
    def test_compute_delay(self):
        """Test delay computation."""
        flow = 800
        delay = self.vdf.compute_delay(flow, self.capacity, self.free_flow_time)
        self.assertGreaterEqual(delay, 0)
    
    def test_synthesize_time_varying(self):
        """Test time-varying synthesis."""
        static_flows = np.array([500, 800, 1000, 800, 500])
        time_steps = np.linspace(0, 1, 5)
        
        delays, queues, speeds = self.vdf.synthesize_time_varying(
            static_flows, self.capacity, self.free_flow_time, time_steps
        )
        
        self.assertEqual(len(delays), len(static_flows))
        self.assertEqual(len(queues), len(static_flows))
        self.assertEqual(len(speeds), len(static_flows))


if __name__ == '__main__':
    unittest.main()
