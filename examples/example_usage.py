"""
Example usage of AI4TFM for congestion analysis.

This script demonstrates how to use the AI4TFM module to:
1. Model traffic flow with Polynomial Arrival Queue
2. Apply Queue-based VDF
3. Identify bottlenecks
4. Compare corridor performance
5. Visualize speed heatmaps
"""

import numpy as np
import sys
sys.path.insert(0, '/home/runner/work/AI4TFM/AI4TFM')

from ai4tfm import (
    PolynomialArrivalQueue,
    QueueBasedVDF,
    BottleneckAnalyzer,
    CorridorComparison,
    TimeVaryingMetrics,
    SpeedHeatmap
)


def generate_sample_data():
    """Generate sample traffic data for demonstration."""
    # Time steps for one day (24 hours, 15-minute intervals)
    time_steps = np.linspace(0, 24, 96)
    
    # Simulated arrival pattern (peak in morning and evening)
    base_flow = 500
    morning_peak = 1200 * np.exp(-((time_steps - 8)**2) / 4)
    evening_peak = 1400 * np.exp(-((time_steps - 17)**2) / 4)
    arrivals = base_flow + morning_peak + evening_peak
    
    return time_steps, arrivals


def example_polynomial_arrival_queue():
    """Demonstrate Polynomial Arrival Queue model."""
    print("=" * 60)
    print("Example 1: Polynomial Arrival Queue Model")
    print("=" * 60)
    
    # Initialize model
    paq = PolynomialArrivalQueue(
        capacity=1000,  # veh/h
        free_flow_speed=60,  # mph
        jam_density=200  # veh/mile
    )
    
    # Generate sample data
    time_steps, arrivals = generate_sample_data()
    segment_length = 2.0  # miles
    
    # Compute metrics
    queue_lengths, departures = paq.compute_queue(arrivals, time_steps)
    delays = paq.compute_delay(arrivals, time_steps, segment_length)
    speeds = paq.compute_speed(arrivals, time_steps, segment_length)
    
    print(f"Time period: {time_steps[0]:.1f} to {time_steps[-1]:.1f} hours")
    print(f"Max queue length: {np.max(queue_lengths):.1f} vehicles")
    print(f"Max delay: {np.max(delays) * 60:.1f} minutes")
    print(f"Min speed: {np.min(speeds):.1f} mph")
    print(f"Avg speed: {np.mean(speeds):.1f} mph")
    print()


def example_queue_based_vdf():
    """Demonstrate Queue-based Volume-Delay Function."""
    print("=" * 60)
    print("Example 2: Queue-based Volume-Delay Function")
    print("=" * 60)
    
    # Initialize VDF
    vdf = QueueBasedVDF(alpha=0.15, beta=4.0)
    
    # Sample flows for different congestion levels
    flows = np.array([500, 800, 1000, 1200, 1400])
    capacity = 1000
    free_flow_time = 0.05  # hours (3 minutes)
    
    print(f"Capacity: {capacity} veh/h")
    print(f"Free-flow time: {free_flow_time * 60:.1f} minutes")
    print()
    print("Flow (veh/h) | Travel Time (min) | Delay (min) | V/C Ratio")
    print("-" * 65)
    
    for flow in flows:
        travel_time = vdf.compute_travel_time(flow, capacity, free_flow_time)
        delay = vdf.compute_delay(flow, capacity, free_flow_time)
        vc_ratio = flow / capacity
        print(f"{flow:12.0f} | {travel_time * 60:17.2f} | {delay * 60:11.2f} | {vc_ratio:9.2f}")
    print()


def example_bottleneck_analysis():
    """Demonstrate bottleneck identification."""
    print("=" * 60)
    print("Example 3: Bottleneck Identification")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = BottleneckAnalyzer(
        speed_threshold=45.0,
        min_duration=0.25,  # 15 minutes
        severity_threshold=0.3
    )
    
    # Generate sample data
    time_steps, arrivals = generate_sample_data()
    paq = PolynomialArrivalQueue(capacity=1000, free_flow_speed=60)
    speeds = paq.compute_speed(arrivals, time_steps, 2.0)
    
    # Identify bottlenecks
    bottlenecks = analyzer.identify_bottlenecks(
        speeds, time_steps, location="TMC_001", free_flow_speed=60
    )
    
    print(f"Found {len(bottlenecks)} bottleneck(s):")
    for i, bn in enumerate(bottlenecks, 1):
        print(f"\nBottleneck {i}:")
        print(f"  Location: {bn.location}")
        print(f"  Time window: {bn.start_time:.2f} - {bn.end_time:.2f} hours")
        print(f"  Duration: {(bn.end_time - bn.start_time) * 60:.1f} minutes")
        print(f"  Severity: {bn.severity:.3f}")
        print(f"  Avg speed: {bn.avg_speed:.1f} mph")
        print(f"  Speed drop: {bn.speed_drop:.1f} mph")
    print()


def example_corridor_comparison():
    """Demonstrate corridor performance comparison."""
    print("=" * 60)
    print("Example 4: Corridor Performance Comparison")
    print("=" * 60)
    
    # Initialize comparison tool
    comparator = CorridorComparison()
    
    # Generate sample data for multiple TMCs
    time_steps, _ = generate_sample_data()
    tmc_data = {}
    
    for i, tmc in enumerate(['TMC_001', 'TMC_002', 'TMC_003']):
        capacity = 900 + i * 100
        paq = PolynomialArrivalQueue(capacity=capacity, free_flow_speed=60)
        
        # Add some variation
        arrivals = generate_sample_data()[1] * (0.9 + i * 0.1)
        speeds = paq.compute_speed(arrivals, time_steps, 2.0)
        delays = paq.compute_delay(arrivals, time_steps, 2.0)
        
        tmc_data[tmc] = {
            'speeds': speeds,
            'flows': arrivals,
            'delays': delays
        }
    
    # Compare TMCs
    comparison_df = comparator.compare_tmcs(tmc_data)
    print("TMC Comparison:")
    print(comparison_df.to_string(index=False))
    print()


def example_time_varying_metrics():
    """Demonstrate time-varying metrics computation."""
    print("=" * 60)
    print("Example 5: Time-Varying Metrics")
    print("=" * 60)
    
    # Initialize metrics calculator
    metrics_calc = TimeVaryingMetrics()
    
    # Generate sample data
    time_steps, arrivals = generate_sample_data()
    paq = PolynomialArrivalQueue(capacity=1000, free_flow_speed=60)
    speeds = paq.compute_speed(arrivals, time_steps, 2.0)
    
    # Compute comprehensive metrics
    all_metrics = metrics_calc.compute_performance_measures(
        speeds=speeds,
        flows=arrivals,
        time_steps=time_steps,
        length=2.0,
        capacity=1000,
        free_flow_speed=60
    )
    
    print("Performance Measures Summary:")
    print(f"Total delay: {np.sum(all_metrics['delays']) * 60:.1f} vehicle-minutes")
    print(f"Max queue: {np.max(all_metrics['queues']):.1f} vehicles")
    print(f"Avg travel time: {np.mean(all_metrics['travel_times']) * 60:.1f} minutes")
    print(f"Max V/C ratio: {np.max(all_metrics['v_c_ratio']):.3f}")
    
    # Count by Level of Service
    los_counts = {}
    for los in ['A', 'B', 'C', 'D', 'E', 'F']:
        count = np.sum(all_metrics['level_of_service'] == los)
        if count > 0:
            los_counts[los] = count
    
    print("\nLevel of Service Distribution:")
    for los, count in los_counts.items():
        percentage = (count / len(all_metrics['level_of_service'])) * 100
        print(f"  LOS {los}: {count} periods ({percentage:.1f}%)")
    print()


def example_visualization():
    """Demonstrate speed heatmap visualization."""
    print("=" * 60)
    print("Example 6: Speed Heatmap Visualization")
    print("=" * 60)
    
    # Initialize visualizer
    viz = SpeedHeatmap(figsize=(12, 6))
    
    # Generate sample data for multiple locations
    time_steps, arrivals = generate_sample_data()
    n_locations = 5
    n_times = len(time_steps)
    
    observed_speeds = np.zeros((n_locations, n_times))
    modeled_speeds = np.zeros((n_locations, n_times))
    
    for i in range(n_locations):
        capacity = 900 + i * 50
        paq = PolynomialArrivalQueue(capacity=capacity, free_flow_speed=60)
        
        # Modeled speeds
        modeled_speeds[i, :] = paq.compute_speed(arrivals, time_steps, 2.0)
        
        # Add noise for observed speeds
        noise = np.random.normal(0, 3, n_times)
        observed_speeds[i, :] = modeled_speeds[i, :] + noise
    
    # Create labels
    time_labels = [f"{int(t)}:00" for t in time_steps[::12]]  # Every 3 hours
    location_labels = [f"TMC_{i:03d}" for i in range(1, n_locations + 1)]
    
    # Create comparison heatmap
    fig = viz.plot_comparison_heatmap(
        observed_speeds[:, ::12],  # Sample every 3 hours for clarity
        modeled_speeds[:, ::12],
        time_labels,
        location_labels,
        vmin=20,
        vmax=65
    )
    
    # Save figure
    output_file = '/tmp/speed_heatmap_comparison.png'
    viz.save_figure(fig, output_file)
    print(f"Speed heatmap saved to: {output_file}")
    
    # Create congestion profile for single location
    fig2 = viz.plot_congestion_profile(
        time_steps,
        modeled_speeds[2, :],  # Middle location
        congestion_threshold=45.0,
        free_flow_speed=60.0
    )
    
    output_file2 = '/tmp/congestion_profile.png'
    viz.save_figure(fig2, output_file2)
    print(f"Congestion profile saved to: {output_file2}")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("*" * 60)
    print("AI4TFM: Advanced Traffic Flow Models for MPO Planning")
    print("Example Usage Demonstrations")
    print("*" * 60)
    print()
    
    example_polynomial_arrival_queue()
    example_queue_based_vdf()
    example_bottleneck_analysis()
    example_corridor_comparison()
    example_time_varying_metrics()
    example_visualization()
    
    print("*" * 60)
    print("All examples completed successfully!")
    print("*" * 60)


if __name__ == "__main__":
    main()
