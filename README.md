# AI4TFM: Advanced Traffic Flow Models for MPO Planning

A lightweight Python module to help MPOs (Metropolitan Planning Organizations) analyze congestion with **observed speeds** and **derived flows** using advanced traffic flow models.

## Features

### Core Models
- **Polynomial Arrival Queue (PAQ) Model**: Model traffic flow dynamics with polynomial arrival patterns
- **Queue-based Volume-Delay Function (VDF)**: Synthesize time-varying delay, queue, and speed using static traffic assignment

### Analysis Capabilities
- **Peak-Period Bottleneck Identification**: Automatically detect congestion hotspots and time windows
- **Corridor Performance Comparison**: Compare traffic performance across TMCs (Traffic Message Channels) and days
- **Time-Varying Metrics**: Compute delay, queue length, and speed profiles over time
- **Speed Heatmap Visualization**: Visualize observed vs. modeled speeds with intuitive heatmaps

## Installation

```bash
# Clone the repository
git clone https://github.com/Grieverwzn/AI4TFM.git
cd AI4TFM

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```python
import numpy as np
from ai4tfm import (
    PolynomialArrivalQueue,
    QueueBasedVDF,
    BottleneckAnalyzer,
    CorridorComparison,
    TimeVaryingMetrics,
    SpeedHeatmap
)

# 1. Create a Polynomial Arrival Queue model
paq = PolynomialArrivalQueue(
    capacity=1000,          # vehicles per hour
    free_flow_speed=60,     # mph
    jam_density=200         # vehicles per mile
)

# 2. Generate time steps and arrival rates
time_steps = np.linspace(0, 24, 96)  # 24 hours, 15-min intervals
arrivals = np.array([...])  # Your arrival data

# 3. Compute traffic metrics
queue_lengths, _ = paq.compute_queue(arrivals, time_steps)
delays = paq.compute_delay(arrivals, time_steps, segment_length=2.0)
speeds = paq.compute_speed(arrivals, time_steps, segment_length=2.0)

# 4. Identify bottlenecks
analyzer = BottleneckAnalyzer(speed_threshold=45.0)
bottlenecks = analyzer.identify_bottlenecks(
    speeds, time_steps, location="TMC_001", free_flow_speed=60
)

# 5. Visualize results
viz = SpeedHeatmap()
fig = viz.plot_congestion_profile(
    time_steps, speeds,
    congestion_threshold=45.0,
    free_flow_speed=60.0
)
viz.save_figure(fig, 'congestion_profile.png')
```

## Module Overview

### `ai4tfm.models`

#### PolynomialArrivalQueue
Models traffic queue dynamics using polynomial arrival patterns.

**Methods:**
- `compute_queue(arrivals, time_steps)`: Calculate queue lengths over time
- `compute_delay(arrivals, time_steps, length)`: Calculate time-varying delays
- `compute_speed(arrivals, time_steps, length)`: Calculate time-varying speeds

#### QueueBasedVDF
Queue-based Volume-Delay Function for traffic assignment.

**Methods:**
- `compute_travel_time(flow, capacity, free_flow_time)`: BPR-based travel time
- `compute_speed(flow, capacity, free_flow_speed, length)`: Speed from flow
- `compute_delay(flow, capacity, free_flow_time)`: Delay calculation
- `synthesize_time_varying(...)`: Generate time-varying metrics from static flows

### `ai4tfm.analysis`

#### BottleneckAnalyzer
Identify congestion bottlenecks and peak periods.

**Methods:**
- `identify_bottlenecks(speeds, time_steps, location, free_flow_speed)`: Detect bottlenecks
- `get_congestion_windows(bottlenecks)`: Extract congestion time windows

#### CorridorComparison
Compare performance across locations and time periods.

**Methods:**
- `compare_tmcs(tmc_data)`: Compare multiple TMC segments
- `compare_days(daily_data)`: Compare performance across days
- `compute_reliability_metrics(speeds)`: Calculate reliability indices

### `ai4tfm.metrics`

#### TimeVaryingMetrics
Compute comprehensive time-varying performance measures.

**Methods:**
- `compute_delay_profile(speeds, free_flow_speed, length)`: Delay over time
- `compute_queue_profile(flows, capacity, time_steps)`: Queue length over time
- `compute_speed_profile(flows, capacity, free_flow_speed)`: Speed from flow
- `compute_performance_measures(...)`: All metrics in one call
- `compute_total_delay(delays, flows, time_steps)`: Total vehicle-hours of delay

### `ai4tfm.visualization`

#### SpeedHeatmap
Create visualizations for traffic speed analysis.

**Methods:**
- `plot_single_heatmap(speeds, ...)`: Single speed heatmap
- `plot_comparison_heatmap(observed_speeds, modeled_speeds, ...)`: Compare observed vs. modeled
- `plot_time_series(time_steps, observed_speeds, modeled_speeds, ...)`: Time series plot
- `plot_congestion_profile(time_steps, speeds, ...)`: Congestion visualization
- `save_figure(fig, filename)`: Save visualization to file

## Examples

See the `examples/` directory for complete usage examples:

```bash
python examples/example_usage.py
```

This will demonstrate:
1. Polynomial Arrival Queue modeling
2. Queue-based VDF application
3. Bottleneck identification
4. Corridor performance comparison
5. Time-varying metrics computation
6. Speed heatmap visualization

## Use Cases

### 1. Congestion Analysis for MPO Planning
```python
# Identify where and when congestion occurs
analyzer = BottleneckAnalyzer(speed_threshold=45.0)
bottlenecks = analyzer.identify_bottlenecks(speeds, times, "I-95", 65)

for bn in bottlenecks:
    print(f"Bottleneck at {bn.location}: {bn.start_time}-{bn.end_time}")
    print(f"  Severity: {bn.severity}, Avg Speed: {bn.avg_speed} mph")
```

### 2. Corridor Performance Evaluation
```python
# Compare multiple road segments
comparator = CorridorComparison()
comparison = comparator.compare_tmcs(tmc_data)
print(comparison[['tmc', 'avg_speed', 'total_delay', 'avg_delay']])
```

### 3. Before/After Analysis
```python
# Compare different days or scenarios
daily_comparison = comparator.compare_days({
    'Before': {'speeds': speeds_before, 'flows': flows_before, ...},
    'After': {'speeds': speeds_after, 'flows': flows_after, ...}
})
```

### 4. Traffic Model Calibration
```python
# Compare observed vs. modeled speeds
viz = SpeedHeatmap()
fig = viz.plot_comparison_heatmap(
    observed_speeds, modeled_speeds,
    time_labels, location_labels
)
```

## Data Requirements

The module works with common traffic data formats:

- **Speeds**: mph or km/h (consistent units throughout)
- **Flows**: vehicles per hour
- **Time**: hours (can be decimal, e.g., 7.5 for 7:30 AM)
- **Lengths**: miles or km (consistent with speed units)
- **Capacity**: vehicles per hour

## Dependencies

- numpy >= 1.20.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0

## License

This project is open source and available for use in traffic analysis and planning applications.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Citation

If you use AI4TFM in your research or planning work, please cite:

```
AI4TFM: Advanced Traffic Flow Models for MPO Planning
https://github.com/Grieverwzn/AI4TFM
```

## Support

For questions, issues, or feature requests, please open an issue on GitHub.
