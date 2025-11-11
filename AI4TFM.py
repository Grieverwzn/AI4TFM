#!/usr/bin/env python3
"""
Advanced Traffic Flow Models Module
====================================
Implements S3, PAQ, and QVDF models for research and education

Add to main ai4tfm.py by importing these functions

Models included:
- S3 (S-Shaped Three-Parameter) Model
- PAQ (Polynomial Arrival Queue) Model  
- QVDF (Queue-Based Volume-Delay Function)
- Queue Shockwave Analysis

References:
    - Zhou, X.S., Cheng, Q., Wu, X., Li, P., Belezamo, B., Lu, J. and Abbasi, M., 2022.
    A meso-to-macro cross-resolution performance approach for connecting polynomial arrival queue model to
    volume-delay function with inflow demand-to-capacity ratio.
    Multimodal Transportation, 1(2), p.100017.
    - Wu, X., Dutta, A., Zhang, W., Zhu, H., Livshits, V. and Zhou, X.S., 2021.
    Characterization and calibration of volume-to-capacity ratio in volume-delay functions on freeways based on a
    queue analysis approach.
    In Proc., 100th Annual Meeting of the Transportation Research Board. Washington, DC: Transportation Research Board.
    - Cheng, Q., Liu, Z., Guo, J., Wu, X., Pendyala, R., Belezamo, B. and Zhou, X.S., 2022.
    Estimating key traffic state parameters through parsimonious spatial queue models.
    Transportation Research Part C: Emerging Technologies, 137, p.103596.
    - Cheng, Q., Liu, Z., Lin, Y. and Zhou, X.S., 2021.
    An s-shaped three-parameter (S3) traffic stream model with consistent car following relationship.
    Transportation Research Part B: Methodological, 153, pp.246-271.
Authors: Xin Wu, Xuesong Zhou
Date: October 15, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import seaborn as sns
from pathlib import Path

pd.set_option('display.max_columns', None)  # Show all columns in DataFrame

# ==============================================================================
# ADVANCED MODEL SETUP
# ==============================================================================
# Define peak period hours
am_peak_start = 6  # 6 AM
am_peak_end = 9  # 9 AM
pm_peak_start = 16  # 3 PM
pm_peak_end = 20  # 8 PM
current_period = 'AM'  # 'AM' or 'PM'
# Define time stamps
time_stamps = 5  # e.g., 5-minute intervals
# load TMC identification data
tmc_identification_df = pd.read_csv('input_data/TMC_identification.csv')
# create dictionary for TMC and its free-flow speed
tmc_uf_dict = dict(zip(tmc_identification_df['tmc'], tmc_identification_df['free_speed']))
# create dictionary for TMC and its capacity
tmc_cap_dict = dict(zip(tmc_identification_df['tmc'], tmc_identification_df['capacity']))
# create dictionary for TMC and its length
tmc_length_dict = dict(zip(tmc_identification_df['tmc'], tmc_identification_df['miles']))
# create dictionary for TMC and its number of lanes
tmc_lanes_dict = dict(zip(tmc_identification_df['tmc'], tmc_identification_df['lane']))
# create dictionary for TMC and its facility type
tmc_facility_dict = dict(zip(tmc_identification_df['tmc'], tmc_identification_df['FT']))
# create dictionary for TMC and its area type
tmc_area_dict = dict(zip(tmc_identification_df['tmc'], tmc_identification_df['AT']))


# ==============================================================================
# ADVANCED MODEL 1: S3 MODEL (S-SHAPED THREE-PARAMETER TRAFFIC FLOW MODEL)
# ==============================================================================

def s3_model_derivation(mm=4):
    """
    Assume that S3 traffic flow model has been calibrated using non-linear optimization.
    
    Model: u = uf /(1 + (k/kc)^m )^(2/m))

    q = u * kc *((uf/u)^(m/2) -1 )^(1/m)
    cap = (kc * uf) / 2^(2/m)
    uc = uf / 2^(2/m)
    where:
        u = speed (mph)
        uf = free-flow speed (mph)
        k = density (veh/mi/lane)
        kc = critical density (veh/mi/lane)
        m = shape parameter
        q = flow (veh/hr/lane)
        cap = capacity (veh/hr/lane)
        uc = speed at capacity (mph)
    
    Returns:
        return estimated flow by using the calibrated S3 model parameters.
    """
    mm = 6.0
    print("\n" + "=" * 80)
    print("MODEL 1: Derive Flow and Density Via Traffic Flow (S3) Model")
    print("=" * 80)
    print("using default mm = {:.2f}".format(mm))
    reading_df = pd.read_csv('input_data/Reading.csv')

    tmc_iter = reading_df.groupby('tmc_code')

    extended_reading_df = pd.DataFrame()
    s3_params_list = []
    for tmc, group in tmc_iter:
        uf = tmc_uf_dict.get(tmc, 65)
        max_speed = group['speed'].max()
        uf = max(uf, max_speed + 5)  # ensure uf is at least greater than max observed speed
        cap = tmc_cap_dict.get(tmc, 1800)
        kc = (cap * (2 ** (2 / mm))) / uf  # critical density (veh/mi/lane)
        uc = uf / (2 ** (2 / mm))  # speed at capacity (mph)
        group['flow'] = group.apply(
            lambda row: row['speed'] * kc * ((uf / min(max(row['speed'], 0.1), uf)) ** (mm / 2) - 1) ** (1 / mm),
            axis=1
        )
        group['density'] = group['flow'] / group['speed']
        group['capacity'] = cap
        group['free_flow_speed'] = uf
        group['critical_density'] = kc
        group['speed_at_capacity'] = uc
        extended_reading_df = pd.concat([extended_reading_df, group], ignore_index=True)
        s3_params_list.append([tmc, uf, kc, mm, cap, uc])
        print(f"  ✓ TMC {tmc}: successfully derived flow and density using S3 model.")

    s3_params_df = pd.DataFrame(s3_params_list,
                                columns=['tmc', 'free_flow_speed_uf', 'critical_density_kc',
                                         'shape_parameter_m', 'capacity', 'speed_at_capacity_uc'])
    s3_params_df.to_csv('output/s3_model/s3_parameters.csv', index=False)

    # for each tmc plot the speed-density and flow-density and speed-flow diagrams
    for tmc, group in extended_reading_df.groupby('tmc_code'):
        # depict group['speed'] vs group['density']
        uf = tmc_uf_dict.get(tmc, 65)
        max_speed = group['speed'].max()
        uf = max(uf, max_speed + 5)  # ensure uf is at least greater than max observed speed
        cap = tmc_cap_dict.get(tmc, 1800)
        kc = (cap * (2 ** (2 / mm))) / uf  # critical density (veh/mi/lane)
        uc = uf / (2 ** (2 / mm))  # speed at capacity (mph)
        plt.figure(figsize=(12, 12))
        plt.subplot(2, 2, 1)
        # if density < kc, then blue dot, else red dot
        group_below_kc = group[group['density'] <= kc]
        group_above_kc = group[group['density'] > kc]
        plt.scatter(group_below_kc['density'], group_below_kc['speed'], color='blue',
                    label='Observed Speed-Density (k<=kc)', alpha=0.5)
        plt.scatter(group_above_kc['density'], group_above_kc['speed'], color='red',
                    label='Observed Speed-Density (k>kc)', alpha=0.5)
        plt.title(f'TMC {tmc} - S3 Model Speed-Density Diagram, uf={uf:.0f}, kc={kc:.0f}, m={mm:.1f},'
                  f' cap={cap:.0f}, uc={uc:.0f}', fontsize=8, fontweight='bold')
        plt.legend(fontsize=8)
        plt.xlabel('Density (veh/mi/lane)')
        plt.ylabel('Speed (mph)')
        plt.grid(True, alpha=0.3)

        # depict group['flow'] vs group['density']
        plt.subplot(2, 2, 2)
        # if density < kc, then blue dot, else red dot
        plt.scatter(group_below_kc['density'], group_below_kc['flow'], color='blue',
                    label='Observed Flow-Density (k<=kc)', alpha=0.5)
        plt.scatter(group_above_kc['density'], group_above_kc['flow'], color='red',
                    label='Observed Flow-Density (k>kc)', alpha=0.5)
        plt.xlabel('Density (veh/mi/lane)')
        plt.ylabel('Flow (veh/hr/lane)')
        plt.title(f'TMC {tmc} - S3 Model Flow-Density Diagram, uf={uf:.0f}, kc={kc:.0f}, m={mm:.1f},'
                  f' cap={cap:.0f}, uc={uc:.0f}', fontsize=8)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)

        # depict x-axis: flow vs y-axis: speed
        plt.subplot(2, 2, 3)
        # if speed > uc, then blue dot, else red dot
        plt.scatter(group_below_kc['flow'], group_below_kc['speed'], color='blue',
                    label='Observed Speed-Flow (u>uc)', alpha=0.5)
        plt.scatter(group_above_kc['flow'], group_above_kc['speed'], color='red',
                    label='Observed Speed-Flow (u<=uc)', alpha=0.5)
        plt.xlabel('Flow (veh/hr/lane)')
        plt.ylabel('Speed (mph)')
        plt.title(f'TMC {tmc} - S3 Model Speed-Flow Diagram, uf={uf:.0f}, kc={kc:.0f}, m={mm:.1f},'
                  f' cap={cap:.0f}, uc={uc:.0f}', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(f'output/s3_model/s3_model_tmc_{tmc}_diagrams.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: output/s3_model/s3_model_tmc_{tmc}_diagrams.png")

    extended_reading_df = extended_reading_df[['tmc_code', 'measurement_tstamp', 'flow', 'density', 'speed',
                                               'capacity', 'free_flow_speed', 'critical_density', 'speed_at_capacity']]
    extended_reading_df.to_csv('output/s3_model/reading_with_s3_params.csv', index=False)


# ==============================================================================
# ADVANCED MODEL 2: PAQ (POLYNOMIAL ARRIVAL QUEUE) MODEL
# ==============================================================================

def analyze_tmc_data(period='AM'):
    """
    Demonstrate PAQ model for queue analysis.
    """

    print("\n" + "=" * 80)
    print("ADVANCED MODEL 2: PAQ (Polynomial Arrival Queue) Analysis")
    print("=" * 80)

    # Load calibrated parameters
    extended_reading_df = pd.read_csv('output/s3_model/reading_with_s3_params.csv')
    # step 1: determine the time interval of measurements_tstamp
    extended_reading_df['date'] = pd.to_datetime(extended_reading_df['measurement_tstamp']).dt.date
    extended_reading_df['time_minute'] = pd.to_datetime(extended_reading_df['measurement_tstamp']).dt.hour * 60 + \
                                         pd.to_datetime(extended_reading_df['measurement_tstamp']).dt.minute
    extended_reading_df['time_hour'] = pd.to_datetime(extended_reading_df['measurement_tstamp']).dt.hour

    # "AM" or "PM" or "MD" or "NT"
    def determine_period(row):
        hour = pd.to_datetime(row['measurement_tstamp']).hour
        if am_peak_start <= hour < am_peak_end:
            return 'AM'
        elif pm_peak_start <= hour < pm_peak_end:
            return 'PM'
        elif 9 <= hour < 16:
            return 'MD'
        else:
            return 'NT'

    extended_reading_df['period'] = extended_reading_df.apply(determine_period, axis=1)
    extended_reading_df = extended_reading_df[extended_reading_df['period'] == period]

    iter_df = extended_reading_df.groupby(['tmc_code', 'date'])
    daily_paq_list = []
    for (tmc, date), group in iter_df:
        # t2: the time with the lowest speed
        t2_in_min = group.loc[group['speed'].idxmin()]['time_minute']
        # determine t0 and t3 based on speed profile
        group['below_uc'] = group['speed'] < group['speed_at_capacity']
        # t1: the earliest time when speed drops below uc
        t0_in_min = group[group['below_uc']]['time_minute'].min()
        # if no time below uc, t0_in_min = t2_in_min
        if pd.isna(t0_in_min):
            t0_in_min = t2_in_min
        # t3: the latest time when speed rises above uc
        t3_in_min = group[group['below_uc']]['time_minute'].max()
        # if no time below uc, t3_in_min = t2_in_min
        if pd.isna(t3_in_min):
            t3_in_min = t2_in_min
        # t1: the time with the highest flow before t2 and after t0
        if t0_in_min >= t2_in_min:
            t1_in_min = t0_in_min
        else:
            mask = (group['time_minute'] > t0_in_min) & (group['time_minute'] < t2_in_min)
            t1_in_min = group[mask].loc[group[mask]['flow'].idxmax()]['time_minute']

        congestion_duration_hours = (t3_in_min - t0_in_min) / 60.0
        # queued_demand = the flows between t1 and t3 (ensure convert to time stamp's interval)
        group['flow_in_interval'] = group['flow'] * (time_stamps / 60.0)  # flow in the time stamp interval
        queued_demand = \
            group[(group['time_minute'] >= t0_in_min) & (group['time_minute'] <= t3_in_min)]['flow_in_interval'].sum()
        total_flow = group['flow_in_interval'].sum()
        queued_demand_factor = queued_demand / total_flow if total_flow > 0 else 0

        ultimate_capacity = group['capacity'].iloc[0]
        free_flow_speed = group['free_flow_speed'].iloc[0]
        speed_at_capacity = group['speed_at_capacity'].iloc[0]
        speed_at_t2 = group['speed'].min()
        magnitude_of_speed_reduction = (speed_at_capacity - speed_at_t2) / speed_at_t2
        discharge_rate = min(queued_demand / congestion_duration_hours, ultimate_capacity)
        # average speed during congestion duration
        average_speed_during_congestion = \
            group[(group['time_minute'] >= t0_in_min) & (group['time_minute'] <= t3_in_min)]['speed'].mean()
        daily_paq_list.append([tmc, date, t0_in_min, t1_in_min, t2_in_min, t3_in_min,
                               congestion_duration_hours, queued_demand, queued_demand_factor,
                               ultimate_capacity, free_flow_speed, speed_at_capacity, speed_at_t2,
                               magnitude_of_speed_reduction, discharge_rate, average_speed_during_congestion])

        print(f"\nPAQ Model Parameters for TMC {tmc} on {date} during {period} Peak:")
        print(f"  • Congestion start (t0): {int(t0_in_min // 60):02d}: {int(t0_in_min % 60):02d})")
        print(f"  • Max flow before t2 (t1): {int(t1_in_min // 60):02d}: {int(t1_in_min % 60):02d})")
        print(f"  • Min speed (t2): {int(t2_in_min // 60):02d}: {int(t2_in_min % 60):02d})")
        print(f"  • Congestion end (t3): {int(t3_in_min // 60):02d}: {int(t3_in_min % 60):02d})")
        print(f"  • Duration (P): {congestion_duration_hours:.2f} hours ({congestion_duration_hours * 60:.0f} minutes)")
        print(f"  • Ultimate capacity: {ultimate_capacity:.0f} veh/hr/lane")
        print(f"  • Free-flow speed: {free_flow_speed:.1f} mph")
        print(f"  • Speed at capacity: {speed_at_capacity:.1f} mph")
        print(f"  • Speed at t2 (min speed): {speed_at_t2:.1f} mph")
        print(f"  • Magnitude of speed reduction: {magnitude_of_speed_reduction:.2}")
        print(f"  • Queued demand: {queued_demand:.0f} vehicles")
        print(f"  • Queued demand factor: {queued_demand_factor:.2%}")
        print(f"  • Discharge rate: {discharge_rate:.0f} veh/hr/lane")
        print(f"  • Average speed during congestion: {average_speed_during_congestion:.1f} mph")

        # plot the speed profile for the period
        uc_str = str(round(group['speed_at_capacity'].iloc[0], 0)) + ' mph'
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(group['time_minute'], group['speed'], 'b-o', label='Observed Speed (mph)')
        # plot the speed_at_capacity # using green dashed line
        plt.plot(group['time_minute'], group['speed_at_capacity'], 'g--',
                 label='Speed at Capacity = ' + uc_str)
        plt.title(f'TMC {tmc} - Speed Profile on {date} during {period} Peak', fontsize=12, fontweight='bold')
        # mark t0, t1, t2, t3
        plt.axvline(x=t0_in_min, color='gray', linestyle='--', alpha=0.7, label='t0 Start of Congestion')
        plt.axvline(x=t1_in_min, color='orange', linestyle='--', alpha=0.7, label='t1 Max Flow Before t2')
        plt.axvline(x=t2_in_min, color='red', linestyle='--', alpha=0.7, label='t2 Min Speed')
        plt.axvline(x=t3_in_min, color='gray', linestyle='--', alpha=0.7, label='t3 End of Congestion')
        plt.xlabel('Time (minutes since midnight)', fontsize=11)
        plt.ylabel('Speed (mph)', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)

        # plot the flow_interval profile for the period (area under curve represents total queued demand)
        plt.subplot(2, 1, 2)
        plt.plot(group['time_minute'], group['flow_in_interval'], 'g-o', label='Observed Flow in Interval (veh)')
        # highlight the queued demand area
        plt.fill_between(group['time_minute'], group['flow_in_interval'], where=(group['time_minute'] >= t0_in_min) &
                                                                                (group['time_minute'] <= t3_in_min),
                         color='orange', alpha=0.3,
                         label='Queued Demand Area')
        plt.title(f'TMC {tmc} - Flow Profile on {date} during {period} Peak', fontsize=12, fontweight='bold')
        # mark t0, t1, t2, t3
        plt.axvline(x=t0_in_min, color='gray', linestyle='--', alpha=0.7, label='t0 Start of Congestion')
        plt.axvline(x=t1_in_min, color='orange', linestyle='--', alpha=0.7, label='t1 Max Flow Before t2')
        plt.axvline(x=t2_in_min, color='red', linestyle='--', alpha=0.7, label='t2 Min Speed')
        plt.axvline(x=t3_in_min, color='gray', linestyle='--', alpha=0.7, label='t3 End of Congestion')

        plt.xlabel('Time (minutes since midnight)', fontsize=11)
        plt.ylabel('Flow in Interval (veh)', fontsize=11)
        # set y-axis limit to max flow_in_interval + 10% and min flow_in_interval -10%
        plt.ylim(group['flow_in_interval'].min() * 0.9, group['flow_in_interval'].max() * 1.1)

        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(f'output/paq_model/paq_speed_flow_tmc_{tmc}_{date}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: output/paq_model/paq_speed_flow_tmc_{tmc}_{date}.png")
    daily_paq_df = pd.DataFrame(daily_paq_list,
                                columns=['tmc_code', 'date', 't0_in_min', 't1_in_min', 't2_in_min', 't3_in_min',
                                         'congestion_duration_hours', 'queued_demand_veh',
                                         'queued_demand_factor', 'ultimate_capacity_veh_hr_lane',
                                         'free_flow_speed_mph', 'speed_at_capacity_mph', 'speed_at_t2_mph',
                                         'magnitude_of_speed_reduction', 'discharge_rate_veh_hr_lane',
                                         'average_speed_during_congestion_mph'])
    daily_paq_df.to_csv('output/paq_model/daily_paq_parameters.csv', index=False)
    print("\n✓ Saved: output/paq_model/daily_paq_parameters.csv")


def analyze_paq_model(period='AM'):
    """

    Calibrate PAQ model for queue analysis.
    """

    print("\n" + "=" * 80)
    print("ADVANCED MODEL 2: PAQ (Polynomial Arrival Queue) Calibration")
    print("=" * 80)

    """
    # congestion_duration_hours = f_d * (D/C)^n
    # magnitude_of_speed_reduction = f_p * (congestion_duration_hours)^s
    # stage 1: calibrate f_d and n using daily_paq_df
    # stage 2: calibrate f_p and s using daily_paq_df
    """
    daily_paq_df = pd.read_csv('output/paq_model/daily_paq_parameters.csv')
    updated_daily_paq_df = pd.DataFrame()
    tmc_params_list = []
    for tmc, group in daily_paq_df.groupby('tmc_code'):
        def objective_stage1(params):
            f_d, n = params
            D_over_C = group['queued_demand_veh'] / group['ultimate_capacity_veh_hr_lane']
            predicted_duration = f_d * (D_over_C) ** n
            observed_duration = group['congestion_duration_hours']
            return np.sum((predicted_duration - observed_duration) ** 2)

        def objective_stage2(params):
            f_p, s = params
            predicted_magnitude = f_p * (group['congestion_duration_hours']) ** s
            observed_magnitude = group['magnitude_of_speed_reduction']
            return np.sum((predicted_magnitude - observed_magnitude) ** 2)

        bounds_stage1 = [(0.01, 10), (0.1, 5)]
        result_stage1 = differential_evolution(objective_stage1, bounds_stage1)
        f_d_opt, n_opt = result_stage1.x
        print(f"\nCalibrated PAQ Parameters for TMC {tmc} at {period} Peak:")
        print(f"  • f_d: {f_d_opt:.4f}")
        print(f"  • n: {n_opt:.4f}")
        group['f_d'] = f_d_opt
        group['n_calibrated'] = n_opt

        bounds_stage2 = [(0.01, 10), (4.1, 6.5)]
        result_stage2 = differential_evolution(objective_stage2, bounds_stage2)
        f_p_opt, s_opt = result_stage2.x
        print(f"  • f_p: {f_p_opt:.4f}")
        print(f"  • s: {s_opt:.4f}")
        group['f_p_calibrated'] = f_p_opt
        group['s_calibrated'] = s_opt
        length = tmc_length_dict.get(tmc, 1.0)  # default 1 mile
        t0 = group['t0_in_min'].mean()
        t1 = max(group['t1_in_min'].mean(), t0)
        t2 = max(group['t2_in_min'].mean(), t1)
        t3 = max(group['t3_in_min'].mean(), t2)
        queue_demand_factor = group['queued_demand_factor'].mean()
        # qvdf alpha and beta
        # alpha = (64 / 120) * f_p * f_d ^ s
        alpha = (64 / 120) * f_p_opt * (f_d_opt ** s_opt)
        # beta = n * s
        beta = n_opt * s_opt

        # ensure t0 <= t1 <= t2 <= t3

        print(f"  • t0: {int(t0 // 60):02d}:{int(t0 % 60):02d}")
        print(f"  • t1: {int(t1 // 60):02d}:{int(t1 % 60):02d}")
        print(f"  • t2: {int(t2 // 60):02d}:{int(t2 % 60):02d}")
        print(f"  • t3: {int(t3 // 60):02d}:{int(t3 % 60):02d}")
        print(f"  • Queued Demand Factor: {queue_demand_factor:.2%}")
        print(f"  • QVDF alpha: {alpha:.4f}")
        print(f"  • QVDF beta: {beta:.4f}")

        tmc_params_list.append([tmc, f_d_opt, n_opt, f_p_opt, s_opt,
                                group['ultimate_capacity_veh_hr_lane'].iloc[0],
                                group['free_flow_speed_mph'].iloc[0], length,
                                t0, t1, t2, t3, queue_demand_factor, group['speed_at_capacity_mph'].iloc[0],
                                alpha, beta])

        print(f"  ✓ TMC {tmc}: successfully calibrated PAQ model parameters.")

        # plot the scatter plot of observed vs calibrated curve
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        D_over_C = group['queued_demand_veh'] / group['ultimate_capacity_veh_hr_lane']
        plt.scatter(D_over_C, group['congestion_duration_hours'], color='blue', label='Observed Data', alpha=0.6)
        D_over_C_range = np.linspace(0, D_over_C.max(), 100)
        calibrated_duration = f_d_opt * D_over_C_range ** n_opt
        plt.plot(D_over_C_range, calibrated_duration, color='red', label='Calibrated PAQ Curve', linewidth=2)
        plt.title(f'TMC {tmc} - PAQ Calibration at {period} Peak, f_d={f_d_opt:.4f}, n={n_opt:.4f}', fontsize=8)
        plt.xlabel('Queued Demand / Capacity (D/C)', fontsize=8)
        plt.ylabel('Congestion Duration (hours)', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)

        plt.subplot(2, 2, 2)
        plt.scatter(group['congestion_duration_hours'], group['magnitude_of_speed_reduction'],
                    color='blue', label='Observed Data', alpha=0.6)
        duration_range = np.linspace(0, group['congestion_duration_hours'].max(), 100)
        calibrated_magnitude = f_p_opt * duration_range ** s_opt
        plt.plot(duration_range, calibrated_magnitude, color='red', label='Calibrated PAQ Curve', linewidth=2)
        plt.title(f'TMC {tmc} - PAQ Calibration at {period} Peak, f_p={f_p_opt:.4f}, s={s_opt:.4f}', fontsize=8)
        plt.xlabel('Congestion Duration (hours)', fontsize=8)
        plt.ylabel('Magnitude of Speed Reduction', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)

        plt.subplot(2, 2, 3)
        # plot D/C vs average speed during congestion duration
        plt.scatter(D_over_C, group['average_speed_during_congestion_mph'],
                    color='blue', label='Observed Data', alpha=0.6)
        # plot the curve average speed during congestion  = speed_at_capacity /(1+ alpha*(D/C)^beta)
        est_D_over_C = np.linspace(0, D_over_C.max(), 100)
        est_avg_speed = group['speed_at_capacity_mph'].iloc[0] / (1 + alpha * (est_D_over_C ** beta))
        plt.plot(est_D_over_C, est_avg_speed, color='red', label='Estimated Curve', linewidth=2)
        plt.title(f'TMC {tmc} - Average Speed During Congestion vs D/C at {period} Peak, '
                  f'alpha={alpha:.4f}, beta={beta:.4f}', fontsize=8)
        plt.xlabel('Queued Demand / Capacity (D/C)', fontsize=8)
        plt.ylabel('Average Speed During Congestion (mph)', fontsize=8)
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.savefig(f'output/paq_model/paq_calibration_tmc_{tmc}_{period}_peak.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: output/paq_model/paq_calibration_tmc_{tmc}_{period}_peak.png")

    tmc_params_df = pd.DataFrame(tmc_params_list,
                                 columns=['tmc', 'f_d', 'n', 'f_p', 's',
                                          'ultimate_capacity_veh_hr_lane', 'free_flow_speed_mph',
                                          'link_length_miles', 't0_in_min', 't1_in_min', 't2_in_min',
                                          't3_in_min', 'queued_demand_factor',
                                          'speed_at_capacity_mph', 'qvdf_alpha', 'qvdf_beta'])

    tmc_params_df.to_csv('output/paq_model/tmc_paq_parameters.csv', index=False)
    print("\n✓ Saved: output/paq_model/tmc_paq_parameters.csv")


# ==============================================================================
# ADVANCED MODEL 3: QVDF (QUEUE-BASED VOLUME-DELAY FUNCTION)
# ==============================================================================

def derive_qvdf_series(period, d_over_c_ratio, tmc_code):
    df = pd.read_csv('output/paq_model/tmc_paq_parameters.csv')
    row = df[df['tmc'] == tmc_code].iloc[0]
    f_d = row['f_d']
    f_p = row['f_p']
    n = row['n']
    s = row['s']
    C = row['ultimate_capacity_veh_hr_lane']
    D = d_over_c_ratio * C
    P = f_d * d_over_c_ratio  # congestion duration hours
    mu = min(D / ((P * 60) / time_stamps), C * time_stamps / 60.0)  # discharge rate in veh per time stamp
    L = row['link_length_miles']
    uc = row['speed_at_capacity_mph']
    uf = row['free_flow_speed_mph']
    # gamma = 64 * mu * (Length/speed_at_capacity) * fp*(P)^(s-4)
    gamma = 64 * mu * (L / uc) * f_p * P ** (s - 4)

    # qvdf alpha
    alpha = (64.0 / 120.0) * f_p * f_d ** s
    beta = n - s

    if period == 'AM':
        start_minute = am_peak_start * 60
        end_minute = am_peak_end * 60
    if period == 'PM':
        start_minute = pm_peak_start * 60
        end_minute = pm_peak_end * 60
    # w(t) = (gamma / (4 * mu)) * ((t - t0) ** 2) * ((t - t3) ** 2)
    time_range = np.arange(start_minute, end_minute + 1, 5)
    w_t_list = []
    for t in time_range:
        t0 = row['t0_in_min']
        t3 = row['t3_in_min']
        if t0 <= t <= t3:
            w_t = (gamma / (4 * mu)) * ((t / 60 - t0 / 60) ** 2) * ((t / 60 - t3 / 60) ** 2)
        else:
            w_t = 0
        w_t_list.append(w_t)
    # Q(t) = (gamma / 4) * ((t - t0) ** 2) * ((t - t3) ** 2)
    Q_t_list = []
    for t in time_range:
        t0 = row['t0_in_min']
        t3 = row['t3_in_min']
        if t0 <= t <= t3:
            Q_t = (gamma / 4) * ((t / 60 - t0 / 60) ** 2) * ((t / 60 - t3 / 60) ** 2)
        else:
            Q_t = 0
        Q_t_list.append(Q_t)

    # v(t) = L/ ((L/uc) + w(t))
    v_t_list = []
    for idx, t in enumerate(time_range):
        t0 = row['t0_in_min']
        t3 = row['t3_in_min']
        if t0 <= t <= t3:
            w_t = w_t_list[idx]
            v_t = L / ((L / uc) + w_t)
        if t < t0:
            # v_t is linear between free flow speed and speed at capacity
            v_t = uf - (uf - uc) * ((t - start_minute) / (t0 - start_minute))
        if t > t3:
            # v_t is linear between speed at capacity and free flow speed
            v_t = uc + (uf - uc) * ((t - t3) / (end_minute - t3))
        v_t_list.append(v_t)

    return time_range, w_t_list, Q_t_list, v_t_list, mu, gamma


def analyze_qvdf_model(period='AM'):
    """Demonstrate QVDF model for various D/C ratios."""

    print("\n" + "=" * 80)
    print("ADVANCED MODEL 3: QVDF (Queue-Based Volume-Delay Function)")
    print("=" * 80)

    # D/C from 0 to 5 with increment of 0.1
    dc_ratios = np.arange(0.1, 1.9, 0.2)

    # discharge rates = min[D/P, C] P = congestion duration hours = f_d * (D/C)^n
    for idx, row in pd.read_csv('output/paq_model/tmc_paq_parameters.csv').iterrows():
        # for each tmc, calculate the congestion duration hours and discharge rate
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        for dc in dc_ratios:
            time_range, w_t_list, Q_t_list, v_t_list, mu, gamma = derive_qvdf_series(period, dc, row['tmc'])
            # change color for different D/C ratios
            axs[0].plot(time_range, w_t_list, label=f'D/C = {dc:.1f}')
            # text box to indicate D/C ratio
            axs[0].set_title(f'TMC {row["tmc"]} - QVDF Delay w(t) at D/Cs during {period} Peak',
                             fontsize=8, fontweight='bold')
            axs[0].set_xlabel('Time (minutes since midnight)', fontsize=8)
            axs[0].set_ylabel('Delay w(t) (hours)', fontsize=8)
            axs[0].grid(True, alpha=0.3)
            axs[0].legend(fontsize=8)

            axs[1].plot(time_range, Q_t_list, label='Queue Length Q(t) (vehicles)')
            axs[1].set_title(f'TMC {row["tmc"]} - QVDF Queue Length Q(t) at D/Cs during {period} Peak',
                             fontsize=8, fontweight='bold')
            axs[1].set_xlabel('Time (minutes since midnight)', fontsize=8)
            axs[1].set_ylabel('Queue Length Q(t) (vehicles)', fontsize=8)
            axs[1].grid(True, alpha=0.3)

            axs[2].plot(time_range, v_t_list, label='Speed v(t) (mph)')
            axs[2].set_title(f'TMC {row["tmc"]} - QVDF Speed v(t) at D/Cs during {period} Peak',
                             fontsize=12, fontweight='bold')
            axs[2].set_xlabel('Time (minutes since midnight)', fontsize=8)
            axs[2].set_ylabel('Speed v(t) (mph)', fontsize=8)
            axs[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'output/qvdf_model/qvdf_tmc_{row["tmc"]}_{period}_peak.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: output/qvdf_model/qvdf_tmc_{row['tmc']}_{period}_peak.png")

    # read output/paq_model/daily_paq_parameters.csv
    # get the D/C ratios each day for each tmc

    daily_df = pd.read_csv('output/paq_model/daily_paq_parameters.csv')
    reading_df = pd.read_csv('output/s3_model/reading_with_s3_params.csv')
    reading_df['date'] = pd.to_datetime(reading_df['measurement_tstamp']).dt.date
    reading_df['time_minute'] = pd.to_datetime(reading_df['measurement_tstamp']).dt.hour * 60 + \
                                pd.to_datetime(reading_df['measurement_tstamp']).dt.minute

    iter_df = daily_df.groupby('date')
    dc_summary_list = []
    # creat heatmap of speed v(t) from QVDF and compare with reading_df speed for each day
    for date, group in iter_df:
        daily_obs_speed_list = []
        daily_est_speed_list = []
        for tmc in group['tmc_code'].unique():
            # get d/c ratio for the day
            dc_ratio = group[group['tmc_code'] == tmc]['queued_demand_veh'].iloc[0] / \
                       group[group['tmc_code'] == tmc]['ultimate_capacity_veh_hr_lane'].iloc[0]
            time_range, w_t_list, Q_t_list, v_t_list, mu, gamma = derive_qvdf_series(period, dc_ratio, tmc)
            # get observed speed for the day from reading_df
            daily_reading = reading_df[(reading_df['tmc_code'] == tmc) &
                                       (reading_df['date'] == pd.to_datetime(date).date())]
            obs_speed_series = []
            est_speed_series = []
            obs_speed_dict = dict(zip(daily_reading['time_minute'], daily_reading['speed']))
            for t in time_range:
                obs_speed = obs_speed_dict.get(t, np.nan)
                obs_speed_series.append(obs_speed)
            daily_obs_speed_list.append(obs_speed_series)
            for v in v_t_list:
                est_speed_series.append(v)
            daily_est_speed_list.append(est_speed_series)
        # create heatmap
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        sns.heatmap(daily_obs_speed_list, cmap='RdYlGn_r', cbar_kws={'label': 'Observed Speed (mph)'},
                    xticklabels=60, yticklabels=group['tmc_code'].unique())
        plt.title(f'Observed Speed Heatmap on {date} during {period} Peak', fontsize=12, fontweight='bold')
        plt.xlabel('Time (minutes since midnight)', fontsize=11)
        plt.ylabel('TMC Code', fontsize=11)
        plt.subplot(2, 1, 2)
        sns.heatmap(daily_est_speed_list, cmap='RdYlGn_r', cbar_kws={'label': 'Estimated Speed (mph)'},
                    xticklabels=60, yticklabels=group['tmc_code'].unique())
        plt.title(f'Estimated Speed Heatmap on {date} during {period} Peak', fontsize=12, fontweight='bold')
        plt.xlabel('Time (minutes since midnight)', fontsize=11)
        plt.ylabel('TMC Code', fontsize=11)
        plt.tight_layout()
        plt.savefig(f'output/qvdf_model/qvdf_speed_heatmap_{date}_{period}_peak.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: output/qvdf_model/qvdf_speed_heatmap_{date}_{period}_peak.png")


def model_performance(period='AM'):
    # read TMC_identification.csv to get tmc_code and tmc_name
    tmc_id_df = pd.read_csv('input_data/TMC_identification.csv')
    # read s3_model/s3_params.csv
    s3_params_df = pd.read_csv('output/s3_model/s3_parameters.csv')
    # attach s3_params to tmc_id_df (do not attach duplicated columns)

    s3_params_df = s3_params_df[['tmc', 'free_flow_speed_uf',
                                 'critical_density_kc', 'shape_parameter_m', 'speed_at_capacity_uc']]
    tmc_summary_df = pd.merge(tmc_id_df, s3_params_df, left_on='tmc', right_on='tmc', how='left')

    # read paq_model/daily_paq_parameters.csv
    daily_paq_df = pd.read_csv('output/paq_model/daily_paq_parameters.csv')
    # calculate average t0_in_min, t1_in_min, t2_in_min t3_in_min, congestion period, queued_demand_veh, queue_demand_factor,
    # speed_at_t2_mph, magnitude_of_speed_reduction, discharge_rate_veh_hr_lane, average_speed_during_congestion_mph
    # for each tmc_code
    paq_summary_df = daily_paq_df.groupby('tmc_code').agg({
        't0_in_min': 'mean',
        't1_in_min': 'mean',
        't2_in_min': 'mean',
        't3_in_min': 'mean',
        'congestion_duration_hours': 'mean',
        'queued_demand_veh': 'mean',
        'queued_demand_factor': 'mean',
        'speed_at_t2_mph': 'mean',
        'magnitude_of_speed_reduction': 'mean',
        'discharge_rate_veh_hr_lane': 'mean',
        'average_speed_during_congestion_mph': 'mean'
    }).reset_index()
    # merge paq_summary_df to tmc_summary_df
    tmc_summary_df = pd.merge(tmc_summary_df, paq_summary_df, left_on='tmc', right_on='tmc_code', how='left')
    # paq_model/tmc_paq_parameters.csv
    tmc_paq_params_df = pd.read_csv('output/paq_model/tmc_paq_parameters.csv')
    tmc_paq_params_df = tmc_paq_params_df[['tmc', 'f_d', 'n', 'f_p', 's', 'qvdf_alpha', 'qvdf_beta']]
    # merge tmc_paq_params_df to tmc_summary_df
    tmc_summary_df = pd.merge(tmc_summary_df, tmc_paq_params_df, left_on='tmc', right_on='tmc', how='left')
    # save tmc_summary_df to output/model_performance_summary.csv
    tmc_summary_df['period'] = period
    tmc_summary_df['geometry'] = ('LINESTRING (' + tmc_summary_df['start_longitude'].astype(str) +
                                  ' ' + tmc_summary_df['start_latitude'].astype(str) + ', ' +
                                  tmc_summary_df['end_longitude'].astype(str) + ' ' +
                                  tmc_summary_df['end_latitude'].astype(str) + ')')
    tmc_summary_df.to_csv('output/model_performance_summary.csv', index=False)
    print("\n✓ Saved: output/model_performance_summary.csv")


# ==============================================================================
# MAIN FUNCTION TO RUN ALL ADVANCED MODELS
# ==============================================================================

def run_advanced_models():
    """Execute all advanced traffic flow models."""

    print("\n" + "=" * 80)
    print("RUNNING ADVANCED TRAFFIC FLOW MODELS".center(80))
    print("=" * 80)
    print("\nModels to be executed:")
    print("  1. S3 Model - Derive estimated flow via TMC speed data")
    print("  2. PAQ Model - Polynomial queue dynamics")
    print("  3. QVDF Model - Queue-based delay function")

    # Ensure output directory exists
    Path('output').mkdir(exist_ok=True)
    # create three sub-folders for each model
    Path('output/s3_model').mkdir(exist_ok=True)
    Path('output/paq_model').mkdir(exist_ok=True)
    Path('output/qvdf_model').mkdir(exist_ok=True)

    try:
        # Model 1: using S3 model to derive flow and density for reading data for TMCs
        s3_model_derivation(mm=6.0)

        # Model 2: PAQ
        analyze_tmc_data(period=current_period)
        analyze_paq_model(period=current_period)

        # Model 3: QVDF
        analyze_qvdf_model()

        # Output performance files
        model_performance(period=current_period)

        print("\n" + "=" * 80)
        print("✅ ALL ADVANCED MODELS COMPLETED!".center(80))
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n❌ Error in advanced models: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys

    success = run_advanced_models()
    sys.exit(0 if success else 1)
