"""
Example usage of the Telemetry Analyzer components
Demonstrates programmatic data generation, logging, and analysis
"""

import numpy as np
from datetime import datetime
from pathlib import Path
import json

# Add parent directory to path if running from examples folder
import sys
sys.path.insert(0, '..')

from telemetry_analyzer import (
    TelemetryLogger, LapAnalyzer, TrackDefinition, 
    FuelTechImporter, TelemetryFrame
)


def generate_synthetic_lap_data(lap_time_s: float = 90.0, 
                                track_length_km: float = 3.0,
                                sample_rate_hz: float = 10.0) -> tuple:
    """
    Generate synthetic telemetry data for one lap
    Returns: (timestamps, field_names, data_dict)
    """
    
    # Calculate number of samples
    n_samples = int(lap_time_s * sample_rate_hz)
    time = np.linspace(0, lap_time_s, n_samples)
    
    # Simulate distance (linear progression with slight variation)
    distance = np.linspace(0, track_length_km, n_samples)
    distance += np.random.normal(0, 0.001, n_samples).cumsum()  # GPS noise
    distance = np.clip(distance, 0, track_length_km)
    
    # Simulate RPM (varying with throttle and gear)
    base_rpm = 4000
    rpm_variation = 2000 * np.sin(2 * np.pi * time / lap_time_s * 8)  # 8 shifts per lap
    rpm = base_rpm + rpm_variation + np.random.normal(0, 50, n_samples)
    rpm = np.clip(rpm, 1000, 8000)
    
    # Simulate TPS (throttle follows track sections)
    tps_pattern = 50 + 30 * np.sin(2 * np.pi * time / lap_time_s * 3)
    tps_pattern += 20 * np.sin(2 * np.pi * time / lap_time_s * 7)
    tps = np.clip(tps_pattern + np.random.normal(0, 2, n_samples), 0, 100)
    
    # Simulate speed (correlates with TPS and RPM)
    speed_base = 80 + (rpm - 4000) / 100 + tps / 2
    speed = np.clip(speed_base + np.random.normal(0, 3, n_samples), 0, 250)
    
    # Simulate gear (based on speed)
    gear = np.ones(n_samples, dtype=int)
    gear[speed > 50] = 2
    gear[speed > 80] = 3
    gear[speed > 120] = 4
    gear[speed > 160] = 5
    gear[speed > 200] = 6
    
    # Simulate G-forces (based on speed changes and cornering)
    g_accel = np.gradient(speed) * 0.03  # Longitudinal
    g_accel += np.random.normal(0, 0.1, n_samples)
    
    # Lateral G follows sinusoidal pattern (corners)
    g_lateral = 0.8 * np.sin(2 * np.pi * time / lap_time_s * 12)
    g_lateral += np.random.normal(0, 0.1, n_samples)
    
    # Simulate MAP (manifold pressure, correlates with TPS)
    map_pressure = 40 + tps * 0.6 + np.random.normal(0, 2, n_samples)
    map_pressure = np.clip(map_pressure, 20, 100)
    
    # Simulate temperatures (gradually increasing)
    engine_temp = 85 + time / lap_time_s * 5 + np.random.normal(0, 0.5, n_samples)
    oil_temp = 90 + time / lap_time_s * 3 + np.random.normal(0, 0.5, n_samples)
    
    # Simulate wheel speeds (similar to vehicle speed with slight variations)
    wheel_speed_fl = speed + np.random.normal(0, 2, n_samples)
    wheel_speed_fr = speed + np.random.normal(0, 2, n_samples)
    wheel_speed_rl = speed + np.random.normal(0, 2, n_samples)
    wheel_speed_rr = speed + np.random.normal(0, 2, n_samples)
    
    # Simulate brake pressure (high in braking zones)
    brake_pressure = np.zeros(n_samples)
    brake_zones = (tps < 30) & (np.gradient(speed) < -0.5)
    brake_pressure[brake_zones] = 80 + np.random.normal(0, 10, np.sum(brake_zones))
    
    # Simulate GPS coordinates (circular track)
    center_lat, center_lon = -23.7011, -46.6972  # Interlagos area
    radius_deg = track_length_km / (2 * np.pi * 111.32)  # Approximate
    angle = 2 * np.pi * distance / track_length_km
    latitude = center_lat + radius_deg * np.cos(angle)
    longitude = center_lon + radius_deg * np.sin(angle)
    
    # Simulate lambda/mixture data
    lambda_correction = 1.0 + np.random.normal(0, 0.05, n_samples)
    exhaust_o2 = 0.9 + 0.1 * (tps / 100) + np.random.normal(0, 0.02, n_samples)
    
    # Simulate injection time
    inj_time_a = 5 + rpm / 1000 + tps / 20 + np.random.normal(0, 0.2, n_samples)
    inj_time_b = inj_time_a + np.random.normal(0, 0.1, n_samples)
    
    # Simulate shock positions (suspension travel)
    shock_fl = 50 + 20 * np.sin(2 * np.pi * time / lap_time_s * 15) + np.random.normal(0, 2, n_samples)
    shock_fr = 50 + 20 * np.sin(2 * np.pi * time / lap_time_s * 15 + 0.1) + np.random.normal(0, 2, n_samples)
    shock_rl = 50 + 15 * np.sin(2 * np.pi * time / lap_time_s * 15 + 0.2) + np.random.normal(0, 2, n_samples)
    shock_rr = 50 + 15 * np.sin(2 * np.pi * time / lap_time_s * 15 + 0.3) + np.random.normal(0, 2, n_samples)
    
    # Create data dictionary
    data = {
        'rpm': rpm,
        'tps': tps,
        'map': map_pressure / 100,  # Convert to bar
        'engine_temp': engine_temp,
        'oil_temp1': oil_temp,
        'oil_pressure': np.ones(n_samples) * 4.5,  # Constant for simplicity
        'gear': gear,
        'wheel_speed_fl': wheel_speed_fl,
        'wheel_speed_fr': wheel_speed_fr,
        'wheel_speed_rl': wheel_speed_rl,
        'wheel_speed_rr': wheel_speed_rr,
        'g_accel': g_accel,
        'g_lateral': g_lateral,
        'brake_pressure': brake_pressure / 1000,  # Convert to bar
        'distance_km': distance,
        'latitude': latitude,
        'longitude': longitude,
        'gps_speed_knots': speed / 1.852,
        'lambda_correction': lambda_correction,
        'exhaust_o2': exhaust_o2,
        'inj_time_bank_a': inj_time_a,
        'inj_time_bank_b': inj_time_b,
        'shock_fl': shock_fl / 1000,  # Convert to meters
        'shock_fr': shock_fr / 1000,
        'shock_rl': shock_rl / 1000,
        'shock_rr': shock_rr / 1000,
    }
    
    field_names = list(data.keys())
    
    return time, field_names, data


def example_create_synthetic_session():
    """Example: Create a synthetic telemetry session with multiple laps"""
    
    print("="*60)
    print("Creating synthetic telemetry session")
    print("="*60)
    
    # Session parameters
    n_laps = 5
    lap_time_base = 90.0  # seconds
    track_length = 3.0  # km
    sample_rate = 10.0  # Hz
    
    # Create session directory
    session_id = f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = Path('sessions') / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    log_path = session_dir / 'telemetry.tlog'
    logger = TelemetryLogger(str(log_path))
    
    # Generate first lap to get field names
    time, field_names, data = generate_synthetic_lap_data(
        lap_time_base, track_length, sample_rate
    )
    
    # Create log with metadata
    metadata = {
        'session_id': session_id,
        'start_time': datetime.now().isoformat(),
        'source': 'synthetic_generator',
        'track_name': 'Synthetic Test Track',
        'track_length_km': track_length,
        'n_laps': n_laps
    }
    logger.create_log(field_names, metadata)
    
    print(f"Created log file: {log_path}")
    print(f"Generating {n_laps} laps...")
    
    # Generate and log multiple laps
    cumulative_time = 0.0
    
    for lap_num in range(n_laps):
        # Vary lap time slightly
        lap_time = lap_time_base + np.random.normal(0, 2)
        
        time, _, data = generate_synthetic_lap_data(lap_time, track_length, sample_rate)
        
        # Write frames
        for i in range(len(time)):
            timestamp = cumulative_time + time[i]
            frame_data = {key: float(data[key][i]) for key in field_names}
            logger.append_frame(timestamp, frame_data)
            
        cumulative_time += lap_time
        print(f"  Lap {lap_num + 1}/{n_laps}: {lap_time:.3f}s")
    
    logger.close()
    
    print(f"\nSession complete!")
    print(f"Total time: {cumulative_time:.1f}s")
    print(f"Total samples: {int(cumulative_time * sample_rate)}")
    print(f"File size: {log_path.stat().st_size / 1024:.1f} KB")
    print(f"\nTo analyze: python telemetry_analyzer.py")
    print(f"Then: File → Open Log → {log_path}")
    
    return log_path


def example_read_and_analyze_log():
    """Example: Read a log file and perform basic analysis"""
    
    print("\n" + "="*60)
    print("Reading and analyzing log file")
    print("="*60)
    
    # Find most recent synthetic session
    sessions_dir = Path('sessions')
    if not sessions_dir.exists():
        print("No sessions directory found. Run example_create_synthetic_session() first.")
        return
        
    synthetic_sessions = sorted(sessions_dir.glob('synthetic_*/telemetry.tlog'))
    if not synthetic_sessions:
        print("No synthetic sessions found. Run example_create_synthetic_session() first.")
        return
        
    log_path = synthetic_sessions[-1]
    print(f"Reading: {log_path}")
    
    # Read log
    field_names, data, metadata = TelemetryLogger.read_log(str(log_path))
    
    print(f"\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    print(f"\nData shape: {data.shape}")
    print(f"Duration: {data[-1, 0] - data[0, 0]:.1f} seconds")
    print(f"Sample rate: ~{len(data) / (data[-1, 0] - data[0, 0]):.1f} Hz")
    
    # Calculate basic statistics
    print(f"\nChannel Statistics:")
    print(f"{'Channel':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("-" * 60)
    
    for i, name in enumerate(field_names):
        col_data = data[:, i + 1]  # +1 to skip timestamp
        print(f"{name:<20} {col_data.min():>10.2f} {col_data.max():>10.2f} "
              f"{col_data.mean():>10.2f} {col_data.std():>10.2f}")
    
    # Define track and segment laps
    track = TrackDefinition(
        name=metadata.get('track_name', 'Unknown'),
        length_km=metadata.get('track_length_km', 3.0),
        start_finish_lat=data[0, field_names.index('latitude') + 1],
        start_finish_lon=data[0, field_names.index('longitude') + 1],
        tolerance_m=50.0
    )
    
    print(f"\nSegmenting laps...")
    laps = LapAnalyzer.segment_laps(data, field_names, track)
    
    print(f"Found {len(laps)} laps:")
    print(f"{'Lap':<6} {'Time':>10} {'Delta':>10}")
    print("-" * 30)
    
    best_time = min(lap.lap_time for lap in laps)
    for lap in laps:
        delta = lap.lap_time - best_time
        delta_str = f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}"
        print(f"{lap.lap_number:<6} {lap.lap_time:>10.3f} {delta_str:>10}")
    
    print(f"\nBest lap: {best_time:.3f}s")


def example_convert_fueltech_csv():
    """Example: Convert FuelTech CSV to internal format"""
    
    print("\n" + "="*60)
    print("Converting FuelTech CSV to internal format")
    print("="*60)
    
    # Check if example CSV exists
    csv_path = Path('../LOG_EXAMPLE.csv')
    if not csv_path.exists():
        csv_path = Path('LOG_EXAMPLE.csv')
        if not csv_path.exists():
            print("LOG_EXAMPLE.csv not found")
            return
    
    print(f"Reading: {csv_path}")
    
    # Import CSV
    field_names, data, metadata = FuelTechImporter.import_csv(str(csv_path))
    
    print(f"Imported {len(data)} samples")
    print(f"Channels: {len(field_names)}")
    
    # Create output session
    session_id = f"fueltech_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = Path('sessions') / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as binary log
    log_path = session_dir / 'telemetry.tlog'
    logger = TelemetryLogger(str(log_path))
    logger.create_log(field_names, metadata)
    
    print(f"Converting to: {log_path}")
    
    for i in range(len(data)):
        timestamp = float(data[i, 0])
        frame_data = {field_names[j]: float(data[i, j + 1]) 
                     for j in range(len(field_names))}
        logger.append_frame(timestamp, frame_data)
    
    logger.close()
    
    print(f"Conversion complete!")
    print(f"Original size: {csv_path.stat().st_size / 1024:.1f} KB")
    print(f"Binary size: {log_path.stat().st_size / 1024:.1f} KB")
    print(f"Compression: {log_path.stat().st_size / csv_path.stat().st_size * 100:.1f}%")


def main():
    """Run all examples"""
    
    print("Professional Telemetry Analyzer - Examples")
    print("=" * 60)
    
    # Example 1: Create synthetic session
    log_path = example_create_synthetic_session()
    
    # Example 2: Read and analyze
    example_read_and_analyze_log()
    
    # Example 3: Convert FuelTech CSV
    example_convert_fueltech_csv()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run the analyzer: python telemetry_analyzer.py")
    print("2. Open the synthetic log file")
    print("3. Configure track and segment laps")
    print("4. Explore different analysis views")


if __name__ == '__main__':
    main()
