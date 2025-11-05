# Professional Motorsport Telemetry Analyzer

A standalone, professional-grade telemetry analysis and visualization application for Windows, designed for motorsport engineering environments. Features real-time data streaming at 10 Hz, offline analysis up to 100 Hz, GPS-based lap segmentation, and advanced multi-channel visualization.

## Features

### Core Capabilities
- **Real-time Telemetry Streaming**: Connect to live telemetry service at ~10 Hz
- **Offline Log Analysis**: Analyze recorded sessions at rates up to 100 Hz
- **GPS-Based Lap Segmentation**: Automatic lap detection using distance integration
- **FuelTech Import**: Native support for FuelTech datalogger CSV files
- **Professional Dark Theme**: Modern UI optimized for motorsport engineering

### Visualization Groups

1. **Primary View** (4-Graph Layout)
   - Engine RPM
   - Vehicle Speed with Gear Overlay
   - Throttle Position
   - Longitudinal & Lateral G-Forces
   - Time-window slider for detailed inspection

2. **Dashboard View**
   - RPM and Speed gauges
   - GG Diagram for g-force visualization
   - Real-time analog-style displays

3. **Mixture Tuning**
   - RPM vs MAP correlation
   - Exhaust O2 vs Lambda Correction
   - Injection time traces (Bank A & B)
   - Lambda scatter plot colored by TPS

4. **Histogram Analysis**
   - Engine operating point distribution
   - RPM histogram
   - Throttle position distribution

5. **Suspension Analysis**
   - 4-corner shock travel
   - Wheel speed comparison
   - Frequency-domain analysis ready

6. **Track Map**
   - Vector-based track visualization
   - Speed-based color coding
   - Brake and TPS event markers
   - No satellite imagery required

### Advanced Features
- **Lap Comparison**: Compare multiple laps side-by-side
- **Delta Analysis**: Real-time delta to best lap
- **Session Management**: Save and reload analysis sessions
- **Multi-Channel Overlay**: Overlay any channels on graphs
- **Data Decimation**: Smooth performance with large datasets
- **Custom Scaling**: Adjustable Y-axis ranges
- **Zoom and Pan**: Interactive plot navigation
- **Export Capability**: Export plots and data

## Installation

### Prerequisites
- Windows 10 or later (64-bit)
- Python 3.8 or higher
- 4 GB RAM minimum (8 GB recommended)
- 1920x1080 display minimum

### Step 1: Install Python
Download and install Python 3.8+ from [python.org](https://www.python.org/downloads/)

**Important**: Check "Add Python to PATH" during installation

### Step 2: Install Dependencies
Open Command Prompt or PowerShell and navigate to the application directory:

```bash
cd path\to\telemetry_analyzer
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python telemetry_analyzer.py
```

The application should launch with the main window.

## Usage

### Quick Start

#### 1. Opening a Log File
- **File → Open Log...** to load a `.tlog` binary log file
- **File → Import FuelTech CSV...** to import FuelTech datalogger files

#### 2. Real-Time Streaming
1. Connect your telemetry hardware to a COM port
2. **Session → Connect to Telemetry...**
3. Enter COM port (e.g., `COM3`)
4. **Session → Start Recording** to log data
5. **Session → Stop Recording** when finished

#### 3. Configuring Track
1. Load or record a session with GPS data
2. **Analysis → Configure Track...**
3. Use **Auto-Detect** to estimate parameters from GPS trace
4. Adjust track length and start/finish line position
5. Click **OK**

#### 4. Lap Segmentation
1. Configure track first (see above)
2. **Analysis → Segment Laps**
3. Laps appear in the right-side dock
4. Click any lap to view detailed analysis

#### 5. Analyzing Data
- Use the **tab bar** to switch between analysis views
- **Time slider** adjusts visible window in Primary View
- **Position slider** seeks through recorded data
- Select laps from the **Lap List** to compare
- Right-click plots for zoom/pan options

### Data Formats

#### Internal Log Format (.tlog)
Binary format optimized for fast read/write:
- Header with field names and metadata
- Timestamp (float64) + values (float32) per frame
- Efficient for both real-time logging and replay

#### FuelTech CSV Import
Automatically maps FuelTech Portuguese field names:
- `RPM` → rpm
- `TPS` → tps
- `Força_G_aceleração` → g_accel
- And many more...

### Keyboard Shortcuts
- **Ctrl+O**: Open log file
- **Ctrl+I**: Import FuelTech CSV
- **Ctrl+Q**: Quit application
- **Space**: Play/Pause (when implemented)
- **Left/Right Arrow**: Seek backward/forward

### Configuration Persistence
The application automatically saves:
- Window size and position
- Selected channels and scales
- Track definitions
- Display preferences

Settings are stored in: `%APPDATA%/MotorsportTech/TelemetryAnalyzer/`

## Telemetry Service Integration

The application interfaces with `telemetry_service.py` for real-time data acquisition. 

### Expected Frame Format
```python
{
    "timestamp": 1234567890.123,
    "data": {
        "rpm": 6500,
        "tps": 85.5,
        "map": 95.2,
        "engine_temp": 92.3,
        "wheel_speed_fl": 120.5,
        # ... more channels
    }
}
```

### Supported Channels
The analyzer recognizes these standard telemetry channels:
- **Engine**: rpm, tps, map, air_temp, engine_temp, oil_pressure, fuel_pressure
- **Drivetrain**: gear, wheel_speed_fl/fr/rl/rr, tc_slip, tc_retard
- **Chassis**: g_accel, g_lateral, yaw_rate_frontal, yaw_rate_lateral
- **Suspension**: shock_fl/fr/rl/rr
- **Brakes**: brake_pressure
- **GPS**: latitude, longitude, distance_km, gps_speed_knots
- **Mixture**: exhaust_o2, lambda_correction, inj_time_bank_a/b
- **Temperatures**: oil_temp1/2, trans_temp

## Performance Optimization

### For Real-Time Streaming (10 Hz)
- Data buffering with 1000-sample deque
- Automatic decimation for display
- 100ms update timer balances smoothness and CPU usage

### For Offline Analysis (100 Hz)
- NumPy arrays for efficient storage
- Vectorized operations for fast computation
- Progressive rendering for large datasets
- Memory-mapped files for huge logs (future enhancement)

### Display Optimization
- PyQtGraph hardware acceleration
- Automatic downsampling for zoomed-out views
- Configurable plot update rates
- Efficient binary log format

## Track Configuration

### GPS-Based Lap Detection
The system uses cumulative distance with tolerance-based correction:

1. **Distance Integration**: GPS coordinates → distance traveled
2. **Lap Detection**: Distance rollover detected when `current < previous - tolerance`
3. **Drift Correction**: Tolerance parameter prevents false detections
4. **Start/Finish Line**: Manually positioned or auto-detected

### Recommended Tolerance Values
- Short tracks (<2 km): 30-50m
- Medium tracks (2-5 km): 50-100m
- Long tracks (>5 km): 100-200m

## Troubleshooting

### COM Port Issues
- **Error: "Access Denied"**: Close other applications using the port
- **Error: "Port not found"**: Check Device Manager for correct port number
- Use Device Manager → Ports to verify COM port

### Import Errors
- **FuelTech CSV**: Ensure file uses comma separators, not semicolons
- **Character encoding**: File must be UTF-8 (Excel may save as ANSI)
- **Missing columns**: Some channels may not map; this is normal

### Performance Issues
- **Slow plotting**: Reduce visible time window
- **High CPU**: Increase update timer interval (edit source)
- **Memory usage**: Large files may require 64-bit Python

### Display Issues
- **Plots not visible**: Check monitor scaling (125% recommended)
- **Text too small**: Adjust Windows display scaling
- **Dark theme**: Works best on Windows 10+

## Development

### Project Structure
```
telemetry_analyzer/
├── telemetry_analyzer.py    # Main application
├── telemetry_service.py     # Serial telemetry decoder (from project)
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── sessions/               # Recorded sessions (created at runtime)
    └── YYYYMMDD_HHMMSS/
        ├── telemetry.tlog  # Binary log
        └── metadata.json   # Session info
```

### Extending the Analyzer

#### Adding Custom Channels
Edit the `FIELD_MAPPING` in `FuelTechImporter` class:
```python
FIELD_MAPPING = {
    'YourCustomField': 'custom_field',
    # ...
}
```

#### Creating New Analysis Views
Add a new method in `TelemetryAnalyzer`:
```python
def create_custom_view(self):
    tab = QWidget()
    layout = QVBoxLayout()
    
    # Add your plots
    custom_plot = TelemetryPlotWidget("Title", "Units")
    custom_plot.add_channel('your_channel', 'color', 2)
    layout.addWidget(custom_plot)
    
    tab.setLayout(layout)
    self.tab_widget.addTab(tab, "Custom View")
```

#### Customizing Colors
Modify `apply_dark_theme()` method with your palette preferences.

## Technical Specifications

### Data Rates
- **Live Streaming**: 10 Hz (100ms intervals)
- **Offline Playback**: Up to 100 Hz
- **Display Update**: 10 Hz (configurable)

### File Sizes
- **Binary logs**: ~400 bytes/frame (40 channels × 4 bytes + timestamp)
- **1 hour @ 10 Hz**: ~14.4 MB
- **1 hour @ 100 Hz**: ~144 MB

### System Requirements
- **CPU**: Dual-core 2 GHz minimum, Quad-core recommended
- **RAM**: 4 GB minimum, 8 GB for large files
- **Disk**: 50 MB installation + log storage
- **Display**: 1920×1080 minimum, 2560×1440 recommended

## License

This software is provided for motorsport engineering and data analysis purposes.

## Support

For issues, feature requests, or questions:
1. Check the Troubleshooting section
2. Review the telemetry_service.py documentation
3. Verify all dependencies are correctly installed

## Acknowledgments

- **PyQtGraph**: High-performance plotting library
- **PyQt5**: Professional GUI framework
- **NumPy/SciPy**: Numerical computing foundation

---

**Version**: 1.0  
**Last Updated**: 2025  
**Compatibility**: Windows 10/11, Python 3.8+
