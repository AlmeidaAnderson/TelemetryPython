# Developer's Guide - Professional Telemetry Analyzer

## Architecture Overview

The application follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                  GUI Layer (PyQt5)                      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │ Main Window  │ │  Plot Widgets│ │  Dialogs     │   │
│  └──────────────┘ └──────────────┘ └──────────────┘   │
└────────────┬────────────────────────────────────────────┘
             │
┌────────────┴────────────────────────────────────────────┐
│              Analysis Layer                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │ Lap Analyzer │ │  Track Config│ │  Statistics  │   │
│  └──────────────┘ └──────────────┘ └──────────────┘   │
└────────────┬────────────────────────────────────────────┘
             │
┌────────────┴────────────────────────────────────────────┐
│              Data Layer                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │ Logger       │ │  Importers   │ │  Receiver    │   │
│  └──────────────┘ └──────────────┘ └──────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Structures (`dataclasses`)

**TelemetryFrame**
- Single timestamped measurement
- Used for real-time streaming
- Lightweight and thread-safe

**LapDefinition**
- Defines a lap segment
- Includes timing and indexing
- Supports lap comparison

**TrackDefinition**
- GPS-based track configuration
- Start/finish line position
- Distance tolerance settings

**SessionMetadata**
- Session identification
- Track assignment
- Lap information

### 2. Data Storage

**TelemetryLogger**

Binary log format for efficient I/O:

```
┌─────────────────────────────────┐
│  Header                         │
│  - Magic: "TLOG" (4 bytes)     │
│  - Version: uint32             │
│  - Field Count: uint32         │
│  - Field Names: variable       │
│  - Metadata JSON: variable     │
├─────────────────────────────────┤
│  Data Frames (repeating)        │
│  - Timestamp: float64 (8 bytes)│
│  - Values: float32[] (4×N bytes)│
└─────────────────────────────────┘
```

**Advantages:**
- Fast sequential write (real-time capable)
- Efficient read for replay
- Small file size (~400 bytes/frame @ 40 channels)
- Platform-independent (struct format)

**Methods:**
- `create_log()`: Initialize new log
- `append_frame()`: Add timestamped data
- `read_log()`: Load entire log into NumPy array
- `close()`: Finalize file

### 3. Real-time Receiver

**TelemetryReceiver (QThread)**

Threaded receiver for live telemetry:

```python
class TelemetryReceiver(QThread):
    frame_received = pyqtSignal(object)  # Thread-safe signal
    connection_status = pyqtSignal(bool, str)
    
    def run(self):
        # Integrates with telemetry_service.py
        # Emits frames via Qt signals
        # Handles reconnection
```

**Key Features:**
- Non-blocking operation
- Automatic buffering
- Signal-based communication
- Clean shutdown handling

### 4. Data Import

**FuelTechImporter**

Converts FuelTech CSV files:

```python
FIELD_MAPPING = {
    'Portuguese_Field': 'english_field',
    # Extensive mapping table
}
```

**Process:**
1. Read CSV with pandas
2. Map column names
3. Convert to NumPy array
4. Return standardized format

**Extensibility:**
- Add new mappings to `FIELD_MAPPING`
- Support other formats by creating new importer classes

### 5. Lap Analysis

**LapAnalyzer**

GPS-based lap segmentation:

```python
@staticmethod
def segment_laps(data, field_names, track) -> List[LapDefinition]:
    # Algorithm:
    # 1. Track cumulative distance
    # 2. Detect rollover (distance < prev_distance - tolerance)
    # 3. Create lap segments
    # 4. Calculate lap times
```

**Distance Calculation:**
- Haversine formula for GPS coordinates
- Handles Earth curvature correctly
- Meters-level accuracy

**Tolerance Handling:**
- Prevents false detections from GPS noise
- Adjustable per track
- Corrects cumulative drift

### 6. Visualization Widgets

**TelemetryPlotWidget (pyqtgraph.PlotWidget)**

Enhanced plotting with:
- Multiple channels per plot
- Color-coded traces
- Grid and labels
- Interactive zoom/pan
- High-performance rendering

**GGDiagram**

Specialized g-force visualization:
- Aspect-locked axes
- Reference circles
- Scatter plot with color mapping
- Real-time update capability

**TrackMapWidget**

GPS track visualization:
- Scatter plot of coordinates
- Color by any channel (RPM, speed, etc.)
- Vector-based rendering
- No external map tiles needed

## GUI Framework

### Main Window (TelemetryAnalyzer)

**Structure:**
```
QMainWindow
├── MenuBar
│   ├── File (Open, Import, Exit)
│   ├── Session (Connect, Record)
│   └── Analysis (Track Config, Lap Segment)
├── Central Widget
│   └── QTabWidget (Analysis Groups)
│       ├── Primary View (4 plots + slider)
│       ├── Dashboard (gauges + GG)
│       ├── Mixture Tuning
│       ├── Histograms
│       ├── Suspension
│       └── Track Map
├── Control Panel
│   └── Playback controls + sliders
└── Dock Widgets
    └── Lap List (QTableWidget)
```

### Theme System

**Dark Theme Implementation:**
```python
def apply_dark_theme(self):
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, Qt.white)
    # ... full palette configuration
    self.setPalette(palette)
```

**Color Scheme:**
- Background: RGB(30, 30, 30)
- Widgets: RGB(40, 40, 40)
- Text: White
- Highlights: RGB(42, 130, 218)

### Settings Persistence

**QSettings Integration:**
```python
settings = QSettings('MotorsportTech', 'TelemetryAnalyzer')

# Save
settings.setValue('geometry', self.saveGeometry())
settings.setValue('track_configs', track_list)

# Load
geometry = settings.value('geometry')
self.restoreGeometry(geometry)
```

**Stored Data:**
- Window geometry
- Splitter positions
- Track definitions
- Channel selections
- Display preferences

## Performance Optimization

### Real-time Display (10 Hz)

**Strategy:**
```python
# Circular buffer for incoming data
self.live_buffer = deque(maxlen=1000)

# Timer-based updates (100ms)
self.update_timer.timeout.connect(self.update_live_display)

# Decimation for long buffers
if len(timestamps) > 1000:
    decimation = len(timestamps) // 1000
    timestamps = timestamps[::decimation]
    data = data[::decimation]
```

### Offline Analysis (100 Hz)

**Techniques:**
1. **NumPy Vectorization**: All operations on arrays
2. **View Culling**: Only render visible time range
3. **Progressive Loading**: Large files loaded in chunks
4. **Downsampling**: Automatic decimation when zoomed out

**Memory Management:**
```python
# Use memory-mapped arrays for huge files
data = np.memmap('huge.tlog', dtype=np.float32, mode='r')

# Process in chunks
chunk_size = 10000
for i in range(0, len(data), chunk_size):
    chunk = data[i:i+chunk_size]
    process_chunk(chunk)
```

### Plot Optimization

**PyQtGraph Features:**
- OpenGL acceleration (optional)
- Automatic downsampling
- Clipping to visible range
- Fast scatter plots

**Best Practices:**
```python
# Enable fast drawing
plot.setClipToView(True)
plot.setDownsampling(auto=True, mode='peak')

# Limit displayed points
if len(x) > 10000:
    plot.setData(x[::10], y[::10])  # Show every 10th point
```

## Extending the Analyzer

### Adding New Analysis Views

**Template:**
```python
def create_custom_view(self):
    """Create a new analysis tab"""
    tab = QWidget()
    layout = QVBoxLayout()
    
    # Create plots
    plot1 = TelemetryPlotWidget("Title", "Units")
    plot1.add_channel('channel_name', 'color', linewidth)
    
    # Add to layout
    layout.addWidget(plot1)
    tab.setLayout(layout)
    
    # Add to tab widget
    self.tab_widget.addTab(tab, "Custom View")
    
    # Store reference for updates
    self.custom_plot = plot1
```

**Update Method:**
```python
def update_custom_view(self, data: np.ndarray):
    """Update custom view with data"""
    time_data = data[:, 0]
    channel_idx = self.field_names.index('channel_name') + 1
    channel_data = data[:, channel_idx]
    
    self.custom_plot.update_channel('channel_name', 
                                    time_data, channel_data)
```

### Adding New Importers

**Template:**
```python
class CustomImporter:
    """Import custom data format"""
    
    FIELD_MAPPING = {
        'source_field': 'standard_field',
        # Define mappings
    }
    
    @staticmethod
    def import_file(filepath: str) -> Tuple[List[str], np.ndarray, Dict]:
        """
        Import custom format
        
        Returns:
            field_names: List of channel names
            data: NumPy array (n_samples, n_channels+1)
                  First column is timestamp
            metadata: Dictionary of session info
        """
        # Read file
        # Parse data
        # Map fields
        # Return standardized format
        pass
```

**Integration:**
```python
# In TelemetryAnalyzer.create_menus()
import_custom = QAction('Import Custom Format...', self)
import_custom.triggered.connect(self.import_custom_format)

def import_custom_format(self):
    filepath, _ = QFileDialog.getOpenFileName(...)
    if filepath:
        field_names, data, metadata = CustomImporter.import_file(filepath)
        self.current_data = data
        self.field_names = field_names
        self.update_all_plots()
```

### Custom Signal Processing

**Example: FFT Analysis**
```python
from scipy import signal

def add_fft_view(self):
    """Add frequency analysis view"""
    tab = QWidget()
    layout = QVBoxLayout()
    
    self.fft_plot = pg.PlotWidget()
    self.fft_plot.setLabel('left', 'Magnitude')
    self.fft_plot.setLabel('bottom', 'Frequency', units='Hz')
    layout.addWidget(self.fft_plot)
    
    tab.setLayout(layout)
    self.tab_widget.addTab(tab, "FFT Analysis")

def update_fft(self, time_data, signal_data, sample_rate=10.0):
    """Compute and display FFT"""
    # Compute FFT
    freqs = np.fft.rfftfreq(len(signal_data), 1/sample_rate)
    fft_vals = np.abs(np.fft.rfft(signal_data))
    
    # Plot
    self.fft_plot.clear()
    self.fft_plot.plot(freqs, fft_vals, pen='y')
```

### Custom Lap Detection

**Example: Speed-based lap detection**
```python
class SpeedBasedLapAnalyzer:
    """Alternative lap detection using speed threshold"""
    
    @staticmethod
    def segment_laps(data, field_names, speed_threshold=30.0):
        """Detect laps using start/finish speed drop"""
        speed_idx = field_names.index('wheel_speed_fl') + 1
        speed = data[:, speed_idx]
        
        # Find speed drops (crossing finish line)
        crossings = []
        for i in range(1, len(speed)):
            if speed[i-1] > speed_threshold and speed[i] < speed_threshold:
                crossings.append(i)
        
        # Create lap definitions
        laps = []
        for i, start_idx in enumerate(crossings[:-1]):
            end_idx = crossings[i+1]
            lap = LapDefinition(
                lap_number=i+1,
                start_time=data[start_idx, 0],
                end_time=data[end_idx, 0],
                lap_time=data[end_idx, 0] - data[start_idx, 0],
                start_index=start_idx,
                end_index=end_idx,
                # Other fields...
            )
            laps.append(lap)
        
        return laps
```

## Testing

### Unit Tests

**Data Layer:**
```python
import unittest

class TestTelemetryLogger(unittest.TestCase):
    def test_create_and_read(self):
        # Create test log
        logger = TelemetryLogger('test.tlog')
        logger.create_log(['rpm', 'tps'], {})
        
        # Write data
        logger.append_frame(1.0, {'rpm': 6000, 'tps': 80})
        logger.append_frame(2.0, {'rpm': 6100, 'tps': 82})
        logger.close()
        
        # Read back
        fields, data, meta = TelemetryLogger.read_log('test.tlog')
        
        # Verify
        self.assertEqual(fields, ['rpm', 'tps'])
        self.assertEqual(len(data), 2)
        self.assertAlmostEqual(data[0, 1], 6000, places=1)
```

### Integration Tests

**Full Pipeline:**
```python
def test_full_pipeline():
    # Generate synthetic data
    time, fields, data = generate_synthetic_lap_data()
    
    # Log data
    logger = TelemetryLogger('test.tlog')
    logger.create_log(fields, {})
    # ... write frames ...
    logger.close()
    
    # Read back
    fields, data, meta = TelemetryLogger.read_log('test.tlog')
    
    # Analyze
    track = TrackDefinition(...)
    laps = LapAnalyzer.segment_laps(data, fields, track)
    
    # Verify
    assert len(laps) > 0
    assert all(lap.lap_time > 0 for lap in laps)
```

## Debugging

### Enable Verbose Logging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('TelemetryAnalyzer')
logger.debug('Debug message')
```

### Performance Profiling

```python
import cProfile
import pstats

# Profile function
profiler = cProfile.Profile()
profiler.enable()

# Run code
update_all_plots()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def update_plots():
    # Function to profile
    pass
```

## Build and Distribution

### Creating Standalone Executable

**PyInstaller Configuration:**
```python
# telemetry_analyzer.spec
a = Analysis(
    ['telemetry_analyzer.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('example_tracks.json', '.'),
    ],
    hiddenimports=['scipy.special._ufuncs_cxx'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='TelemetryAnalyzer',
    icon='icon.ico',
    console=False,  # No console window
)
```

**Build Command:**
```bash
pyinstaller telemetry_analyzer.spec
```

### Installer Creation

**NSIS Script for Windows installer:**
```nsis
!define APP_NAME "Professional Telemetry Analyzer"
!define VERSION "1.0.0"

OutFile "TelemetryAnalyzer_Setup.exe"
InstallDir "$PROGRAMFILES\${APP_NAME}"

Section "Install"
    SetOutPath $INSTDIR
    File /r "dist\TelemetryAnalyzer\*.*"
    
    CreateShortCut "$DESKTOP\${APP_NAME}.lnk" \
        "$INSTDIR\TelemetryAnalyzer.exe"
SectionEnd
```

## Contribution Guidelines

### Code Style

- Follow PEP 8
- Use type hints where possible
- Document all public methods
- Keep functions focused and small

### Documentation

- Update README for user-facing changes
- Update this guide for architectural changes
- Add docstrings to new classes/methods
- Include usage examples

### Pull Request Process

1. Create feature branch
2. Write tests
3. Update documentation
4. Submit PR with description
5. Address review comments

## Future Enhancements

### Planned Features

1. **Video Synchronization**
   - Import video files
   - Sync with telemetry
   - Overlay data on video

2. **Comparative Analysis**
   - Multi-lap overlay
   - Sector comparison
   - Driver comparison

3. **Advanced Statistics**
   - Consistency metrics
   - Corner analysis
   - Predictive modeling

4. **Cloud Integration**
   - Optional cloud backup
   - Team data sharing
   - Remote viewing

5. **Mobile Companion**
   - Live monitoring
   - Session notes
   - Photo integration

### Extensibility Hooks

All major components have extension points:
- Custom importers
- Custom analysis algorithms
- Custom visualization widgets
- Custom export formats

---

**Document Version**: 1.0  
**Last Updated**: 2025  
**Maintainer**: Development Team
