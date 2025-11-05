"""
Professional Motorsport Telemetry Analysis Application
Native Windows application for real-time and offline telemetry visualization
"""

import sys
import os
import json
import time
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import struct
import traceback

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QAction, QFileDialog, QDockWidget, QTabWidget,
    QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox, QComboBox,
    QListWidget, QSplitter, QDialog, QDialogButtonBox, QFormLayout,
    QLineEdit, QMessageBox, QProgressDialog, QGroupBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit, QTreeWidget,
    QTreeWidgetItem
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QSettings
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon, QTextCursor

import pyqtgraph as pg
from pyqtgraph import PlotWidget, mkPen, mkBrush


# ============================================================================
# Logging Setup
# ============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TelemetryAnalyzer')


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TelemetryFrame:
    """Single telemetry data frame"""
    timestamp: float
    data: Dict[str, float]


@dataclass
class LapDefinition:
    """Lap segment definition"""
    lap_number: int
    start_distance: float
    end_distance: float
    start_time: float
    end_time: float
    lap_time: float
    start_index: int
    end_index: int


@dataclass
class TrackDefinition:
    """Track configuration"""
    name: str
    length_km: float
    start_finish_lat: float
    start_finish_lon: float
    tolerance_m: float = 50.0


@dataclass
class SessionMetadata:
    """Session metadata"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    track: Optional[TrackDefinition]
    sample_count: int
    duration_s: float
    laps: List[LapDefinition]


# ============================================================================
# Data Storage
# ============================================================================

class TelemetryLogger:
    """Efficient binary telemetry log storage"""
    
    HEADER_MAGIC = b'TLOG'
    VERSION = 1
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None
        self.field_names = []
        self.field_count = 0
        self.metadata = {}
        
    def create_log(self, field_names: List[str], metadata: Dict):
        """Create new log file"""
        self.field_names = field_names
        self.field_count = len(field_names)
        self.metadata = metadata
        
        # Create header
        self.file = open(self.filepath, 'wb')
        
        # Magic and version
        self.file.write(self.HEADER_MAGIC)
        self.file.write(struct.pack('<I', self.VERSION))
        
        # Field count and names
        self.file.write(struct.pack('<I', self.field_count))
        field_str = '\n'.join(field_names).encode('utf-8')
        self.file.write(struct.pack('<I', len(field_str)))
        self.file.write(field_str)
        
        # Metadata
        meta_str = json.dumps(metadata).encode('utf-8')
        self.file.write(struct.pack('<I', len(meta_str)))
        self.file.write(meta_str)
        
        self.file.flush()
        
    def append_frame(self, timestamp: float, data: Dict[str, float]):
        """Append telemetry frame"""
        if not self.file:
            return
            
        # Write timestamp
        self.file.write(struct.pack('<d', timestamp))
        
        # Write data fields in order
        for field in self.field_names:
            value = data.get(field, 0.0)
            self.file.write(struct.pack('<f', value))
            
    def close(self):
        """Close log file"""
        if self.file:
            self.file.close()
            self.file = None
            
    @staticmethod
    def read_log(filepath: str) -> Tuple[List[str], np.ndarray, Dict]:
        """Read log file and return field names, data array, and metadata"""
        with open(filepath, 'rb') as f:
            # Read magic
            magic = f.read(4)
            if magic != TelemetryLogger.HEADER_MAGIC:
                raise ValueError("Invalid log file format")
                
            # Read version
            version = struct.unpack('<I', f.read(4))[0]
            
            # Read field names
            field_count = struct.unpack('<I', f.read(4))[0]
            field_str_len = struct.unpack('<I', f.read(4))[0]
            field_str = f.read(field_str_len).decode('utf-8')
            field_names = field_str.split('\n')
            
            # Read metadata
            meta_len = struct.unpack('<I', f.read(4))[0]
            metadata = json.loads(f.read(meta_len).decode('utf-8'))
            
            # Read data frames
            frames = []
            record_size = 8 + field_count * 4  # timestamp + fields
            
            while True:
                chunk = f.read(record_size)
                if len(chunk) < record_size:
                    break
                    
                timestamp = struct.unpack('<d', chunk[:8])[0]
                values = struct.unpack(f'<{field_count}f', chunk[8:])
                frames.append([timestamp] + list(values))
                
        # Convert to numpy array
        data = np.array(frames, dtype=np.float32)
        
        return field_names, data, metadata


# ============================================================================
# Real-time Data Receiver
# ============================================================================

class TelemetryReceiver(QThread):
    """Real-time telemetry data receiver thread"""
    
    frame_received = pyqtSignal(object)  # TelemetryFrame
    connection_status = pyqtSignal(bool, str)
    
    def __init__(self, port: str, baudrate: int = 115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.running = False
        self.service_handle = None
        
    def run(self):
        """Run receiver thread"""
        try:
            # Import telemetry service
            import sys
            sys.path.insert(0, '/mnt/project')
            from telemetry_service import start_telemetry_service
            
            self.running = True
            self.connection_status.emit(True, f"Connected to {self.port}")
            
            # Start service
            self.service_handle = start_telemetry_service(
                port=self.port,
                baudrate=self.baudrate,
                on_frame=self._on_frame,
                log_path_pattern=None  # We handle logging ourselves
            )
            
            # Keep thread alive
            while self.running:
                time.sleep(0.1)
                
        except Exception as e:
            self.connection_status.emit(False, f"Error: {str(e)}")
        finally:
            if self.service_handle:
                self.service_handle.stop()
                
    def _on_frame(self, frame: Dict):
        """Handle received frame"""
        if self.running:
            tel_frame = TelemetryFrame(
                timestamp=frame['timestamp'],
                data=frame['data']
            )
            self.frame_received.emit(tel_frame)
            
    def stop(self):
        """Stop receiver"""
        self.running = False
        self.wait()


# ============================================================================
# FuelTech CSV Import
# ============================================================================

class FuelTechImporter:
    """Import FuelTech datalogger CSV files"""
    
    # FuelTech to internal field mapping (Portuguese to English)
    FIELD_MAPPING = {
        'TIME': 'timestamp',
        'RPM': 'rpm',
        'TPS': 'tps',
        'Posição_do_acelerador': 'tps',
        'MAP': 'map',
        'Temp._do_motor': 'engine_temp',
        'Temp._do_Ar': 'air_temp',
        'Pressão_de_Óleo': 'oil_pressure',
        'Marcha': 'gear',
        'Marcha_Atual': 'gear',
        'Velocidade_da_roda_frontal_esquerda': 'wheel_speed_fl',
        'Velocidade_da_roda_traseira_esquerda': 'wheel_speed_rl',
        'Velocidade_da_roda_traseira_direita': 'wheel_speed_rr',
        'Força_G_aceleração': 'g_accel',
        'Força_G_lateral': 'g_lateral',
        'Controle_de_tração_-_Slip': 'tc_slip',
        'Amortecedor_dianteiro_esquerdo': 'shock_fl',
        'Amortecedor_dianteiro_direito': 'shock_fr',
        'Amortecedor_traseiro_esquerdo': 'shock_rl',
        'Amortecedor_traseiro_direito': 'shock_rr',
        'Pressão_do_freio': 'brake_pressure',
        'Temperatura_da_transmissão': 'trans_temp',
        'Tempo_de_Injeção_Banco_A': 'inj_time_bank_a',
        'Sonda_Malha_Fechada': 'exhaust_o2',
        'Correção_do_malha_fechada': 'lambda_correction',
    }
    
    @staticmethod
    def import_csv(filepath: str) -> Tuple[List[str], np.ndarray, Dict]:
        """Import FuelTech CSV file"""
        import pandas as pd

        # Read CSV
        df = pd.read_csv(filepath)

        # Map column names
        mapped_columns = {}
        for col in df.columns:
            if col in FuelTechImporter.FIELD_MAPPING:
                mapped_columns[col] = FuelTechImporter.FIELD_MAPPING[col]
            else:
                # Keep original name if no mapping
                mapped_columns[col] = col.lower().replace(' ', '_')

        df = df.rename(columns=mapped_columns)

        # Convert string values to numeric
        def convert_value(val):
            """Convert string values to numeric"""
            if pd.isna(val):
                return 0.0
            if isinstance(val, str):
                val_upper = val.upper().strip()
                if val_upper in ['OFF', 'DESLIGADO', 'NO', 'FALSE']:
                    return 0.0
                elif val_upper in ['ON', 'LIGADO', 'YES', 'TRUE']:
                    return 1.0
                else:
                    try:
                        return float(val)
                    except ValueError:
                        return 0.0
            return float(val)

        # Apply conversion to all columns
        for col in df.columns:
            df[col] = df[col].apply(convert_value)

        # Extract field names (exclude timestamp)
        field_names = [c for c in df.columns if c != 'timestamp']

        # Convert to numpy array
        data = df.to_numpy(dtype=np.float32)

        # Metadata
        metadata = {
            'source': 'FuelTech CSV Import',
            'filename': os.path.basename(filepath),
            'import_time': datetime.now().isoformat()
        }

        return field_names, data, metadata


# ============================================================================
# Lap Analysis
# ============================================================================

class LapAnalyzer:
    """GPS-based lap segmentation and analysis"""
    
    @staticmethod
    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS points (Haversine formula)"""
        R = 6371000  # Earth radius in meters
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    @staticmethod
    def segment_laps(data: np.ndarray, field_names: List[str], 
                     track: TrackDefinition) -> List[LapDefinition]:
        """Segment telemetry into laps based on distance"""
        
        # Find column indices
        try:
            time_idx = 0  # timestamp is always first
            dist_idx = field_names.index('distance_km') + 1
        except ValueError:
            return []
            
        laps = []
        track_length_km = track.length_km
        tolerance_km = track.tolerance_m / 1000.0
        
        # Find lap transitions
        lap_num = 1
        lap_start_idx = 0
        lap_start_dist = data[0, dist_idx]
        lap_start_time = data[0, time_idx]
        
        for i in range(1, len(data)):
            current_dist = data[i, dist_idx]
            prev_dist = data[i-1, dist_idx]
            
            # Check for lap completion (distance rolls over)
            if current_dist < prev_dist - tolerance_km:
                # Lap completed
                lap_end_idx = i - 1
                lap_end_dist = data[lap_end_idx, dist_idx]
                lap_end_time = data[lap_end_idx, time_idx]
                
                lap = LapDefinition(
                    lap_number=lap_num,
                    start_distance=lap_start_dist,
                    end_distance=lap_end_dist,
                    start_time=lap_start_time,
                    end_time=lap_end_time,
                    lap_time=lap_end_time - lap_start_time,
                    start_index=lap_start_idx,
                    end_index=lap_end_idx
                )
                laps.append(lap)
                
                # Start new lap
                lap_num += 1
                lap_start_idx = i
                lap_start_dist = current_dist
                lap_start_time = data[i, time_idx]
                
        # Add final incomplete lap if exists
        if lap_start_idx < len(data) - 1:
            lap = LapDefinition(
                lap_number=lap_num,
                start_distance=lap_start_dist,
                end_distance=data[-1, dist_idx],
                start_time=lap_start_time,
                end_time=data[-1, time_idx],
                lap_time=data[-1, time_idx] - lap_start_time,
                start_index=lap_start_idx,
                end_index=len(data) - 1
            )
            laps.append(lap)
            
        return laps


# ============================================================================
# Track Configuration Dialog
# ============================================================================

# ============================================================================
# Serial Configuration Dialog
# ============================================================================

class SerialConfigDialog(QDialog):
    """Modern serial port configuration dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.port = None
        self.baudrate = 9600
        self.bytesize = 8
        self.parity = 'N'
        self.stopbits = 1
        self.timeout = 1.0

        self.setWindowTitle("Serial Connection Configuration")
        self.setModal(True)
        self.setMinimumWidth(500)

        # Set dialog background to dark
        self.setStyleSheet("""
            QDialog {
                background-color: #2A2A2A;
                color: #DCDCDC;
            }
            QLabel {
                color: #DCDCDC;
            }
        """)

        self.init_ui()
        self.scan_ports()
        
    def init_ui(self):
        """Initialize modern UI"""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # Title section
        title = QLabel("Configure Telemetry Connection")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setStyleSheet("color: white; padding: 10px;")
        layout.addWidget(title)
        
        # Port selection group
        port_group = QGroupBox("Serial Port")
        port_group.setStyleSheet("""
            QGroupBox {
                color: white;
                font-weight: bold;
                border: 2px solid #3daee9;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        port_layout = QVBoxLayout()
        
        # Port combo box
        port_label = QLabel("Available Ports:")
        port_label.setStyleSheet("color: white; margin-left: 5px;")
        port_layout.addWidget(port_label)
        
        self.port_combo = QComboBox()
        self.port_combo.setStyleSheet("""
            QComboBox {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #3daee9;
                border-radius: 4px;
                padding: 8px;
                min-height: 25px;
                font-size: 11pt;
            }
            QComboBox:hover {
                border: 1px solid #4fc3f7;
                background-color: #353535;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #3daee9;
                margin-right: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                color: white;
                selection-background-color: #3daee9;
                border: 1px solid #3daee9;
            }
        """)
        port_layout.addWidget(self.port_combo)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Ports")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #3daee9;
                border-radius: 4px;
                padding: 6px;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #3daee9;
                border-color: #4fc3f7;
            }
            QPushButton:pressed {
                background-color: #2b8cbd;
            }
        """)
        refresh_btn.clicked.connect(self.scan_ports)
        port_layout.addWidget(refresh_btn)
        
        port_group.setLayout(port_layout)
        layout.addWidget(port_group)
        
        # Connection parameters group
        params_group = QGroupBox("Connection Parameters")
        params_group.setStyleSheet("""
            QGroupBox {
                color: white;
                font-weight: bold;
                border: 2px solid #3daee9;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        params_layout = QFormLayout()
        params_layout.setSpacing(12)
        
        # Baud rate
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(['300', '1200', '2400', '4800', '9600', '19200', '38400', '57600', '115200', '230400'])
        self.baud_combo.setCurrentText('9600')
        self.baud_combo.setStyleSheet(self._get_combo_style())
        
        # Data bits
        self.databits_combo = QComboBox()
        self.databits_combo.addItems(['5', '6', '7', '8'])
        self.databits_combo.setCurrentText('8')
        self.databits_combo.setStyleSheet(self._get_combo_style())
        
        # Parity
        self.parity_combo = QComboBox()
        self.parity_combo.addItems(['None', 'Even', 'Odd', 'Mark', 'Space'])
        self.parity_combo.setCurrentText('None')
        self.parity_combo.setStyleSheet(self._get_combo_style())
        
        # Stop bits
        self.stopbits_combo = QComboBox()
        self.stopbits_combo.addItems(['1', '1.5', '2'])
        self.stopbits_combo.setCurrentText('1')
        self.stopbits_combo.setStyleSheet(self._get_combo_style())
        
        # Timeout
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(0.1, 10.0)
        self.timeout_spin.setValue(1.0)
        self.timeout_spin.setSuffix(' s')
        self.timeout_spin.setDecimals(1)
        self.timeout_spin.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #3daee9;
                border-radius: 4px;
                padding: 6px;
                min-height: 20px;
            }
            QDoubleSpinBox:hover {
                border: 1px solid #4fc3f7;
            }
        """)
        
        # Add to form
        params_layout.addRow(self._create_label("Baud Rate:"), self.baud_combo)
        params_layout.addRow(self._create_label("Data Bits:"), self.databits_combo)
        params_layout.addRow(self._create_label("Parity:"), self.parity_combo)
        params_layout.addRow(self._create_label("Stop Bits:"), self.stopbits_combo)
        params_layout.addRow(self._create_label("Timeout:"), self.timeout_spin)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #ffaa00; padding: 5px; font-style: italic;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # Modern buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        test_btn = QPushButton("🔌 Test Connection")
        test_btn.setStyleSheet(self._get_secondary_button_style())
        test_btn.clicked.connect(self.test_connection)
        button_layout.addWidget(test_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(self._get_secondary_button_style())
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        connect_btn = QPushButton("Connect")
        connect_btn.setStyleSheet(self._get_primary_button_style())
        connect_btn.setDefault(True)
        connect_btn.clicked.connect(self.accept_connection)
        button_layout.addWidget(connect_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def _create_label(self, text):
        """Create styled label"""
        label = QLabel(text)
        label.setStyleSheet("color: white; font-size: 10pt;")
        return label
    
    def _get_combo_style(self):
        """Get combo box stylesheet"""
        return """
            QComboBox {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #3daee9;
                border-radius: 4px;
                padding: 6px;
                min-height: 20px;
            }
            QComboBox:hover {
                border: 1px solid #4fc3f7;
                background-color: #353535;
            }
            QComboBox::drop-down {
                border: none;
                width: 25px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #3daee9;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                color: white;
                selection-background-color: #3daee9;
                border: 1px solid #3daee9;
            }
        """
    
    def _get_primary_button_style(self):
        """Get primary button stylesheet"""
        return """
            QPushButton {
                background-color: #3daee9;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 30px;
                font-size: 11pt;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #4fc3f7;
            }
            QPushButton:pressed {
                background-color: #2b8cbd;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
        """
    
    def _get_secondary_button_style(self):
        """Get secondary button stylesheet"""
        return """
            QPushButton {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #3daee9;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 10pt;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #353535;
                border-color: #4fc3f7;
            }
            QPushButton:pressed {
                background-color: #252525;
            }
        """
    
    def scan_ports(self):
        """Scan for available serial ports"""
        try:
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            
            self.port_combo.clear()
            
            if ports:
                for port in ports:
                    # Format: "COM3 - USB Serial Device"
                    display_name = f"{port.device}"
                    if port.description and port.description != port.device:
                        display_name += f" - {port.description}"
                    self.port_combo.addItem(display_name, port.device)
                
                self.status_label.setText(f"✓ Found {len(ports)} port(s)")
                self.status_label.setStyleSheet("color: #4caf50; padding: 5px;")
                logger.info(f"Found {len(ports)} serial port(s)")
            else:
                self.port_combo.addItem("No ports available", None)
                self.status_label.setText("⚠ No serial ports detected")
                self.status_label.setStyleSheet("color: #ff9800; padding: 5px;")
                logger.warning("No serial ports found")
                
        except Exception as e:
            self.status_label.setText(f"❌ Error scanning ports: {str(e)}")
            self.status_label.setStyleSheet("color: #f44336; padding: 5px;")
            logger.error(f"Error scanning ports: {str(e)}")
    
    def test_connection(self):
        """Test serial connection"""
        port = self.port_combo.currentData()
        if not port:
            self.status_label.setText("⚠ No port selected")
            self.status_label.setStyleSheet("color: #ff9800; padding: 5px;")
            return
        
        try:
            self.status_label.setText("🔄 Testing connection...")
            self.status_label.setStyleSheet("color: #2196f3; padding: 5px;")
            QApplication.processEvents()
            
            import serial
            ser = serial.Serial(
                port=port,
                baudrate=int(self.baud_combo.currentText()),
                bytesize=int(self.databits_combo.currentText()),
                parity=self._get_parity(),
                stopbits=float(self.stopbits_combo.currentText()),
                timeout=self.timeout_spin.value()
            )
            ser.close()
            
            self.status_label.setText(f"✓ Connection test successful!")
            self.status_label.setStyleSheet("color: #4caf50; padding: 5px;")
            logger.info(f"Test connection successful: {port}")
            
        except Exception as e:
            self.status_label.setText(f"❌ Connection failed: {str(e)}")
            self.status_label.setStyleSheet("color: #f44336; padding: 5px;")
            logger.error(f"Test connection failed: {str(e)}")
    
    def _get_parity(self):
        """Get parity character"""
        parity_map = {
            'None': 'N',
            'Even': 'E',
            'Odd': 'O',
            'Mark': 'M',
            'Space': 'S'
        }
        return parity_map.get(self.parity_combo.currentText(), 'N')
    
    def accept_connection(self):
        """Accept and validate connection"""
        port = self.port_combo.currentData()
        if not port:
            QMessageBox.warning(self, 'No Port', 'Please select a serial port')
            return
        
        self.port = port
        self.baudrate = int(self.baud_combo.currentText())
        self.bytesize = int(self.databits_combo.currentText())
        self.parity = self._get_parity()
        self.stopbits = float(self.stopbits_combo.currentText())
        self.timeout = self.timeout_spin.value()
        
        logger.info(f"Accepting connection: {port} @ {self.baudrate} baud")
        self.accept()


# ============================================================================
# Track Configuration Dialog
# ============================================================================

class TrackConfigDialog(QDialog):
    """Dialog for configuring track parameters"""

    def __init__(self, data: Optional[np.ndarray], field_names: List[str], parent=None):
        super().__init__(parent)
        self.data = data
        self.field_names = field_names
        self.track = None

        self.setWindowTitle("Track Configuration")
        self.setModal(True)
        self.resize(600, 400)

        # Set dialog background to dark
        self.setStyleSheet("""
            QDialog {
                background-color: #2A2A2A;
                color: #DCDCDC;
            }
            QLabel {
                color: #DCDCDC;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Form for track parameters
        form_layout = QFormLayout()
        
        self.name_edit = QLineEdit("Unknown Track")
        self.length_spin = QDoubleSpinBox()
        self.length_spin.setRange(0.1, 50.0)
        self.length_spin.setValue(3.0)
        self.length_spin.setSuffix(" km")
        self.length_spin.setDecimals(3)
        
        self.sf_lat_spin = QDoubleSpinBox()
        self.sf_lat_spin.setRange(-90, 90)
        self.sf_lat_spin.setDecimals(6)
        
        self.sf_lon_spin = QDoubleSpinBox()
        self.sf_lon_spin.setRange(-180, 180)
        self.sf_lon_spin.setDecimals(6)
        
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(10, 500)
        self.tolerance_spin.setValue(50)
        self.tolerance_spin.setSuffix(" m")
        
        form_layout.addRow("Track Name:", self.name_edit)
        form_layout.addRow("Track Length:", self.length_spin)
        form_layout.addRow("Start/Finish Latitude:", self.sf_lat_spin)
        form_layout.addRow("Start/Finish Longitude:", self.sf_lon_spin)
        form_layout.addRow("Distance Tolerance:", self.tolerance_spin)
        
        layout.addLayout(form_layout)
        
        # Auto-detect button
        if data is not None:
            auto_btn = QPushButton("Auto-Detect from GPS Data")
            auto_btn.clicked.connect(self.auto_detect)
            layout.addWidget(auto_btn)
            
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
        # Auto-detect on startup
        if data is not None:
            self.auto_detect()
            
    def auto_detect(self):
        """Auto-detect track parameters from GPS data"""
        if self.data is None:
            return
            
        try:
            # Find GPS columns
            lat_idx = self.field_names.index('latitude') + 1
            lon_idx = self.field_names.index('longitude') + 1
            
            # Use first point as start/finish
            self.sf_lat_spin.setValue(float(self.data[0, lat_idx]))
            self.sf_lon_spin.setValue(float(self.data[0, lon_idx]))
            
            # Calculate approximate track length from total distance
            if 'distance_km' in self.field_names:
                dist_idx = self.field_names.index('distance_km') + 1
                max_dist = float(np.max(self.data[:, dist_idx]))
                self.length_spin.setValue(max_dist)
                
        except (ValueError, IndexError):
            pass
            
    def accept(self):
        """Create track definition"""
        self.track = TrackDefinition(
            name=self.name_edit.text(),
            length_km=self.length_spin.value(),
            start_finish_lat=self.sf_lat_spin.value(),
            start_finish_lon=self.sf_lon_spin.value(),
            tolerance_m=self.tolerance_spin.value()
        )
        super().accept()


# ============================================================================
# Serial Connection Dialog
# ============================================================================


# ============================================================================
# Serial Configuration Dialog
# ============================================================================

class TelemetryPlotWidget(pg.PlotWidget):
    """Enhanced plot widget for telemetry data"""
    
    def __init__(self, title: str = "", ylabel: str = ""):
        super().__init__()
        
        self.setBackground('k')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setLabel('left', ylabel)
        self.setLabel('bottom', 'Time', units='s')
        self.setTitle(title, color='w', size='12pt')
        
        # Configure appearance
        self.getAxis('bottom').setPen(pg.mkPen(color='w', width=1))
        self.getAxis('left').setPen(pg.mkPen(color='w', width=1))
        
        self.plots = {}
        self.data_sources = {}
        
    def add_channel(self, name: str, color: str = 'y', width: int = 2):
        """Add a data channel to the plot"""
        pen = pg.mkPen(color=color, width=width)
        plot = self.plot([], [], pen=pen, name=name)
        self.plots[name] = plot
        self.data_sources[name] = ([], [])
        
    def update_channel(self, name: str, x: np.ndarray, y: np.ndarray):
        """Update channel data"""
        if name in self.plots:
            self.plots[name].setData(x, y)
            self.data_sources[name] = (x, y)
            
    def clear_all(self):
        """Clear all data"""
        for name in self.plots:
            self.plots[name].setData([], [])
            self.data_sources[name] = ([], [])


class GGDiagram(pg.PlotWidget):
    """GG Diagram for lateral/longitudinal g-force visualization"""
    
    def __init__(self):
        super().__init__()
        
        self.setBackground('k')
        self.setAspectLocked(True)
        self.setLabel('left', 'Lateral G', units='g')
        self.setLabel('bottom', 'Longitudinal G', units='g')
        self.setTitle('GG Diagram', color='w', size='12pt')
        
        # Draw reference circles
        for g in [0.5, 1.0, 1.5, 2.0]:
            circle = pg.QtWidgets.QGraphicsEllipseItem(-g, -g, 2*g, 2*g)
            circle.setPen(pg.mkPen(color=(80, 80, 80), width=1))
            self.addItem(circle)
            
        self.scatter = pg.ScatterPlotItem(size=3, pen=None)
        self.addItem(self.scatter)
        
    def update_data(self, g_long: np.ndarray, g_lat: np.ndarray, colors: np.ndarray = None):
        """Update GG diagram data"""
        if colors is not None:
            brushes = [pg.mkBrush(c) for c in colors]
            self.scatter.setData(g_long, g_lat, brush=brushes)
        else:
            self.scatter.setData(g_long, g_lat, brush=pg.mkBrush('y'))


class TrackMapWidget(pg.PlotWidget):
    """Track map visualization"""
    
    def __init__(self):
        super().__init__()
        
        self.setBackground('k')
        self.setAspectLocked(True)
        self.setLabel('left', 'Latitude')
        self.setLabel('bottom', 'Longitude')
        self.setTitle('Track Map', color='w', size='12pt')
        
        self.track_plot = self.plot([], [], pen=None)
        self.scatter = pg.ScatterPlotItem(size=5, pen=None)
        self.addItem(self.scatter)
        
    def update_map(self, lat: np.ndarray, lon: np.ndarray, colors: np.ndarray):
        """Update track map with colored points"""
        brushes = [pg.mkBrush(c) for c in colors]
        self.scatter.setData(lon, lat, brush=brushes)


# ============================================================================
# Debug/Developer Widget
# ============================================================================

class LogHandler(logging.Handler):
    """Custom logging handler that emits to Qt signal"""
    
    def __init__(self):
        super().__init__()
        self.log_signal = None
        
    def emit(self, record):
        """Emit log record"""
        if self.log_signal:
            msg = self.format(record)
            self.log_signal.emit(msg, record.levelname)


class DebugWidget(QWidget):
    """Developer/Debug mode widget for inspecting data and system state"""
    
    log_received = pyqtSignal(str, str)  # message, level
    
    def __init__(self):
        super().__init__()
        
        self.current_frame = None
        self.performance_metrics = {}
        self.frame_count = 0
        self.start_time = time.time()
        
        self.init_ui()
        self.setup_logging()
        
    def init_ui(self):
        """Initialize debug UI"""
        layout = QVBoxLayout()
        
        # Create tab widget for different debug views
        self.debug_tabs = QTabWidget()
        
        # Tab 1: Real-time Frame Inspector
        self.create_frame_inspector_tab()
        
        # Tab 2: Data Structure Viewer
        self.create_data_structure_tab()
        
        # Tab 3: Performance Monitor
        self.create_performance_tab()
        
        # Tab 4: Log Viewer
        self.create_log_viewer_tab()
        
        # Tab 5: System Info
        self.create_system_info_tab()
        
        layout.addWidget(self.debug_tabs)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.clear_log_btn = QPushButton("Clear Logs")
        self.clear_log_btn.clicked.connect(self.clear_logs)
        button_layout.addWidget(self.clear_log_btn)
        
        self.copy_data_btn = QPushButton("Copy Current Frame")
        self.copy_data_btn.clicked.connect(self.copy_current_frame)
        button_layout.addWidget(self.copy_data_btn)
        
        self.export_debug_btn = QPushButton("Export Debug Info")
        self.export_debug_btn.clicked.connect(self.export_debug_info)
        button_layout.addWidget(self.export_debug_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def create_frame_inspector_tab(self):
        """Create real-time frame inspector tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Frame info
        info_group = QGroupBox("Current Frame Info")
        info_layout = QFormLayout()
        
        self.frame_count_label = QLabel("0")
        self.frame_count_label.setStyleSheet("color: white;")
        self.frame_timestamp_label = QLabel("N/A")
        self.frame_timestamp_label.setStyleSheet("color: white;")
        self.frame_rate_label = QLabel("0.0 Hz")
        self.frame_rate_label.setStyleSheet("color: white;")
        
        info_layout.addRow(self.create_white_label("Frame Count:"), self.frame_count_label)
        info_layout.addRow(self.create_white_label("Timestamp:"), self.frame_timestamp_label)
        info_layout.addRow(self.create_white_label("Frame Rate:"), self.frame_rate_label)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Frame data display (JSON format)
        data_group = QGroupBox("Frame Data (JSON)")
        data_layout = QVBoxLayout()
        
        self.frame_data_text = QTextEdit()
        self.frame_data_text.setReadOnly(True)
        self.frame_data_text.setFont(QFont("Courier New", 9))
        self.frame_data_text.setStyleSheet("background-color: #1e1e1e; color: #00ff00;")
        
        data_layout.addWidget(self.frame_data_text)
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        tab.setLayout(layout)
        self.debug_tabs.addTab(tab, "Frame Inspector")
        
    def create_data_structure_tab(self):
        """Create data structure viewer tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Field list
        fields_group = QGroupBox("Telemetry Fields")
        fields_layout = QVBoxLayout()
        
        self.fields_tree = QTreeWidget()
        self.fields_tree.setHeaderLabels(["Field Name", "Type", "Min", "Max", "Mean", "Std"])
        self.fields_tree.setStyleSheet("background-color: #2d2d2d; color: white;")
        
        fields_layout.addWidget(self.fields_tree)
        fields_group.setLayout(fields_layout)
        layout.addWidget(fields_group)
        
        # Array info
        array_group = QGroupBox("Data Array Info")
        array_layout = QFormLayout()
        
        self.array_shape_label = QLabel("N/A")
        self.array_shape_label.setStyleSheet("color: white;")
        self.array_size_label = QLabel("N/A")
        self.array_size_label.setStyleSheet("color: white;")
        self.array_memory_label = QLabel("N/A")
        self.array_memory_label.setStyleSheet("color: white;")
        
        array_layout.addRow(self.create_white_label("Shape:"), self.array_shape_label)
        array_layout.addRow(self.create_white_label("Size:"), self.array_size_label)
        array_layout.addRow(self.create_white_label("Memory:"), self.array_memory_label)
        
        array_group.setLayout(array_layout)
        layout.addWidget(array_group)
        
        tab.setLayout(layout)
        self.debug_tabs.addTab(tab, "Data Structure")
        
    def create_performance_tab(self):
        """Create performance monitor tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Performance metrics
        perf_group = QGroupBox("Performance Metrics")
        perf_layout = QFormLayout()
        
        self.uptime_label = QLabel("0.0 s")
        self.uptime_label.setStyleSheet("color: white;")
        self.avg_frame_rate_label = QLabel("0.0 Hz")
        self.avg_frame_rate_label.setStyleSheet("color: white;")
        self.buffer_size_label = QLabel("0")
        self.buffer_size_label.setStyleSheet("color: white;")
        self.update_time_label = QLabel("0.0 ms")
        self.update_time_label.setStyleSheet("color: white;")
        
        perf_layout.addRow(self.create_white_label("Uptime:"), self.uptime_label)
        perf_layout.addRow(self.create_white_label("Avg Frame Rate:"), self.avg_frame_rate_label)
        perf_layout.addRow(self.create_white_label("Buffer Size:"), self.buffer_size_label)
        perf_layout.addRow(self.create_white_label("Update Time:"), self.update_time_label)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        # Memory usage
        memory_group = QGroupBox("Memory Usage")
        memory_layout = QFormLayout()
        
        self.data_memory_label = QLabel("0 MB")
        self.data_memory_label.setStyleSheet("color: white;")
        self.buffer_memory_label = QLabel("0 MB")
        self.buffer_memory_label.setStyleSheet("color: white;")
        
        memory_layout.addRow(self.create_white_label("Data Array:"), self.data_memory_label)
        memory_layout.addRow(self.create_white_label("Buffer:"), self.buffer_memory_label)
        
        memory_group.setLayout(memory_layout)
        layout.addWidget(memory_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.debug_tabs.addTab(tab, "Performance")
        
    def create_log_viewer_tab(self):
        """Create log viewer tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Log level filter
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(self.create_white_label("Log Level:"))
        
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
        self.log_level_combo.setCurrentText('INFO')
        self.log_level_combo.currentTextChanged.connect(self.change_log_level)
        filter_layout.addWidget(self.log_level_combo)
        
        filter_layout.addStretch()
        layout.addLayout(filter_layout)
        
        # Log display
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier New", 9))
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #cccccc;")
        
        layout.addWidget(self.log_text)
        
        tab.setLayout(layout)
        self.debug_tabs.addTab(tab, "Logs")
        
    def create_system_info_tab(self):
        """Create system information tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # System info
        sys_group = QGroupBox("System Information")
        sys_layout = QFormLayout()
        
        # Python info
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.python_label = QLabel(python_version)
        self.python_label.setStyleSheet("color: white;")
        
        # Qt version
        from PyQt5.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
        self.qt_label = QLabel(QT_VERSION_STR)
        self.qt_label.setStyleSheet("color: white;")
        self.pyqt_label = QLabel(PYQT_VERSION_STR)
        self.pyqt_label.setStyleSheet("color: white;")
        
        # NumPy version
        self.numpy_label = QLabel(np.__version__)
        self.numpy_label.setStyleSheet("color: white;")
        
        sys_layout.addRow(self.create_white_label("Python:"), self.python_label)
        sys_layout.addRow(self.create_white_label("Qt:"), self.qt_label)
        sys_layout.addRow(self.create_white_label("PyQt5:"), self.pyqt_label)
        sys_layout.addRow(self.create_white_label("NumPy:"), self.numpy_label)
        
        sys_group.setLayout(sys_layout)
        layout.addWidget(sys_group)
        
        # Paths info
        paths_group = QGroupBox("Paths")
        paths_layout = QVBoxLayout()
        
        self.paths_text = QTextEdit()
        self.paths_text.setReadOnly(True)
        self.paths_text.setFont(QFont("Courier New", 8))
        self.paths_text.setMaximumHeight(150)
        self.paths_text.setStyleSheet("background-color: #2d2d2d; color: white;")
        
        paths_info = f"""Working Directory: {os.getcwd()}
User Home: {Path.home()}
App Data: {QSettings().fileName()}"""
        self.paths_text.setPlainText(paths_info)
        
        paths_layout.addWidget(self.paths_text)
        paths_group.setLayout(paths_layout)
        layout.addWidget(paths_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.debug_tabs.addTab(tab, "System Info")
        
    def create_white_label(self, text: str) -> QLabel:
        """Create a label with white text for dark mode"""
        label = QLabel(text)
        label.setStyleSheet("color: white;")
        return label
        
    def setup_logging(self):
        """Setup logging handler"""
        # Create custom handler
        self.log_handler = LogHandler()
        self.log_handler.log_signal = self.log_received
        self.log_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        
        # Add to logger
        logger.addHandler(self.log_handler)
        
        # Connect signal
        self.log_received.connect(self.append_log)
        
    def change_log_level(self, level_name: str):
        """Change logging level"""
        level = getattr(logging, level_name)
        self.log_handler.setLevel(level)
        logger.info(f"Log level changed to {level_name}")
        
    def update_frame(self, frame: TelemetryFrame):
        """Update with new telemetry frame"""
        self.current_frame = frame
        self.frame_count += 1
        
        # Update frame inspector
        self.frame_count_label.setText(str(self.frame_count))
        self.frame_timestamp_label.setText(f"{frame.timestamp:.3f}")
        
        # Calculate frame rate
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            rate = self.frame_count / elapsed
            self.frame_rate_label.setText(f"{rate:.2f} Hz")
            self.avg_frame_rate_label.setText(f"{rate:.2f} Hz")
        
        # Format frame data as JSON
        frame_dict = {
            "timestamp": frame.timestamp,
            "data": frame.data
        }
        json_str = json.dumps(frame_dict, indent=2, sort_keys=True)
        self.frame_data_text.setPlainText(json_str)
        
    def update_data_structure(self, field_names: List[str], data: np.ndarray):
        """Update data structure information"""
        if data is None or len(data) == 0:
            return
            
        # Update array info
        self.array_shape_label.setText(str(data.shape))
        self.array_size_label.setText(f"{data.size:,}")
        memory_mb = data.nbytes / (1024 * 1024)
        self.array_memory_label.setText(f"{memory_mb:.2f} MB")
        self.data_memory_label.setText(f"{memory_mb:.2f} MB")
        
        # Update field tree
        self.fields_tree.clear()
        
        for i, field_name in enumerate(field_names):
            col_idx = i + 1  # Skip timestamp column
            if col_idx < data.shape[1]:
                col_data = data[:, col_idx]
                
                item = QTreeWidgetItem([
                    field_name,
                    str(col_data.dtype),
                    f"{col_data.min():.3f}",
                    f"{col_data.max():.3f}",
                    f"{col_data.mean():.3f}",
                    f"{col_data.std():.3f}"
                ])
                self.fields_tree.addTopLevelItem(item)
                
        # Expand all
        self.fields_tree.expandAll()
        
    def update_performance(self, metrics: Dict):
        """Update performance metrics"""
        self.performance_metrics.update(metrics)
        
        # Update uptime
        uptime = time.time() - self.start_time
        self.uptime_label.setText(f"{uptime:.1f} s")
        
        # Update metrics
        if 'buffer_size' in metrics:
            self.buffer_size_label.setText(str(metrics['buffer_size']))
            
        if 'update_time_ms' in metrics:
            self.update_time_label.setText(f"{metrics['update_time_ms']:.2f} ms")
            
        if 'buffer_memory_mb' in metrics:
            self.buffer_memory_label.setText(f"{metrics['buffer_memory_mb']:.2f} MB")
            
    def append_log(self, message: str, level: str):
        """Append log message with color coding"""
        # Color code by level
        color_map = {
            'DEBUG': '#888888',
            'INFO': '#00ff00',
            'WARNING': '#ffff00',
            'ERROR': '#ff0000',
            'CRITICAL': '#ff00ff'
        }
        
        color = color_map.get(level, '#cccccc')
        formatted = f'<span style="color: {color};">{message}</span>'
        
        self.log_text.append(formatted)
        
        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
        
    def clear_logs(self):
        """Clear log display"""
        self.log_text.clear()
        logger.info("Logs cleared")
        
    def copy_current_frame(self):
        """Copy current frame to clipboard"""
        if self.current_frame:
            frame_dict = {
                "timestamp": self.current_frame.timestamp,
                "data": self.current_frame.data
            }
            json_str = json.dumps(frame_dict, indent=2)
            
            clipboard = QApplication.clipboard()
            clipboard.setText(json_str)
            
            logger.info("Current frame copied to clipboard")
        else:
            logger.warning("No frame data available to copy")
            
    def export_debug_info(self):
        """Export all debug information to file"""
        try:
            filepath, _ = QFileDialog.getSaveFileName(
                self, 'Export Debug Info', '', 'JSON Files (*.json)')
                
            if filepath:
                debug_info = {
                    'timestamp': datetime.now().isoformat(),
                    'frame_count': self.frame_count,
                    'uptime': time.time() - self.start_time,
                    'performance_metrics': self.performance_metrics,
                    'current_frame': {
                        'timestamp': self.current_frame.timestamp if self.current_frame else None,
                        'data': self.current_frame.data if self.current_frame else None
                    }
                }
                
                with open(filepath, 'w') as f:
                    json.dump(debug_info, f, indent=2)
                    
                logger.info(f"Debug info exported to {filepath}")
                
        except Exception as e:
            logger.error(f"Failed to export debug info: {str(e)}")
            QMessageBox.critical(self, 'Export Error', str(e))


# ============================================================================
# Main Analysis Window
# ============================================================================

class TelemetryAnalyzer(QMainWindow):
    """Main telemetry analysis application"""
    
    def __init__(self):
        super().__init__()
        
        # Application state
        self.current_data = None
        self.field_names = []
        self.metadata = {}
        self.current_track = None
        self.laps = []
        self.recording = False
        self.receiver = None
        self.logger = None
        self.live_buffer = deque(maxlen=1000)
        
        # Debug mode
        self.debug_mode = False
        self.debug_widget = None
        self.frame_count = 0
        self.last_frame = None
        self.last_update_time = time.time()
        self.performance_stats = {
            'frames_received': 0,
            'frames_processed': 0,
            'update_time_ms': 0,
            'memory_mb': 0
        }
        
        # Settings
        self.settings = QSettings('MotorsportTech', 'TelemetryAnalyzer')
        
        # Initialize UI
        self.init_ui()
        
        # Timer for live updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_live_display)
        
        # Load previous session if exists
        self.load_settings()
        
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("Professional Telemetry Analyzer")
        self.setGeometry(100, 100, 1600, 900)
        
        # Apply dark theme
        self.apply_dark_theme()
        
        # Create menu bar
        self.create_menus()
        
        # Create main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Create tab widget for different analysis groups
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create analysis tabs
        self.create_primary_view()
        self.create_dashboard_view()
        self.create_mixture_view()
        self.create_histogram_view()
        self.create_suspension_view()
        self.create_track_view()
        
        # Create control panel
        self.create_control_panel()
        main_layout.addWidget(self.control_panel)
        
        # Create lap list dock
        self.create_lap_dock()
        
        # Create debug dock (hidden by default)
        self.create_debug_dock()
        
        # Status bar
        self.statusBar().showMessage('Ready')
        
    def apply_dark_theme(self):
        """Apply professional dark theme with proper contrast"""
        palette = QPalette()
        
        # Dark backgrounds
        palette.setColor(QPalette.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.WindowText, QColor(220, 220, 220))  # Light gray text
        palette.setColor(QPalette.Base, QColor(40, 40, 40))
        palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
        palette.setColor(QPalette.Text, QColor(220, 220, 220))  # Light gray text
        palette.setColor(QPalette.Button, QColor(50, 50, 50))
        palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))  # Light gray text
        palette.setColor(QPalette.BrightText, Qt.white)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        
        # Disabled state
        palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(120, 120, 120))
        palette.setColor(QPalette.Disabled, QPalette.Text, QColor(120, 120, 120))
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120, 120, 120))
        
        self.setPalette(palette)
        QApplication.instance().setPalette(palette)
        
        # Additional stylesheet for better contrast and modern look
        self.setStyleSheet("""
            QLabel {
                color: #DCDCDC;
            }
            QGroupBox {
                color: #DCDCDC;
                border: 1px solid #555555;
                border-radius: 3px;
                margin-top: 7px;
                padding-top: 5px;
            }
            QGroupBox::title {
                color: #42A5DA;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QMenuBar {
                background-color: #2A2A2A;
                color: #DCDCDC;
                padding: 2px;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QMenuBar::item:selected {
                background-color: #3daee9;
                color: white;
            }
            QMenuBar::item:pressed {
                background-color: #2b8cbd;
            }
            QMenu {
                background-color: #2A2A2A;
                color: #DCDCDC;
                border: 1px solid #3daee9;
            }
            QMenu::item {
                padding: 6px 25px;
            }
            QMenu::item:selected {
                background-color: #3daee9;
                color: white;
            }
            QPushButton {
                background-color: #3daee9;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4fc3f7;
            }
            QPushButton:pressed {
                background-color: #2b8cbd;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            QComboBox {
                background-color: #2d2d2d;
                color: #DCDCDC;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 4px;
                min-height: 20px;
            }
            QComboBox:hover {
                border: 1px solid #3daee9;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #DCDCDC;
            }
            QComboBox QAbstractItemView {
                background-color: #2A2A2A;
                color: #DCDCDC;
                selection-background-color: #3daee9;
                border: 1px solid #3daee9;
            }
            QStatusBar {
                background-color: #2A2A2A;
                color: #DCDCDC;
            }
            QDockWidget {
                color: #DCDCDC;
                titlebar-close-icon: url(close.png);
                titlebar-normal-icon: url(float.png);
            }
            QDockWidget::title {
                background-color: #3A3A3A;
                text-align: center;
                padding: 5px;
            }
            QTableWidget {
                background-color: #282828;
                alternate-background-color: #2E2E2E;
                color: #DCDCDC;
                gridline-color: #3A3A3A;
            }
            QTableWidget::item:selected {
                background-color: #42A5DA;
            }
            QHeaderView::section {
                background-color: #3A3A3A;
                color: #DCDCDC;
                padding: 5px;
                border: 1px solid #555555;
            }
            QTextEdit, QPlainTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: 1px solid #3A3A3A;
            }
            QSlider::groove:horizontal {
                background: #3A3A3A;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #3daee9;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #4fc3f7;
            }
        """)
        
    def create_menus(self):
        """Create application menus"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        open_action = QAction('&Open Log...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_log)
        file_menu.addAction(open_action)
        
        import_action = QAction('&Import FuelTech CSV...', self)
        import_action.setShortcut('Ctrl+I')
        import_action.triggered.connect(self.import_fueltech)
        file_menu.addAction(import_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Session menu
        session_menu = menubar.addMenu('&Session')
        
        connect_action = QAction('&Connect to Telemetry...', self)
        connect_action.triggered.connect(self.connect_telemetry)
        session_menu.addAction(connect_action)
        
        start_rec_action = QAction('&Start Recording', self)
        start_rec_action.triggered.connect(self.start_recording)
        session_menu.addAction(start_rec_action)
        
        stop_rec_action = QAction('S&top Recording', self)
        stop_rec_action.triggered.connect(self.stop_recording)
        session_menu.addAction(stop_rec_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu('&Analysis')
        
        config_track_action = QAction('Configure &Track...', self)
        config_track_action.triggered.connect(self.configure_track)
        analysis_menu.addAction(config_track_action)
        
        segment_laps_action = QAction('&Segment Laps', self)
        segment_laps_action.triggered.connect(self.segment_laps)
        analysis_menu.addAction(segment_laps_action)
        
        # Debug menu
        debug_menu = menubar.addMenu('&Debug')
        
        toggle_debug_action = QAction('&Enable Debug Mode', self)
        toggle_debug_action.setCheckable(True)
        toggle_debug_action.setChecked(False)
        toggle_debug_action.triggered.connect(self.toggle_debug_mode)
        debug_menu.addAction(toggle_debug_action)
        self.toggle_debug_action = toggle_debug_action
        
        debug_menu.addSeparator()
        
        show_frame_data_action = QAction('Show Raw &Frame Data', self)
        show_frame_data_action.triggered.connect(self.show_frame_data)
        debug_menu.addAction(show_frame_data_action)
        
        show_stats_action = QAction('Show &Performance Stats', self)
        show_stats_action.triggered.connect(self.show_performance_stats)
        debug_menu.addAction(show_stats_action)
        
        show_structure_action = QAction('Show Data &Structure', self)
        show_structure_action.triggered.connect(self.show_data_structure)
        debug_menu.addAction(show_structure_action)
        
        debug_menu.addSeparator()
        
        clear_buffer_action = QAction('Clear Live &Buffer', self)
        clear_buffer_action.triggered.connect(self.clear_live_buffer)
        debug_menu.addAction(clear_buffer_action)
        
    def create_primary_view(self):
        """Create primary 4-graph view"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Create 4 stacked plots
        self.rpm_plot = TelemetryPlotWidget("Engine RPM", "RPM")
        self.rpm_plot.add_channel('rpm', 'r', 2)
        self.rpm_plot.setYRange(0, 8000)

        self.speed_plot = TelemetryPlotWidget("Vehicle Speed", "km/h")
        self.speed_plot.add_channel('speed', 'c', 2)
        self.speed_plot.add_channel('gear', 'y', 1)
        self.speed_plot.setYRange(0, 300)

        self.tps_plot = TelemetryPlotWidget("Throttle Position", "%")
        self.tps_plot.add_channel('tps', 'g', 2)
        self.tps_plot.setYRange(0, 100)

        self.g_plot = TelemetryPlotWidget("G-Forces", "g")
        self.g_plot.add_channel('g_accel', 'r', 2)
        self.g_plot.add_channel('g_lateral', 'b', 2)
        self.g_plot.setYRange(-3, 3)
        
        # Add plots to layout
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.rpm_plot)
        splitter.addWidget(self.speed_plot)
        splitter.addWidget(self.tps_plot)
        splitter.addWidget(self.g_plot)
        
        layout.addWidget(splitter)
        
        # Add time slider
        slider_layout = QHBoxLayout()
        time_window_label = QLabel("Time Window:")
        time_window_label.setStyleSheet("color: white;")
        slider_layout.addWidget(time_window_label)
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 100)
        self.time_slider.setValue(100)
        self.time_slider.valueChanged.connect(self.update_time_window)
        slider_layout.addWidget(self.time_slider)
        
        layout.addLayout(slider_layout)
        
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "Primary View")
        
    def create_dashboard_view(self):
        """Create gauge dashboard view"""
        tab = QWidget()
        layout = QHBoxLayout()
        
        # RPM gauge (using plot widget as circular gauge alternative)
        rpm_gauge = TelemetryPlotWidget("RPM", "RPM")
        rpm_gauge.add_channel('rpm', 'r', 3)
        rpm_gauge.setYRange(0, 8000)
        
        # Speed gauge
        speed_gauge = TelemetryPlotWidget("Speed", "km/h")
        speed_gauge.add_channel('speed', 'c', 3)
        speed_gauge.setYRange(0, 300)
        
        # GG diagram
        self.gg_diagram = GGDiagram()
        
        layout.addWidget(rpm_gauge)
        layout.addWidget(speed_gauge)
        layout.addWidget(self.gg_diagram)
        
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "Dashboard")
        
    def create_mixture_view(self):
        """Create mixture tuning analysis view"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Create plots
        splitter = QSplitter(Qt.Vertical)

        self.rpm_map_plot = TelemetryPlotWidget("RPM vs MAP", "MAP (kPa)")
        self.rpm_map_plot.add_channel('map', 'y', 1)
        self.rpm_map_plot.setYRange(0, 200)

        self.o2_lambda_plot = TelemetryPlotWidget("O2 vs Lambda Correction", "%")
        self.o2_lambda_plot.add_channel('exhaust_o2', 'g', 2)
        self.o2_lambda_plot.add_channel('lambda_correction', 'r', 2)
        self.o2_lambda_plot.setYRange(-20, 20)

        self.inj_time_plot = TelemetryPlotWidget("Injection Time", "ms")
        self.inj_time_plot.add_channel('inj_time_bank_a', 'c', 2)
        self.inj_time_plot.add_channel('inj_time_bank_b', 'm', 2)
        self.inj_time_plot.setYRange(0, 20)

        splitter.addWidget(self.rpm_map_plot)
        splitter.addWidget(self.o2_lambda_plot)
        splitter.addWidget(self.inj_time_plot)

        layout.addWidget(splitter)
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "Mixture Tuning")
        
    def create_histogram_view(self):
        """Create histogram analysis view"""
        tab = QWidget()
        layout = QHBoxLayout()
        
        # RPM histogram
        rpm_hist = pg.PlotWidget()
        rpm_hist.setBackground('k')
        rpm_hist.setTitle("RPM Distribution", color='w')
        rpm_hist.setLabel('left', 'Count')
        rpm_hist.setLabel('bottom', 'RPM')
        self.rpm_hist_plot = rpm_hist
        
        # TPS histogram
        tps_hist = pg.PlotWidget()
        tps_hist.setBackground('k')
        tps_hist.setTitle("Throttle Position Distribution", color='w')
        tps_hist.setLabel('left', 'Count')
        tps_hist.setLabel('bottom', 'TPS (%)')
        self.tps_hist_plot = tps_hist
        
        layout.addWidget(rpm_hist)
        layout.addWidget(tps_hist)
        
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "Histograms")
        
    def create_suspension_view(self):
        """Create suspension analysis view"""
        tab = QWidget()
        layout = QVBoxLayout()

        splitter = QSplitter(Qt.Vertical)

        self.shock_plot = TelemetryPlotWidget("Shock Travel", "mm")
        self.shock_plot.add_channel('shock_fl', 'r', 2)
        self.shock_plot.add_channel('shock_fr', 'g', 2)
        self.shock_plot.add_channel('shock_rl', 'b', 2)
        self.shock_plot.add_channel('shock_rr', 'y', 2)
        self.shock_plot.setYRange(-50, 50)

        self.wheel_speed_plot = TelemetryPlotWidget("Wheel Speeds", "km/h")
        self.wheel_speed_plot.add_channel('wheel_speed_fl', 'r', 2)
        self.wheel_speed_plot.add_channel('wheel_speed_fr', 'g', 2)
        self.wheel_speed_plot.add_channel('wheel_speed_rl', 'b', 2)
        self.wheel_speed_plot.add_channel('wheel_speed_rr', 'y', 2)
        self.wheel_speed_plot.setYRange(0, 300)

        splitter.addWidget(self.shock_plot)
        splitter.addWidget(self.wheel_speed_plot)

        layout.addWidget(splitter)
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "Suspension")
        
    def create_track_view(self):
        """Create track map view"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        self.track_map = TrackMapWidget()
        layout.addWidget(self.track_map)
        
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "Track Map")
        
    def create_control_panel(self):
        """Create playback control panel"""
        self.control_panel = QWidget()
        layout = QHBoxLayout()
        
        # Playback controls
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_playback)
        layout.addWidget(self.stop_btn)
        
        # Position slider
        position_label = QLabel("Position:")
        position_label.setStyleSheet("color: white;")
        layout.addWidget(position_label)
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.valueChanged.connect(self.seek_position)
        layout.addWidget(self.position_slider)
        
        # Time display
        self.time_label = QLabel("00:00.000")
        self.time_label.setStyleSheet("color: white;")
        layout.addWidget(self.time_label)
        
        # Playback speed
        speed_label = QLabel("Speed:")
        speed_label.setStyleSheet("color: white;")
        layout.addWidget(speed_label)
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(['0.25x', '0.5x', '1x', '2x', '4x'])
        self.speed_combo.setCurrentText('1x')
        layout.addWidget(self.speed_combo)
        
        layout.addStretch()
        self.control_panel.setLayout(layout)
        
    def create_lap_dock(self):
        """Create lap list dock widget"""
        dock = QDockWidget("Laps", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        self.lap_table = QTableWidget()
        self.lap_table.setColumnCount(3)
        self.lap_table.setHorizontalHeaderLabels(['Lap', 'Time', 'Delta'])
        self.lap_table.horizontalHeader().setStretchLastSection(True)
        self.lap_table.itemSelectionChanged.connect(self.on_lap_selected)
        
        dock.setWidget(self.lap_table)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        
    def create_debug_dock(self):
        """Create comprehensive debug/developer console dock widget"""
        self.debug_dock = QDockWidget("Developer Console", self)
        self.debug_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)
        
        # Create comprehensive debug widget
        self.debug_widget = DebugWidget()
        self.debug_dock.setWidget(self.debug_widget)
        
        # Initially hidden
        self.debug_dock.hide()
        
        # Add to main window
        self.addDockWidget(Qt.BottomDockWidgetArea, self.debug_dock)
        
        logger.info("Debug console created")
        
    def connect_telemetry(self):
        """Connect to real-time telemetry with modern configuration dialog"""
        dialog = SerialConfigDialog(self)
        
        if dialog.exec_() == QDialog.Accepted:
            if dialog.port:
                try:
                    self.receiver = TelemetryReceiver(
                        port=dialog.port,
                        baudrate=dialog.baudrate
                    )
                    self.receiver.frame_received.connect(self.on_telemetry_frame)
                    self.receiver.connection_status.connect(self.on_connection_status)
                    self.receiver.start()
                    
                    self.statusBar().showMessage(f'Connecting to {dialog.port} @ {dialog.baudrate} baud...')
                    logger.info(f"Connecting to {dialog.port} @ {dialog.baudrate} baud")
                    
                except Exception as e:
                    logger.error(f"Failed to connect: {str(e)}")
                    QMessageBox.critical(self, 'Connection Error', f'Failed to connect:\n{str(e)}')
            
    def on_connection_status(self, connected: bool, message: str):
        """Handle connection status change"""
        self.statusBar().showMessage(message)
        if connected:
            self.update_timer.start(100)  # Update display at 10 Hz
    
    # Note: on_telemetry_frame is now defined in the debug section with enhanced logging
            
    def update_live_display(self):
        """Update live display with buffered data"""
        if not self.live_buffer:
            return
            
        # Convert buffer to arrays
        timestamps = np.array([f.timestamp for f in self.live_buffer])
        
        # Update primary plots
        if len(timestamps) > 1:
            # RPM
            rpm_data = np.array([f.data.get('rpm', 0) for f in self.live_buffer])
            self.rpm_plot.update_channel('rpm', timestamps, rpm_data)
            
            # Speed
            speed_data = np.array([f.data.get('gps_speed_knots', 0) * 1.852 
                                  for f in self.live_buffer])
            self.speed_plot.update_channel('speed', timestamps, speed_data)
            
            # TPS
            tps_data = np.array([f.data.get('tps', 0) for f in self.live_buffer])
            self.tps_plot.update_channel('tps', timestamps, tps_data)
            
            # G-forces
            g_accel = np.array([f.data.get('g_accel', 0) for f in self.live_buffer])
            g_lat = np.array([f.data.get('g_lateral', 0) for f in self.live_buffer])
            self.g_plot.update_channel('g_accel', timestamps, g_accel)
            self.g_plot.update_channel('g_lateral', timestamps, g_lat)
            
            # Update GG diagram
            self.gg_diagram.update_data(g_accel, g_lat)
            
        # Update debug widget if enabled
        if self.debug_mode and self.debug_widget:
            # Calculate performance metrics
            update_time = (time.time() - self.last_update_time) * 1000  # ms
            self.last_update_time = time.time()
            
            buffer_memory = sum(sys.getsizeof(f) for f in self.live_buffer) / (1024 * 1024)
            
            metrics = {
                'buffer_size': len(self.live_buffer),
                'update_time_ms': update_time,
                'buffer_memory_mb': buffer_memory
            }
            self.debug_widget.update_performance(metrics)
            
    def start_recording(self):
        """Start recording telemetry to file"""
        if not self.receiver or not self.receiver.running:
            QMessageBox.warning(self, 'Error', 'Not connected to telemetry service')
            return
            
        # Create session directory
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_dir = Path('sessions') / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        log_path = session_dir / 'telemetry.tlog'
        self.logger = TelemetryLogger(str(log_path))
        
        # Get field names from first frame
        if self.live_buffer:
            field_names = list(self.live_buffer[0].data.keys())
            metadata = {
                'session_id': session_id,
                'start_time': datetime.now().isoformat(),
                'source': 'live_telemetry'
            }
            self.logger.create_log(field_names, metadata)
            
        self.recording = True
        self.statusBar().showMessage(f'Recording to {log_path}')
        
    def stop_recording(self):
        """Stop recording telemetry"""
        if self.logger:
            self.logger.close()
            self.logger = None
            
        self.recording = False
        self.statusBar().showMessage('Recording stopped')
        
    def open_log(self):
        """Open telemetry log file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, 'Open Telemetry Log', '', 'Telemetry Logs (*.tlog);;All Files (*)')
            
        if filepath:
            self.load_log(filepath)
            
    def import_fueltech(self):
        """Import FuelTech CSV file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, 'Import FuelTech CSV', '', 'CSV Files (*.csv);;All Files (*)')
            
        if filepath:
            try:
                progress = QProgressDialog("Importing CSV...", "Cancel", 0, 100, self)
                progress.setWindowModality(Qt.WindowModal)
                progress.show()
                
                field_names, data, metadata = FuelTechImporter.import_csv(filepath)
                
                progress.setValue(100)
                
                self.current_data = data
                self.field_names = field_names
                self.metadata = metadata
                
                self.update_all_plots()
                self.statusBar().showMessage(f'Imported {len(data)} samples from {filepath}')
                
            except Exception as e:
                QMessageBox.critical(self, 'Import Error', str(e))
                
    def load_log(self, filepath: str):
        """Load telemetry log"""
        try:
            field_names, data, metadata = TelemetryLogger.read_log(filepath)
            
            self.current_data = data
            self.field_names = field_names
            self.metadata = metadata
            
            self.update_all_plots()
            
            # Update debug widget if enabled
            if self.debug_mode and self.debug_widget:
                self.debug_widget.update_data_structure(field_names, data)
            
            self.statusBar().showMessage(f'Loaded {len(data)} samples')
            logger.info(f"Loaded log file: {filepath}, {len(data)} samples")
            
        except Exception as e:
            logger.error(f"Failed to load log: {str(e)}")
            QMessageBox.critical(self, 'Load Error', str(e))
            
    def configure_track(self):
        """Configure track parameters"""
        dialog = TrackConfigDialog(self.current_data, self.field_names, self)
        if dialog.exec_() == QDialog.Accepted:
            self.current_track = dialog.track
            self.statusBar().showMessage(f'Track configured: {self.current_track.name}')
            
    def segment_laps(self):
        """Segment telemetry into laps"""
        if self.current_data is None:
            QMessageBox.warning(self, 'Error', 'No data loaded')
            return
            
        if self.current_track is None:
            QMessageBox.warning(self, 'Error', 'Track not configured')
            return
            
        self.laps = LapAnalyzer.segment_laps(
            self.current_data, self.field_names, self.current_track)
            
        # Update lap table
        self.update_lap_table()
        self.statusBar().showMessage(f'Segmented {len(self.laps)} laps')
        
    def update_lap_table(self):
        """Update lap table with lap times"""
        self.lap_table.setRowCount(len(self.laps))
        
        for i, lap in enumerate(self.laps):
            self.lap_table.setItem(i, 0, QTableWidgetItem(str(lap.lap_number)))
            self.lap_table.setItem(i, 1, QTableWidgetItem(f'{lap.lap_time:.3f}s'))
            
            # Calculate delta to best lap
            if i > 0:
                best_time = min(l.lap_time for l in self.laps[:i+1])
                delta = lap.lap_time - best_time
                delta_str = f'+{delta:.3f}' if delta > 0 else f'{delta:.3f}'
                self.lap_table.setItem(i, 2, QTableWidgetItem(delta_str))
                
    def on_lap_selected(self):
        """Handle lap selection"""
        selected = self.lap_table.selectedItems()
        if selected:
            lap_idx = selected[0].row()
            lap = self.laps[lap_idx]
            
            # Update plots to show selected lap
            self.update_plots_for_lap(lap)
            
    def update_plots_for_lap(self, lap: LapDefinition):
        """Update plots to show specific lap"""
        if self.current_data is None:
            return
            
        # Extract lap data
        lap_data = self.current_data[lap.start_index:lap.end_index+1]
        time_data = lap_data[:, 0] - lap_data[0, 0]  # Relative time
        
        # Update plots
        self.update_plot_data(time_data, lap_data)
        
    def update_all_plots(self):
        """Update all plots with current data"""
        if self.current_data is None:
            return
            
        time_data = self.current_data[:, 0] - self.current_data[0, 0]
        self.update_plot_data(time_data, self.current_data)
        
    def update_plot_data(self, time_data: np.ndarray, data: np.ndarray):
        """Update plots with data"""
        # Helper to get channel data
        def get_channel(name: str) -> Optional[np.ndarray]:
            try:
                idx = self.field_names.index(name) + 1
                return data[:, idx]
            except (ValueError, IndexError):
                return None
                
        # Update RPM plot
        rpm = get_channel('rpm')
        if rpm is not None:
            self.rpm_plot.update_channel('rpm', time_data, rpm)
            
        # Update speed plot
        speed = get_channel('wheel_speed_fl')
        if speed is not None:
            self.speed_plot.update_channel('speed', time_data, speed)
            
        gear = get_channel('gear')
        if gear is not None:
            self.speed_plot.update_channel('gear', time_data, gear * 10)  # Scale for visibility
            
        # Update TPS plot
        tps = get_channel('tps')
        if tps is not None:
            self.tps_plot.update_channel('tps', time_data, tps)
            
        # Update G-force plot
        g_accel = get_channel('g_accel')
        g_lat = get_channel('g_lateral')
        if g_accel is not None and g_lat is not None:
            self.g_plot.update_channel('g_accel', time_data, g_accel)
            self.g_plot.update_channel('g_lateral', time_data, g_lat)
            
            # Update GG diagram
            self.gg_diagram.update_data(g_accel, g_lat)
            
        # Update track map
        lat = get_channel('latitude')
        lon = get_channel('longitude')
        if lat is not None and lon is not None and rpm is not None:
            # Color by RPM
            colors = self.rpm_to_color(rpm)
            self.track_map.update_map(lat, lon, colors)
            
        # Update histograms
        if rpm is not None:
            y, x = np.histogram(rpm, bins=50)
            self.rpm_hist_plot.clear()
            self.rpm_hist_plot.plot(x, y, stepMode=True, fillLevel=0,
                                   brush=(255, 255, 0, 150))

        if tps is not None:
            y, x = np.histogram(tps, bins=50)
            self.tps_hist_plot.clear()
            self.tps_hist_plot.plot(x, y, stepMode=True, fillLevel=0,
                                   brush=(0, 255, 255, 150))

        # Update mixture tuning plots
        map_data = get_channel('map')
        if map_data is not None:
            self.rpm_map_plot.update_channel('map', time_data, map_data)

        o2 = get_channel('exhaust_o2')
        if o2 is not None:
            self.o2_lambda_plot.update_channel('exhaust_o2', time_data, o2)

        lambda_corr = get_channel('lambda_correction')
        if lambda_corr is not None:
            self.o2_lambda_plot.update_channel('lambda_correction', time_data, lambda_corr)

        inj_a = get_channel('inj_time_bank_a')
        if inj_a is not None:
            self.inj_time_plot.update_channel('inj_time_bank_a', time_data, inj_a)

        inj_b = get_channel('inj_time_bank_b')
        if inj_b is not None:
            self.inj_time_plot.update_channel('inj_time_bank_b', time_data, inj_b)

        # Update suspension plots
        for shock in ['shock_fl', 'shock_fr', 'shock_rl', 'shock_rr']:
            shock_data = get_channel(shock)
            if shock_data is not None:
                self.shock_plot.update_channel(shock, time_data, shock_data)

        for wheel in ['wheel_speed_fl', 'wheel_speed_fr', 'wheel_speed_rl', 'wheel_speed_rr']:
            wheel_data = get_channel(wheel)
            if wheel_data is not None:
                self.wheel_speed_plot.update_channel(wheel, time_data, wheel_data)

    def rpm_to_color(self, rpm: np.ndarray) -> np.ndarray:
        """Convert RPM values to colors"""
        # Normalize RPM to 0-1
        rpm_norm = (rpm - rpm.min()) / (rpm.max() - rpm.min() + 1e-6)
        
        # Create color map (blue to red)
        colors = np.zeros((len(rpm), 3), dtype=np.uint8)
        colors[:, 0] = (rpm_norm * 255).astype(np.uint8)  # Red
        colors[:, 2] = ((1 - rpm_norm) * 255).astype(np.uint8)  # Blue
        
        return colors
        
    def update_time_window(self, value: int):
        """Update visible time window"""
        # Zoom all plots
        pass  # Implement zoom logic
        
    def toggle_playback(self):
        """Toggle playback mode"""
        pass  # Implement playback
        
    def stop_playback(self):
        """Stop playback"""
        pass  # Implement playback stop
        
    def seek_position(self, value: int):
        """Seek to position"""
        pass  # Implement seek
        
    def toggle_debug_mode(self, enabled: bool):
        """Toggle debug mode on/off"""
        self.debug_mode = enabled
        
        if enabled:
            self.debug_dock.show()
            self.statusBar().showMessage('Debug mode enabled')
            logger.info("Debug mode enabled")
            
            # Update debug widget with current data if available
            if self.current_data is not None and len(self.field_names) > 0:
                self.debug_widget.update_data_structure(self.field_names, self.current_data)
        else:
            self.debug_dock.hide()
            self.statusBar().showMessage('Debug mode disabled')
            logger.info("Debug mode disabled")
            
    def show_frame_data(self):
        """Display raw frame data in debug console"""
        if not self.debug_mode:
            self.toggle_debug_action.setChecked(True)
            self.toggle_debug_mode(True)
            
        # Switch to Frame Inspector tab
        if self.debug_widget:
            self.debug_widget.debug_tabs.setCurrentIndex(0)
        
    def show_performance_stats(self):
        """Display performance statistics"""
        if not self.debug_mode:
            self.toggle_debug_action.setChecked(True)
            self.toggle_debug_mode(True)
            
        # Switch to Performance tab
        if self.debug_widget:
            self.debug_widget.debug_tabs.setCurrentIndex(2)
        
    def show_data_structure(self):
        """Display data structure information"""
        if not self.debug_mode:
            self.toggle_debug_action.setChecked(True)
            self.toggle_debug_mode(True)
            
        self.debug_dock.widget().setCurrentIndex(2)  # Switch to Data Structure tab
        self.update_data_structure()
        
    def clear_live_buffer(self):
        """Clear the live data buffer"""
        self.live_buffer.clear()
        self.frame_count = 0
        self.statusBar().showMessage('Live buffer cleared')
        
    def update_debug_info(self):
        """Update all debug information displays"""
        if not self.debug_mode:
            return
            
        # Update frame data
        if self.last_frame:
            self.update_frame_data(self.last_frame)
            
        # Update performance stats
        self.update_performance_stats()
        
        # Update data structure
        self.update_data_structure()
        
        # Update field info
        self.update_field_info()
        
        # Update buffer status
        self.update_buffer_status()
        
    def update_frame_data(self, frame: TelemetryFrame):
        """Update raw frame data display"""
        import json
        
        output = []
        output.append(f"{'='*60}")
        output.append(f"Frame #{self.frame_count}")
        output.append(f"Timestamp: {frame.timestamp:.3f} seconds")
        output.append(f"{'='*60}")
        output.append("")
        
        # Display data in formatted JSON
        output.append("Data Package:")
        output.append(json.dumps(frame.data, indent=2, sort_keys=True))
        output.append("")
        
        # Summary statistics
        output.append(f"{'='*60}")
        output.append("Frame Summary:")
        output.append(f"  Fields: {len(frame.data)}")
        output.append(f"  Non-zero: {sum(1 for v in frame.data.values() if v != 0)}")
        output.append(f"  Min value: {min(frame.data.values()):.3f}")
        output.append(f"  Max value: {max(frame.data.values()):.3f}")
        output.append(f"{'='*60}")
        output.append("")
        
        # Append to existing text
        self.frame_data_text.appendPlainText("\n".join(output))
        
    def update_performance_stats(self):
        """Update performance statistics display"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        output = []
        output.append(f"{'='*60}")
        output.append("PERFORMANCE STATISTICS")
        output.append(f"{'='*60}")
        output.append("")
        
        # Frame statistics
        output.append("Frame Statistics:")
        output.append(f"  Total Frames Received: {self.frame_count}")
        output.append(f"  Frames in Buffer: {len(self.live_buffer)}")
        output.append(f"  Buffer Capacity: {self.live_buffer.maxlen}")
        output.append(f"  Buffer Fill: {len(self.live_buffer)/self.live_buffer.maxlen*100:.1f}%")
        output.append("")
        
        # Memory statistics
        output.append("Memory Usage:")
        output.append(f"  RSS: {memory_info.rss / 1024 / 1024:.1f} MB")
        output.append(f"  VMS: {memory_info.vms / 1024 / 1024:.1f} MB")
        
        if hasattr(memory_info, 'shared'):
            output.append(f"  Shared: {memory_info.shared / 1024 / 1024:.1f} MB")
        output.append("")
        
        # CPU statistics
        cpu_percent = process.cpu_percent(interval=0.1)
        output.append("CPU Usage:")
        output.append(f"  Process: {cpu_percent:.1f}%")
        output.append(f"  Threads: {process.num_threads()}")
        output.append("")
        
        # Loaded data statistics
        if self.current_data is not None:
            data_size = self.current_data.nbytes / 1024 / 1024
            output.append("Loaded Data:")
            output.append(f"  Array Size: {data_size:.1f} MB")
            output.append(f"  Shape: {self.current_data.shape}")
            output.append(f"  Fields: {len(self.field_names)}")
            output.append(f"  Samples: {len(self.current_data)}")
            if len(self.current_data) > 0:
                duration = self.current_data[-1, 0] - self.current_data[0, 0]
                output.append(f"  Duration: {duration:.1f} seconds")
                output.append(f"  Sample Rate: ~{len(self.current_data)/duration:.1f} Hz")
        output.append("")
        
        # Recording status
        output.append("Recording Status:")
        output.append(f"  Active: {'Yes' if self.recording else 'No'}")
        output.append(f"  Receiver Running: {'Yes' if self.receiver and self.receiver.running else 'No'}")
        output.append("")
        
        # Update timer
        output.append("Update Timer:")
        output.append(f"  Active: {'Yes' if self.update_timer.isActive() else 'No'}")
        output.append(f"  Interval: {self.update_timer.interval()} ms")
        output.append("")
        
        output.append(f"{'='*60}")
        
        self.performance_text.setPlainText("\n".join(output))
        
    def update_data_structure(self):
        """Update data structure display"""
        output = []
        output.append(f"{'='*60}")
        output.append("DATA STRUCTURE INFORMATION")
        output.append(f"{'='*60}")
        output.append("")
        
        # Metadata
        if self.metadata:
            output.append("Metadata:")
            for key, value in self.metadata.items():
                output.append(f"  {key}: {value}")
            output.append("")
        
        # Field names and types
        if self.field_names:
            output.append(f"Fields ({len(self.field_names)}):")
            for i, field in enumerate(self.field_names):
                if self.current_data is not None and len(self.current_data) > 0:
                    col_data = self.current_data[:, i + 1]
                    dtype = col_data.dtype
                    min_val = col_data.min()
                    max_val = col_data.max()
                    mean_val = col_data.mean()
                    output.append(f"  [{i:2d}] {field:25s} | {dtype} | "
                                f"min={min_val:8.2f} max={max_val:8.2f} mean={mean_val:8.2f}")
                else:
                    output.append(f"  [{i:2d}] {field}")
            output.append("")
        
        # Track information
        if self.current_track:
            output.append("Track Configuration:")
            output.append(f"  Name: {self.current_track.name}")
            output.append(f"  Length: {self.current_track.length_km} km")
            output.append(f"  Start/Finish: ({self.current_track.start_finish_lat:.6f}, "
                         f"{self.current_track.start_finish_lon:.6f})")
            output.append(f"  Tolerance: {self.current_track.tolerance_m} m")
            output.append("")
        
        # Lap information
        if self.laps:
            output.append(f"Laps ({len(self.laps)}):")
            for lap in self.laps[:10]:  # Show first 10 laps
                output.append(f"  Lap {lap.lap_number:2d}: {lap.lap_time:.3f}s "
                             f"({lap.end_index - lap.start_index} samples)")
            if len(self.laps) > 10:
                output.append(f"  ... and {len(self.laps) - 10} more laps")
            output.append("")
        
        # Live buffer structure
        if self.live_buffer:
            output.append("Live Buffer Structure:")
            output.append(f"  Size: {len(self.live_buffer)} frames")
            output.append(f"  Max Size: {self.live_buffer.maxlen} frames")
            if len(self.live_buffer) > 0:
                first_frame = self.live_buffer[0]
                output.append(f"  First Timestamp: {first_frame.timestamp:.3f}")
                output.append(f"  Fields: {len(first_frame.data)}")
                output.append(f"  Field Names: {', '.join(list(first_frame.data.keys())[:10])}")
                if len(first_frame.data) > 10:
                    output.append(f"    ... and {len(first_frame.data) - 10} more fields")
        output.append("")
        
        output.append(f"{'='*60}")
        
        self.structure_text.setPlainText("\n".join(output))
        
    def update_field_info(self):
        """Update field information table"""
        if not self.field_names or self.current_data is None:
            return
            
        self.field_info_table.setRowCount(len(self.field_names))
        
        for i, field in enumerate(self.field_names):
            # Field name
            self.field_info_table.setItem(i, 0, QTableWidgetItem(field))
            
            # Data type
            col_data = self.current_data[:, i + 1]
            self.field_info_table.setItem(i, 1, QTableWidgetItem(str(col_data.dtype)))
            
            # Min value
            self.field_info_table.setItem(i, 2, QTableWidgetItem(f"{col_data.min():.3f}"))
            
            # Max value
            self.field_info_table.setItem(i, 3, QTableWidgetItem(f"{col_data.max():.3f}"))
            
            # Current value (from last frame or buffer)
            current = ""
            if self.live_buffer and len(self.live_buffer) > 0:
                last_frame = self.live_buffer[-1]
                if field in last_frame.data:
                    current = f"{last_frame.data[field]:.3f}"
            elif len(col_data) > 0:
                current = f"{col_data[-1]:.3f}"
            self.field_info_table.setItem(i, 4, QTableWidgetItem(current))
            
    def update_buffer_status(self):
        """Update buffer status display"""
        output = []
        output.append(f"{'='*60}")
        output.append("BUFFER STATUS")
        output.append(f"{'='*60}")
        output.append("")
        
        output.append("Live Buffer:")
        output.append(f"  Current Size: {len(self.live_buffer)} frames")
        output.append(f"  Maximum Size: {self.live_buffer.maxlen} frames")
        output.append(f"  Fill Percentage: {len(self.live_buffer)/self.live_buffer.maxlen*100:.1f}%")
        
        if len(self.live_buffer) > 0:
            first_time = self.live_buffer[0].timestamp
            last_time = self.live_buffer[-1].timestamp
            duration = last_time - first_time
            output.append(f"  Time Span: {duration:.3f} seconds")
            if duration > 0:
                output.append(f"  Effective Rate: {len(self.live_buffer)/duration:.1f} Hz")
        output.append("")
        
        # Show last few frames summary
        if len(self.live_buffer) >= 5:
            output.append("Last 5 Frames:")
            for i, frame in enumerate(list(self.live_buffer)[-5:]):
                non_zero = sum(1 for v in frame.data.values() if abs(v) > 0.001)
                output.append(f"  Frame {len(self.live_buffer)-5+i}: "
                             f"t={frame.timestamp:.3f}s, {non_zero}/{len(frame.data)} fields active")
        output.append("")
        
        output.append(f"{'='*60}")
        
        self.buffer_text.setPlainText("\n".join(output))
        
    def on_telemetry_frame(self, frame: TelemetryFrame):
        """Handle received telemetry frame"""
        self.live_buffer.append(frame)
        self.frame_count += 1
        self.last_frame = frame
        
        # Update debug widget if enabled
        if self.debug_mode and self.debug_widget:
            self.debug_widget.update_frame(frame)
        
        # Log if recording
        if self.recording and self.logger:
            self.logger.append_frame(frame.timestamp, frame.data)
            
        logger.debug(f"Frame received: {frame.timestamp:.3f}, {len(frame.data)} channels")
            
    def load_settings(self):
        """Load application settings"""
        geometry = self.settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
            
    def save_settings(self):
        """Save application settings"""
        self.settings.setValue('geometry', self.saveGeometry())
        
    def closeEvent(self, event):
        """Handle application close"""
        # Stop receiver
        if self.receiver and self.receiver.running:
            self.receiver.stop()
            
        # Stop recording
        if self.recording:
            self.stop_recording()
            
        # Save settings
        self.save_settings()
        
        event.accept()


# ============================================================================
# Application Entry Point
# ============================================================================

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Telemetry Analyzer")
    app.setOrganizationName("MotorsportTech")
    
    # Set application font
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    # Create and show main window
    window = TelemetryAnalyzer()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()