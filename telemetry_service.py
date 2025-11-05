# telemetry_service.py

import os
import time
import json
import struct
import threading
from datetime import datetime
from typing import Dict, List, Optional, Callable, Iterable

import serial  # pip install pyserial

# ---- Frame constants ----
HEADER = b'\xFF\xFF'
TRAILER = b'\xFF\xFF'
FRAME_LEN = 94
PAYLOAD_LEN = FRAME_LEN - len(HEADER) - len(TRAILER)  # 90 bytes

# 36 int16, 2 float32, 1 uint64, 1 int16 (little-endian by default)
DEFAULT_ENDIAN = '<'
STRUCT_FMT = '36hffQh'

FIELD_NAMES = [
    "tps", "map", "air_temp", "engine_temp", "oil_pressure", "fuel_pressure", "water_pressure",
    "gear", "exhaust_o2", "rpm", "oil_temp1", "pit_limit",
    "wheel_speed_fr", "wheel_speed_fl", "wheel_speed_rr", "wheel_speed_rl",
    "tc_slip", "tc_retard", "tc_cut", "heading",
    "shock_fr", "shock_fl", "shock_rr", "shock_rl",
    "g_accel", "g_lateral",
    "yaw_rate_frontal", "yaw_rate_lateral",
    "lambda_correction", "fuel_flow_total",
    "inj_time_bank_a", "inj_time_bank_b",
    "oil_temp2", "trans_temp",
    "fuel_consumption", "brake_pressure",
    "latitude", "longitude", "distance_km", "gps_speed_knots",
]

# Scaling factors
SCALES = {
    "tps": 0.1, "map": 0.001, "air_temp": 0.1, "engine_temp": 0.1,
    "oil_pressure": 0.001, "fuel_pressure": 0.001, "water_pressure": 0.001,
    "gear": None,
    "exhaust_o2": 0.001, "rpm": 1.0, "oil_temp1": 0.1,
    "pit_limit": "bool",
    "wheel_speed_fr": 0.1, "wheel_speed_fl": 1.0, "wheel_speed_rr": 1.0, "wheel_speed_rl": 1.0,
    "tc_slip": None, "tc_retard": None, "tc_cut": None, "heading": None,
    "shock_fr": 0.001, "shock_fl": 0.001, "shock_rr": 0.001, "shock_rl": 0.001,
    "g_accel": 0.001, "g_lateral": 0.001,
    "yaw_rate_frontal": None, "yaw_rate_lateral": None,
    "lambda_correction": None, "fuel_flow_total": None,
    "inj_time_bank_a": 0.01, "inj_time_bank_b": 0.01,
    "oil_temp2": 0.1, "trans_temp": 0.1,
    "fuel_consumption": None, "brake_pressure": 0.001,
    "latitude": 1.0, "longitude": 1.0,
    "distance_km": 1.0 / 76000000.0,
    "gps_speed_knots": 0.1,
}


# --------------------------
# Low-level decoder (stream)
# --------------------------
class TelemetryDecoder:
    """
    Robust streaming decoder for 94-byte frames delimited by:
    - Header: 0xFF 0xFF
    - Trailer: 0x00 0x00
    Handles partial frames, noise, and resynchronization.
    """

    def __init__(self, endian: str = DEFAULT_ENDIAN, return_raw: bool = False):
        if endian not in ('<', '>'):
            raise ValueError("endian must be '<' (little) or '>' (big)")
        self.endian = endian
        self.payload_struct = struct.Struct(endian + STRUCT_FMT)
        self.buffer = bytearray()
        self.return_raw = return_raw

    def reset(self):
        self.buffer.clear()

    def _scale_fields(self, values: List):
        raw_map = dict(zip(FIELD_NAMES, values))
        scaled = {}
        for k, v in raw_map.items():
            s = SCALES.get(k)
            if s is None:
                scaled[k] = v
            elif s == "bool":
                scaled[k] = bool(v)
            else:
                scaled[k] = v * s
        return scaled, raw_map

    def _parse_payload(self, payload: bytes) -> Optional[Dict]:
        if len(payload) != PAYLOAD_LEN:
            return None
        try:
            unpacked = self.payload_struct.unpack(payload)
        except struct.error:
            return None

        scaled, raw = self._scale_fields(unpacked)

        # Optional sanity checks (enable or adjust as needed):
        # lat, lon = scaled["latitude"], scaled["longitude"]
        # if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        #     return None

        return {
            "timestamp": time.time(),
            "data": scaled if not self.return_raw else {"scaled": scaled, "raw": raw}
        }

    def feed(self, data: bytes) -> List[Dict]:
        """
        Feed incoming bytes and return a list of decoded frames.
        """
        frames: List[Dict] = []
        if not data:
            return frames

        self.buffer.extend(data)

        while True:
            start = self.buffer.find(HEADER)
            if start == -1:
                # keep last byte in case it's 0xFF starting the next header
                if len(self.buffer) > 1:
                    self.buffer[:] = self.buffer[-1:]
                break

            # Not enough data for a full frame yet
            if len(self.buffer) - start < FRAME_LEN:
                if start > 0:
                    del self.buffer[:start]  # drop noise before header
                break

            candidate = self.buffer[start:start + FRAME_LEN]
            if candidate[-2:] != TRAILER:
                # False header; discard 1 byte and rescan
                del self.buffer[:start + 1]
                continue
            payload = candidate[len(HEADER):-len(TRAILER)]
            parsed = self._parse_payload(payload)
            if parsed:
                frames.append(parsed)

            # Remove processed slice and continue
            del self.buffer[:start + FRAME_LEN]

        return frames


# --------------------------
# Serial utilities
# --------------------------
def open_serial(port: str, baudrate: int = 115200, timeout: float = 0.05) -> serial.Serial:
    """
    Open a serial port with typical defaults (8N1).
    """
    return serial.Serial(
        port=port,
        baudrate=baudrate,
        timeout=timeout,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
    )


# --------------------------
# File Logger (daily rotate)
# --------------------------
class FileLogger:
    """
    Daily-rotating file logger supporting CSV or JSON Lines.
    Use a strftime pattern in 'path_pattern' (e.g., 'logs/telemetry_%Y-%m-%d.csv').
    For CSV, writes a header when creating a new file for the day.
    """

    def __init__(self, path_pattern: Optional[str], fmt: str = 'csv'):
        self.path_pattern = path_pattern
        self.fmt = fmt.lower()
        if self.path_pattern is None:
            self._fh = None
            self._current_path = None
            return

        if self.fmt not in ('csv', 'jsonl'):
            raise ValueError("fmt must be 'csv' or 'jsonl'")

        self._fh = None
        self._current_path = None

        # Precompute CSV header list
        self._csv_headers = ['timestamp'] + FIELD_NAMES

    def _render_path(self) -> str:
        return datetime.now().strftime(self.path_pattern)

    def _ensure_open(self):
        if self.path_pattern is None:
            return

        path = self._render_path()
        if path != self._current_path:
            # Rotate: close previous and open new file
            if self._fh:
                try:
                    self._fh.flush()
                    self._fh.close()
                except Exception:
                    pass
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            new_file = not os.path.exists(path) or os.path.getsize(path) == 0
            self._fh = open(path, 'a', encoding='utf-8', newline='')
            self._current_path = path

            # For CSV, write header on new file
            if self.fmt == 'csv' and new_file:
                header_line = ','.join(self._csv_headers)
                self._fh.write(header_line + '\n')
                self._fh.flush()

    def log(self, frame: Dict):
        """
        frame = {"timestamp": float, "data": {...scaled fields...}}
        """
        if self.path_pattern is None:
            return

        self._ensure_open()
        if not self._fh:
            return

        ts = frame.get("timestamp", time.time())
        data = frame.get("data", {})

        if self.fmt == 'jsonl':
            rec = {"timestamp": ts, **data}
            self._fh.write(json.dumps(rec, ensure_ascii=False, separators=(',', ':')) + '\n')
            self._fh.flush()
        else:  # CSV
            # Ensure stable order and empty if missing
            row = [ts] + [data.get(k, '') for k in FIELD_NAMES]
            # Simple CSV without quoting (values are numeric/bool)
            line = ','.join(str(v) for v in row)
            self._fh.write(line + '\n')
            self._fh.flush()

    def close(self):
        if self._fh:
            try:
                self._fh.flush()
                self._fh.close()
            finally:
                self._fh = None
                self._current_path = None


# --------------------------
# Public APIs for consumers
# --------------------------
def read_frames(port: str,
                baudrate: int = 115200,
                endian: str = DEFAULT_ENDIAN,
                return_raw: bool = False,
                chunk_size: int = 1024) -> Iterable[Dict]:
    """
    Pull-based generator that yields decoded frames as dictionaries:
        {"timestamp": float, "data": {field: value, ...}}
    Usage (consumer drives the loop):
        for frame in read_frames('/dev/ttyUSB0'):
            dashboard.push(frame)
    """
    ser = open_serial(port, baudrate=baudrate)
    decoder = TelemetryDecoder(endian=endian, return_raw=return_raw)
    try:
        while True:
            chunk = ser.read(chunk_size)
            if not chunk:
                continue
            frames = decoder.feed(chunk)
            for f in frames:
                yield f
    finally:
        ser.close()


class TelemetryServiceHandle:
    """
    Handle returned by start_telemetry_service to stop the background thread cleanly.
    """
    def __init__(self, thread: threading.Thread, stop_event: threading.Event, logger: FileLogger):
        self._thread = thread
        self._stop_event = stop_event
        self._logger = logger

    def stop(self, join_timeout: Optional[float] = 2.0):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=join_timeout)
        self._logger.close()


def start_telemetry_service(
    port: str,
    baudrate: int = 115200,
    endian: str = DEFAULT_ENDIAN,
    on_frame: Optional[Callable[[Dict], None]] = None,
    log_path_pattern: Optional[str] = None,   # e.g., "logs/telemetry_%Y-%m-%d.csv"
    log_format: str = 'csv',
    chunk_size: int = 1024,
    return_raw: bool = False,
) -> TelemetryServiceHandle:
    """
    Push-based API: reads serial in a background thread, decodes frames,
    calls 'on_frame(frame)' for each decoded frame, and logs to file if enabled.

    Returns a handle with .stop() for clean shutdown.

    - on_frame: a callback taking a frame dict {"timestamp": ..., "data": {...}}.
    - log_path_pattern: strftime pattern file path for daily rotation.
      Examples: "logs/telemetry_%Y-%m-%d.csv" or "logs/telemetry_%Y-%m-%d.jsonl"
    - log_format: 'csv' or 'jsonl'
    """
    stop_event = threading.Event()
    logger = FileLogger(log_path_pattern, fmt=log_format)
    ser = open_serial(port, baudrate=baudrate)
    decoder = TelemetryDecoder(endian=endian, return_raw=return_raw)

    def _worker():
        try:
            while not stop_event.is_set():
                chunk = ser.read(chunk_size)
                if not chunk:
                    continue
                frames = decoder.feed(chunk)
                for f in frames:
                    if on_frame:
                        try:
                            on_frame(f)
                        except Exception:
                            # Keep service alive even if the consumer callback fails
                            pass
                    try:
                        logger.log(f)
                    except Exception:
                        # Keep service alive on logging errors
                        pass
        finally:
            try:
                ser.close()
            except Exception:
                pass
            logger.close()

    t = threading.Thread(target=_worker, name="TelemetryServiceThread", daemon=True)
    t.start()
    return TelemetryServiceHandle(t, stop_event, logger)


# --------------------------
# Optional: simple CLI usage
# --------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Telemetry serial decoder service")
    parser.add_argument('--port', required=True, help="Serial port (e.g., COM3, /dev/ttyUSB0)")
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--endian', choices=['<', '>'], default='<', help="'<' little-endian, '>' big-endian")
    parser.add_argument('--mode', choices=['pull', 'push'], default='push')
    parser.add_argument('--log', default='logs/telemetry_%Y-%m-%d.csv', help="Path pattern for logs or '' to disable")
    parser.add_argument('--fmt', choices=['csv', 'jsonl'], default='csv', help="Log file format")
    parser.add_argument('--raw', action='store_true', help="Include raw values in output")
    args = parser.parse_args()

    if args.mode == 'pull':
        # Pull mode: print frames to stdout
        for frame in read_frames(args.port, baudrate=args.baud, endian=args.endian, return_raw=args.raw):
            print(json.dumps(frame, ensure_ascii=False, separators=(',', ':')))
    else:
        # Push mode: start service and print frames via callback
        def on_frame_cb(frame: Dict):
            print(json.dumps(frame, ensure_ascii=False, separators=(',', ':')))

        log_pattern = args.log if args.log.strip() else None
        handle = start_telemetry_service(
            port=args.port,
            baudrate=args.baud,
            endian=args.endian,
            on_frame=on_frame_cb,
            log_path_pattern=log_pattern,
            log_format=args.fmt,
            return_raw=args.raw,
        )

        print("Telemetry service running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("Stopping service...")
            handle.stop()
            print("Stopped.")
