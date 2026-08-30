"""Dedicated 1090ES result workspace using the shared IQ recording contract."""

from __future__ import annotations

import json
import queue
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_sa.config.input_frontend import InputPowerCorrection
from pluto_sa.standards.adsb1090.analysis import ADSB1090Analyzer
from pluto_sa.standards.adsb1090.decoder import (
    decode_global_airborne_cpr,
    decode_local_airborne_cpr,
)
from pluto_sa.standards.adsb1090.metadata import (
    AircraftMetadata,
    AircraftMetadataDatabase,
)
from pluto_sa.standards.adsb1090.route import (
    ADSBDBRouteClient,
    FlightRoute,
    RouteAirport,
    normalize_callsign,
)
from pluto_sa.standards.adsb1090.leaflet_map import LeafletAircraftMap
from pluto_sa.standards.adsb1090.model import (
    ADSB1090AnalysisResult,
    ADSB1090Message,
    ADSB1090Settings,
)
from pluto_sa.vsa.model import IQRecording
from pluto_sa.vsa.pluto_source import (
    CaptureCancelledError,
    PlutoCaptureSettings,
    PlutoLiveSource,
)
from pluto_sa.vsa.sources import FileIQSource
from pluto_sa.vsa.ui.measurement_chrome import (
    install_measurement_plot_menu,
    make_measurement_dock,
    make_measurement_plot,
)


_TRACE_COLOR = "y"
_BIT_COLOR = (0, 255, 160)
_ZERO_BIT_COLOR = (0, 210, 255)
_PACKET_GROUP_GAP_S = 0.020
_STREAM_BLOCK_DURATION_S = 0.050
_STREAM_OVERLAP_S = 160e-6
_SINGLE_PRETRIGGER_S = 1e-3
_IQ_POWER_DISPLAY_FLOOR_DBM = -120.0
_MAX_POWER_PLOT_POINTS = 25_000
_STREAM_DISPLAY_BATCH_MS = 100
_MODE_S_HEADER_FIELDS = (
    "flight_status",
    "capability",
    "control_field",
    "vertical_status",
    "cross_link_capability",
    "sensitivity_level",
)
_FLIGHT_STATUS_DESCRIPTIONS = {
    0: "No alert, no SPI, airborne",
    1: "No alert, no SPI, on ground",
    2: "Alert, no SPI, airborne",
    3: "Alert, no SPI, on ground",
    4: "Alert, SPI",
    5: "No alert, SPI",
    6: "Reserved",
    7: "Not assigned",
}

_MAX_LOCAL_CPR_RANGE_NM = 180.0


def _great_circle_distance_nm(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    latitude_a_rad, latitude_b_rad = np.radians([latitude_a, latitude_b])
    delta_latitude = latitude_b_rad - latitude_a_rad
    delta_longitude = np.radians(longitude_b - longitude_a)
    haversine = (
        np.sin(delta_latitude / 2.0) ** 2
        + np.cos(latitude_a_rad)
        * np.cos(latitude_b_rad)
        * np.sin(delta_longitude / 2.0) ** 2
    )
    central_angle = 2.0 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
    return float(3440.065 * central_angle)


def _peak_envelope_decimate(
    x: np.ndarray,
    y: np.ndarray,
    maximum_points: int = _MAX_POWER_PLOT_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Bound GUI trace size while preserving each bucket's low/high peaks."""

    count = min(x.size, y.size)
    if count <= maximum_points:
        return x[:count], y[:count]
    bucket_count = max(1, maximum_points // 2)
    bucket_size = int(np.ceil(count / bucket_count))
    padded_count = int(np.ceil(count / bucket_size) * bucket_size)
    pad = padded_count - count
    x_values = np.pad(x[:count], (0, pad), mode="edge").reshape(-1, bucket_size)
    y_values = np.pad(y[:count], (0, pad), mode="edge").reshape(-1, bucket_size)
    low_index = np.argmin(y_values, axis=1)
    high_index = np.argmax(y_values, axis=1)
    order = np.stack((low_index, high_index), axis=1)
    order.sort(axis=1)
    rows = np.arange(y_values.shape[0])[:, None]
    return x_values[rows, order].reshape(-1), y_values[rows, order].reshape(-1)


@dataclass(frozen=True)
class _ADSBCaptureBatch:
    recording: IQRecording


@dataclass(frozen=True)
class _ADSBStreamView:
    recording: IQRecording
    result: ADSB1090AnalysisResult
    append: bool
    elapsed_base_s: float
    capture_started_at: datetime
    single_complete: bool = False


@dataclass(frozen=True)
class _ADSBPacketEntry:
    message: ADSB1090Message
    result: ADSB1090AnalysisResult
    recording: IQRecording
    elapsed_s: float
    wall_time: datetime
    on_pulse_power_dbm: float


@dataclass
class _ADSBAircraftState:
    icao_address: str
    first_elapsed_s: float
    first_wall_time: datetime
    last_elapsed_s: float
    last_wall_time: datetime
    message_count: int = 0
    parity_verified_count: int = 0
    callsign: str | None = None
    emitter_category: int | None = None
    latest_altitude_ft: int | None = None
    latest_ground_speed_kt: float | None = None
    latest_track_deg: float | None = None
    latest_vertical_rate_fpm: float | None = None
    latest_vertical_rate_source: str | None = None
    latest_air_ground: str | None = None
    latest_latitude_deg: float | None = None
    latest_longitude_deg: float | None = None
    latest_position_elapsed_s: float | None = None
    latest_position_source: str | None = None
    latest_position_reference: tuple[float, float] | None = None
    airborne_cpr_even: tuple[float, int, int] | None = None
    airborne_cpr_odd: tuple[float, int, int] | None = None
    position_history: list[tuple[float, float, float, int | None]] = field(
        default_factory=list
    )
    metadata: AircraftMetadata | None = None
    route: FlightRoute | None = None
    latest_power_dbm: float = float("nan")
    power_sum_mw: float = 0.0
    peak_power_dbm: float = -float("inf")
    latest_snr_db: float = float("nan")
    snr_sum_db: float = 0.0
    peak_snr_db: float = -float("inf")
    latest_correlation: float = float("nan")
    latest_raw_message: str = ""
    latest_fields: dict[str, object] = field(default_factory=dict)
    downlink_formats: set[int] = field(default_factory=set)
    type_codes: set[int] = field(default_factory=set)


class _ADSBRouteLookupThread(QtCore.QThread):
    route_ready = QtCore.Signal(str, object)
    route_failed = QtCore.Signal(str, str)

    def __init__(
        self,
        client: ADSBDBRouteClient,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._client = client
        self._queue: queue.Queue[str | None] = queue.Queue()

    def enqueue(self, callsign: str) -> None:
        self._queue.put(normalize_callsign(callsign))

    def stop(self) -> None:
        self.requestInterruption()
        self._queue.put(None)

    def run(self) -> None:
        while not self.isInterruptionRequested():
            callsign = self._queue.get()
            if callsign is None:
                break
            try:
                route = self._client.lookup(callsign)
            except Exception as error:
                self.route_failed.emit(callsign, str(error))
            else:
                self.route_ready.emit(callsign, route)


class _AircraftMetadataThread(QtCore.QThread):
    completed = QtCore.Signal(int)
    failed = QtCore.Signal(str)

    def __init__(
        self,
        database: AircraftMetadataDatabase,
        *,
        csv_path: str | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._database = database
        self._csv_path = csv_path

    def run(self) -> None:
        try:
            if self._csv_path is None:
                count = self._database.download_and_import()
            else:
                count = self._database.import_opensky_csv(self._csv_path)
        except Exception as error:
            self.failed.emit(str(error))
            return
        self.completed.emit(count)


class _ReceiverLocationDialog(QtWidgets.QDialog):
    MAP_RESULT = 2
    CLEAR_RESULT = 3

    def __init__(
        self,
        latitude: float | None,
        longitude: float | None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("ADS-B Receiver Location")
        layout = QtWidgets.QFormLayout(self)
        self.latitude_spin = QtWidgets.QDoubleSpinBox()
        self.latitude_spin.setRange(-90.0, 90.0)
        self.latitude_spin.setDecimals(6)
        self.latitude_spin.setSingleStep(0.0001)
        self.latitude_spin.setValue(0.0 if latitude is None else latitude)
        self.longitude_spin = QtWidgets.QDoubleSpinBox()
        self.longitude_spin.setRange(-180.0, 180.0)
        self.longitude_spin.setDecimals(6)
        self.longitude_spin.setSingleStep(0.0001)
        self.longitude_spin.setValue(0.0 if longitude is None else longitude)
        layout.addRow("Latitude (degree)", self.latitude_spin)
        layout.addRow("Longitude (degree)", self.longitude_spin)
        note = QtWidgets.QLabel(
            "Used as the Local CPR reference. Airborne targets must be within "
            "180 NM of this location."
        )
        note.setWordWrap(True)
        layout.addRow(note)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        map_button = buttons.addButton(
            "Select on Map...", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
        )
        clear_button = buttons.addButton(
            "Clear", QtWidgets.QDialogButtonBox.ButtonRole.ResetRole
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        map_button.clicked.connect(lambda: self.done(self.MAP_RESULT))
        clear_button.clicked.connect(lambda: self.done(self.CLEAR_RESULT))
        layout.addRow(buttons)

    @property
    def coordinates(self) -> tuple[float, float]:
        return self.latitude_spin.value(), self.longitude_spin.value()


class _ADSBPlutoCaptureThread(QtCore.QThread):
    capture_ready = QtCore.Signal(object)
    capture_failed = QtCore.Signal(str)
    capture_cancelled = QtCore.Signal()

    def __init__(
        self,
        source: PlutoLiveSource,
        settings: PlutoCaptureSettings,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._settings = settings

    def run(self) -> None:
        fresh = True
        stream_settings = replace(
            self._settings,
            capture_length_s=_STREAM_BLOCK_DURATION_S,
        )
        while not self.isInterruptionRequested():
            try:
                recording = self._source.capture_single(
                    stream_settings,
                    cancelled=self.isInterruptionRequested,
                    fresh=fresh,
                )
            except CaptureCancelledError:
                self.capture_cancelled.emit()
                break
            except Exception as error:
                self.capture_failed.emit(str(error))
                break
            self.capture_ready.emit(
                _ADSBCaptureBatch(recording=recording)
            )
            fresh = False


class _ADSBStreamProcessor:
    """Stateful ADS-B DSP pipeline with no dependency on Qt widgets."""

    def __init__(
        self,
        settings: PlutoCaptureSettings,
        analysis_settings: ADSB1090Settings,
        *,
        continuous: bool,
        scan_started_wall_time: datetime,
    ) -> None:
        self.settings = settings
        self.analysis_settings = analysis_settings
        self.continuous = bool(continuous)
        self.scan_started_wall_time = scan_started_wall_time
        self.analyzer = ADSB1090Analyzer()
        self.sample_rate_hz = float(settings.requested_sample_rate_hz)
        self.total_samples = 0
        self.ring_start_sample = 0
        self.ring_iq = np.empty(0, dtype=np.complex64)
        self.tail_start_sample = 0
        self.tail_iq = np.empty(0, dtype=np.complex64)
        self.last_reported_start_sample = -1
        self.single_trigger_sample: int | None = None
        self.single_messages: list[tuple[int, ADSB1090Message]] = []
        self.single_complete = False

    def process(self, recording: IQRecording) -> _ADSBStreamView | None:
        if self.single_complete:
            return None
        sample_rate_hz = float(recording.sample_rate_hz)
        if not np.isclose(self.sample_rate_hz, sample_rate_hz):
            raise ValueError("ADS-B stream sample rate changed during acquisition")
        block_iq = np.asarray(recording.iq, dtype=np.complex64)
        block_start = self.total_samples
        self.total_samples += block_iq.size
        self._append_ring(block_iq)

        if self.tail_iq.size:
            analysis_iq = np.concatenate((self.tail_iq, block_iq))
            analysis_start = self.tail_start_sample
        else:
            analysis_iq = block_iq
            analysis_start = block_start
        analysis_recording = replace(
            recording,
            iq=analysis_iq,
            start_sample_index=analysis_start,
            trigger_sample_index=None,
            source="VSA Pluto ADS-B Stream",
        )
        analysis_result = self.analyzer.analyze(
            analysis_recording,
            self.analysis_settings,
        )
        new_messages: list[tuple[int, ADSB1090Message]] = []
        for message in analysis_result.messages:
            absolute_start = analysis_start + message.start_sample
            if absolute_start <= self.last_reported_start_sample:
                continue
            new_messages.append((absolute_start, message))
        if new_messages:
            self.last_reported_start_sample = max(
                absolute_start for absolute_start, _message in new_messages
            )

        view: _ADSBStreamView | None = None
        if self.continuous and new_messages:
            view_start = max(
                self.ring_start_sample,
                self.total_samples - self.settings.capture_samples,
            )
            view = self._make_view(
                recording,
                view_start,
                self.total_samples,
                new_messages,
                append=True,
            )
        elif not self.continuous:
            if new_messages:
                self.single_messages.extend(new_messages)
                if self.single_trigger_sample is None:
                    self.single_trigger_sample = new_messages[0][0]
            if self.single_trigger_sample is not None:
                target_stop = (
                    self.single_trigger_sample + self.settings.capture_samples
                )
                if self.total_samples >= target_stop:
                    pretrigger = int(round(_SINGLE_PRETRIGGER_S * sample_rate_hz))
                    view_start = max(
                        self.ring_start_sample,
                        self.single_trigger_sample - pretrigger,
                    )
                    view = self._make_view(
                        recording,
                        view_start,
                        target_stop,
                        self.single_messages,
                        append=False,
                        single_complete=True,
                    )
                    self.single_complete = True

        overlap_samples = max(1, int(round(_STREAM_OVERLAP_S * sample_rate_hz)))
        keep = min(overlap_samples, analysis_iq.size)
        self.tail_iq = analysis_iq[-keep:].copy()
        self.tail_start_sample = self.total_samples - keep
        return view

    def _append_ring(self, block_iq: np.ndarray) -> None:
        if self.ring_iq.size:
            self.ring_iq = np.concatenate((self.ring_iq, block_iq))
        else:
            self.ring_iq = block_iq.copy()
        pretrigger = int(
            round(_SINGLE_PRETRIGGER_S * self.settings.requested_sample_rate_hz)
        )
        stream_block = int(
            round(_STREAM_BLOCK_DURATION_S * self.settings.requested_sample_rate_hz)
        )
        maximum = self.settings.capture_samples + pretrigger + 2 * stream_block
        excess = self.ring_iq.size - maximum
        if excess > 0:
            self.ring_iq = self.ring_iq[excess:].copy()
            self.ring_start_sample += excess

    def _make_view(
        self,
        template: IQRecording,
        start_sample: int,
        stop_sample: int,
        messages: list[tuple[int, ADSB1090Message]],
        *,
        append: bool,
        single_complete: bool = False,
    ) -> _ADSBStreamView | None:
        start_sample = max(start_sample, self.ring_start_sample)
        stop_sample = min(stop_sample, self.total_samples)
        lo = start_sample - self.ring_start_sample
        hi = stop_sample - self.ring_start_sample
        if hi <= lo:
            return None
        view_recording = replace(
            template,
            iq=self.ring_iq[lo:hi].copy(),
            start_sample_index=start_sample,
            trigger_sample_index=None,
            source="VSA Pluto ADS-B Stream",
        )
        relative_messages = tuple(
            replace(message, start_sample=absolute_start - start_sample)
            for absolute_start, message in messages
            if start_sample <= absolute_start < stop_sample
        )
        linear_power = (
            np.abs(view_recording.iq) / float(view_recording.full_scale)
        ) ** 2
        result = ADSB1090AnalysisResult(
            time_s=np.arange(view_recording.sample_count, dtype=np.float64)
            / view_recording.sample_rate_hz,
            power_dbfs=10.0
            * np.log10(np.maximum(linear_power, np.finfo(np.float64).tiny)),
            messages=relative_messages,
            metadata={
                "source": view_recording.source,
                "stream_start_sample": start_sample,
            },
        )
        elapsed_base_s = start_sample / view_recording.sample_rate_hz
        return _ADSBStreamView(
            recording=view_recording,
            result=result,
            append=append,
            elapsed_base_s=elapsed_base_s,
            capture_started_at=self.scan_started_wall_time
            + timedelta(seconds=elapsed_base_s),
            single_complete=single_complete,
        )


class _ADSBStreamAnalysisThread(QtCore.QThread):
    """Consume IQ blocks away from the GUI thread and emit display snapshots."""

    view_ready = QtCore.Signal(object)
    analysis_failed = QtCore.Signal(str)

    def __init__(
        self,
        processor: _ADSBStreamProcessor,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.processor = processor
        self._queue: queue.Queue[IQRecording | None] = queue.Queue()
        self._accepting = True

    def enqueue(self, payload: object) -> None:
        if not self._accepting:
            return
        if not isinstance(payload, _ADSBCaptureBatch):
            self.analysis_failed.emit("capture returned an invalid IQ recording")
            return
        self._queue.put(payload.recording)

    def stop(self) -> None:
        self._accepting = False
        self.requestInterruption()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(None)

    def run(self) -> None:
        while not self.isInterruptionRequested():
            recording = self._queue.get()
            if recording is None:
                break
            try:
                view = self.processor.process(recording)
            except Exception as error:
                self.analysis_failed.emit(str(error))
                break
            if view is not None:
                self.view_ready.emit(view)
                if view.single_complete:
                    break
        self._accepting = False


class ADSB1090Window(QtWidgets.QMainWindow):
    """Protocol workspace kept separate from generic modulation analysis."""

    analysis_mode_requested = QtCore.Signal(str)
    application_close_requested = QtCore.Signal()
    shutdown_ready = QtCore.Signal()

    def __init__(
        self,
        recording: IQRecording | None = None,
        pluto_source: PlutoLiveSource | None = None,
        owns_pluto_source: bool = True,
        preferences: QtCore.QSettings | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Pluto VSA - ADS-B 1090ES")
        self.resize(1400, 850)
        self.recording = recording
        self.result: ADSB1090AnalysisResult | None = None
        self._analyzer = ADSB1090Analyzer()
        self._pluto_source = pluto_source or PlutoLiveSource()
        self._owns_pluto_source = bool(owns_pluto_source)
        self._pluto_target = ""
        # ADS-B receiver settings intentionally use their own application
        # store.  They must not inherit the generic VSA Pluto frontend setup.
        self._preferences = preferences or QtCore.QSettings(
            "PlutoSA", "PlutoVSA-ADSB1090"
        )
        metadata_path = str(
            self._preferences.value("metadata/sqlite_path", "")
        ).strip()
        if not metadata_path:
            application_data = QtCore.QStandardPaths.writableLocation(
                QtCore.QStandardPaths.StandardLocation.AppDataLocation
            )
            metadata_path = str(Path(application_data) / "aircraft_metadata.sqlite")
        self._aircraft_metadata_database = AircraftMetadataDatabase(metadata_path)
        self._aircraft_metadata_thread: _AircraftMetadataThread | None = None
        self._route_client = ADSBDBRouteClient()
        self._route_lookup_thread: _ADSBRouteLookupThread | None = None
        self._route_cache: dict[str, FlightRoute | None] = {}
        self._route_pending: set[str] = set()
        self._capture_thread: _ADSBPlutoCaptureThread | None = None
        self._analysis_stream_thread: _ADSBStreamAnalysisThread | None = None
        self._sync_stream_processor: _ADSBStreamProcessor | None = None
        self._packet_history: list[_ADSBPacketEntry] = []
        self._aircraft_states: dict[str, _ADSBAircraftState] = {}
        self._aircraft_row_by_icao: dict[str, int] = {}
        self._receiver_latitude_deg: float | None = None
        self._receiver_longitude_deg: float | None = None
        self._continuous_scan = False
        self._scan_started_wall_time: datetime | None = None
        self._stream_sample_rate_hz: float | None = None
        self._stream_total_samples = 0
        self._stream_ring_start_sample = 0
        self._stream_ring_iq = np.empty(0, dtype=np.complex64)
        self._stream_tail_start_sample = 0
        self._stream_tail_iq = np.empty(0, dtype=np.complex64)
        self._last_reported_start_sample = -1
        self._single_trigger_sample: int | None = None
        self._single_messages: list[tuple[int, ADSB1090Message]] = []
        self._single_complete = False
        self._plot_initial_ranges: dict[
            str, tuple[list[float], list[float]]
        ] = {}
        self._plot_context_actions: dict[str, dict[str, QtGui.QAction]] = {}
        self._closing = False
        self._shutdown_finalized = False
        self._shutdown_ready_emitted = False
        self._packet_selection_connected = False
        self._aircraft_selection_connected = False
        self._dock_resize_pending = False
        self._pending_stream_views: list[_ADSBStreamView] = []
        self._stream_display_timer = QtCore.QTimer(self)
        self._stream_display_timer.setSingleShot(True)
        self._stream_display_timer.setInterval(_STREAM_DISPLAY_BATCH_MS)
        self._stream_display_timer.timeout.connect(self._flush_stream_views)
        self._build_menu()
        self._build_ui()
        self._restore_user_settings()
        self._connect_user_setting_persistence()
        if recording is not None:
            self.analyze_recording(recording)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_action = file_menu.addAction("Open IQ...")
        open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_iq)
        self.export_packet_list_action = file_menu.addAction("Export Packet List...")
        self.export_packet_list_action.triggered.connect(self._export_packet_list)
        metadata_menu = file_menu.addMenu("Aircraft Database")
        self.import_aircraft_database_action = metadata_menu.addAction(
            "Import OpenSky CSV..."
        )
        self.import_aircraft_database_action.triggered.connect(
            self._import_aircraft_database
        )
        self.update_aircraft_database_action = metadata_menu.addAction(
            "Download / Update from OpenSky..."
        )
        self.update_aircraft_database_action.triggered.connect(
            self._download_aircraft_database
        )
        file_menu.addSeparator()
        close_action = file_menu.addAction("Close")
        close_action.triggered.connect(self.application_close_requested.emit)
        run_menu = self.menuBar().addMenu("Sweep / Run")
        self.run_single_action = run_menu.addAction("Run Single (Pluto)")
        self.run_single_action.setShortcut("F6")
        self.run_single_action.triggered.connect(self._run_pluto_single)
        self.run_continuous_action = run_menu.addAction("Run Continuous (Pluto)")
        self.run_continuous_action.setShortcut("F7")
        self.run_continuous_action.triggered.connect(self._run_pluto_continuous)
        refresh = run_menu.addAction("Refresh Analysis")
        refresh.setShortcut("F5")
        refresh.triggered.connect(self._refresh)
        mode_menu = self.menuBar().addMenu("Analysis Mode")
        generic_action = mode_menu.addAction("Generic FSK / PSK VSA")
        generic_action.triggered.connect(
            lambda: self.analysis_mode_requested.emit("generic")
        )
        bluetooth_action = mode_menu.addAction("Bluetooth Dedicated Analyzer...")
        bluetooth_action.triggered.connect(
            lambda: self.analysis_mode_requested.emit("bluetooth")
        )
        mode_menu.addSeparator()
        adsb_action = mode_menu.addAction("ADS-B 1090ES")
        adsb_action.setCheckable(True)
        adsb_action.setChecked(True)
        adsb_action.setEnabled(False)
        mode_menu.addSeparator()
        receiver_action = mode_menu.addAction("Receiver Location...")
        receiver_action.triggered.connect(self._edit_receiver_location)

    def _dock(self, title: str, widget: QtWidgets.QWidget) -> QtWidgets.QDockWidget:
        return make_measurement_dock(
            title,
            widget,
            self,
            object_prefix="adsb",
            closable=False,
        )

    def _build_ui(self) -> None:
        self.setCentralWidget(QtWidgets.QWidget())
        toolbar = QtWidgets.QToolBar("1090ES Capture", self)
        toolbar.setMovable(False)
        toolbar.addWidget(QtWidgets.QLabel("Center: 1090 MHz   Fs:"))
        self.sample_rate_combo = QtWidgets.QComboBox()
        self.sample_rate_combo.addItem("8 MS/s", 8)
        self.sample_rate_combo.addItem("16 MS/s", 16)
        toolbar.addWidget(self.sample_rate_combo)
        toolbar.addWidget(QtWidgets.QLabel("   Capture:"))
        self.capture_length_spin = QtWidgets.QDoubleSpinBox()
        self.capture_length_spin.setRange(1.0, 2000.0)
        self.capture_length_spin.setValue(250.0)
        self.capture_length_spin.setSuffix(" ms")
        toolbar.addWidget(self.capture_length_spin)
        toolbar.addWidget(QtWidgets.QLabel("   Internal Gain:"))
        self.internal_gain_spin = QtWidgets.QDoubleSpinBox()
        self.internal_gain_spin.setRange(0.0, 70.0)
        self.internal_gain_spin.setValue(50.0)
        self.internal_gain_spin.setSuffix(" dB")
        toolbar.addWidget(self.internal_gain_spin)
        toolbar.addWidget(QtWidgets.QLabel("   Preamble SNR Threshold:"))
        self.preamble_snr_spin = QtWidgets.QDoubleSpinBox()
        self.preamble_snr_spin.setRange(-20.0, 40.0)
        self.preamble_snr_spin.setDecimals(1)
        self.preamble_snr_spin.setSingleStep(0.5)
        self.preamble_snr_spin.setValue(5.0)
        self.preamble_snr_spin.setSuffix(" dB")
        self.preamble_snr_spin.setToolTip(
            "Minimum pulse/quiet power ratio accepted for the 8 us Mode S preamble. "
            "This is a receiver detection threshold, not a fixed ICAO limit."
        )
        toolbar.addWidget(self.preamble_snr_spin)
        toolbar.addSeparator()
        self.receiver_location_button = QtWidgets.QToolButton()
        self.receiver_location_button.setText("Receiver: Not set")
        self.receiver_location_button.setToolTip(
            "Set the reference latitude/longitude used for Local CPR decoding"
        )
        self.receiver_location_button.clicked.connect(self._edit_receiver_location)
        toolbar.addWidget(self.receiver_location_button)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, toolbar)
        self.power_plot = make_measurement_plot(
            "IQ Power (dBm)", "Measurement Elapsed Time (ms)"
        )
        self.ppm_plot = make_measurement_plot(
            "First / Second Chip Power (dB)", "Data Bit Index"
        )
        self.packet_table = QtWidgets.QTableWidget(0, 11)
        self.packet_table.setHorizontalHeaderLabels(
            [
                "#",
                "Elapsed (s)",
                "OS Time",
                "Raw Message",
                "DF",
                "ICAO",
                "TC",
                "Parity / CRC",
                "ON Power (dBm)",
                "SNR (dB)",
                "Decoded",
            ]
        )
        self.packet_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.packet_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.packet_table.verticalHeader().setVisible(False)
        packet_header = self.packet_table.horizontalHeader()
        for column in range(10):
            packet_header.setSectionResizeMode(
                column,
                QtWidgets.QHeaderView.ResizeMode.Interactive,
            )
        for column, width in enumerate(
            (45, 105, 205, 235, 45, 80, 45, 105, 110, 75)
        ):
            self.packet_table.setColumnWidth(column, width)
        packet_header.setSectionResizeMode(
            10,
            QtWidgets.QHeaderView.ResizeMode.Stretch,
        )
        self.packet_table.itemSelectionChanged.connect(self._selected_packet_changed)
        self._packet_selection_connected = True
        self.packet_table.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.packet_table.customContextMenuRequested.connect(
            self._show_packet_table_context_menu
        )
        self.summary_table = QtWidgets.QTableWidget(0, 2)
        self.summary_table.setHorizontalHeaderLabels(["Parameter", "Current"])
        summary_header = self.summary_table.horizontalHeader()
        summary_header.setSectionResizeMode(
            0,
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents,
        )
        summary_header.setSectionResizeMode(
            1,
            QtWidgets.QHeaderView.ResizeMode.Stretch,
        )
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.aircraft_table = QtWidgets.QTableWidget(0, 14)
        self.aircraft_table.setHorizontalHeaderLabels(
            [
                "ICAO",
                "Callsign",
                "Air/Ground",
                "Latitude",
                "Longitude",
                "Last Seen (s)",
                "Messages",
                "Altitude (ft / m)",
                "Speed (kt / km/h)",
                "V/Rate (ft/min / m/s)",
                "Power (dBm)",
                "SNR (dB)",
                "Origin",
                "Destination",
            ]
        )
        self.aircraft_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.aircraft_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.aircraft_table.verticalHeader().setVisible(False)
        aircraft_header = self.aircraft_table.horizontalHeader()
        for column in range(14):
            aircraft_header.setSectionResizeMode(
                column,
                QtWidgets.QHeaderView.ResizeMode.Interactive,
            )
        for column, width in enumerate(
            (75, 100, 85, 95, 95, 105, 80, 130, 145, 165, 100, 80, 90, 90)
        ):
            self.aircraft_table.setColumnWidth(column, width)
        aircraft_header.setSectionResizeMode(
            1,
            QtWidgets.QHeaderView.ResizeMode.Stretch,
        )
        self.aircraft_table.itemSelectionChanged.connect(
            self._selected_aircraft_changed
        )
        self._aircraft_selection_connected = True
        self.aircraft_summary_table = QtWidgets.QTableWidget(0, 2)
        self.aircraft_summary_table.setHorizontalHeaderLabels(
            ["Parameter", "Current"]
        )
        aircraft_summary_header = self.aircraft_summary_table.horizontalHeader()
        aircraft_summary_header.setSectionResizeMode(
            0,
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents,
        )
        aircraft_summary_header.setSectionResizeMode(
            1,
            QtWidgets.QHeaderView.ResizeMode.Stretch,
        )
        self.aircraft_summary_table.verticalHeader().setVisible(False)
        self.aircraft_summary_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.aircraft_map = LeafletAircraftMap()
        self.aircraft_map.receiver_location_selected.connect(
            self._receiver_location_picked
        )
        self.aircraft_detail_tabs = QtWidgets.QTabWidget()
        self.aircraft_detail_tabs.addTab(self.aircraft_summary_table, "Details")
        self.aircraft_detail_tabs.addTab(self.aircraft_map, "Position History")
        self.aircraft_detail_tabs.currentChanged.connect(
            self._aircraft_detail_tab_changed
        )
        # Keep Python references as well as Qt parentage.  PySide can otherwise
        # collect a locally-created QDockWidget and delete its child ViewBox while
        # the PlotWidget wrapper is still reachable from this window.
        self.power_dock = self._dock("IQ Power", self.power_plot)
        self.ppm_dock = self._dock("PPM Demodulation", self.ppm_plot)
        self.packet_dock = self._dock("Packet List", self.packet_table)
        self.summary_dock = self._dock("Message Summary", self.summary_table)
        self.aircraft_dock = self._dock("Detected Aircraft", self.aircraft_table)
        self.aircraft_summary_dock = self._dock(
            "Aircraft Details", self.aircraft_detail_tabs
        )
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.power_dock)
        self.splitDockWidget(
            self.power_dock,
            self.packet_dock,
            QtCore.Qt.Orientation.Horizontal,
        )
        self.splitDockWidget(
            self.packet_dock,
            self.aircraft_dock,
            QtCore.Qt.Orientation.Horizontal,
        )
        self.splitDockWidget(
            self.power_dock,
            self.ppm_dock,
            QtCore.Qt.Orientation.Vertical,
        )
        self.splitDockWidget(
            self.packet_dock,
            self.summary_dock,
            QtCore.Qt.Orientation.Vertical,
        )
        self.splitDockWidget(
            self.aircraft_dock,
            self.aircraft_summary_dock,
            QtCore.Qt.Orientation.Vertical,
        )
        self._configure_plot_context_menus()
        for widget in (
            self.power_plot,
            self.ppm_plot,
            self.packet_table,
            self.summary_table,
            self.aircraft_table,
            self.aircraft_detail_tabs,
        ):
            widget.setMinimumSize(0, 0)
        for dock in (
            self.power_dock,
            self.ppm_dock,
            self.packet_dock,
            self.summary_dock,
            self.aircraft_dock,
            self.aircraft_summary_dock,
        ):
            dock.setMinimumSize(0, 0)
        QtCore.QTimer.singleShot(0, self._equalize_result_docks)
        self.statusBar().showMessage("Ready - load 1090 MHz IQ or pass the current VSA capture")

    def _equalize_result_docks(self) -> None:
        self._dock_resize_pending = False
        top_row = (self.power_dock, self.packet_dock, self.aircraft_dock)
        bottom_row = (
            self.ppm_dock,
            self.summary_dock,
            self.aircraft_summary_dock,
        )
        column_width = max(self.width() // 3, 1)
        self.resizeDocks(
            list(top_row), [column_width] * 3, QtCore.Qt.Orientation.Horizontal
        )
        self.resizeDocks(
            list(bottom_row), [column_width] * 3, QtCore.Qt.Orientation.Horizontal
        )
        for upper, lower in zip(top_row, bottom_row):
            self.resizeDocks(
                [upper, lower], [400, 400], QtCore.Qt.Orientation.Vertical
            )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if hasattr(self, "power_dock") and not self._dock_resize_pending:
            self._dock_resize_pending = True
            QtCore.QTimer.singleShot(0, self._equalize_result_docks)

    def _restore_user_settings(self) -> None:
        sample_rate_msps = int(
            self._preferences.value("capture/sample_rate_msps", 8, type=int)
        )
        sample_rate_index = self.sample_rate_combo.findData(sample_rate_msps)
        if sample_rate_index >= 0:
            self.sample_rate_combo.setCurrentIndex(sample_rate_index)
        self.capture_length_spin.setValue(
            float(self._preferences.value("capture/length_ms", 250.0, type=float))
        )
        self.internal_gain_spin.setValue(
            float(self._preferences.value("capture/internal_gain_db", 50.0, type=float))
        )
        self.preamble_snr_spin.setValue(
            float(self._preferences.value("detection/preamble_snr_db", 5.0, type=float))
        )
        if self._preferences.contains("receiver/latitude_deg") and self._preferences.contains(
            "receiver/longitude_deg"
        ):
            latitude = float(
                self._preferences.value("receiver/latitude_deg", 0.0, type=float)
            )
            longitude = float(
                self._preferences.value("receiver/longitude_deg", 0.0, type=float)
            )
            if -90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0:
                self._set_receiver_location(latitude, longitude, persist=False)
        else:
            self._update_receiver_location_button()

    def _connect_user_setting_persistence(self) -> None:
        self.sample_rate_combo.currentIndexChanged.connect(self._save_user_settings)
        self.capture_length_spin.valueChanged.connect(self._save_user_settings)
        self.internal_gain_spin.valueChanged.connect(self._save_user_settings)
        self.preamble_snr_spin.valueChanged.connect(self._save_user_settings)

    @QtCore.Slot()
    def _save_user_settings(self) -> None:
        self._preferences.setValue(
            "capture/sample_rate_msps", int(self.sample_rate_combo.currentData())
        )
        self._preferences.setValue(
            "capture/length_ms", float(self.capture_length_spin.value())
        )
        self._preferences.setValue(
            "capture/internal_gain_db", float(self.internal_gain_spin.value())
        )
        self._preferences.setValue(
            "detection/preamble_snr_db", float(self.preamble_snr_spin.value())
        )
        if (
            self._receiver_latitude_deg is None
            or self._receiver_longitude_deg is None
        ):
            self._preferences.remove("receiver/latitude_deg")
            self._preferences.remove("receiver/longitude_deg")
        else:
            self._preferences.setValue(
                "receiver/latitude_deg", self._receiver_latitude_deg
            )
            self._preferences.setValue(
                "receiver/longitude_deg", self._receiver_longitude_deg
            )
        self._preferences.sync()

    def _edit_receiver_location(self) -> None:
        dialog = _ReceiverLocationDialog(
            self._receiver_latitude_deg,
            self._receiver_longitude_deg,
            self,
        )
        result = dialog.exec()
        if result == QtWidgets.QDialog.DialogCode.Accepted:
            self._set_receiver_location(*dialog.coordinates)
        elif result == _ReceiverLocationDialog.CLEAR_RESULT:
            self._set_receiver_location(None, None)
        elif result == _ReceiverLocationDialog.MAP_RESULT:
            self.aircraft_detail_tabs.setCurrentIndex(1)
            self.aircraft_map.begin_receiver_selection()
            self.statusBar().showMessage(
                "Receiver location selection: click the Position History map"
            )

    @QtCore.Slot(float, float)
    def _receiver_location_picked(self, latitude: float, longitude: float) -> None:
        self.aircraft_map.cancel_receiver_selection()
        answer = QtWidgets.QMessageBox.question(
            self,
            "ADS-B Receiver Location",
            f"Use this receiver location?\n\n"
            f"Latitude:  {latitude:.6f}\nLongitude: {longitude:.6f}",
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self._set_receiver_location(latitude, longitude)
            self.statusBar().showMessage(
                f"Receiver location set - {latitude:.6f}, {longitude:.6f}"
            )
        else:
            self.aircraft_map.set_receiver_location(
                self._receiver_latitude_deg, self._receiver_longitude_deg
            )
            self.statusBar().showMessage("Receiver location selection cancelled")

    def _set_receiver_location(
        self,
        latitude: float | None,
        longitude: float | None,
        *,
        persist: bool = True,
    ) -> None:
        if latitude is None or longitude is None:
            self._receiver_latitude_deg = None
            self._receiver_longitude_deg = None
        else:
            latitude = float(latitude)
            longitude = float(longitude)
            if not (-90.0 <= latitude <= 90.0):
                raise ValueError("receiver latitude must be between -90 and 90")
            if not (-180.0 <= longitude <= 180.0):
                raise ValueError("receiver longitude must be between -180 and 180")
            self._receiver_latitude_deg = latitude
            self._receiver_longitude_deg = longitude
        self.aircraft_map.set_receiver_location(
            self._receiver_latitude_deg, self._receiver_longitude_deg
        )
        self._update_receiver_location_button()
        if self._receiver_latitude_deg is not None:
            self._decode_stored_cpr_with_receiver()
        if persist:
            self._save_user_settings()

    def _update_receiver_location_button(self) -> None:
        if (
            self._receiver_latitude_deg is None
            or self._receiver_longitude_deg is None
        ):
            self.receiver_location_button.setText("Receiver: Not set")
            return
        self.receiver_location_button.setText(
            f"RX: {self._receiver_latitude_deg:.4f}, "
            f"{self._receiver_longitude_deg:.4f}"
        )

    def _decode_stored_cpr_with_receiver(self) -> None:
        """Apply a newly configured receiver to offline/saved single frames."""

        if (
            self._receiver_latitude_deg is None
            or self._receiver_longitude_deg is None
        ):
            return
        updated: list[_ADSBAircraftState] = []
        for state in self._aircraft_states.values():
            if state.latest_latitude_deg is not None:
                continue
            frames = [
                (False, state.airborne_cpr_even),
                (True, state.airborne_cpr_odd),
            ]
            available = [
                (is_odd, frame) for is_odd, frame in frames if frame is not None
            ]
            if not available:
                continue
            is_odd, frame = max(available, key=lambda item: item[1][0])
            position = decode_local_airborne_cpr(
                frame[1],
                frame[2],
                is_odd=is_odd,
                reference_latitude_deg=self._receiver_latitude_deg,
                reference_longitude_deg=self._receiver_longitude_deg,
            )
            if position is None or _great_circle_distance_nm(
                self._receiver_latitude_deg,
                self._receiver_longitude_deg,
                position[0],
                position[1],
            ) > _MAX_LOCAL_CPR_RANGE_NM:
                continue
            state.latest_latitude_deg, state.latest_longitude_deg = position
            state.latest_position_elapsed_s = frame[0]
            state.latest_position_source = "Local CPR (Receiver)"
            state.latest_position_reference = (
                self._receiver_latitude_deg,
                self._receiver_longitude_deg,
            )
            state.position_history.append(
                (frame[0], position[0], position[1], state.latest_altitude_ft)
            )
            updated.append(state)
        for state in updated:
            self._update_aircraft_row(state)
        selected_icao = self._selected_aircraft_icao()
        if selected_icao is not None and selected_icao in self._aircraft_states:
            self._show_aircraft_summary(self._aircraft_states[selected_icao])

    def _open_iq(self) -> None:
        directory = str(self._preferences.value("paths/iq_directory", ""))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open 1090 MHz IQ",
            directory,
            "IQ Recordings (*.iq.tar *.npz);;All Files (*)",
        )
        if not path:
            return
        try:
            recording = FileIQSource.load(path)
            self.analyze_recording(recording)
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "ADS-B 1090ES", str(error))
            return
        self._preferences.setValue(
            "paths/iq_directory", str(Path(path).resolve().parent)
        )

    def _import_aircraft_database(self) -> None:
        directory = str(self._preferences.value("paths/metadata_directory", ""))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import OpenSky Aircraft Database",
            directory,
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        self._preferences.setValue(
            "paths/metadata_directory", str(Path(path).resolve().parent)
        )
        self._start_aircraft_database_update(csv_path=path)

    def _download_aircraft_database(self) -> None:
        answer = QtWidgets.QMessageBox.question(
            self,
            "Update Aircraft Database",
            "Download the OpenSky aircraft metadata CSV and rebuild the local "
            "lookup database?\n\nOpenSky states that this snapshot is not current "
            "and is provided without guarantees.",
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._start_aircraft_database_update(csv_path=None)

    def _start_aircraft_database_update(self, *, csv_path: str | None) -> None:
        if self._aircraft_metadata_thread is not None:
            self.statusBar().showMessage("Aircraft database update is already running")
            return
        self.import_aircraft_database_action.setEnabled(False)
        self.update_aircraft_database_action.setEnabled(False)
        self.statusBar().showMessage(
            "Importing aircraft metadata..."
            if csv_path is not None
            else "Downloading and indexing OpenSky aircraft metadata..."
        )
        thread = _AircraftMetadataThread(
            self._aircraft_metadata_database,
            csv_path=csv_path,
            parent=self,
        )
        thread.completed.connect(self._aircraft_database_updated)
        thread.failed.connect(self._aircraft_database_update_failed)
        thread.finished.connect(self._aircraft_database_thread_finished)
        self._aircraft_metadata_thread = thread
        thread.start()

    @QtCore.Slot(int)
    def _aircraft_database_updated(self, count: int) -> None:
        self._preferences.setValue(
            "metadata/sqlite_path", str(self._aircraft_metadata_database.path)
        )
        for state in self._aircraft_states.values():
            state.metadata = self._aircraft_metadata_database.lookup(
                state.icao_address
            )
            self._update_aircraft_row(state)
        selected_icao = self._selected_aircraft_icao()
        self._show_aircraft_summary(
            self._aircraft_states.get(selected_icao)
            if selected_icao is not None
            else None
        )
        self.statusBar().showMessage(
            f"Aircraft database ready - {count:,} records indexed"
        )

    @QtCore.Slot(str)
    def _aircraft_database_update_failed(self, message: str) -> None:
        self.statusBar().showMessage("Aircraft database update failed")
        QtWidgets.QMessageBox.critical(self, "Aircraft Database", message)

    @QtCore.Slot()
    def _aircraft_database_thread_finished(self) -> None:
        thread = self._aircraft_metadata_thread
        self._aircraft_metadata_thread = None
        self.import_aircraft_database_action.setEnabled(True)
        self.update_aircraft_database_action.setEnabled(True)
        if thread is not None:
            thread.deleteLater()

    def _show_packet_table_context_menu(self, position: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self.packet_table)
        menu.addAction(self.export_packet_list_action)
        menu.exec(self.packet_table.viewport().mapToGlobal(position))

    def _export_packet_list(self) -> None:
        if not self._packet_history:
            self.statusBar().showMessage("Packet List is empty - nothing to export")
            return
        directory = str(self._preferences.value("paths/export_directory", ""))
        suggested = str(Path(directory) / "adsb1090_packets.jsonl") if directory else "adsb1090_packets.jsonl"
        path_text, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export ADS-B Packet List",
            suggested,
            "JSON Lines (*.jsonl);;All Files (*)",
        )
        if not path_text:
            return
        path = Path(path_text)
        if not path.suffix:
            path = path.with_suffix(".jsonl")
        try:
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                for index, entry in enumerate(self._packet_history, start=1):
                    record = self._packet_export_record(index, entry)
                    stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                    stream.write("\n")
        except OSError as error:
            QtWidgets.QMessageBox.critical(self, "Packet List Export Error", str(error))
            return
        self._preferences.setValue(
            "paths/export_directory", str(path.resolve().parent)
        )
        self.statusBar().showMessage(
            f"Exported {len(self._packet_history)} ADS-B packets to {path.name}"
        )

    @staticmethod
    def _packet_export_record(index: int, entry: _ADSBPacketEntry) -> dict[str, object]:
        message = entry.message
        return {
            "schema": "pluto-vsa.adsb1090.packet",
            "version": 1,
            "index": int(index),
            "elapsed_s": float(entry.elapsed_s),
            "os_time": entry.wall_time.isoformat(timespec="microseconds"),
            "raw_message": message.raw_hex,
            "bit_length": message.bit_length,
            "downlink_format": message.downlink_format,
            "icao_address": message.icao_address,
            "icao_address_source": message.icao_address_source,
            "icao_confirmed": message.icao_confirmed,
            "type_code": message.type_code,
            "parity": {
                "kind": message.parity_kind,
                "display": message.parity_display,
                "remainder_hex": f"{message.crc_remainder:06X}",
                "verified": message.parity_ok,
                "interrogator_identifier": message.interrogator_identifier,
            },
            "mean_on_pulse_power_dbm": float(entry.on_pulse_power_dbm),
            "preamble_snr_db": float(message.preamble_snr_db),
            "preamble_correlation": float(message.preamble_correlation),
            "decoded_fields": ADSB1090Window._json_compatible(dict(message.fields)),
        }

    @staticmethod
    def _json_compatible(value: object) -> object:
        if isinstance(value, dict):
            return {
                str(key): ADSB1090Window._json_compatible(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [ADSB1090Window._json_compatible(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _refresh(self) -> None:
        if self.recording is not None:
            self.analyze_recording(self.recording)

    def _pluto_settings(self) -> PlutoCaptureSettings:
        sample_rate_msps = int(self.sample_rate_combo.currentData())
        return PlutoCaptureSettings(
            center_frequency_hz=1_090_000_000.0,
            symbol_rate_hz=1_000_000.0,
            samples_per_symbol=sample_rate_msps,
            capture_length_s=self.capture_length_spin.value() / 1e3,
            rf_bandwidth_hz=4_000_000.0,
            sdr_uri=self._pluto_target or None,
            power_correction=InputPowerCorrection(
                internal_gain_db=self.internal_gain_spin.value(),
                external_attenuation_db=0.0,
                external_gain_db=0.0,
            ),
        )

    def set_pluto_target(self, target: str | None) -> None:
        """Share the shell's selected physical receiver with ADS-B mode."""

        self._pluto_target = str(target or "").strip()

    def _analysis_settings(self) -> ADSB1090Settings:
        return ADSB1090Settings(
            minimum_preamble_snr_db=self.preamble_snr_spin.value(),
        )

    def _run_pluto_single(self) -> None:
        self._run_pluto(continuous=False)

    def _run_pluto_continuous(self) -> None:
        self._run_pluto(continuous=True)

    def _run_pluto(self, *, continuous: bool) -> None:
        if self._capture_thread is not None and self._capture_thread.isRunning():
            self._capture_thread.requestInterruption()
            self.run_single_action.setEnabled(False)
            self.run_continuous_action.setEnabled(False)
            self.statusBar().showMessage("Stopping Pluto capture...")
            return
        try:
            settings = self._pluto_settings()
        except ValueError as error:
            QtWidgets.QMessageBox.critical(self, "Pluto Capture", str(error))
            return
        self._continuous_scan = bool(continuous)
        self._scan_started_wall_time = datetime.now().astimezone()
        self._clear_packet_history()
        self._reset_stream_state(settings.requested_sample_rate_hz)
        if continuous:
            self.run_continuous_action.setText("Stop Continuous")
            self.run_single_action.setEnabled(False)
        else:
            self.run_single_action.setText("Stop Single")
            self.run_continuous_action.setEnabled(False)
        self.statusBar().showMessage(
            f"{'Continuously scanning' if continuous else 'Capturing'} 1090 MHz - "
            f"{settings.requested_sample_rate_hz / 1e6:.0f} MS/s, "
            f"{settings.capture_length_s * 1e3:.0f} ms"
        )
        thread = _ADSBPlutoCaptureThread(
            self._pluto_source,
            settings,
            parent=self,
        )
        processor = _ADSBStreamProcessor(
            settings,
            self._analysis_settings(),
            continuous=continuous,
            scan_started_wall_time=self._scan_started_wall_time,
        )
        analysis_thread = _ADSBStreamAnalysisThread(processor, parent=self)
        analysis_thread.view_ready.connect(self._stream_analysis_ready)
        analysis_thread.analysis_failed.connect(self._pluto_capture_failed)
        analysis_thread.finished.connect(self._stream_analysis_stopped)
        analysis_thread.finished.connect(analysis_thread.deleteLater)
        self._analysis_stream_thread = analysis_thread
        thread.capture_ready.connect(analysis_thread.enqueue)
        thread.capture_failed.connect(self._pluto_capture_failed)
        thread.capture_cancelled.connect(
            lambda: self.statusBar().showMessage("Pluto capture cancelled")
        )
        thread.finished.connect(self._pluto_capture_stopped)
        thread.finished.connect(thread.deleteLater)
        self._capture_thread = thread
        analysis_thread.start()
        thread.start()

    def _reset_stream_state(self, sample_rate_hz: float) -> None:
        self._stream_sample_rate_hz = float(sample_rate_hz)
        self._stream_total_samples = 0
        self._stream_ring_start_sample = 0
        self._stream_ring_iq = np.empty(0, dtype=np.complex64)
        self._stream_tail_start_sample = 0
        self._stream_tail_iq = np.empty(0, dtype=np.complex64)
        self._last_reported_start_sample = -1
        self._single_trigger_sample = None
        self._single_messages.clear()
        self._single_complete = False
        self._sync_stream_processor = None

    def _pluto_capture_ready(self, payload: object) -> None:
        if not isinstance(payload, _ADSBCaptureBatch):
            self._pluto_capture_failed("capture returned an invalid IQ recording")
            return
        self._process_stream_block(payload.recording)

    def _process_stream_block(self, recording: IQRecording) -> None:
        if self._single_complete:
            return
        sample_rate_hz = float(recording.sample_rate_hz)
        if (
            self._stream_sample_rate_hz is None
            or not np.isclose(self._stream_sample_rate_hz, sample_rate_hz)
        ):
            self._reset_stream_state(sample_rate_hz)
        block_iq = np.asarray(recording.iq, dtype=np.complex64)
        block_start = self._stream_total_samples
        self._stream_total_samples += block_iq.size
        self._append_stream_ring(block_iq)

        if self._stream_tail_iq.size:
            analysis_iq = np.concatenate((self._stream_tail_iq, block_iq))
            analysis_start = self._stream_tail_start_sample
        else:
            analysis_iq = block_iq
            analysis_start = block_start
        analysis_recording = replace(
            recording,
            iq=analysis_iq,
            start_sample_index=analysis_start,
            trigger_sample_index=None,
            source="VSA Pluto ADS-B Stream",
        )
        try:
            analysis_result = self._analyzer.analyze(
                analysis_recording,
                self._analysis_settings(),
            )
        except Exception as error:
            self._pluto_capture_failed(str(error))
            if self._capture_thread is not None:
                self._capture_thread.requestInterruption()
            return

        new_messages: list[tuple[int, ADSB1090Message]] = []
        for message in analysis_result.messages:
            absolute_start = analysis_start + message.start_sample
            if absolute_start <= self._last_reported_start_sample:
                continue
            new_messages.append((absolute_start, message))
        if new_messages:
            self._last_reported_start_sample = max(
                absolute_start for absolute_start, _message in new_messages
            )
            if self._continuous_scan:
                display_samples = self._pluto_settings().capture_samples
                view_start = max(
                    self._stream_ring_start_sample,
                    self._stream_total_samples - display_samples,
                )
                self._display_stream_view(
                    recording,
                    view_start,
                    self._stream_total_samples,
                    new_messages,
                    append=True,
                )
            else:
                self._single_messages.extend(new_messages)
                if self._single_trigger_sample is None:
                    self._single_trigger_sample = new_messages[0][0]

        if not self._continuous_scan and self._single_trigger_sample is not None:
            target_stop = (
                self._single_trigger_sample + self._pluto_settings().capture_samples
            )
            if self._stream_total_samples >= target_stop:
                pretrigger = int(round(_SINGLE_PRETRIGGER_S * sample_rate_hz))
                view_start = max(
                    self._stream_ring_start_sample,
                    self._single_trigger_sample - pretrigger,
                )
                self._display_stream_view(
                    recording,
                    view_start,
                    target_stop,
                    self._single_messages,
                    append=False,
                )
                self._single_complete = True
                if self._capture_thread is not None:
                    self._capture_thread.requestInterruption()

        overlap_samples = max(1, int(round(_STREAM_OVERLAP_S * sample_rate_hz)))
        keep = min(overlap_samples, analysis_iq.size)
        self._stream_tail_iq = analysis_iq[-keep:].copy()
        self._stream_tail_start_sample = self._stream_total_samples - keep

    def _append_stream_ring(self, block_iq: np.ndarray) -> None:
        if self._stream_ring_iq.size:
            self._stream_ring_iq = np.concatenate((self._stream_ring_iq, block_iq))
        else:
            self._stream_ring_iq = block_iq.copy()
        settings = self._pluto_settings()
        pretrigger = int(round(_SINGLE_PRETRIGGER_S * settings.requested_sample_rate_hz))
        stream_block = int(
            round(_STREAM_BLOCK_DURATION_S * settings.requested_sample_rate_hz)
        )
        maximum = settings.capture_samples + pretrigger + 2 * stream_block
        excess = self._stream_ring_iq.size - maximum
        if excess > 0:
            self._stream_ring_iq = self._stream_ring_iq[excess:].copy()
            self._stream_ring_start_sample += excess

    def _display_stream_view(
        self,
        template: IQRecording,
        start_sample: int,
        stop_sample: int,
        messages: list[tuple[int, ADSB1090Message]],
        *,
        append: bool,
    ) -> None:
        start_sample = max(start_sample, self._stream_ring_start_sample)
        stop_sample = min(stop_sample, self._stream_total_samples)
        lo = start_sample - self._stream_ring_start_sample
        hi = stop_sample - self._stream_ring_start_sample
        if hi <= lo:
            return
        view_recording = replace(
            template,
            iq=self._stream_ring_iq[lo:hi],
            start_sample_index=start_sample,
            trigger_sample_index=None,
            source="VSA Pluto ADS-B Stream",
        )
        relative_messages = tuple(
            replace(message, start_sample=absolute_start - start_sample)
            for absolute_start, message in messages
            if start_sample <= absolute_start < stop_sample
        )
        full_scale = float(view_recording.full_scale)
        linear_power = (np.abs(view_recording.iq) / full_scale) ** 2
        power_dbfs = 10.0 * np.log10(
            np.maximum(linear_power, np.finfo(np.float64).tiny)
        )
        result = ADSB1090AnalysisResult(
            time_s=np.arange(view_recording.sample_count, dtype=np.float64)
            / view_recording.sample_rate_hz,
            power_dbfs=power_dbfs,
            messages=relative_messages,
            metadata={
                "source": view_recording.source,
                "stream_start_sample": start_sample,
            },
        )
        self.recording = view_recording
        self.result = result
        scan_wall = self._scan_started_wall_time or datetime.now().astimezone()
        elapsed_base_s = start_sample / view_recording.sample_rate_hz
        self._display_result(
            result,
            view_recording,
            append=append,
            capture_started_at=scan_wall + timedelta(seconds=elapsed_base_s),
            elapsed_base_s=elapsed_base_s,
            fit_latest_group=False,
        )
        valid = sum(message.parity_ok is True for message in relative_messages)
        self.statusBar().showMessage(
            f"{'Continuous scan' if append else 'Single complete'} - "
            f"{len(relative_messages)} new messages, {valid} parity verified, "
            f"{len(self._packet_history)} total"
        )

    @QtCore.Slot(object)
    def _stream_analysis_ready(self, payload: object) -> None:
        if not isinstance(payload, _ADSBStreamView) or self._closing:
            return
        self._pending_stream_views.append(payload)
        if not self._stream_display_timer.isActive():
            self._stream_display_timer.start()

    def _flush_stream_views(self) -> None:
        if self._closing or not self._pending_stream_views:
            self._pending_stream_views.clear()
            return
        pending = self._pending_stream_views
        self._pending_stream_views = []
        latest = pending[-1]
        selected_packet_row = self.packet_table.currentRow()
        follow_latest_packet = (
            selected_packet_row < 0
            or selected_packet_row == len(self._packet_history) - 1
        )
        self.packet_table.setUpdatesEnabled(False)
        self.aircraft_table.setUpdatesEnabled(False)
        try:
            for index, payload in enumerate(pending):
                is_latest = index == len(pending) - 1
                self.recording = payload.recording
                self.result = payload.result
                self._display_result(
                    payload.result,
                    payload.recording,
                    append=payload.append,
                    capture_started_at=payload.capture_started_at,
                    elapsed_base_s=payload.elapsed_base_s,
                    fit_latest_group=False,
                    update_power_plot=is_latest,
                    update_selection=is_latest,
                )
        finally:
            self.packet_table.setUpdatesEnabled(True)
            self.aircraft_table.setUpdatesEnabled(True)
        if latest.append and follow_latest_packet and self._packet_history:
            latest_row = len(self._packet_history) - 1
            selection_blocker = QtCore.QSignalBlocker(self.packet_table)
            self.packet_table.selectRow(latest_row)
            del selection_blocker
            latest_entry = self._packet_history[latest_row]
            self._show_message_plot(latest_entry)
            self._show_summary(latest_entry)
        valid = sum(
            message.parity_ok is True
            for payload in pending
            for message in payload.result.messages
        )
        new_message_count = sum(len(payload.result.messages) for payload in pending)
        self.statusBar().showMessage(
            f"{'Continuous scan' if latest.append else 'Single complete'} - "
            f"{new_message_count} new messages, "
            f"{valid} parity verified, {len(self._packet_history)} total"
        )
        if latest.single_complete:
            self._single_complete = True
            processor = (
                self._analysis_stream_thread.processor
                if self._analysis_stream_thread is not None
                else None
            )
            if processor is not None:
                self._single_trigger_sample = processor.single_trigger_sample
            if self._capture_thread is not None:
                self._capture_thread.requestInterruption()

    def _pluto_capture_failed(self, message: str) -> None:
        self.statusBar().showMessage(f"Pluto capture failed: {message}")
        if self._capture_thread is not None:
            self._capture_thread.requestInterruption()
        if self._analysis_stream_thread is not None:
            self._analysis_stream_thread.stop()
        if self._closing:
            return
        QtWidgets.QMessageBox.critical(self, "Pluto Capture", message)

    def _pluto_capture_stopped(self) -> None:
        self._capture_thread = None
        if (
            self._analysis_stream_thread is not None
            and self._analysis_stream_thread.isRunning()
        ):
            self._analysis_stream_thread.stop()
            self.statusBar().showMessage("Stopping ADS-B analysis...")
            return
        self._finalize_pluto_run_controls()

    def _stream_analysis_stopped(self) -> None:
        self._analysis_stream_thread = None
        if self._capture_thread is None:
            self._finalize_pluto_run_controls()

    def _finalize_pluto_run_controls(self) -> None:
        self.run_single_action.setText("Run Single (Pluto)")
        self.run_continuous_action.setText("Run Continuous (Pluto)")
        self.run_single_action.setEnabled(True)
        self.run_continuous_action.setEnabled(True)
        self._continuous_scan = False

    def analyze_recording(
        self,
        recording: IQRecording,
        *,
        append: bool = False,
        capture_started_at: datetime | None = None,
        elapsed_base_s: float = 0.0,
    ) -> None:
        self.statusBar().showMessage("Analyzing 1090 MHz capture...")
        QtWidgets.QApplication.processEvents()
        try:
            result = self._analyzer.analyze(
                recording,
                self._analysis_settings(),
            )
        except Exception as error:
            self.statusBar().showMessage(f"Analysis failed: {error}")
            QtWidgets.QMessageBox.critical(self, "ADS-B 1090ES", str(error))
            return
        self.recording = recording
        self.result = result
        started_at = capture_started_at or datetime.now().astimezone()
        self._display_result(
            result,
            recording,
            append=append,
            capture_started_at=started_at,
            elapsed_base_s=float(elapsed_base_s),
        )
        valid = sum(message.parity_ok is True for message in result.messages)
        self.statusBar().showMessage(
            f"{'Continuous scan' if append else 'Analysis complete'} - "
            f"{len(result.messages)} new messages, {valid} parity verified, "
            f"{len(self._packet_history)} total"
        )

    def _clear_packet_history(self) -> None:
        self._packet_history.clear()
        blocker = QtCore.QSignalBlocker(self.packet_table)
        self.packet_table.clearSelection()
        self.packet_table.setRowCount(0)
        del blocker
        self._aircraft_states.clear()
        self._aircraft_row_by_icao.clear()
        aircraft_blocker = QtCore.QSignalBlocker(self.aircraft_table)
        self.aircraft_table.clearSelection()
        self.aircraft_table.setRowCount(0)
        del aircraft_blocker
        self._show_aircraft_summary(None)
        self.power_plot.clear()
        self.ppm_plot.clear()
        self._plot_initial_ranges.clear()
        self._show_summary(None)

    def _display_result(
        self,
        result: ADSB1090AnalysisResult,
        recording: IQRecording,
        *,
        append: bool,
        capture_started_at: datetime,
        elapsed_base_s: float,
        fit_latest_group: bool = True,
        update_power_plot: bool = True,
        update_selection: bool = True,
    ) -> None:
        if not append:
            self._clear_packet_history()
        if update_power_plot:
            self.power_plot.clear()
            time_ms = (elapsed_base_s + result.time_s) * 1e3
            display_power = np.maximum(
                result.power_dbfs + recording.dbfs_to_dbm_offset_db,
                _IQ_POWER_DISPLAY_FLOOR_DBM,
            )
            plot_time_ms, plot_power = _peak_envelope_decimate(
                time_ms,
                display_power,
            )
            self.power_plot.plot(plot_time_ms, plot_power, pen=_TRACE_COLOR)
        new_entries: list[_ADSBPacketEntry] = []
        for message in result.messages:
            elapsed_s = elapsed_base_s + message.start_time_s
            entry = _ADSBPacketEntry(
                message=message,
                result=result,
                recording=recording,
                elapsed_s=elapsed_s,
                wall_time=capture_started_at + timedelta(seconds=message.start_time_s),
                on_pulse_power_dbm=self._on_pulse_power_dbm(
                    result,
                    recording,
                    message,
                ),
            )
            new_entries.append(entry)
            if update_power_plot:
                line = pg.InfiniteLine(
                    pos=elapsed_s * 1e3,
                    angle=90,
                    pen=pg.mkPen(_BIT_COLOR, width=1),
                )
                self.power_plot.addItem(line)
        first_new_row = len(self._packet_history)
        previous_selected_row = self.packet_table.currentRow()
        follow_latest = append and (
            previous_selected_row < 0
            or previous_selected_row == first_new_row - 1
        )
        self._packet_history.extend(new_entries)
        selection_blocker = QtCore.QSignalBlocker(self.packet_table)
        if not append:
            self.packet_table.clearSelection()
        self.packet_table.setRowCount(len(self._packet_history))
        for offset, entry in enumerate(new_entries):
            row = first_new_row + offset
            message = entry.message
            decoded = message.fields.get("callsign")
            if not decoded and message.fields.get("altitude_ft") is not None:
                decoded = f"{message.fields['altitude_ft']} ft"
            values = (
                str(row + 1),
                f"{entry.elapsed_s:.6f}",
                entry.wall_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                message.raw_hex,
                str(message.downlink_format),
                message.icao_address or "-",
                "-" if message.type_code is None else str(message.type_code),
                message.parity_display,
                f"{entry.on_pulse_power_dbm:+.2f}",
                f"{message.preamble_snr_db:.1f}",
                str(decoded or ""),
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.packet_table.setItem(row, column, item)
        selected_entry = (
            new_entries[-1]
            if append and new_entries and follow_latest
            else None
        )
        selected_row = (
            len(self._packet_history) - 1 if selected_entry is not None else -1
        )
        if not append and new_entries:
            selected_entry = new_entries[0]
            selected_row = 0
        if update_selection and selected_row >= 0:
            self.packet_table.selectRow(selected_row)
        del selection_blocker
        self._update_aircraft_states(new_entries, refresh_selection=update_selection)
        if update_power_plot and fit_latest_group:
            self._set_latest_group_power_range(result, elapsed_base_s)
        elif update_power_plot and result.time_s.size:
            self.power_plot.setXRange(
                elapsed_base_s * 1e3,
                (elapsed_base_s + float(result.time_s[-1])) * 1e3,
                padding=0.0,
            )
        if update_power_plot:
            self._remember_plot_range("power", self.power_plot)
        if update_selection and selected_entry is not None:
            self._show_message_plot(selected_entry)
            self._show_summary(selected_entry)
        elif update_selection and not self._packet_history:
            self._show_summary(None)

    def _selected_packet_changed(self) -> None:
        if self._closing:
            return
        row = self.packet_table.currentRow()
        if not 0 <= row < len(self._packet_history):
            return
        entry = self._packet_history[row]
        self._show_message_plot(entry)
        self._show_summary(entry)

    def _update_aircraft_states(
        self,
        entries: list[_ADSBPacketEntry],
        *,
        refresh_selection: bool = True,
    ) -> None:
        selected_icao = self._selected_aircraft_icao()
        updated: set[str] = set()
        for entry in entries:
            message = entry.message
            icao = message.icao_address
            if not icao or not message.icao_confirmed:
                continue
            state = self._aircraft_states.get(icao)
            if state is None:
                state = _ADSBAircraftState(
                    icao_address=icao,
                    first_elapsed_s=entry.elapsed_s,
                    first_wall_time=entry.wall_time,
                    last_elapsed_s=entry.elapsed_s,
                    last_wall_time=entry.wall_time,
                )
                state.metadata = self._aircraft_metadata_database.lookup(icao)
                self._aircraft_states[icao] = state
                row = self.aircraft_table.rowCount()
                self._aircraft_row_by_icao[icao] = row
                self.aircraft_table.insertRow(row)
            state.last_elapsed_s = entry.elapsed_s
            state.last_wall_time = entry.wall_time
            state.message_count += 1
            state.parity_verified_count += int(message.parity_ok is True)
            state.downlink_formats.add(message.downlink_format)
            if message.type_code is not None:
                state.type_codes.add(message.type_code)
            fields = dict(message.fields)
            callsign = str(fields.get("callsign", "")).strip()
            if callsign:
                state.callsign = normalize_callsign(callsign)
                cached_route = self._route_cache.get(state.callsign)
                if cached_route is not None:
                    state.route = cached_route
                self._request_route(state.callsign)
            if fields.get("emitter_category") is not None:
                state.emitter_category = int(fields["emitter_category"])
            if fields.get("altitude_ft") is not None:
                state.latest_altitude_ft = int(fields["altitude_ft"])
            if fields.get("ground_speed_kt") is not None:
                state.latest_ground_speed_kt = float(fields["ground_speed_kt"])
            if fields.get("track_deg") is not None:
                state.latest_track_deg = float(fields["track_deg"])
            if fields.get("vertical_rate_fpm") is not None:
                state.latest_vertical_rate_fpm = float(fields["vertical_rate_fpm"])
                state.latest_vertical_rate_source = str(
                    fields.get("vertical_rate_source", "")
                ) or None
            if fields.get("air_ground") is not None:
                state.latest_air_ground = str(fields["air_ground"])
            elif fields.get("flight_status") in {0, 2}:
                state.latest_air_ground = "airborne"
            elif fields.get("flight_status") in {1, 3}:
                state.latest_air_ground = "ground"
            elif fields.get("vertical_status") == 0:
                state.latest_air_ground = "airborne"
            elif fields.get("vertical_status") == 1:
                state.latest_air_ground = "ground"
            if (
                fields.get("position_type") == "airborne"
                and fields.get("cpr_format") in {"even", "odd"}
            ):
                cpr = (
                    entry.elapsed_s,
                    int(fields["cpr_latitude"]),
                    int(fields["cpr_longitude"]),
                )
                if fields["cpr_format"] == "even":
                    state.airborne_cpr_even = cpr
                else:
                    state.airborne_cpr_odd = cpr
                even = state.airborne_cpr_even
                odd = state.airborne_cpr_odd
                position: tuple[float, float] | None = None
                position_source: str | None = None
                position_reference: tuple[float, float] | None = None
                if even is not None and odd is not None and abs(even[0] - odd[0]) <= 10.0:
                    position = decode_global_airborne_cpr(
                        even[1], even[2], odd[1], odd[2],
                        use_odd=odd[0] >= even[0],
                    )
                    if position is not None:
                        position_source = "Global CPR"
                if position is None:
                    references: list[tuple[str, float, float]] = []
                    if (
                        state.latest_latitude_deg is not None
                        and state.latest_longitude_deg is not None
                    ):
                        references.append(
                            (
                                "Local CPR (Previous Position)",
                                state.latest_latitude_deg,
                                state.latest_longitude_deg,
                            )
                        )
                    if (
                        self._receiver_latitude_deg is not None
                        and self._receiver_longitude_deg is not None
                    ):
                        references.append(
                            (
                                "Local CPR (Receiver)",
                                self._receiver_latitude_deg,
                                self._receiver_longitude_deg,
                            )
                        )
                    for source, reference_latitude, reference_longitude in references:
                        candidate = decode_local_airborne_cpr(
                            int(fields["cpr_latitude"]),
                            int(fields["cpr_longitude"]),
                            is_odd=fields["cpr_format"] == "odd",
                            reference_latitude_deg=reference_latitude,
                            reference_longitude_deg=reference_longitude,
                        )
                        if candidate is None:
                            continue
                        if _great_circle_distance_nm(
                            reference_latitude,
                            reference_longitude,
                            candidate[0],
                            candidate[1],
                        ) > _MAX_LOCAL_CPR_RANGE_NM:
                            continue
                        if (
                            state.latest_latitude_deg is not None
                            and state.latest_longitude_deg is not None
                            and state.latest_position_elapsed_s is not None
                        ):
                            elapsed = max(
                                entry.elapsed_s - state.latest_position_elapsed_s, 0.0
                            )
                            plausible_distance = max(
                                2.0, 1500.0 * elapsed / 3600.0
                            )
                            if _great_circle_distance_nm(
                                state.latest_latitude_deg,
                                state.latest_longitude_deg,
                                candidate[0],
                                candidate[1],
                            ) > plausible_distance:
                                continue
                        position = candidate
                        position_source = source
                        position_reference = (
                            reference_latitude,
                            reference_longitude,
                        )
                        break
                if position is not None:
                    state.latest_latitude_deg, state.latest_longitude_deg = position
                    state.latest_position_elapsed_s = entry.elapsed_s
                    state.latest_position_source = position_source
                    state.latest_position_reference = position_reference
                    history_point = (
                        entry.elapsed_s,
                        position[0],
                        position[1],
                        state.latest_altitude_ft,
                    )
                    if (
                        not state.position_history
                        or state.position_history[-1][1:3]
                        != history_point[1:3]
                    ):
                        state.position_history.append(history_point)
                        if len(state.position_history) > 10_000:
                            del state.position_history[:-10_000]
            state.latest_power_dbm = entry.on_pulse_power_dbm
            state.power_sum_mw += 10.0 ** (entry.on_pulse_power_dbm / 10.0)
            state.peak_power_dbm = max(
                state.peak_power_dbm, entry.on_pulse_power_dbm
            )
            state.latest_snr_db = message.preamble_snr_db
            state.snr_sum_db += message.preamble_snr_db
            state.peak_snr_db = max(state.peak_snr_db, message.preamble_snr_db)
            state.latest_correlation = message.preamble_correlation
            state.latest_raw_message = message.raw_hex
            state.latest_fields.update(fields)
            updated.add(icao)

        blocker = QtCore.QSignalBlocker(self.aircraft_table)
        for icao in updated:
            self._update_aircraft_row(self._aircraft_states[icao])
        if selected_icao is not None:
            selected_row = self._aircraft_row_by_icao.get(selected_icao)
            if selected_row is not None:
                self.aircraft_table.selectRow(selected_row)
        elif self.aircraft_table.rowCount() > 0:
            self.aircraft_table.selectRow(0)
            selected_icao = self.aircraft_table.item(0, 0).text()
        del blocker
        if refresh_selection and selected_icao is not None:
            self._show_aircraft_summary(self._aircraft_states.get(selected_icao))

    def _request_route(self, callsign: str) -> None:
        normalized = normalize_callsign(callsign)
        if (
            not normalized
            or normalized in self._route_cache
            or normalized in self._route_pending
        ):
            return
        if self._route_lookup_thread is None:
            thread = _ADSBRouteLookupThread(self._route_client, parent=self)
            thread.route_ready.connect(self._route_lookup_ready)
            thread.route_failed.connect(self._route_lookup_failed)
            thread.finished.connect(thread.deleteLater)
            self._route_lookup_thread = thread
            thread.start()
        self._route_pending.add(normalized)
        self._route_lookup_thread.enqueue(normalized)

    @QtCore.Slot(str, object)
    def _route_lookup_ready(self, callsign: str, route: object) -> None:
        normalized = normalize_callsign(callsign)
        self._route_pending.discard(normalized)
        resolved = route if isinstance(route, FlightRoute) else None
        self._route_cache[normalized] = resolved
        updated_states = [
            state
            for state in self._aircraft_states.values()
            if normalize_callsign(state.callsign or "") == normalized
        ]
        for state in updated_states:
            state.route = resolved
            self._update_aircraft_row(state)
        selected_icao = self._selected_aircraft_icao()
        selected = self._aircraft_states.get(selected_icao)
        if selected in updated_states:
            self._show_aircraft_summary(selected)

    @QtCore.Slot(str, str)
    def _route_lookup_failed(self, callsign: str, _message: str) -> None:
        # Route enrichment is optional. Treat network/API failures exactly like
        # an unknown callsign and keep ADS-B capture/decoding uninterrupted.
        self._route_lookup_ready(callsign, None)

    @staticmethod
    def _format_route_airport(airport: RouteAirport, *, detail: bool) -> str:
        if not detail:
            return airport.compact_name
        codes = " / ".join(
            code for code in (airport.iata_code, airport.icao_code) if code
        )
        location = ", ".join(
            value for value in (airport.municipality, airport.country) if value
        )
        parts = [value for value in (airport.name, codes, location) if value]
        return " | ".join(parts) or "-"

    def _update_aircraft_row(self, state: _ADSBAircraftState) -> None:
        row = self._aircraft_row_by_icao[state.icao_address]
        altitude = (
            "-"
            if state.latest_altitude_ft is None
            else f"{state.latest_altitude_ft} / {state.latest_altitude_ft * 0.3048:.0f}"
        )
        speed = (
            "-"
            if state.latest_ground_speed_kt is None
            else f"{state.latest_ground_speed_kt:.1f} / {state.latest_ground_speed_kt * 1.852:.1f}"
        )
        vertical_rate = (
            "-"
            if state.latest_vertical_rate_fpm is None
            else f"{state.latest_vertical_rate_fpm:+.0f} / {state.latest_vertical_rate_fpm * 0.00508:+.2f}"
        )
        origin = "-"
        destination = "-"
        if state.callsign in self._route_pending:
            origin = destination = "..."
        elif state.route is not None:
            origin = self._format_route_airport(state.route.origin, detail=False)
            destination = self._format_route_airport(
                state.route.destination, detail=False
            )
        values = (
            state.icao_address,
            state.callsign or "-",
            state.latest_air_ground or "-",
            "-" if state.latest_latitude_deg is None else f"{state.latest_latitude_deg:.5f}",
            "-" if state.latest_longitude_deg is None else f"{state.latest_longitude_deg:.5f}",
            f"{state.last_elapsed_s:.3f}",
            str(state.message_count),
            altitude,
            speed,
            vertical_rate,
            f"{state.latest_power_dbm:+.2f}",
            f"{state.latest_snr_db:.1f}",
            origin,
            destination,
        )
        for column, value in enumerate(values):
            item = self.aircraft_table.item(row, column)
            if item is None:
                item = QtWidgets.QTableWidgetItem()
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.aircraft_table.setItem(row, column, item)
            item.setText(value)

    def _selected_aircraft_icao(self) -> str | None:
        row = self.aircraft_table.currentRow()
        if not 0 <= row < self.aircraft_table.rowCount():
            return None
        item = self.aircraft_table.item(row, 0)
        return item.text() if item is not None else None

    def _selected_aircraft_changed(self) -> None:
        if self._closing:
            return
        icao = self._selected_aircraft_icao()
        self._show_aircraft_summary(
            self._aircraft_states.get(icao) if icao is not None else None
        )

    @QtCore.Slot(int)
    def _aircraft_detail_tab_changed(self, index: int) -> None:
        if index == 1:
            self.aircraft_map.activate()
            icao = self._selected_aircraft_icao()
            self._show_aircraft_map(
                self._aircraft_states.get(icao) if icao is not None else None
            )

    def _show_aircraft_summary(
        self, state: _ADSBAircraftState | None
    ) -> None:
        self._show_aircraft_map(state)
        if state is None:
            rows: list[tuple[str, object]] = []
        else:
            average_power_dbm = 10.0 * np.log10(
                max(
                    state.power_sum_mw / max(state.message_count, 1),
                    np.finfo(np.float64).tiny,
                )
            )
            average_snr_db = state.snr_sum_db / max(state.message_count, 1)
            fields = state.latest_fields
            metadata = state.metadata
            receiver_distance_nm = None
            if (
                self._receiver_latitude_deg is not None
                and self._receiver_longitude_deg is not None
                and state.latest_latitude_deg is not None
                and state.latest_longitude_deg is not None
            ):
                receiver_distance_nm = _great_circle_distance_nm(
                    self._receiver_latitude_deg,
                    self._receiver_longitude_deg,
                    state.latest_latitude_deg,
                    state.latest_longitude_deg,
                )
            rows = [
                ("ICAO Address", state.icao_address),
                ("Callsign", state.callsign or "-"),
                (
                    "Route Origin (ADSBDB)",
                    "-"
                    if state.route is None
                    else self._format_route_airport(state.route.origin, detail=True),
                ),
                (
                    "Route Destination (ADSBDB)",
                    "-"
                    if state.route is None
                    else self._format_route_airport(
                        state.route.destination, detail=True
                    ),
                ),
                (
                    "Route Airline (ADSBDB)",
                    "-"
                    if state.route is None
                    else state.route.airline_name or "-",
                ),
                (
                    "Registration (DB)",
                    "-" if metadata is None else metadata.registration or "-",
                ),
                (
                    "Manufacturer (DB)",
                    "-" if metadata is None else metadata.manufacturer or "-",
                ),
                (
                    "Model (DB)",
                    "-" if metadata is None else metadata.model or "-",
                ),
                (
                    "Type Code (DB)",
                    "-" if metadata is None else metadata.type_code or "-",
                ),
                (
                    "Serial Number (DB)",
                    "-" if metadata is None else metadata.serial_number or "-",
                ),
                (
                    "Operator (DB)",
                    "-" if metadata is None else metadata.operator or "-",
                ),
                (
                    "Operator Callsign (DB)",
                    "-" if metadata is None else metadata.operator_callsign or "-",
                ),
                (
                    "Owner (DB)",
                    "-" if metadata is None else metadata.owner or "-",
                ),
                (
                    "Country (DB)",
                    "-" if metadata is None else metadata.country or "-",
                ),
                (
                    "Emitter Category",
                    "-" if state.emitter_category is None else state.emitter_category,
                ),
                ("First Seen", f"{state.first_elapsed_s:.6f} s"),
                ("First OS Time", state.first_wall_time.strftime("%Y-%m-%d %H:%M:%S.%f")),
                ("Last Seen", f"{state.last_elapsed_s:.6f} s"),
                ("Last OS Time", state.last_wall_time.strftime("%Y-%m-%d %H:%M:%S.%f")),
                ("Messages", state.message_count),
                ("Parity Verified", state.parity_verified_count),
                ("Downlink Formats", ", ".join(map(str, sorted(state.downlink_formats)))),
                ("Type Codes", ", ".join(map(str, sorted(state.type_codes))) or "-"),
                (
                    "Latest Altitude",
                    "-" if state.latest_altitude_ft is None else f"{state.latest_altitude_ft} ft / {state.latest_altitude_ft * 0.3048:.0f} m",
                ),
                (
                    "Latest Ground Speed",
                    "-" if state.latest_ground_speed_kt is None else f"{state.latest_ground_speed_kt:.1f} kt / {state.latest_ground_speed_kt * 1.852:.1f} km/h",
                ),
                (
                    "Latest Vertical Rate",
                    "-" if state.latest_vertical_rate_fpm is None else f"{state.latest_vertical_rate_fpm:+.0f} ft/min / {state.latest_vertical_rate_fpm * 0.00508:+.2f} m/s",
                ),
                ("Vertical Rate Source", state.latest_vertical_rate_source or "-"),
                ("Air/Ground", state.latest_air_ground or "-"),
                (
                    "Latitude",
                    "-" if state.latest_latitude_deg is None else f"{state.latest_latitude_deg:.6f} degree",
                ),
                (
                    "Longitude",
                    "-" if state.latest_longitude_deg is None else f"{state.latest_longitude_deg:.6f} degree",
                ),
                ("Position Source", state.latest_position_source or "-"),
                (
                    "Position Reference",
                    "-"
                    if state.latest_position_reference is None
                    else f"{state.latest_position_reference[0]:.6f}, "
                    f"{state.latest_position_reference[1]:.6f}",
                ),
                (
                    "Distance from Receiver",
                    "-"
                    if receiver_distance_nm is None
                    else f"{receiver_distance_nm:.1f} NM / "
                    f"{receiver_distance_nm * 1.852:.1f} km",
                ),
                (
                    "Latest Track",
                    "-" if state.latest_track_deg is None else f"{state.latest_track_deg:.1f} degree",
                ),
                ("Latest ON Power", f"{state.latest_power_dbm:+.2f} dBm"),
                ("Average ON Power", f"{average_power_dbm:+.2f} dBm"),
                ("Peak ON Power", f"{state.peak_power_dbm:+.2f} dBm"),
                ("Latest Preamble SNR", f"{state.latest_snr_db:.2f} dB"),
                ("Average Preamble SNR", f"{average_snr_db:.2f} dB"),
                ("Peak Preamble SNR", f"{state.peak_snr_db:.2f} dB"),
                ("Latest Correlation", f"{state.latest_correlation:.3f}"),
                ("Position Type", fields.get("position_type", "-")),
                ("CPR Format", fields.get("cpr_format", "-")),
                ("CPR Latitude (Raw)", fields.get("cpr_latitude", "-")),
                ("CPR Longitude (Raw)", fields.get("cpr_longitude", "-")),
                ("Latest Raw Message", state.latest_raw_message),
            ]
        self.aircraft_summary_table.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            self.aircraft_summary_table.setItem(
                row, 0, QtWidgets.QTableWidgetItem(str(name))
            )
            self.aircraft_summary_table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(str(value))
            )

    def _show_aircraft_map(self, state: _ADSBAircraftState | None) -> None:
        render = self.aircraft_detail_tabs.currentIndex() == 1
        if state is None or not state.position_history:
            self.aircraft_map.set_aircraft_track(
                icao=None if state is None else state.icao_address,
                callsign=None if state is None else state.callsign,
                track_deg=None if state is None else state.latest_track_deg,
                points=[],
                render=render,
            )
            return
        history = state.position_history[-5_000:]
        self.aircraft_map.set_aircraft_track(
            icao=state.icao_address,
            callsign=state.callsign,
            track_deg=state.latest_track_deg,
            points=[
                {
                    "elapsed_s": elapsed_s,
                    "latitude": latitude,
                    "longitude": longitude,
                    "altitude_ft": altitude_ft,
                }
                for elapsed_s, latitude, longitude, altitude_ft in history
            ],
            render=render,
        )

    def _show_message_plot(self, entry: _ADSBPacketEntry) -> None:
        message = entry.message
        result = entry.result
        self.ppm_plot.clear()
        linear_power = np.power(10.0, result.power_dbfs / 10.0)
        samples_per_us = message.sample_rate_hz * 1e-6
        data_start = message.start_sample + 8.0 * samples_per_us
        first_chip = np.empty(message.bit_length, dtype=np.float64)
        second_chip = np.empty(message.bit_length, dtype=np.float64)
        for bit_index in range(message.bit_length):
            symbol_start = data_start + bit_index * samples_per_us
            midpoint = symbol_start + 0.5 * samples_per_us
            symbol_stop = symbol_start + samples_per_us
            first_chip[bit_index] = self._fractional_window_mean(
                linear_power, symbol_start, midpoint
            )
            second_chip[bit_index] = self._fractional_window_mean(
                linear_power, midpoint, symbol_stop
            )
        epsilon = np.finfo(np.float64).tiny
        chip_ratio_db = 10.0 * np.log10(
            np.maximum(first_chip, epsilon) / np.maximum(second_chip, epsilon)
        )
        bit_index = np.arange(message.bit_length, dtype=np.float64)
        self.ppm_plot.addItem(
            pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen((160, 160, 160), width=1))
        )
        self.ppm_plot.plot(
            bit_index,
            chip_ratio_db,
            pen=pg.mkPen(_TRACE_COLOR, width=1),
        )
        one = message.bits == 1
        zero = ~one
        self.ppm_plot.plot(
            bit_index[one],
            chip_ratio_db[one],
            pen=None,
            symbol="o",
            symbolSize=6,
            symbolPen=None,
            symbolBrush=_BIT_COLOR,
        )
        self.ppm_plot.plot(
            bit_index[zero],
            chip_ratio_db[zero],
            pen=None,
            symbol="o",
            symbolSize=6,
            symbolPen=None,
            symbolBrush=_ZERO_BIT_COLOR,
        )
        self.ppm_plot.setXRange(-1.0, float(message.bit_length), padding=0.0)
        limit = max(3.0, 1.1 * float(np.max(np.abs(chip_ratio_db))))
        self.ppm_plot.setYRange(-limit, limit, padding=0.0)
        self._remember_plot_range("ppm", self.ppm_plot)

    @staticmethod
    def _fractional_window_mean(
        values: np.ndarray, start: float, stop: float
    ) -> float:
        lo = max(0, int(np.floor(start)))
        hi = min(values.size, max(lo + 1, int(np.ceil(stop))))
        return float(np.mean(values[lo:hi]))

    def _show_summary(self, entry: _ADSBPacketEntry | None) -> None:
        if entry is None:
            rows: list[tuple[str, object]] = []
        else:
            message = entry.message
            rows = [
                ("Measurement Elapsed", f"{entry.elapsed_s:.6f} s"),
                ("OS Time", entry.wall_time.strftime("%Y-%m-%d %H:%M:%S.%f")),
                ("Raw Message", message.raw_hex),
                ("Length", f"{message.bit_length} bit"),
                ("Downlink Format", message.downlink_format),
            ]
            rows.extend(
                (
                    key.replace("_", " ").title(),
                    self._format_mode_s_header_field(key, message.fields[key]),
                )
                for key in _MODE_S_HEADER_FIELDS
                if key in message.fields
            )
            rows.extend([
                ("ICAO Address", message.icao_address or "-"),
                (
                    "ICAO Address Source",
                    (message.icao_address_source or "-").replace("_", " ").title(),
                ),
                ("Type Code", message.type_code if message.type_code is not None else "-"),
                ("Parity / CRC", message.parity_display),
                ("Mean ON Pulse Power", f"{entry.on_pulse_power_dbm:+.2f} dBm"),
                ("Preamble SNR", f"{message.preamble_snr_db:.2f} dB"),
                ("Preamble Correlation", f"{message.preamble_correlation:.3f}"),
            ])
            fixed_fields = {"type_code", "icao_address", *_MODE_S_HEADER_FIELDS}
            rows.extend(
                (key.replace("_", " ").title(), value)
                for key, value in message.fields.items()
                if key not in fixed_fields
            )
        self.summary_table.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            self.summary_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(name)))
            self.summary_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(value)))

    @staticmethod
    def _format_mode_s_header_field(key: str, value: object) -> object:
        if key != "flight_status":
            return value
        numeric = int(value)
        description = _FLIGHT_STATUS_DESCRIPTIONS.get(numeric, "Unknown")
        return f"{numeric} ({description})"

    def _on_pulse_power_dbm(
        self,
        result: ADSB1090AnalysisResult,
        recording: IQRecording,
        message: ADSB1090Message,
    ) -> float:
        linear_power = np.power(10.0, result.power_dbfs / 10.0)
        samples_per_us = message.sample_rate_hz * 1e-6
        data_start = message.start_sample + 8.0 * samples_per_us
        pulse_powers = np.empty(message.bit_length, dtype=np.float64)
        for bit_index, bit in enumerate(message.bits):
            pulse_start = data_start + bit_index * samples_per_us
            if bit == 0:
                pulse_start += 0.5 * samples_per_us
            pulse_powers[bit_index] = self._fractional_window_mean(
                linear_power,
                pulse_start,
                pulse_start + 0.5 * samples_per_us,
            )
        mean_dbfs = 10.0 * np.log10(
            max(float(np.mean(pulse_powers)), np.finfo(np.float64).tiny)
        )
        return mean_dbfs + recording.dbfs_to_dbm_offset_db

    def _set_latest_group_power_range(
        self,
        result: ADSB1090AnalysisResult,
        elapsed_base_s: float,
    ) -> None:
        if result.time_s.size == 0:
            return
        capture_start_s = elapsed_base_s + float(result.time_s[0])
        capture_stop_s = elapsed_base_s + float(result.time_s[-1])
        messages = result.messages
        if not messages:
            lower_s, upper_s = capture_start_s, capture_stop_s
        else:
            group_start = len(messages) - 1
            while group_start > 0:
                gap = (
                    messages[group_start].start_time_s
                    - messages[group_start - 1].start_time_s
                )
                if gap > _PACKET_GROUP_GAP_S:
                    break
                group_start -= 1
            first_s = elapsed_base_s + messages[group_start].start_time_s
            last = messages[-1]
            last_s = (
                elapsed_base_s
                + last.start_time_s
                + (8.0 + last.bit_length) * 1e-6
            )
            span_s = max(last_s - first_s, 1e-3)
            margin_s = max(1e-3, 0.1 * span_s)
            lower_s = max(capture_start_s, first_s - margin_s)
            upper_s = min(capture_stop_s, last_s + margin_s)
        if upper_s <= lower_s:
            upper_s = lower_s + 1e-3
        self.power_plot.setXRange(lower_s * 1e3, upper_s * 1e3, padding=0.0)

    def _configure_plot_context_menus(self) -> None:
        for name, plot in (("power", self.power_plot), ("ppm", self.ppm_plot)):
            self._plot_context_actions[name] = install_measurement_plot_menu(
                plot,
                reset=lambda plot_name=name, target=plot: self._reset_plot(
                    plot_name, target
                ),
            )
    def _remember_plot_range(self, name: str, plot: pg.PlotWidget) -> None:
        plot.getViewBox().updateAutoRange()
        x_range, y_range = plot.viewRange()
        self._plot_initial_ranges[name] = (list(x_range), list(y_range))

    def _reset_plot(self, name: str, plot: pg.PlotWidget) -> None:
        ranges = self._plot_initial_ranges.get(name)
        if ranges is None:
            return
        x_range, y_range = ranges
        plot.setRange(xRange=x_range, yRange=y_range, padding=0.0)

    def prepare_for_shutdown(self) -> None:
        """Stop UI callbacks before Qt starts deleting child dock widgets."""

        if self._closing:
            return
        self._save_user_settings()
        self._closing = True
        self._stream_display_timer.stop()
        self._pending_stream_views.clear()
        if self._route_lookup_thread is not None:
            self._route_lookup_thread.stop()
        for thread in (
            self._capture_thread,
            self._analysis_stream_thread,
            self._aircraft_metadata_thread,
            self._route_lookup_thread,
        ):
            if thread is None or not thread.isRunning():
                continue
            try:
                thread.finished.connect(
                    self._shutdown_worker_finished,
                    QtCore.Qt.ConnectionType.UniqueConnection,
                )
            except (RuntimeError, TypeError):
                pass
        if self._packet_selection_connected:
            try:
                self.packet_table.itemSelectionChanged.disconnect(
                    self._selected_packet_changed
                )
            except (RuntimeError, TypeError):
                pass
            self._packet_selection_connected = False
        if self._aircraft_selection_connected:
            try:
                self.aircraft_table.itemSelectionChanged.disconnect(
                    self._selected_aircraft_changed
                )
            except (RuntimeError, TypeError):
                pass
            self._aircraft_selection_connected = False
        try:
            self.packet_table.blockSignals(True)
        except RuntimeError:
            pass
        try:
            self.aircraft_table.blockSignals(True)
        except RuntimeError:
            pass
        self.aircraft_map.shutdown()

    def request_shutdown(self) -> None:
        """Request worker termination without blocking the GUI thread."""

        self.prepare_for_shutdown()
        if self._capture_thread is not None and self._capture_thread.isRunning():
            self._capture_thread.requestInterruption()
        if (
            self._analysis_stream_thread is not None
            and self._analysis_stream_thread.isRunning()
        ):
            self._analysis_stream_thread.stop()
        if self._route_lookup_thread is not None:
            self._route_lookup_thread.stop()

    def shutdown_busy_reason(self) -> str | None:
        if self._capture_thread is not None and self._capture_thread.isRunning():
            return "ADS-B Pluto capture"
        if (
            self._analysis_stream_thread is not None
            and self._analysis_stream_thread.isRunning()
        ):
            return "ADS-B stream analysis"
        if (
            self._aircraft_metadata_thread is not None
            and self._aircraft_metadata_thread.isRunning()
        ):
            return "ADS-B aircraft database update"
        if (
            self._route_lookup_thread is not None
            and self._route_lookup_thread.isRunning()
        ):
            return "ADS-B route lookup"
        return None

    def finalize_shutdown(self) -> None:
        if self._shutdown_finalized:
            return
        self._shutdown_finalized = True
        self._route_lookup_thread = None
        if self._owns_pluto_source:
            self._pluto_source.close()

    def _shutdown_worker_finished(self) -> None:
        if not self._closing or self.shutdown_busy_reason() is not None:
            return
        if not self._shutdown_ready_emitted:
            self._shutdown_ready_emitted = True
            self.shutdown_ready.emit()
        if self._owns_pluto_source:
            self.close()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.request_shutdown()
        busy = self.shutdown_busy_reason()
        if busy is not None:
            self.statusBar().showMessage(f"Stopping {busy} before closing...")
            event.ignore()
            return
        self.finalize_shutdown()
        super().closeEvent(event)
