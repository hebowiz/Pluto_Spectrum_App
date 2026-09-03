"""Bluetooth dedicated-analyzer result assembled from Generic VSA products."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from collections.abc import Callable
from typing import Mapping

import numpy as np

from pluto_protocol.model import (
    FieldStatus,
    IssueSeverity,
    PacketAnalysisResult,
    PacketField,
    PacketIntegritySummary,
    PacketIssue,
    PacketSourceInfo,
    PacketSummaryItem,
)
from pluto_protocol.bluetooth.common import le_whitening_sequence
from pluto_protocol.bluetooth.hdt import (
    HDT_DEFINITIONS,
    HDT_RF_TEST_CRC32_INIT,
    HDT_RF_TEST_PCA,
    HDTRate,
    convolutional_encode,
    hdt_coded_payload_bit_count,
    hdt_crc24,
    hdt_crc32,
    hdt_definition,
    hdt_rf_test_format0_bits,
    hdt_rf_test_training_symbols,
    map_hdt_symbols,
    puncture,
)
from pluto_sa.vsa.model import VSAAnalysisResult
from pluto_sa.vsa.mapping import (
    BLUETOOTH_EDR_MAPPING,
    BLUETOOTH_HDT_MAPPING,
    phase_indices_to_logical_symbols,
    psk_constellation,
    reverse_symbol_bits,
)
from pluto_sa.vsa.model import IQRecording, ModulationKind, SignalDescription
from pluto_sa.vsa.pattern import (
    DemodulationSettings,
    IQPowerTriggerSettings,
    KnownPattern,
    MatchSelectionPolicy,
    MeasurementFilterMode,
    PatternSearchMode,
    PatternSearchSettings,
    ResultRangeSettings,
    prepare_psk_iq,
)
from pluto_sa.vsa.profiles.bluetooth_br import (
    BluetoothBRProfile,
    access_code_bits,
    prbs9_period,
)
from pluto_sa.vsa.profiles.bluetooth_edr import edr_sync_symbols
from pluto_sa.vsa.protocol import analyze_demodulated_packet_bits
from pluto_sa.vsa.session import VSASession
from pluto_sa.vsa.protocol_modes.bluetooth.rf_measurement.hdt import (
    apply_hdt_payload_estimate,
    apply_hdt_reference,
    build_hdt_evm_result,
    estimate_hdt_payload,
    estimate_hdt_reference,
)
from pluto_sa.vsa.protocol_modes.bluetooth.rf_measurement import (
    BluetoothRFMeasurementFilterProfile,
    BluetoothRFMeasurementResult,
    BluetoothRFTestAccumulator,
    EDRConformanceResult,
    HDTPlotData,
    RFTestEligibility,
    RFTestVerdict,
    build_fm_measurement_trace,
    measure_edr_devm,
    measure_edr_guard_time,
    measure_burst_power,
    measure_carrier_drift,
    measure_initial_carrier_frequency,
    measure_modulation_characteristics,
    measure_observed_fsk_deviation,
    measure_pre_packet_emissions,
)
from pluto_sa.vsa.protocol_modes.bluetooth.rf_measurement.limits import (
    HDT_FREQUENCY_OFFSET_CHANGE_LIMIT_HZ,
    HDT_HEADER_EVM_LIMIT_DB,
    HDT_PAYLOAD_EVM_LIMIT_DB,
    HDT_PRE_PACKET_EMISSIONS_LIMIT_S,
    OUTPUT_POWER_LIMIT_DEPENDENCY,
    TEST_SUITE_REVISION,
)
from pluto_sa.vsa.protocol_modes.bluetooth.summary import (
    BluetoothSummaryRow,
    build_bluetooth_summary,
)


class BluetoothAnalysisProfile(StrEnum):
    RF_PHY_TEST = "rf_phy_test"
    GENERAL_PACKET = "general_packet"


class BluetoothClassicPhy(StrEnum):
    BR = "BR"
    EDR_2M = "EDR 2M"
    EDR_3M = "EDR 3M"


class BluetoothLEPhy(StrEnum):
    LE_1M = "LE 1M"
    LE_2M = "LE 2M"


@dataclass(frozen=True)
class BluetoothMetric:
    metric_id: str
    label: str
    display: str
    limit: str = "\N{EM DASH}"
    result: str = "\N{EM DASH}"
    group: str | None = None


@dataclass(frozen=True)
class BluetoothDedicatedResult:
    profile: BluetoothAnalysisProfile
    vsa_result: VSAAnalysisResult
    packet: PacketAnalysisResult
    metrics: tuple[BluetoothMetric, ...]
    metadata: Mapping[str, object]
    summary_rows: tuple[BluetoothSummaryRow, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile", BluetoothAnalysisProfile(self.profile))
        object.__setattr__(self, "metrics", tuple(self.metrics))
        metadata = dict(self.metadata)
        metadata.setdefault("bluetooth_test_suite_revision", TEST_SUITE_REVISION)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))
        object.__setattr__(self, "summary_rows", build_bluetooth_summary(self))


def _finite_stat(values: np.ndarray, *, peak: bool = False) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.max(finite) if peak else np.mean(finite))


def _mean_power_dbm(values: np.ndarray) -> float | None:
    """Average calibrated power in the linear domain and return dBm."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    mean_mw = float(np.mean(np.power(10.0, finite / 10.0)))
    if not np.isfinite(mean_mw) or mean_mw <= 0.0:
        return None
    return float(10.0 * np.log10(mean_mw))


def _recording_range_power_dbm(
    recording: IQRecording, start_sample: int, stop_sample: int
) -> float | None:
    values = np.asarray(
        recording.iq[max(0, int(start_sample)) : min(recording.sample_count, int(stop_sample))],
        dtype=np.complex128,
    )
    if values.size == 0:
        return None
    mean_power = float(np.mean(np.abs(values / recording.full_scale) ** 2))
    if not np.isfinite(mean_power) or mean_power <= 0.0:
        return None
    return float(10.0 * np.log10(mean_power) + recording.dbfs_to_dbm_offset_db)


def _recording_range_spectrum_dbm(
    recording: IQRecording,
    start_sample: int,
    stop_sample: int,
    *,
    fft_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an absolute-frequency spectrum for one packet region."""

    values = np.asarray(
        recording.iq[
            max(0, int(start_sample)) : min(recording.sample_count, int(stop_sample))
        ],
        dtype=np.complex128,
    )
    if values.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    used = min(values.size, int(fft_size))
    start = max(0, (values.size - used) // 2)
    window = np.hanning(used) if used > 1 else np.ones(used, dtype=np.float64)
    transform = np.fft.fftshift(
        np.fft.fft(values[start : start + used] * window, n=int(fft_size))
    )
    amplitude = np.abs(transform) / (
        max(float(np.sum(window)), 1.0) * float(recording.full_scale)
    )
    frequency_hz = (
        np.fft.fftshift(
            np.fft.fftfreq(int(fft_size), d=1.0 / recording.sample_rate_hz)
        )
        + recording.center_frequency_hz
    )
    spectrum_dbm = (
        20.0 * np.log10(np.maximum(amplitude, np.finfo(np.float64).tiny))
        + recording.dbfs_to_dbm_offset_db
    )
    return frequency_hz, spectrum_dbm


def _display(value: float | None, unit: str, scale: float = 1.0) -> str:
    return "--" if value is None or not np.isfinite(value) else f"{value / scale:+.3f} {unit}"


def _display_evm(value: object) -> str:
    try:
        percent = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(percent) or percent < 0.0:
        return "--"
    db_text = (
        "-inf"
        if percent == 0.0
        else f"{20.0 * np.log10(percent / 100.0):.1f}"
    )
    return f"{percent:.2f} % / {db_text} dB"


def analyze_bluetooth_session(
    session: VSASession,
    *,
    profile: BluetoothAnalysisProfile,
    protocol_id: str,
    phy_name: str,
    context: Mapping[str, object] | None = None,
) -> BluetoothDedicatedResult:
    """Build one dedicated result from the currently synchronized VSA packet."""

    recording = session.recording
    base_result = session.result
    if recording is None or base_result is None:
        raise RuntimeError("Generic VSA has no completed analysis result")
    profile = BluetoothAnalysisProfile(profile)
    pattern = session.pattern_result
    result = session.carrier_corrected_pattern_range_result or session.pattern_range_result or base_result
    if pattern is None:
        bits = result.decoded_bits
        start_sample, stop_sample = 0, recording.sample_count
        cfo_hz = result.frequency_error_hz
        correlation = "--"
    else:
        bits = pattern.decoded_bits
        start_sample, stop_sample = pattern.result_start_sample, pattern.result_stop_sample
        cfo_hz = float(pattern.carrier_frequency_offset_hz)
        correlation = f"{100.0 * float(pattern.correlation):.2f} %"
    if bits.size == 0:
        raise RuntimeError("The VSA result does not contain demodulated bits")
    packet = analyze_demodulated_packet_bits(
        bits,
        protocol_id=protocol_id,
        phy_name=phy_name,
        context=dict(context or {}),
        packet_index=0,
        center_frequency_hz=recording.center_frequency_hz,
        start_sample=start_sample,
        stop_sample=stop_sample,
    )
    try:
        rate_error = float(result.metadata.get("symbol_rate_error_ppm"))
    except (TypeError, ValueError):
        rate_error = None
    duration_ms = max(0, stop_sample - start_sample) / recording.sample_rate_hz * 1e3
    metrics = (
        BluetoothMetric("profile", "Analysis Profile", profile.value),
        BluetoothMetric("packet_power", "Packet Average Power", _display(_mean_power_dbm(result.power_dbm), "dBm")),
        BluetoothMetric("peak_power", "Peak Power", _display(_finite_stat(result.power_dbm, peak=True), "dBm")),
        BluetoothMetric("cfo", "Carrier Frequency Offset", _display(cfo_hz, "kHz", 1e3)),
        BluetoothMetric("symbol_rate_error", "Symbol Rate Error", _display(rate_error, "ppm")),
        BluetoothMetric("duration", "Packet Duration", f"{duration_ms:.3f} ms"),
        BluetoothMetric("correlation", "Synchronization Correlation", correlation),
    )
    return BluetoothDedicatedResult(
        profile=profile,
        vsa_result=result,
        packet=packet,
        metrics=metrics,
        metadata={
            "source": recording.source,
            "sample_rate_hz": recording.sample_rate_hz,
            "center_frequency_hz": recording.center_frequency_hz,
            "semantic_decode_is_independent": True,
        },
    )


def _classic_signal(phy: BluetoothClassicPhy) -> SignalDescription:
    if phy is BluetoothClassicPhy.BR:
        return SignalDescription(
            modulation=ModulationKind.FSK,
            symbol_rate_hz=1_000_000.0,
            frequency_deviation_hz=160_000.0,
            tx_filter="Gaussian",
            filter_parameter=0.5,
            symbol_mapping="Natural",
        )
    modulation = (
        ModulationKind.PI4_DQPSK
        if phy is BluetoothClassicPhy.EDR_2M
        else ModulationKind.DPSK8
    )
    return SignalDescription(
        modulation=modulation,
        symbol_rate_hz=1_000_000.0,
        tx_filter="Root Raised Cosine",
        filter_parameter=0.4,
        symbol_mapping=BLUETOOTH_EDR_MAPPING,
    )


def _le_access_bits(access_address: int) -> np.ndarray:
    """Return an LE access address in over-the-air bit order."""

    value = int(access_address)
    if not 0 <= value <= 0xFFFFFFFF:
        raise ValueError("LE access address must be a 32-bit value")
    octets = value.to_bytes(4, byteorder="little")
    return np.unpackbits(np.frombuffer(octets, dtype=np.uint8), bitorder="little")


def _le_sync_bits(phy: BluetoothLEPhy, access_address: int) -> np.ndarray:
    access = _le_access_bits(access_address)
    preamble_count = 16 if phy is BluetoothLEPhy.LE_2M else 8
    if int(access_address) == 0x71764129:
        # The uncoded RF PHY Test Packet uses the prescribed alternating
        # preamble paired with Sync Word 0x71764129.
        preamble = np.resize(
            np.asarray([1, 0], dtype=np.uint8), preamble_count
        )
        return np.concatenate((preamble, access))
    # Core Vol 6, Part B: the alternating preamble is selected so its last
    # transmitted bit differs from the first access-address bit.  In the OTA
    # arrays used here that means the first preamble bit is the complement of
    # access[0] (Adv AA therefore starts with 10101010).
    preamble = (
        1 - int(access[0]) + np.arange(preamble_count, dtype=np.uint8)
    ) & 1
    return np.concatenate((preamble, access))


def _le_signal(phy: BluetoothLEPhy) -> SignalDescription:
    rate = 2_000_000.0 if phy is BluetoothLEPhy.LE_2M else 1_000_000.0
    return SignalDescription(
        modulation=ModulationKind.FSK,
        symbol_rate_hz=rate,
        frequency_deviation_hz=(
            500_000.0 if phy is BluetoothLEPhy.LE_2M else 250_000.0
        ),
        tx_filter="Gaussian",
        filter_parameter=0.5,
        symbol_mapping="Natural",
    )


def _rf_test_eligibility(
    *,
    profile: BluetoothAnalysisProfile | str,
    whitening_enabled: bool,
    sample_rate_hz: float,
    minimum_sample_rate_hz: float,
    extra_reasons: tuple[str, ...] = (),
) -> RFTestEligibility:
    reasons = list(extra_reasons)
    if BluetoothAnalysisProfile(profile) is not BluetoothAnalysisProfile.RF_PHY_TEST:
        reasons.append("RF / PHY Test profile is not selected")
    if whitening_enabled:
        reasons.append("RF test packet must have whitening disabled")
    if float(sample_rate_hz) <= float(minimum_sample_rate_hz):
        reasons.append(
            f"sample rate must exceed {minimum_sample_rate_hz / 1e6:.3f} MS/s"
        )
    return RFTestEligibility.from_reasons(reasons)


def _sig_fsk_measurements(
    recording: IQRecording,
    packet: PacketAnalysisResult,
    *,
    profile: BluetoothAnalysisProfile | str,
    whitening_enabled: bool,
    symbol_rate_hz: float,
    filter_profile: BluetoothRFMeasurementFilterProfile,
    packet_start_sample: int,
    packet_stop_sample: int,
    p0_sample: float | None = None,
    extra_eligibility_reasons: tuple[str, ...] = (),
    drift_block_symbols: int = 50,
) -> tuple[tuple[BluetoothMetric, ...], tuple[BluetoothRFMeasurementResult, ...]]:
    """Measure BR/LE RF properties without decoder frequency-model fitting."""

    scale = 2.0 if filter_profile is BluetoothRFMeasurementFilterProfile.LE_2M else 1.0
    eligibility = _rf_test_eligibility(
        profile=profile,
        whitening_enabled=whitening_enabled,
        sample_rate_hz=recording.sample_rate_hz,
        minimum_sample_rate_hz=4_000_000.0 * scale,
        extra_reasons=extra_eligibility_reasons,
    )
    eligibility_text = (
        "Eligible" if eligibility.eligible else "N/A - " + "; ".join(eligibility.reasons)
    )
    metrics: list[BluetoothMetric] = [
        BluetoothMetric("sig_eligibility", "SIG RF Test Eligibility", eligibility_text)
    ]
    payload = _packet_field_by_id(packet.root_fields, "payload_body")
    if payload is None:
        payload = _packet_field_by_id(packet.root_fields, "payload")
    if payload is None or payload.raw_bits.size < 16:
        unavailable = RFTestEligibility.from_reasons(
            (*eligibility.reasons, "decoded payload is too short")
        )
        return tuple(metrics), (
            BluetoothRFMeasurementResult(
                "bluetooth.fsk",
                unavailable,
                metadata={"reason": "decoded payload is too short"},
            ),
        )
    try:
        trace = build_fm_measurement_trace(
            recording.iq,
            sample_rate_hz=recording.sample_rate_hz,
            symbol_rate_hz=symbol_rate_hz,
            p0_sample=(
                float(packet_start_sample)
                if p0_sample is None
                else float(p0_sample)
            ),
            profile=filter_profile,
        )
        modulation = measure_modulation_characteristics(
            trace,
            payload.raw_bits,
            payload_start_symbol=int(payload.start_bit),
        )
        if modulation.payload_pattern is None:
            eligibility = RFTestEligibility.from_reasons(
                (
                    *eligibility.reasons,
                    "payload is not an RF test 11110000 or 10101010 pattern",
                )
            )
            metrics[0] = BluetoothMetric(
                "sig_eligibility",
                "SIG RF Test Eligibility",
                "N/A - " + "; ".join(eligibility.reasons),
            )
        preamble_symbols = (
            4
            if filter_profile is BluetoothRFMeasurementFilterProfile.BR_1M
            else 16
            if filter_profile is BluetoothRFMeasurementFilterProfile.LE_2M
            else 8
        )
        initial = measure_initial_carrier_frequency(
            trace,
            packet.raw_bits[: min(preamble_symbols, packet.raw_bits.size)],
            nominal_frequency_hz=recording.center_frequency_hz,
            start_symbol=0,
        )
        observed_deviation = measure_observed_fsk_deviation(
            trace,
            payload_start_symbol=int(payload.start_bit),
            payload_symbol_count=int(payload.raw_bits.size),
            carrier_frequency_offset_hz=initial.error_hz,
        )
        drift = measure_carrier_drift(
            trace,
            payload.raw_bits,
            nominal_frequency_hz=recording.center_frequency_hz,
            start_symbol=int(payload.start_bit),
            block_symbols=drift_block_symbols,
        )
        power = measure_burst_power(
            recording.iq,
            full_scale=recording.full_scale,
            dbfs_to_dbm_offset_db=recording.dbfs_to_dbm_offset_db,
            start_sample=packet_start_sample,
            stop_sample=packet_stop_sample,
        )
    except (ValueError, RuntimeError) as error:
        unavailable = RFTestEligibility.from_reasons(
            (*eligibility.reasons, str(error))
        )
        return tuple(metrics), (
            BluetoothRFMeasurementResult(
                "bluetooth.fsk", unavailable, metadata={"reason": str(error)}
            ),
        )

    metrics.extend(
        (
            BluetoothMetric("sig_pavg", "SIG PAVG", _display(power.average_dbm, "dBm")),
            BluetoothMetric("sig_ppk", "SIG PPK", _display(power.peak_dbm, "dBm")),
            BluetoothMetric("sig_ppk_minus_pavg", "SIG PPK - PAVG", _display(power.peak_to_average_db, "dB")),
            BluetoothMetric("sig_initial_carrier_error", "SIG Initial Carrier Error f0", _display(initial.error_hz, "kHz", 1e3)),
            BluetoothMetric("sig_max_drift", "SIG Max Drift from f0", _display(drift.max_drift_from_f0_hz, "kHz", 1e3)),
            BluetoothMetric("sig_max_drift_rate", "SIG Max Drift Step", _display(drift.max_drift_rate_hz, "kHz", 1e3)),
        )
    )
    if modulation.delta_f1_avg_hz is not None:
        metrics.append(
            BluetoothMetric("sig_delta_f1_avg", "SIG Delta f1avg", _display(modulation.delta_f1_avg_hz, "kHz", 1e3))
        )

    if modulation.delta_f2_avg_hz is not None:
        metrics.append(
            BluetoothMetric("sig_delta_f2_avg", "SIG Delta f2avg", _display(modulation.delta_f2_avg_hz, "kHz", 1e3))
        )
    if modulation.delta_f2_max_hz.size:
        metrics.append(
            BluetoothMetric(
                "sig_delta_f2_p999",
                "SIG Delta f2max 99.9% Floor",
                _display(float(np.percentile(modulation.delta_f2_max_hz, 0.1)), "kHz", 1e3),
            )
        )
    result = BluetoothRFMeasurementResult(
        test_case_id="bluetooth.fsk",
        eligibility=eligibility,
        verdict=RFTestVerdict.NOT_APPLICABLE,
        metrics={
            "delta_f1_avg_hz": modulation.delta_f1_avg_hz,
            "delta_f2_avg_hz": modulation.delta_f2_avg_hz,
            "initial_carrier_error_hz": initial.error_hz,
            "max_drift_from_f0_hz": drift.max_drift_from_f0_hz,
            "max_drift_rate_hz": drift.max_drift_rate_hz,
            "mean_abs_fsk_deviation_hz": observed_deviation.mean_abs_hz,
            "p999_abs_fsk_deviation_hz": observed_deviation.percentile_99_9_hz,
            "max_abs_fsk_deviation_hz": observed_deviation.max_abs_hz,
            "pavg_dbm": power.average_dbm,
            "ppk_dbm": power.peak_dbm,
        },
        arrays={
            "delta_f2_max_hz": modulation.delta_f2_max_hz,
            "fn_hz": drift.fn_hz,
            "frequency_hz": trace.frequency_hz,
            "observed_fsk_deviation_hz": observed_deviation.deviations_hz,
            "f0_selected_bit_indices": initial.selected_bit_indices,
        },
        metadata={
            "payload_pattern": modulation.payload_pattern,
            "filter_profile": filter_profile.value,
            "aggregation_required": True,
        },
    )
    return tuple(metrics), (result,)


def _attach_rf_capture_aggregates(
    results: list[BluetoothDedicatedResult],
) -> tuple[BluetoothDedicatedResult, ...]:
    """Attach capture-wide SIG evidence without inventing incomplete verdicts."""

    accumulator = BluetoothRFTestAccumulator()
    for item in results:
        for measurement in item.metadata.get("rf_measurements", ()):
            if isinstance(measurement, BluetoothRFMeasurementResult):
                accumulator.add(measurement)
    test_ids = {measurement.test_case_id for measurement in accumulator.results}
    aggregates: dict[str, BluetoothRFMeasurementResult] = {}
    if "bluetooth.fsk" in test_ids:
        aggregates["bluetooth.fsk"] = accumulator.aggregate_fsk()
    if "bluetooth.edr" in test_ids:
        aggregates["bluetooth.edr"] = accumulator.aggregate_edr()
    if "bluetooth.hdt.evm" in test_ids:
        aggregates["bluetooth.hdt.evm"] = accumulator.aggregate_hdt()

    attached: list[BluetoothDedicatedResult] = []
    for item in results:
        measurements = tuple(item.metadata.get("rf_measurements", ()))
        aggregate = next(
            (
                aggregates[measurement.test_case_id]
                for measurement in measurements
                if isinstance(measurement, BluetoothRFMeasurementResult)
                and measurement.test_case_id in aggregates
            ),
            None,
        )
        if aggregate is None:
            attached.append(item)
            continue
        if aggregate.test_case_id == "bluetooth.hdt.evm.aggregate":
            evaluated = int(aggregate.metrics["eligible_packet_count"])
            required = int(aggregate.metadata["required_packets"])
            metadata = dict(item.metadata)
            metadata["rf_capture_aggregate"] = aggregate
            metadata["hdt_rms_evm_packets_evaluated"] = evaluated
            metadata["hdt_rms_evm_packets_required"] = required
            metadata["hdt_rms_evm_aggregate_status"] = (
                aggregate.verdict.value.upper()
                if evaluated >= required
                else f"MEASURING {evaluated} / {required}"
            )
            attached.append(
                replace(
                    item,
                    metrics=tuple(
                        replace(
                            metric,
                            display=f"{evaluated} / {required}",
                        )
                        if metric.metric_id == "sig_hdt_evm_packets_evaluated"
                        else metric
                        for metric in item.metrics
                    ),
                    metadata=metadata,
                )
            )
            continue
        block_count = aggregate.metrics.get("block_count")
        suffix = f", {int(block_count)} DEVM blocks" if block_count is not None else ""
        reasons = "; ".join(aggregate.eligibility.reasons)
        display = aggregate.verdict.value
        if reasons:
            display += f" - {reasons}"
        display += suffix
        metadata = dict(item.metadata)
        metadata["rf_capture_aggregate"] = aggregate
        attached.append(
            replace(
                item,
                metrics=(
                    *item.metrics,
                    BluetoothMetric(
                        "sig_capture_aggregate",
                        "SIG Capture Aggregate",
                        display,
                    ),
                ),
                metadata=metadata,
            )
        )
    return tuple(attached)

def _trim_le_packet_bits(
    bits: np.ndarray,
    *,
    phy: BluetoothLEPhy,
    whitening_enabled: bool,
    channel_index: int,
) -> np.ndarray:
    """Trim a synchronized LE result using the decoded PDU length field."""

    values = np.asarray(bits, dtype=np.uint8)
    prefix = (16 if phy is BluetoothLEPhy.LE_2M else 8) + 32
    if values.size < prefix + 16:
        return values
    encoded = values[prefix:]
    logical = (
        encoded ^ le_whitening_sequence(int(channel_index), encoded.size)
        if whitening_enabled
        else encoded
    )
    payload_octets = sum(int(logical[8 + bit]) << bit for bit in range(8))
    packet_bits = prefix + 16 + payload_octets * 8 + 24
    return values[: min(values.size, packet_bits)]


def _analyze_known_pattern(
    recording: IQRecording,
    signal: SignalDescription,
    symbols: np.ndarray,
    *,
    result_length: int,
    minimum_correlation: float,
    match_index: int = 1,
    match_selection: MatchSelectionPolicy = MatchSelectionPolicy.INDEX,
    iq_power_trigger: IQPowerTriggerSettings | None = None,
) -> VSASession:
    session = VSASession(name="Bluetooth dedicated")
    session.set_recording(recording)
    session.set_signal(signal)
    # Bluetooth FSK already contains the Gaussian transmitter shaping.  A
    # second Gaussian "measurement" filter reduces the recovered deviation,
    # most visibly in an alternating 01 sequence.  Keep the wide/unfiltered
    # discriminator path for BR/LE FSK.  EDR PSK continues to use the matched
    # receive filter selected from the PHY's TX-filter description.
    demodulation = DemodulationSettings(
        measurement_filter=(
            MeasurementFilterMode.NONE
            if signal.modulation is ModulationKind.FSK
            else MeasurementFilterMode.AUTO
        )
    )
    session.configure_pattern_analysis(
        PatternSearchSettings(
            pattern=KnownPattern(tuple(map(int, symbols))),
            mode=PatternSearchMode.ON,
            correlation_threshold_auto=False,
            iq_correlation_threshold=float(minimum_correlation),
            match_selection=MatchSelectionPolicy(match_selection),
            match_index=max(1, int(match_index)),
        ),
        ResultRangeSettings(result_length=max(1, int(result_length))),
        demodulation=demodulation,
        iq_power_trigger=iq_power_trigger,
    )
    session.analyze()
    if session.pattern_result is None:
        raise RuntimeError("Bluetooth synchronization pattern was not found")
    return session


def _edr_candidate_for_type(packet_type: int) -> BluetoothClassicPhy | None:
    """Return the EDR PHY which *could* use a Classic TYPE value.

    Classic TYPE is not a unique BR/EDR discriminator.  Every value returned
    here is only a candidate: the packet is promoted to EDR only when the
    corresponding PSK synchronization word is present immediately after the
    BR header and guard interval.  Otherwise it remains a BR packet.
    """

    if int(packet_type) in {0x4, 0xA, 0xE}:
        return BluetoothClassicPhy.EDR_2M
    if int(packet_type) in {0x8, 0xB, 0xF}:
        return BluetoothClassicPhy.EDR_3M
    return None


def _edr_sync_search_bounds(
    *,
    expected_start_sample: int,
    sync_symbol_count: int,
    recording_sample_count: int,
    samples_per_br_symbol: float,
    samples_per_psk_symbol: float,
) -> tuple[int, int, int]:
    """Return a local EDR-sync search window and its timing tolerance.

    The pre-roll is deliberately wider than the accepted timing error so the
    matched filter can settle.  The stop is limited to the EDR synchronization
    word plus a short post-roll.  This prevents a later packet's PSK sync from
    changing the PHY decision for the current BR header.
    """

    # The nominal guard is five BR symbol periods, but a real capture also
    # carries RX/TX filter group delay, fractional timing error and power-ramp
    # transients around the PHY switch.  Two symbols proved too narrow for
    # Pluto captures even though ideal generated IQ passed.  Eight symbols is
    # still a strictly local boundary test (and therefore cannot reach the
    # next packet), while covering the practical acquisition uncertainty.
    timing_tolerance = max(1, int(round(8.0 * samples_per_br_symbol)))
    filter_preroll = max(
        timing_tolerance,
        int(round(8.0 * samples_per_psk_symbol)),
    )
    postroll = max(
        timing_tolerance,
        int(round(8.0 * samples_per_psk_symbol)),
    )
    start = max(0, int(expected_start_sample - filter_preroll))
    stop = min(
        int(recording_sample_count),
        int(
            expected_start_sample
            + sync_symbol_count * samples_per_psk_symbol
            + postroll
        ),
    )
    return start, stop, timing_tolerance


def _analyze_edr_payload_at_sync(
    recording: IQRecording,
    signal: SignalDescription,
    sync: np.ndarray,
    *,
    result_length: int,
    expected_sync_sample: int,
    minimum_correlation: float = 0.72,
) -> VSASession:
    """Analyze the EDR payload match nearest an already-confirmed sync.

    A 10-symbol EDR synchronization word is short enough that a long random
    payload can contain a stronger accidental correlation.  Selecting the
    strongest match over the whole payload therefore occasionally attached
    the analysis to the wrong location (or rejected the packet as BR).  The
    narrow boundary pass has already confirmed the physical sync, so the long
    pass must select the eligible match closest to that position.
    """

    session = _analyze_known_pattern(
        recording,
        signal,
        sync,
        result_length=result_length,
        minimum_correlation=minimum_correlation,
        match_index=1,
        match_selection=MatchSelectionPolicy.STRONGEST,
    )
    starts = tuple(
        int(value)
        for value in session.pattern_result.metadata.get(
            "eligible_match_start_samples", ()
        )
    )
    if not starts:
        return session
    nearest_index = min(
        range(len(starts)),
        key=lambda index: abs(starts[index] - int(expected_sync_sample)),
    )
    selected_start = int(session.pattern_result.pattern_start_sample)
    if selected_start == starts[nearest_index]:
        return session
    return _analyze_known_pattern(
        recording,
        signal,
        sync,
        result_length=result_length,
        minimum_correlation=minimum_correlation,
        match_index=nearest_index + 1,
        match_selection=MatchSelectionPolicy.INDEX,
    )


def _symbols_to_air_bits(symbols: np.ndarray, order: int) -> np.ndarray:
    """Serialize logical PSK symbols in Bluetooth over-the-air bit order.

    Generic VSA keeps ``decoded_bits`` in the user-selected table ordering
    (LSB by default).  Protocol decoding must not depend on that display
    preference: Bluetooth EDR groups the incoming air bits MSB-first into a
    2- or 3-bit differential symbol.
    """

    bit_count = int(round(np.log2(int(order))))
    values = np.asarray(symbols, dtype=np.int16)
    shifts = np.arange(bit_count - 1, -1, -1, dtype=np.int16)
    return ((values[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)


def _packet_field_by_id(
    fields: tuple[PacketField, ...], field_id: str
) -> PacketField | None:
    """Return a decoded field without coupling DSP code to the UI tree."""

    for field in fields:
        if field.field_id == field_id:
            return field
        nested = _packet_field_by_id(field.children, field_id)
        if nested is not None:
            return nested
    return None


def _exact_edr_result_symbols(
    packet: PacketAnalysisResult, *, bits_per_symbol: int
) -> int | None:
    """Derive the complete EDR PSK extent from the decoded Length field.

    ``payload.stop_bit`` ends after the payload CRC.  Bluetooth EDR appends a
    two-symbol trailer, which is deliberately excluded by the semantic
    decoder.  Converting that exact air-bit stop back to PSK symbols prevents
    idle samples or a following packet from entering the vector/EVM result.
    """

    payload = _packet_field_by_id(packet.root_fields, "payload")
    length = _packet_field_by_id(packet.root_fields, "length")
    if payload is None or length is None or not packet.integrity.complete:
        return None
    try:
        int(length.value)
    except (TypeError, ValueError):
        return None
    edr_air_bits = int(payload.stop_bit) + 2 * int(bits_per_symbol) - 126
    if edr_air_bits <= 0:
        return None
    return int(np.ceil(edr_air_bits / float(bits_per_symbol)))


def _exact_br_result_symbols(packet: PacketAnalysisResult) -> int | None:
    """Return the decoded BR packet length in one-bit GFSK symbols.

    TYPE selects the payload-header format and slot family, but the decoded
    ACL Length field determines the actual end of the packet.  The payload
    field's stop bit includes its header, body and CRC, so it is also the
    exact BR result-symbol count measured from the access-code start.
    """

    payload = _packet_field_by_id(packet.root_fields, "payload")
    length = _packet_field_by_id(packet.root_fields, "length")
    if payload is None or length is None or not packet.integrity.complete:
        return None
    try:
        int(length.value)
    except (TypeError, ValueError):
        return None
    stop_bit = int(payload.stop_bit)
    return stop_bit if stop_bit > 0 else None


_HDT_SYMBOL_RATE_HZ = 2_000_000.0
_HDT_TRAINING_SYMBOLS = 74
_HDT_CONTROL_SYMBOLS = 62
_HDT_TERMINATING_SYMBOLS = 2
_HDT_PAYLOAD_START_SYMBOL = (
    _HDT_TRAINING_SYMBOLS + _HDT_CONTROL_SYMBOLS + _HDT_TERMINATING_SYMBOLS
)
def _hdt_qpsk_constellation(symbol_indices: np.ndarray) -> np.ndarray:
    even_phases = np.asarray(
        [np.pi / 4.0, 3.0 * np.pi / 4.0, -np.pi / 4.0, -3.0 * np.pi / 4.0]
    )
    odd_phases = np.asarray([np.pi / 2.0, np.pi, 0.0, -np.pi / 2.0])
    indices = np.asarray(symbol_indices, dtype=np.int64)
    return np.where(
        (indices[:, None] & 1) == 0,
        np.exp(1j * even_phases)[None, :],
        np.exp(1j * odd_phases)[None, :],
    )


def _hdt_sample_symbols(
    recording: IQRecording,
    first_center_sample: float,
    count: int,
) -> np.ndarray:
    samples_per_symbol = recording.sample_rate_hz / _HDT_SYMBOL_RATE_HZ
    centers = first_center_sample + np.arange(max(0, int(count))) * samples_per_symbol
    axis = np.arange(recording.sample_count, dtype=np.float64)
    return (
        np.interp(centers, axis, recording.iq.real)
        + 1j * np.interp(centers, axis, recording.iq.imag)
    )


def _hdt_training_matches(recording: IQRecording) -> tuple[tuple[int, float], ...]:
    """Locate complete HDT training sequences and return first symbol centers."""

    samples_per_symbol = recording.sample_rate_hz / _HDT_SYMBOL_RATE_HZ
    integer_sps = int(round(samples_per_symbol))
    if integer_sps < 2 or not np.isclose(samples_per_symbol, integer_sps, atol=1e-6):
        raise ValueError("HDT analysis requires an integer samples-per-symbol ratio")
    filtered_iq, filtered_rate_hz = prepare_psk_iq(
        recording.iq,
        sample_rate_hz=recording.sample_rate_hz,
        symbol_rate_hz=_HDT_SYMBOL_RATE_HZ,
        tx_filter="Root Raised Cosine",
        filter_parameter=0.4,
        samples_per_symbol=integer_sps,
        apply_measurement_filter=True,
    )
    if not np.isclose(filtered_rate_hz, recording.sample_rate_hz):
        raise RuntimeError("Unexpected HDT matched-filter sample rate")
    reference = hdt_rf_test_training_symbols().astype(np.complex128)
    short_reference = reference[:36]
    short_energy = float(np.sum(np.abs(short_reference) ** 2))
    candidates: list[tuple[float, int]] = []
    for phase in range(integer_sps):
        sampled = np.asarray(filtered_iq[phase::integer_sps], dtype=np.complex128)
        if sampled.size < reference.size:
            continue
        correlation = np.correlate(sampled, short_reference, mode="valid")
        energy = np.convolve(
            np.abs(sampled) ** 2,
            np.ones(short_reference.size, dtype=np.float64),
            mode="valid",
        )
        scores = np.abs(correlation) / np.sqrt(
            np.maximum(energy * short_energy, np.finfo(np.float64).tiny)
        )
        for index in np.flatnonzero(scores >= 0.65):
            if index + reference.size > sampled.size:
                continue
            observed = sampled[index : index + reference.size]
            phase_error = np.unwrap(np.angle(observed * np.conj(reference)))
            axis = np.arange(reference.size, dtype=np.float64)
            phase_step, phase_intercept = np.polyfit(axis, phase_error, 1)
            corrected = observed * np.exp(
                -1j * (phase_intercept + phase_step * axis)
            )
            score = float(
                np.abs(np.vdot(reference, corrected))
                / max(
                    np.linalg.norm(reference) * np.linalg.norm(corrected),
                    np.finfo(np.float64).tiny,
                )
            )
            if score >= 0.80:
                candidates.append((score, int(phase + index * integer_sps)))
    selected: list[tuple[float, int]] = []
    guard_samples = int(round(_HDT_TRAINING_SYMBOLS * samples_per_symbol))
    for score, center in sorted(candidates, reverse=True):
        if all(abs(center - other_center) >= guard_samples for _, other_center in selected):
            selected.append((score, center))
    return tuple((center, score) for score, center in sorted(selected, key=lambda item: item[1]))


def _viterbi_decode_hdt_control(encoded: np.ndarray) -> tuple[np.ndarray, int]:
    """Hard-decision K=6 Viterbi decode for one standard HDT Control Header."""

    values = np.asarray(encoded, dtype=np.uint8)[:124]
    if values.size < 124:
        raise RuntimeError("HDT control header is incomplete")
    infinity = 1_000_000
    costs = np.full(32, infinity, dtype=np.int64)
    costs[0] = 0
    history: list[tuple[np.ndarray, np.ndarray]] = []
    for received in values.reshape(-1, 2):
        next_costs = np.full(32, infinity, dtype=np.int64)
        previous_state = np.zeros(32, dtype=np.int16)
        previous_bit = np.zeros(32, dtype=np.uint8)
        for state in range(32):
            if costs[state] >= infinity:
                continue
            taps = np.asarray([(state >> bit) & 1 for bit in range(5)], dtype=np.uint8)
            for value in (0, 1):
                registers = np.concatenate(([value], taps))
                expected = (
                    registers[0] ^ registers[2] ^ registers[4] ^ registers[5],
                    registers[0] ^ registers[1] ^ registers[2] ^ registers[3] ^ registers[5],
                )
                next_state = ((state << 1) & 0x1F) | value
                distance = int(expected[0] != received[0]) + int(
                    expected[1] != received[1]
                )
                candidate = int(costs[state]) + distance
                if candidate < next_costs[next_state]:
                    next_costs[next_state] = candidate
                    previous_state[next_state] = state
                    previous_bit[next_state] = value
        costs = next_costs
        history.append((previous_state, previous_bit))
    state = int(np.argmin(costs))
    decoded: list[int] = []
    for previous_state, previous_bit in reversed(history):
        decoded.append(int(previous_bit[state]))
        state = int(previous_state[state])
    return np.asarray(decoded[::-1], dtype=np.uint8), int(np.min(costs))


def _hdt_signal(rate: HDTRate) -> SignalDescription:
    modulation = {
        "pi/4-QPSK": ModulationKind.PI4_QPSK,
        "8PSK": ModulationKind.PSK8,
        "16QAM": ModulationKind.QAM16,
    }[hdt_definition(rate).modulation]
    return SignalDescription(
        modulation=modulation,
        symbol_rate_hz=_HDT_SYMBOL_RATE_HZ,
        tx_filter="Root Raised Cosine",
        filter_parameter=0.4,
        symbol_mapping=BLUETOOTH_HDT_MAPPING,
    )


_HDT_PUNCTURE_MASKS = {
    "1/2": (1, 1),
    "2/3": (1, 1, 0, 1),
    "3/4": (1, 1, 0, 1, 0, 1),
    "15/16": (
        1, 1, 0, 1, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 1, 0, 0, 1,
    ),
}


def _hdt_lsb_value(bits: np.ndarray) -> int:
    return int(sum(int(bit) << index for index, bit in enumerate(bits)))


def _hdt_msb_value(bits: np.ndarray) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _viterbi_decode_hdt_punctured(
    received: np.ndarray,
    *,
    logical_bit_count: int,
    code_rate: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Decode a terminated K=6 HDT stream with hard erasures at punctured bits."""

    step_count = int(logical_bit_count) + 5
    encoded_count = 2 * step_count
    mask = np.resize(
        np.asarray(_HDT_PUNCTURE_MASKS[str(code_rate)], dtype=bool), encoded_count
    )
    values = np.asarray(received, dtype=np.uint8)[: int(np.count_nonzero(mask))]
    if values.size < int(np.count_nonzero(mask)):
        raise RuntimeError("Bluetooth HDT coded PDU is incomplete")
    observed = np.zeros(encoded_count, dtype=np.uint8)
    observed[mask] = values
    infinity = 1_000_000
    costs = np.full(32, infinity, dtype=np.int64)
    costs[0] = 0
    previous_states = np.zeros((step_count, 32), dtype=np.uint8)
    previous_bits = np.zeros((step_count, 32), dtype=np.uint8)
    for step in range(step_count):
        next_costs = np.full(32, infinity, dtype=np.int64)
        retained = mask[2 * step : 2 * step + 2]
        current = observed[2 * step : 2 * step + 2]
        for state in range(32):
            if costs[state] >= infinity:
                continue
            history = np.asarray(
                [(state >> index) & 1 for index in range(5)], dtype=np.uint8
            )
            for bit in (0, 1):
                registers = np.concatenate(([bit], history))
                expected = np.asarray(
                    (
                        registers[0] ^ registers[2] ^ registers[4] ^ registers[5],
                        registers[0] ^ registers[1] ^ registers[2] ^ registers[3] ^ registers[5],
                    ),
                    dtype=np.uint8,
                )
                next_state = ((state << 1) & 0x1F) | bit
                candidate = int(costs[state]) + int(
                    np.count_nonzero(expected[retained] != current[retained])
                )
                if candidate < next_costs[next_state]:
                    next_costs[next_state] = candidate
                    previous_states[step, next_state] = state
                    previous_bits[step, next_state] = bit
        costs = next_costs
    state = 0
    decoded = np.empty(step_count, dtype=np.uint8)
    for step in range(step_count - 1, -1, -1):
        decoded[step] = previous_bits[step, state]
        state = int(previous_states[step, state])
    return decoded[:logical_bit_count], decoded[logical_bit_count:], int(costs[0])


def analyze_bluetooth_hdt_recording(
    recording: IQRecording,
    *,
    profile: BluetoothAnalysisProfile,
    match_index: int = 1,
    _matches: tuple[tuple[int, float], ...] | None = None,
) -> BluetoothDedicatedResult:
    """Synchronize HDT, decode RI/Length, and evaluate its QPSK and payload regions."""

    matches = _hdt_training_matches(recording) if _matches is None else _matches
    if not 1 <= int(match_index) <= len(matches):
        raise RuntimeError("Bluetooth HDT synchronization pattern was not found")
    coarse_first_center, _detection_correlation = matches[int(match_index) - 1]
    samples_per_symbol = recording.sample_rate_hz / _HDT_SYMBOL_RATE_HZ
    filtered_iq, filtered_rate_hz = prepare_psk_iq(
        recording.iq,
        sample_rate_hz=recording.sample_rate_hz,
        symbol_rate_hz=_HDT_SYMBOL_RATE_HZ,
        tx_filter="Root Raised Cosine",
        filter_parameter=0.4,
        samples_per_symbol=int(round(samples_per_symbol)),
        apply_measurement_filter=True,
    )
    if not np.isclose(filtered_rate_hz, recording.sample_rate_hz):
        raise RuntimeError("Unexpected HDT matched-filter sample rate")
    filtered_recording = replace(recording, iq=filtered_iq)
    training_reference = hdt_rf_test_training_symbols()
    hdt_reference = estimate_hdt_reference(
        filtered_iq,
        coarse_first_symbol_center_sample=coarse_first_center,
        samples_per_symbol=samples_per_symbol,
        training_reference=training_reference,
    )
    first_center = hdt_reference.first_symbol_center_sample
    training_correlation = hdt_reference.training_correlation
    header_observed = _hdt_sample_symbols(
        filtered_recording, first_center, _HDT_PAYLOAD_START_SYMBOL
    )
    if header_observed.size < _HDT_PAYLOAD_START_SYMBOL:
        raise RuntimeError("Bluetooth HDT control header is incomplete")
    corrected_header = apply_hdt_reference(
        filtered_iq,
        hdt_reference,
        samples_per_symbol=samples_per_symbol,
        start_symbol=0,
        symbol_count=_HDT_PAYLOAD_START_SYMBOL,
    )
    control_observed = corrected_header[
        _HDT_TRAINING_SYMBOLS : _HDT_TRAINING_SYMBOLS + _HDT_CONTROL_SYMBOLS
    ]
    control_first_center = first_center + _HDT_TRAINING_SYMBOLS * samples_per_symbol
    control_symbol_sample_positions = control_first_center + (
        np.arange(_HDT_CONTROL_SYMBOLS, dtype=np.float64) * samples_per_symbol
    )
    control_trajectory_start = max(
        0, int(np.floor(control_first_center - 0.5 * samples_per_symbol))
    )
    control_trajectory_stop = min(
        recording.sample_count,
        int(
            np.ceil(
                control_first_center
                + (_HDT_CONTROL_SYMBOLS - 0.5) * samples_per_symbol
            )
        ),
    )
    control_trajectory_indices = np.arange(
        control_trajectory_start, control_trajectory_stop, dtype=np.float64
    )
    control_trajectory_axis = (
        control_trajectory_indices - first_center
    ) / samples_per_symbol
    header_trajectory = (
        filtered_iq[control_trajectory_start:control_trajectory_stop]
        * np.exp(
            1j
            * (
                hdt_reference.phase_rad
                + hdt_reference.phase_step_rad_per_symbol
                * control_trajectory_axis
            )
        )
        * hdt_reference.amplitude
    )
    qpsk_alphabets = _hdt_qpsk_constellation(np.arange(_HDT_CONTROL_SYMBOLS))
    control_labels = np.argmin(
        np.abs(control_observed[:, None] - qpsk_alphabets), axis=1
    ).astype(np.int16)
    control_reference = qpsk_alphabets[
        np.arange(_HDT_CONTROL_SYMBOLS), control_labels
    ]
    control_encoded = (
        (control_labels[:, None] >> np.asarray([1, 0], dtype=np.int16)) & 1
    ).astype(np.uint8).reshape(-1)
    control_decoded, control_path_errors = _viterbi_decode_hdt_control(
        control_encoded
    )
    control_data = control_decoded[:57]
    control_tail = control_decoded[57:62]
    pca_a = _hdt_lsb_value(control_data[:16])
    nesn = _hdt_lsb_value(control_data[16:19])
    packet_format_indicator = int(control_data[19])
    rate_indicator = _hdt_lsb_value(control_data[20:23])
    control_rfu = int(control_data[23])
    pdu_octets = _hdt_lsb_value(control_data[24:33])
    received_hec = _hdt_msb_value(control_data[33:57])
    calculated_hec = hdt_crc24(
        control_data[:33], init=HDT_RF_TEST_PCA & 0xFF_FFFF
    )
    hec_valid = received_hec == calculated_hec
    rate = next(
        (
            candidate
            for candidate, definition in HDT_DEFINITIONS.items()
            if definition.rate_indicator == rate_indicator
        ),
        None,
    )
    if rate is None:
        raise RuntimeError(
            f"Unsupported HDT rate indicator 0b{rate_indicator:03b}"
        )
    if packet_format_indicator != 0 or pdu_octets < 1:
        raise RuntimeError("Bluetooth HDT analyzer currently requires packet format 0")
    payload_length = pdu_octets - 1
    definition = hdt_definition(rate)
    coded_payload_bits = hdt_coded_payload_bit_count(rate, payload_length)
    payload_symbol_count = int(
        np.ceil(coded_payload_bits / float(definition.bits_per_symbol))
    )
    payload_start = int(
        round(first_center - 0.5 * samples_per_symbol + _HDT_PAYLOAD_START_SYMBOL * samples_per_symbol)
    )
    payload_stop = int(round(payload_start + payload_symbol_count * samples_per_symbol))
    payload_measurement_symbol_count = min(1000, payload_symbol_count)
    payload_terminating_stop = int(
        round(
            payload_start
            + (payload_symbol_count + _HDT_TERMINATING_SYMBOLS)
            * samples_per_symbol
        )
    )
    if payload_terminating_stop > recording.sample_count:
        raise RuntimeError(
            f"HDT header declares {payload_length} payload byte(s), but the capture is incomplete"
        )

    payload_signal = _hdt_signal(rate)
    payload_session: VSASession | None = None
    format0_bit_count = (pdu_octets + 4) * 8

    def payload_labels(symbols: np.ndarray) -> np.ndarray:
        if definition.modulation in {"8PSK", "16QAM"}:
            alphabet = psk_constellation(
                payload_signal.modulation, BLUETOOTH_HDT_MAPPING
            )
            return np.argmin(
                np.abs(symbols[:, None] - alphabet[None, :]), axis=1
            ).astype(np.int16)
        qpsk_alphabets = _hdt_qpsk_constellation(
            np.arange(payload_symbol_count)
        )
        return np.argmin(
            np.abs(symbols[:, None] - qpsk_alphabets), axis=1
        ).astype(np.int16)

    # RF Test Packets have a known PRBS-9 payload, so Appendix C's internally
    # generated reference is available before any payload decision is made.
    prbs9 = prbs9_period()
    expected_payload_bits = prbs9[
        np.arange(payload_length * 8, dtype=np.int64) % prbs9.size
    ]
    expected_format0_bits = hdt_rf_test_format0_bits(expected_payload_bits)
    expected_coded_bits = puncture(
        convolutional_encode(expected_format0_bits),
        definition.payload_code_rate,
    )[:coded_payload_bits]
    expected_payload_reference = map_hdt_symbols(
        expected_coded_bits, rate
    )[:payload_symbol_count]
    payload_evm_reference = expected_payload_reference[
        :payload_measurement_symbol_count
    ]
    payload_estimate, payload_evm_measured = estimate_hdt_payload(
        filtered_iq,
        hdt_reference,
        samples_per_symbol=samples_per_symbol,
        start_symbol=_HDT_PAYLOAD_START_SYMBOL,
        payload_reference=payload_evm_reference,
    )
    payload_evm_first_center = (
        first_center + _HDT_PAYLOAD_START_SYMBOL * samples_per_symbol
    )
    payload_symbol_sample_positions = payload_evm_first_center + (
        np.arange(payload_measurement_symbol_count, dtype=np.float64)
        * samples_per_symbol
    )
    payload_trajectory_start = max(
        0, int(np.floor(payload_evm_first_center - 0.5 * samples_per_symbol))
    )
    payload_trajectory_stop = min(
        recording.sample_count,
        int(
            np.ceil(
                payload_evm_first_center
                + (payload_measurement_symbol_count - 0.5) * samples_per_symbol
            )
        ),
    )
    payload_trajectory_indices = np.arange(
        payload_trajectory_start, payload_trajectory_stop, dtype=np.float64
    )
    payload_trajectory_axis = (
        payload_trajectory_indices - payload_evm_first_center
    ) / samples_per_symbol
    payload_trajectory = (
        filtered_iq[payload_trajectory_start:payload_trajectory_stop]
        * hdt_reference.amplitude
        * np.exp(
            1j
            * (
                payload_estimate.phase_rad
                + payload_estimate.phase_step_rad_per_symbol
                * payload_trajectory_axis
            )
        )
    )
    payload_sig_corrected_all = apply_hdt_payload_estimate(
        filtered_iq,
        hdt_reference,
        payload_estimate,
        samples_per_symbol=samples_per_symbol,
        start_symbol=_HDT_PAYLOAD_START_SYMBOL,
        payload_symbol_offset=0,
        symbol_count=payload_symbol_count + _HDT_TERMINATING_SYMBOLS,
    )
    payload_sig_corrected = payload_sig_corrected_all[:payload_symbol_count]
    measurement_payload_labels = payload_labels(payload_sig_corrected)
    payload_air_bits = _symbols_to_air_bits(
        measurement_payload_labels, payload_signal.modulation.order
    )[:coded_payload_bits]

    format0_bits, payload_tail, payload_path_errors = _viterbi_decode_hdt_punctured(
        payload_air_bits,
        logical_bit_count=format0_bit_count,
        code_rate=definition.payload_code_rate,
    )
    pdu_bits = format0_bits[: pdu_octets * 8]
    payload_bits = pdu_bits[8:]
    received_crc = _hdt_msb_value(format0_bits[pdu_octets * 8 :])
    calculated_crc = hdt_crc32(pdu_bits, init=HDT_RF_TEST_CRC32_INIT)
    legacy_crc = hdt_crc32(pdu_bits, init=HDT_RF_TEST_PCA & 0xFF_FFFF)
    crc_valid = received_crc == calculated_crc
    legacy_crc_match = received_crc == legacy_crc

    # Appendix C requires an internally generated, fixed transmitted-symbol
    # reference.  Re-encode the decoded format-0 bitstream; do not substitute
    # nearest constellation decisions into the EVM reference.
    fixed_coded_payload_bits = puncture(
        convolutional_encode(format0_bits), definition.payload_code_rate
    )[:coded_payload_bits]
    fixed_payload_reference = map_hdt_symbols(
        fixed_coded_payload_bits, rate
    )[:payload_symbol_count]
    if fixed_payload_reference.size < payload_measurement_symbol_count:
        raise RuntimeError("HDT fixed payload reference is incomplete")
    payload_evm_reference = fixed_payload_reference[:payload_measurement_symbol_count]
    terminating_reference = map_hdt_symbols(
        np.zeros(
            _HDT_TERMINATING_SYMBOLS * definition.bits_per_symbol,
            dtype=np.uint8,
        ),
        rate,
    )
    terminating_measured = payload_sig_corrected_all[payload_symbol_count:]

    # Generic VSA remains a visualization product only.  SIG EVM, packet
    # decoding and the fixed reference above do not consume its resynchronised
    # decisions or carrier/timing estimates.
    if definition.modulation in {"8PSK", "16QAM"} and payload_symbol_count >= 4:
        seed_count = min(24, payload_symbol_count)
        seed_labels = measurement_payload_labels[:seed_count]
        pattern_symbols = reverse_symbol_bits(
            seed_labels, payload_signal.modulation.order
        )
        padding = int(round(8.0 * samples_per_symbol))
        crop_start = max(0, payload_start - padding)
        crop_stop = min(recording.sample_count, payload_stop + padding)
        payload_recording = replace(
            recording,
            iq=recording.iq[crop_start:crop_stop],
            start_sample_index=recording.start_sample_index + crop_start,
            trigger_sample_index=None,
        )
        payload_session = _analyze_known_pattern(
            payload_recording,
            payload_signal,
            pattern_symbols,
            result_length=max(1, payload_symbol_count),
            minimum_correlation=0.60,
            match_selection=MatchSelectionPolicy.STRONGEST,
        )
        payload_pattern = payload_session.pattern_result
        vsa_result = (
            payload_session.carrier_corrected_pattern_range_result
            or payload_session.pattern_range_result
            or payload_session.result
        )
        analysis_sample_offset = crop_start
    else:
        fallback_session = VSASession(name="Bluetooth HDT QPSK")
        fallback_session.set_recording(recording)
        fallback_session.set_signal(payload_signal)
        fallback_session.analyze()
        payload_session = fallback_session
        vsa_result = fallback_session.result
        analysis_sample_offset = 0
    if vsa_result is None:
        raise RuntimeError("Bluetooth HDT payload analysis produced no VSA result")

    fixed_control_reference = map_hdt_symbols(
        convolutional_encode(control_data), HDTRate.HDT2
    )
    hdt_evm = build_hdt_evm_result(
        reference=hdt_reference,
        header_measured_symbols=control_observed,
        header_reference_symbols=fixed_control_reference,
        payload_measured_symbols=payload_evm_measured,
        payload_reference_symbols=payload_evm_reference,
        payload_estimate=payload_estimate,
        terminating_measured_symbols=terminating_measured,
        terminating_reference_symbols=terminating_reference,
        header_corrected_waveform=header_trajectory,
        payload_corrected_waveform=payload_trajectory,
        header_symbol_sample_positions=control_symbol_sample_positions,
        payload_symbol_sample_positions=payload_symbol_sample_positions,
    )
    header_evm = hdt_evm.header_rms_percent
    payload_evm = hdt_evm.payload_rms_percent
    packet_bits = np.concatenate((control_data, format0_bits))
    packet_start = max(0, int(round(first_center - 0.5 * samples_per_symbol)))
    control_header_start_sample = int(
        round(
            first_center
            - 0.5 * samples_per_symbol
            + _HDT_TRAINING_SYMBOLS * samples_per_symbol
        )
    )
    control_header_stop_sample = int(
        round(control_header_start_sample + _HDT_CONTROL_SYMBOLS * samples_per_symbol)
    )
    payload_evm_stop_sample = int(
        round(payload_start + payload_measurement_symbol_count * samples_per_symbol)
    )
    hdt_plot_data = HDTPlotData(
        evm=hdt_evm,
        packet_sample_range=(packet_start, payload_stop),
        training_sample_range=(packet_start, control_header_start_sample),
        control_header_sample_range=(
            control_header_start_sample,
            control_header_stop_sample,
        ),
        payload_sample_range=(payload_start, payload_stop),
        payload_evm_sample_range=(payload_start, payload_evm_stop_sample),
    )
    control_children = (
        PacketField("pca_a", "PCA-A", 0, 16, control_data[:16], f"0x{pca_a:04X}"),
        PacketField("nesn", "NESN", 16, 19, control_data[16:19], nesn),
        PacketField("pfi", "PFI", 19, 20, control_data[19:20], packet_format_indicator, "Packet format 0"),
        PacketField("rate_indicator", "Rate Indicator", 20, 23, control_data[20:23], f"{rate.value} (0b{rate_indicator:03b})", f"{rate.value}: {definition.modulation}, code rate {definition.payload_code_rate}", FieldStatus.VALID),
        PacketField("rfu", "RFU", 23, 24, control_data[23:24], control_rfu),
        PacketField("pdu_control", "PDU Control", 24, 33, control_data[24:33], pdu_octets, f"{pdu_octets} octet(s), excluding CRC"),
        PacketField("hec_c", "HEC-C", 33, 57, control_data[33:57], f"0x{received_hec:06X}", f"Calculated 0x{calculated_hec:06X}", FieldStatus.VALID if hec_valid else FieldStatus.INVALID),
    )
    payload_offset = 57
    pdu_stop = payload_offset + pdu_bits.size
    payload_children = (
        PacketField("pdu_header", "PDU Header", payload_offset, payload_offset + 8, pdu_bits[:8], f"0x{_hdt_lsb_value(pdu_bits[:8]):02X}"),
        PacketField("payload_body", "Payload", payload_offset + 8, pdu_stop, payload_bits, f"{payload_bits.size // 8} byte(s)"),
        PacketField("crc32", "CRC-32", pdu_stop, pdu_stop + 32, format0_bits[-32:], f"0x{received_crc:08X}", f"Calculated 0x{calculated_crc:08X}", FieldStatus.VALID if crc_valid else FieldStatus.INVALID),
    )
    issues: tuple[PacketIssue, ...] = tuple(
        issue
        for issue in (
            None if hec_valid else PacketIssue("invalid_hec_c", "HEC-C does not match the Control Header", IssueSeverity.ERROR, 33, 57),
            None if crc_valid else PacketIssue("invalid_crc32", "CRC-32 does not match the PDU Header and Payload" + ("; received value matches legacy 0x00555555 initialization" if legacy_crc_match else ""), IssueSeverity.ERROR, pdu_stop, pdu_stop + 32),
        )
        if issue is not None
    )
    packet = PacketAnalysisResult(
        "1.0",
        "bluetooth.hdt",
        "Bluetooth HDT",
        rate.value,
        rate.value,
        (
            PacketSummaryItem("protocol", "Protocol", "Bluetooth HDT", "Bluetooth HDT"),
            PacketSummaryItem("phy", "Detected PHY", rate.value, rate.value),
            PacketSummaryItem("payload_length", "Payload Length", payload_length, f"{payload_length} byte(s)"),
            PacketSummaryItem("hec_c", "HEC-C", hec_valid, "Pass" if hec_valid else "Fail", FieldStatus.VALID if hec_valid else FieldStatus.INVALID),
            PacketSummaryItem("crc32", "CRC-32", crc_valid, "Pass" if crc_valid else "Fail", FieldStatus.VALID if crc_valid else FieldStatus.INVALID),
        ),
        (
            PacketField("training", "Training / Preamble", 0, 0, np.empty(0, dtype=np.uint8), "74 symbols", "STS x9 + GI + LTS x2", FieldStatus.VALID),
            PacketField("control_header", "Control Header", 0, 57, control_data, f"RI=0b{rate_indicator:03b}, PDU={pdu_octets} octets", "RF PHY test Control Header", FieldStatus.VALID if hec_valid else FieldStatus.INVALID, control_children),
            PacketField("payload", "PDU Header / Payload / CRC", payload_offset, payload_offset + format0_bits.size, format0_bits, f"{payload_length} payload byte(s)", "Packet format 0 decoded bitstream", FieldStatus.VALID if crc_valid else FieldStatus.INVALID, payload_children),
        ),
        issues,
        PacketIntegritySummary(hec_valid, crc_valid, True),
        PacketSourceInfo("iq", int(match_index) - 1, None, recording.center_frequency_hz, packet_start, payload_stop),
        packet_bits,
    )
    cfo_hz = (
        hdt_reference.carrier_error_rad_per_symbol
        * _HDT_SYMBOL_RATE_HZ
        / (2.0 * np.pi)
    )
    payload_cfo_hz = (
        hdt_evm.payload_estimate.carrier_error_rad_per_symbol
        * _HDT_SYMBOL_RATE_HZ
        / (2.0 * np.pi)
    )
    header_power_dbm = _recording_range_power_dbm(
        recording, packet_start, payload_start
    )
    payload_power_dbm = _recording_range_power_dbm(
        recording, payload_start, payload_stop
    )
    # Provisional whole-packet value for visibility only.  This is deliberately
    # kept separate from the generic burst-power result: the HDT RF PHY Test
    # Suite window/filter/detector requirements and limits are not yet encoded,
    # so this value must not produce a conformance verdict.
    output_power_dbm = _recording_range_power_dbm(
        recording, packet_start, payload_terminating_stop
    )
    relative_power_db = (
        payload_power_dbm - header_power_dbm
        if payload_power_dbm is not None and header_power_dbm is not None
        else None
    )
    hdt_eligibility = _rf_test_eligibility(
        profile=profile,
        whitening_enabled=False,
        sample_rate_hz=recording.sample_rate_hz,
        minimum_sample_rate_hz=4_000_000.0,
    )
    hdt_eligibility_text = (
        "Eligible"
        if hdt_eligibility.eligible
        else "N/A - " + "; ".join(hdt_eligibility.reasons)
    )
    header_evm_db = (
        float("-inf")
        if header_evm == 0.0
        else float(20.0 * np.log10(header_evm / 100.0))
    )
    payload_evm_db = (
        float("-inf")
        if payload_evm == 0.0
        else float(20.0 * np.log10(payload_evm / 100.0))
    )
    header_evm_limit_db = HDT_HEADER_EVM_LIMIT_DB
    payload_evm_limit_db = HDT_PAYLOAD_EVM_LIMIT_DB[rate.value]
    center_frequency_deviation_hz = max(
        (cfo_hz, payload_cfo_hz), key=abs
    )
    frequency_offset_change_hz = abs(payload_cfo_hz - cfo_hz)
    frequency_offset_change_limit_hz = (
        HDT_FREQUENCY_OFFSET_CHANGE_LIMIT_HZ[rate.value]
    )
    pre_packet_emissions_s = measure_pre_packet_emissions(
        recording.iq,
        packet_start_sample=packet_start,
        packet_stop_sample=payload_terminating_stop,
        sample_rate_hz=recording.sample_rate_hz,
    )

    def result_text(passed: bool) -> str:
        if not hdt_eligibility.eligible:
            return "N/A"
        return "PASS" if passed else "FAIL"

    measurement_group = "RF PHY Measurements"
    reference_group = "Reference Information"
    metrics = (
        BluetoothMetric("payload_modulation", "Payload Modulation", definition.modulation),
        BluetoothMetric("payload_code_rate", "Payload Code Rate", definition.payload_code_rate),
        BluetoothMetric("payload_length", "Payload Length", f"{payload_length} byte(s)"),
        BluetoothMetric("pdu_length", "PDU Control Length", f"{pdu_octets} byte(s)"),
        BluetoothMetric("hec_c", "HEC-C", "Pass" if hec_valid else f"Fail (Rx 0x{received_hec:06X}, Calc 0x{calculated_hec:06X})"),
        BluetoothMetric("crc32", "CRC-32", "Pass" if crc_valid else f"Fail (Rx 0x{received_crc:08X}, Calc 0x{calculated_crc:08X}" + (", legacy-init match" if legacy_crc_match else "") + ")"),
        BluetoothMetric("automatic_result_range", "Payload Result Range", f"{payload_symbol_count} symbol(s) (automatic)"),
        BluetoothMetric(
            "sig_hdt_output_power",
            "Output power",
            _display(output_power_dbm, "dBm"),
            OUTPUT_POWER_LIMIT_DEPENDENCY,
            "N/A",
            measurement_group,
        ),
        BluetoothMetric(
            "sig_hdt_header_evm_rms",
            "Control Header RMS EVM",
            _display_evm(header_evm),
            "\N{LESS-THAN OR EQUAL TO} -10 dB",
            result_text(header_evm_db <= header_evm_limit_db),
            measurement_group,
        ),
        BluetoothMetric(
            "sig_hdt_payload_evm_rms",
            "PDU Header and payload RMS EVM",
            _display_evm(payload_evm),
            f"\N{LESS-THAN OR EQUAL TO} {payload_evm_limit_db:.0f} dB",
            result_text(payload_evm_db <= payload_evm_limit_db),
            measurement_group,
        ),
        BluetoothMetric(
            "sig_hdt_center_frequency_deviation",
            "Center frequency deviation",
            _display(center_frequency_deviation_hz, "kHz", 1e3),
            "\N{PLUS-MINUS SIGN}125 kHz",
            result_text(abs(center_frequency_deviation_hz) <= 125_000.0),
            measurement_group,
        ),
        BluetoothMetric(
            "sig_hdt_frequency_offset_change",
            "Center frequency offset change between the preamble and the payload",
            f"{frequency_offset_change_hz / 1e3:.3f} kHz",
            (
                "\N{LESS-THAN OR EQUAL TO} "
                f"{frequency_offset_change_limit_hz / 1e3:.1f} kHz"
            ),
            result_text(
                frequency_offset_change_hz
                <= frequency_offset_change_limit_hz
            ),
            measurement_group,
        ),
        BluetoothMetric(
            "sig_hdt_symbol_timing_accuracy",
            "Symbol timing accuracy",
            "N/A",
            "< \N{PLUS-MINUS SIGN}50 ppm",
            "N/A",
            measurement_group,
        ),
        BluetoothMetric(
            "sig_hdt_pre_packet_emissions",
            "Pre-packet emissions",
            (
                "N/A"
                if pre_packet_emissions_s is None
                else f"{pre_packet_emissions_s / 1e-6:.2f} \N{MICRO SIGN}s"
            ),
            "\N{LESS-THAN OR EQUAL TO} 4 \N{MICRO SIGN}s",
            (
                "N/A"
                if pre_packet_emissions_s is None
                else result_text(
                    pre_packet_emissions_s <= HDT_PRE_PACKET_EMISSIONS_LIMIT_S
                )
            ),
            measurement_group,
        ),
        BluetoothMetric(
            "detected_phy", "Detected PHY", rate.value, group=reference_group
        ),
        BluetoothMetric(
            "sig_eligibility",
            "RF Test Eligibility",
            hdt_eligibility_text,
            group=reference_group,
        ),
        BluetoothMetric(
            "sig_hdt_preamble_carrier_error",
            "Preamble Carrier Frequency Error",
            _display(cfo_hz, "kHz", 1e3),
            group=reference_group,
        ),
        BluetoothMetric(
            "sig_hdt_payload_carrier_error",
            "Payload Carrier Frequency Error",
            _display(payload_cfo_hz, "kHz", 1e3),
            group=reference_group,
        ),
        BluetoothMetric(
            "sig_hdt_header_average_power",
            "Control Header Average Power",
            _display(header_power_dbm, "dBm"),
            group=reference_group,
        ),
        BluetoothMetric(
            "sig_hdt_payload_average_power",
            "PDU Header and Payload Average Power",
            _display(payload_power_dbm, "dBm"),
            group=reference_group,
        ),
        BluetoothMetric(
            "sig_hdt_relative_power",
            "Relative Power (Payload - Header)",
            _display(relative_power_db, "dB"),
            group=reference_group,
        ),
        BluetoothMetric(
            "sig_hdt_training_correlation",
            "Preamble Correlation",
            f"{100.0 * training_correlation:.2f} %",
            group=reference_group,
        ),
        BluetoothMetric(
            "sig_hdt_evm_packets_evaluated",
            "RMS EVM Packets Evaluated",
            f"{1 if hdt_eligibility.eligible else 0} / 1500",
            group=reference_group,
        ),
    )
    header_spectrum_frequency_hz, header_spectrum_dbm = (
        _recording_range_spectrum_dbm(
            recording,
            control_header_start_sample,
            control_header_stop_sample,
        )
    )
    payload_spectrum_frequency_hz, payload_spectrum_dbm = (
        _recording_range_spectrum_dbm(recording, payload_start, payload_stop)
    )
    return BluetoothDedicatedResult(
        profile=BluetoothAnalysisProfile(profile),
        vsa_result=vsa_result,
        packet=packet,
        metrics=metrics,
        metadata={
            "source": recording.source,
            "sample_rate_hz": recording.sample_rate_hz,
            "center_frequency_hz": recording.center_frequency_hz,
            "analysis_session": payload_session,
            "hdt_header_symbols": hdt_evm.header_corrected_symbols,
            "hdt_header_vector_symbols": np.asarray(
                hdt_evm.header_corrected_symbols, dtype=np.complex64
            ),
            "hdt_header_trajectory": hdt_evm.header_corrected_waveform,
            "hdt_payload_symbols": hdt_evm.payload_corrected_symbols,
            "hdt_payload_vector_symbols": hdt_evm.payload_corrected_symbols,
            "hdt_payload_trajectory": hdt_evm.payload_corrected_waveform,
            "hdt_header_reference_symbols": hdt_evm.header_reference_symbols,
            "hdt_payload_reference_symbols": hdt_evm.payload_reference_symbols,
            "hdt_header_symbol_sample_positions": (
                hdt_evm.header_symbol_sample_positions
            ),
            "hdt_payload_symbol_sample_positions": (
                hdt_evm.payload_symbol_sample_positions
            ),
            "hdt_header_spectrum_frequency_hz": header_spectrum_frequency_hz,
            "hdt_header_spectrum_dbm": header_spectrum_dbm,
            "hdt_header_spectrum_sample_range": (
                control_header_start_sample,
                control_header_stop_sample,
            ),
            "hdt_payload_spectrum_frequency_hz": payload_spectrum_frequency_hz,
            "hdt_payload_spectrum_dbm": payload_spectrum_dbm,
            "hdt_payload_spectrum_sample_range": (payload_start, payload_stop),
            "hdt_header_evm_rms_percent": header_evm,
            "hdt_payload_evm_rms_percent": payload_evm,
            "hdt_evm_result": hdt_evm,
            "hdt_plot_data": hdt_plot_data,
            "hdt_reference_amplitude": hdt_reference.amplitude,
            "hdt_reference_phase_rad": hdt_reference.phase_rad,
            "hdt_reference_phase_step_rad_per_symbol": (
                hdt_reference.carrier_error_rad_per_symbol
            ),
            "hdt_reference_timing_offset_samples": (
                hdt_reference.timing_offset_samples
            ),
            "hdt_alpha0": hdt_reference.amplitude,
            "hdt_phi0_rad": hdt_reference.phase_rad,
            "hdt_delta_omega0_rad_per_symbol": (
                hdt_reference.phase_step_rad_per_symbol
            ),
            "hdt_t0_sample": hdt_reference.first_symbol_center_sample,
            "hdt_phi1_rad": hdt_evm.payload_estimate.phase_rad,
            "hdt_delta_omega1_rad_per_symbol": (
                hdt_evm.payload_estimate.phase_step_rad_per_symbol
            ),
            "hdt_preamble_carrier_error_hz": cfo_hz,
            "hdt_payload_carrier_error_hz": payload_cfo_hz,
            "hdt_center_frequency_deviation_hz": (
                center_frequency_deviation_hz
            ),
            "hdt_frequency_offset_change_hz": frequency_offset_change_hz,
            "hdt_frequency_offset_change_limit_hz": (
                frequency_offset_change_limit_hz
            ),
            "hdt_symbol_timing_accuracy_ppm": None,
            "hdt_pre_packet_emissions_s": pre_packet_emissions_s,
            "hdt_output_power_dbm": output_power_dbm,
            "hdt_output_power_measurement_status": "provisional",
            "hdt_output_power_window_start_sample": packet_start,
            "hdt_output_power_window_stop_sample": payload_terminating_stop,
            "hdt_payload_evm_symbol_count": payload_measurement_symbol_count,
            "hdt_payload_evm_stop_sample": payload_evm_stop_sample,
            "hdt_payload_terminating_symbol_count": _HDT_TERMINATING_SYMBOLS,
            "hdt_payload_reference_source": "decoded_reencoded_bits",
            "hdt_rate_indicator": rate_indicator,
            "hdt_pca_a": pca_a,
            "hdt_nesn": nesn,
            "hdt_packet_format_indicator": packet_format_indicator,
            "hdt_pdu_control_octets": pdu_octets,
            "hdt_received_hec_c": received_hec,
            "hdt_calculated_hec_c": calculated_hec,
            "hdt_hec_c_valid": hec_valid,
            "hdt_received_crc32": received_crc,
            "hdt_calculated_crc32": calculated_crc,
            "hdt_legacy_init_crc32": legacy_crc,
            "hdt_legacy_init_crc32_match": legacy_crc_match,
            "hdt_crc32_valid": crc_valid,
            "hdt_control_path_errors": control_path_errors,
            "hdt_control_tail_bits": control_tail,
            "hdt_payload_path_errors": payload_path_errors,
            "hdt_payload_tail_bits": payload_tail,
            "hdt_payload_symbol_count": payload_symbol_count,
            "hdt_payload_start_sample": payload_start,
            "hdt_payload_stop_sample": payload_stop,
            "recording_sample_offset": 0,
            "analysis_sample_offset": analysis_sample_offset,
            "packet_start_sample": packet_start,
            "packet_stop_sample": payload_stop,
            "selected_match_index": int(match_index),
            "eligible_match_count": len(matches),
            "rf_measurements": (
                BluetoothRFMeasurementResult(
                    test_case_id="bluetooth.hdt.evm",
                    eligibility=hdt_eligibility,
                    verdict=(
                        RFTestVerdict.PASS
                        if hdt_eligibility.eligible
                        and header_evm_db <= header_evm_limit_db
                        and payload_evm_db <= payload_evm_limit_db
                        else RFTestVerdict.FAIL
                        if hdt_eligibility.eligible
                        else RFTestVerdict.NOT_APPLICABLE
                    ),
                    metrics={
                        "header_rms_evm_percent": header_evm,
                        "payload_rms_evm_percent": payload_evm,
                        "header_rms_evm_db": header_evm_db,
                        "payload_rms_evm_db": payload_evm_db,
                        "header_limit_db": header_evm_limit_db,
                        "payload_limit_db": payload_evm_limit_db,
                        "header_pass": header_evm_db <= header_evm_limit_db,
                        "payload_pass": payload_evm_db <= payload_evm_limit_db,
                        "carrier_error_hz": cfo_hz,
                        "payload_carrier_error_hz": payload_cfo_hz,
                        "center_frequency_deviation_hz": (
                            center_frequency_deviation_hz
                        ),
                        "frequency_offset_change_hz": (
                            frequency_offset_change_hz
                        ),
                        "frequency_offset_change_limit_hz": (
                            frequency_offset_change_limit_hz
                        ),
                        "symbol_timing_accuracy_ppm": None,
                        "pre_packet_emissions_s": pre_packet_emissions_s,
                        "output_power_dbm": output_power_dbm,
                        "alpha0": hdt_reference.amplitude,
                        "phi0_rad": hdt_reference.phase_rad,
                        "delta_omega0_rad_per_symbol": (
                            hdt_reference.phase_step_rad_per_symbol
                        ),
                        "t0_sample": hdt_reference.first_symbol_center_sample,
                        "phi1_rad": hdt_evm.payload_estimate.phase_rad,
                        "delta_omega1_rad_per_symbol": (
                            hdt_evm.payload_estimate.phase_step_rad_per_symbol
                        ),
                        "header_average_power_dbm": header_power_dbm,
                        "payload_average_power_dbm": payload_power_dbm,
                        "relative_power_db": relative_power_db,
                    },
                    arrays={
                        "header_measured_symbols": hdt_evm.header_measured_symbols,
                        "header_reference_symbols": hdt_evm.header_reference_symbols,
                        "payload_measured_symbols": hdt_evm.payload_measured_symbols,
                        "payload_reference_symbols": hdt_evm.payload_reference_symbols,
                        "header_corrected_waveform": (
                            hdt_evm.header_corrected_waveform
                        ),
                        "payload_corrected_waveform": (
                            hdt_evm.payload_corrected_waveform
                        ),
                        "header_symbol_sample_positions": (
                            hdt_evm.header_symbol_sample_positions
                        ),
                        "payload_symbol_sample_positions": (
                            hdt_evm.payload_symbol_sample_positions
                        ),
                        "terminating_measured_symbols": (
                            hdt_evm.terminating_measured_symbols
                        ),
                        "terminating_reference_symbols": (
                            hdt_evm.terminating_reference_symbols
                        ),
                    },
                    metadata={
                        "fractional_timing_offset_samples": hdt_reference.timing_offset_samples,
                        "terminating_symbols_included": False,
                        "terminating_symbols_held_separately": True,
                        "payload_reference_source": "decoded_reencoded_bits",
                        "payload_evm_symbol_limit": 1000,
                        "output_power_measurement_status": "provisional",
                        "output_power_limit": OUTPUT_POWER_LIMIT_DEPENDENCY,
                        "output_power_window_start_sample": packet_start,
                        "output_power_window_stop_sample": payload_terminating_stop,
                        "appendix_c_final_audit": True,
                    },
                ),
            ),
        },
    )


def analyze_bluetooth_hdt_recordings(
    recording: IQRecording,
    *,
    profile: BluetoothAnalysisProfile,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[BluetoothDedicatedResult, ...]:
    """Return every complete HDT packet found in a capture."""

    matches = _hdt_training_matches(recording)
    results: list[BluetoothDedicatedResult] = []
    for match_index in range(1, len(matches) + 1):
        if cancelled is not None and cancelled():
            break
        try:
            results.append(
                analyze_bluetooth_hdt_recording(
                    recording,
                    profile=profile,
                    match_index=match_index,
                    _matches=matches,
                )
            )
        except RuntimeError:
            continue
    if not results:
        raise RuntimeError("Bluetooth HDT synchronization pattern was not found")
    return _attach_rf_capture_aggregates(results)


def analyze_bluetooth_classic_recording(
    recording: IQRecording,
    *,
    profile: BluetoothAnalysisProfile,
    lap: int,
    uap: int,
    clock_6_1: int,
    whitening_enabled: bool = True,
    result_length: int = 4096,
    match_index: int = 1,
    iq_power_trigger: IQPowerTriggerSettings | None = None,
    _recording_sample_offset: int = 0,
) -> BluetoothDedicatedResult:
    """Decode the BR header first and automatically select BR/EDR PHY.

    Classic Bluetooth always starts with the BR access code and GFSK header.
    Ambiguous TYPE values are disambiguated by requiring the appropriate EDR
    synchronization word to correlate in the following PSK region.
    """

    access = access_code_bits(int(lap) & 0xFFFFFF)
    # Keep a dedicated BR/GFSK result even for EDR packets.  The Bluetooth
    # workspace uses it for the access/header spectrum and modulation panes,
    # while the PSK session below owns the EDR payload products.
    br_analysis_session = _analyze_known_pattern(
        recording,
        _classic_signal(BluetoothClassicPhy.BR),
        access,
        result_length=126,
        minimum_correlation=0.60,
        match_index=match_index,
        iq_power_trigger=iq_power_trigger,
    )
    # Burst Search and its trigger windows are authoritative.  Decode the BR
    # header from the same selected candidate, rather than allowing the
    # profile correlator to pick an earlier noise candidate from the capture.
    frontend_sample_offset = 0
    frontend_recording = recording
    frontend_match_index = match_index
    # Always bind the semantic BR-header decode to the exact access-code
    # candidate selected by the shared VSA synchronizer.  Re-running the BR
    # profile over the complete capture allowed packet 2+ to attach to packet
    # 1's header/EDR payload and also repeated an expensive full-capture
    # search for every result.
    selected = br_analysis_session.pattern_result
    samples_per_br_symbol = recording.sample_rate_hz / 1_000_000.0
    frontend_sample_offset = max(
        0, int(selected.pattern_start_sample - 8 * samples_per_br_symbol)
    )
    frontend_stop = min(
        recording.sample_count,
        int(selected.pattern_start_sample + 192 * samples_per_br_symbol),
    )
    frontend_recording = replace(
        recording,
        iq=recording.iq[frontend_sample_offset:frontend_stop],
        start_sample_index=recording.start_sample_index + frontend_sample_offset,
        trigger_sample_index=None,
    )
    frontend_match_index = 1
    br_frontend = BluetoothBRProfile(access).analyze(
        frontend_recording,
        clock_6_1=int(clock_6_1),
        uap=int(uap) & 0xFF,
        whitening_enabled=bool(whitening_enabled),
        minimum_correlation=0.60,
        match_index=frontend_match_index,
    )
    if br_frontend.header is None:
        raise RuntimeError("Bluetooth Classic header could not be decoded")

    phy = BluetoothClassicPhy.BR
    analysis_session: VSASession | None = None
    analysis_sample_offset = 0
    edr_candidate = _edr_candidate_for_type(br_frontend.header.packet_type)
    edr_error: str | None = None
    detected_edr_sync_start: int | None = None
    expected_edr_sync_symbols = np.empty(0, dtype=np.int16)
    expected_edr_decoded_sync_symbols = np.empty(0, dtype=np.int16)
    if edr_candidate is not None:
        width = 2 if edr_candidate is BluetoothClassicPhy.EDR_2M else 3
        # ``edr_sync_symbols`` is expressed as physical differential phase
        # indices, while PatternAnalyzer consumes the logical symbol numbers
        # selected by SignalDescription.symbol_mapping.  Convert through the
        # Bluetooth mapping before applying the R&S-style LSB symbol display
        # order.  Treating phase indices as logical values made TYPE 0x4/0x8
        # packets fall back to BR (and consequently broke EDR Length/CRC).
        edr_signal = _classic_signal(edr_candidate)
        expected_edr_decoded_sync_symbols = phase_indices_to_logical_symbols(
            edr_signal.modulation,
            BLUETOOTH_EDR_MAPPING,
            edr_sync_symbols(width),
        )
        sync = reverse_symbol_bits(
            expected_edr_decoded_sync_symbols,
            2**width,
        )
        try:
            samples_per_br_symbol = recording.sample_rate_hz / 1_000_000.0
            # EDR carries 2 or 3 bits per symbol, but both PHYs retain the
            # 1-Msym/s symbol clock.  Using the bit rate here shortened the
            # crop and made the shared VSA filter/synchronizer see a partial
            # PSK result range.
            psk_symbol_rate_hz = edr_signal.symbol_rate_hz
            samples_per_psk_symbol = recording.sample_rate_hz / psk_symbol_rate_hz
            # BR/EDR packets switch PHY at a deterministic boundary:
            # 72-symbol access code + 54-symbol BR header + 5 us guard.  The
            # previous broad crop began before the access code and selected
            # the strongest sync anywhere in the following recording.  With
            # multiple packets this could attach the current BR header to a
            # distant EDR payload, corrupting power and vector results.
            edr_sync_start = (
                frontend_sample_offset
                + br_frontend.demodulation.access_start_sample
                + int(round(131.0 * samples_per_br_symbol))
            )
            sync_search_start, sync_search_stop, sync_timing_tolerance = (
                _edr_sync_search_bounds(
                    expected_start_sample=edr_sync_start,
                    sync_symbol_count=int(sync.size),
                    recording_sample_count=recording.sample_count,
                    samples_per_br_symbol=samples_per_br_symbol,
                    samples_per_psk_symbol=samples_per_psk_symbol,
                )
            )
            sync_recording = replace(
                recording,
                iq=recording.iq[sync_search_start:sync_search_stop],
                start_sample_index=(
                    recording.start_sample_index + sync_search_start
                ),
                trigger_sample_index=None,
            )
            sync_session = _analyze_known_pattern(
                sync_recording,
                edr_signal,
                sync,
                result_length=int(sync.size) + 2,
                minimum_correlation=0.72,
                match_index=1,
                # Guard/ramp transients can produce an earlier, merely
                # acceptable 10-symbol correlation.  FIRST then rejects a
                # valid EDR packet as BR before reaching the true sync.  This
                # recording is already restricted to the deterministic PHY
                # boundary, so STRONGEST is safe and is the correct local
                # maximum-likelihood decision.
                match_selection=MatchSelectionPolicy.STRONGEST,
            )
            detected_sync_start = (
                sync_search_start
                + int(sync_session.pattern_result.pattern_start_sample)
            )
            detected_edr_sync_start = detected_sync_start
            expected_edr_sync_symbols = np.asarray(sync, dtype=np.int16)
            sync_timing_error = detected_sync_start - int(edr_sync_start)
            if abs(sync_timing_error) > sync_timing_tolerance:
                raise RuntimeError(
                    "EDR synchronization was not found at the expected "
                    f"post-header boundary (timing error {sync_timing_error} samples)"
                )

            # The narrow pass above decides BR versus EDR. Only after that
            # decision do we open a longer range for payload demodulation.
            # Anchor it to the confirmed sync so a later packet cannot be
            # associated with the current header.
            crop_start = sync_search_start
            crop_stop = min(
                recording.sample_count,
                int(
                    detected_sync_start
                    + (sync.size + result_length + 8) * samples_per_psk_symbol
                ),
            )
            edr_recording = replace(
                recording,
                iq=recording.iq[crop_start:crop_stop],
                start_sample_index=recording.start_sample_index + crop_start,
                trigger_sample_index=None,
            )
            candidate_session = _analyze_edr_payload_at_sync(
                edr_recording,
                edr_signal,
                sync,
                result_length=result_length,
                expected_sync_sample=detected_sync_start - crop_start,
                minimum_correlation=0.72,
            )
            payload_sync_start = (
                crop_start
                + int(candidate_session.pattern_result.pattern_start_sample)
            )
            if abs(payload_sync_start - detected_sync_start) > sync_timing_tolerance:
                raise RuntimeError(
                    "EDR payload analysis did not remain anchored to the "
                    "post-header synchronization word"
                )
            correlation = float(candidate_session.pattern_result.correlation)
            if correlation >= 0.72:
                # First pass establishes EDR sync and decodes the enhanced
                # ACL header.  Its Length field is authoritative for the
                # packet end; the capture/result setting is only a generous
                # discovery bound.
                provisional_pattern = candidate_session.pattern_result
                provisional_air_bits = _symbols_to_air_bits(
                    provisional_pattern.decoded_symbols,
                    candidate_session.signal.modulation.order,
                )
                provisional_packet = analyze_demodulated_packet_bits(
                    np.concatenate(
                        (
                            br_frontend.access_code_bits,
                            br_frontend.header_air_bits,
                            provisional_air_bits,
                        )
                    ),
                    protocol_id="bluetooth.br_edr",
                    phy_name=edr_candidate.value,
                    context={
                        "uap": int(uap) & 0xFF,
                        "clock_6_1": int(clock_6_1),
                        "whitening_enabled": bool(whitening_enabled),
                        "phy": edr_candidate.value,
                    },
                    packet_index=0,
                    center_frequency_hz=recording.center_frequency_hz,
                    start_sample=provisional_pattern.result_start_sample,
                    stop_sample=provisional_pattern.result_stop_sample,
                )
                exact_result_symbols = _exact_edr_result_symbols(
                    provisional_packet, bits_per_symbol=width
                )
                if exact_result_symbols is not None:
                    candidate_session = _analyze_edr_payload_at_sync(
                        edr_recording,
                        edr_signal,
                        sync,
                        result_length=exact_result_symbols,
                        expected_sync_sample=detected_sync_start - crop_start,
                        minimum_correlation=0.72,
                    )
                phy = edr_candidate
                analysis_session = candidate_session
                analysis_sample_offset = crop_start
        except Exception as error:
            edr_error = str(error)

    if analysis_session is None:
        # No EDR synchronization word was present at the deterministic PHY
        # switch boundary, so this is a BR packet even when TYPE is shared
        # with an EDR family.  Decode a generous first pass to obtain the ACL
        # Length, then repeat with the exact packet range.  TYPE/slot capacity
        # must never be used as the observed packet length.
        br_packet_session = _analyze_known_pattern(
            recording,
            _classic_signal(BluetoothClassicPhy.BR),
            access,
            result_length=result_length,
            minimum_correlation=0.60,
            match_index=match_index,
            iq_power_trigger=iq_power_trigger,
        )
        provisional_pattern = br_packet_session.pattern_result
        provisional_packet = analyze_demodulated_packet_bits(
            provisional_pattern.decoded_bits,
            protocol_id="bluetooth.br_edr",
            phy_name=BluetoothClassicPhy.BR.value,
            context={
                "uap": int(uap) & 0xFF,
                "clock_6_1": int(clock_6_1),
                "whitening_enabled": bool(whitening_enabled),
                "phy": BluetoothClassicPhy.BR.value,
            },
            packet_index=0,
            center_frequency_hz=recording.center_frequency_hz,
            start_sample=provisional_pattern.result_start_sample,
            stop_sample=provisional_pattern.result_stop_sample,
        )
        exact_result_symbols = _exact_br_result_symbols(provisional_packet)
        if exact_result_symbols is not None:
            br_packet_session = _analyze_known_pattern(
                recording,
                _classic_signal(BluetoothClassicPhy.BR),
                access,
                result_length=exact_result_symbols,
                minimum_correlation=0.60,
                match_index=match_index,
                iq_power_trigger=iq_power_trigger,
            )
        analysis_session = br_packet_session

    pattern = analysis_session.pattern_result
    if phy is BluetoothClassicPhy.BR:
        packet_bits = pattern.decoded_bits
    else:
        edr_air_bits = _symbols_to_air_bits(
            pattern.decoded_symbols, analysis_session.signal.modulation.order
        )
        packet_bits = np.concatenate(
            (
                br_frontend.access_code_bits,
                br_frontend.header_air_bits,
                edr_air_bits,
            )
        )
    context = {
        "uap": int(uap) & 0xFF,
        "clock_6_1": int(clock_6_1),
        "whitening_enabled": bool(whitening_enabled),
        "phy": phy.value,
    }
    packet = analyze_demodulated_packet_bits(
        packet_bits,
        protocol_id="bluetooth.br_edr",
        phy_name=phy.value,
        context=context,
        packet_index=0,
        center_frequency_hz=recording.center_frequency_hz,
        start_sample=pattern.result_start_sample,
        stop_sample=pattern.result_stop_sample,
    )
    vsa_result = (
        analysis_session.carrier_corrected_pattern_range_result
        or analysis_session.pattern_range_result
        or analysis_session.result
    )
    if vsa_result is None:
        raise RuntimeError("Bluetooth PHY analysis produced no VSA result")
    br_vsa_result = (
        br_analysis_session.carrier_corrected_pattern_range_result
        or br_analysis_session.pattern_range_result
        or br_analysis_session.result
    )
    fsk_power_dbm = (
        _mean_power_dbm(br_vsa_result.power_dbm)
        if br_vsa_result is not None
        else None
    )
    psk_power_dbm = (
        _mean_power_dbm(vsa_result.power_dbm)
        if phy is not BluetoothClassicPhy.BR
        else None
    )
    cfo_hz = float(pattern.carrier_frequency_offset_hz)
    duration_ms = max(0, pattern.result_stop_sample - pattern.result_start_sample) / recording.sample_rate_hz * 1e3
    try:
        rate_error = float(vsa_result.metadata.get("symbol_rate_error_ppm"))
    except (TypeError, ValueError):
        rate_error = None
    metrics = [
        BluetoothMetric("detected_phy", "Detected PHY", phy.value),
        BluetoothMetric(
            "header_type", "Classic Header TYPE", f"0x{br_frontend.header.packet_type:X}"
        ),
        BluetoothMetric("profile", "Analysis Profile", BluetoothAnalysisProfile(profile).value),
        BluetoothMetric("packet_power", "Packet Average Power", _display(_mean_power_dbm(vsa_result.power_dbm), "dBm")),
        BluetoothMetric("peak_power", "Peak Power", _display(_finite_stat(vsa_result.power_dbm, peak=True), "dBm")),
        BluetoothMetric("cfo", "Carrier Frequency Offset", _display(cfo_hz, "kHz", 1e3)),
        BluetoothMetric("symbol_rate_error", "Symbol Rate Error", _display(rate_error, "ppm")),
        BluetoothMetric("duration", "Packet Duration", f"{duration_ms:.3f} ms"),
        BluetoothMetric("correlation", "Synchronization Correlation", f"{100.0 * float(pattern.correlation):.2f} %"),
    ]
    if phy is not BluetoothClassicPhy.BR:
        relative_power_db = (
            psk_power_dbm - fsk_power_dbm
            if psk_power_dbm is not None and fsk_power_dbm is not None
            else None
        )
        metrics.extend(
            (
                BluetoothMetric(
                    "fsk_average_power",
                    "FSK Average Power",
                    _display(fsk_power_dbm, "dBm"),
                ),
                BluetoothMetric(
                    "psk_average_power",
                    "PSK Average Power",
                    _display(psk_power_dbm, "dBm"),
                ),
                BluetoothMetric(
                    "psk_relative_power",
                    "Relative Power (PSK - FSK)",
                    _display(relative_power_db, "dB"),
                ),
                BluetoothMetric(
                    "bluetooth_devm_rms",
                    "Bluetooth DEVM RMS",
                    _display_evm(
                        pattern.metadata.get("bluetooth_devm_rms_percent")
                    ),
                ),
            )
        )
    recording_sample_offset = max(0, int(_recording_sample_offset))
    analysis_sample_offset_global = recording_sample_offset + int(
        analysis_sample_offset
    )
    packet_start_sample = recording_sample_offset + int(
        br_analysis_session.pattern_result.result_start_sample
    )
    packet_stop_sample = (
        analysis_sample_offset_global + int(pattern.result_stop_sample)
        if phy is not BluetoothClassicPhy.BR
        else recording_sample_offset + int(pattern.result_stop_sample)
    )
    rf_metrics: tuple[BluetoothMetric, ...] = ()
    rf_measurements: tuple[BluetoothRFMeasurementResult, ...] = ()
    if phy is BluetoothClassicPhy.BR:
        rf_metrics, rf_measurements = _sig_fsk_measurements(
            recording,
            packet,
            profile=profile,
            whitening_enabled=bool(whitening_enabled),
            symbol_rate_hz=1_000_000.0,
            filter_profile=BluetoothRFMeasurementFilterProfile.BR_1M,
            packet_start_sample=packet_start_sample - recording_sample_offset,
            packet_stop_sample=packet_stop_sample - recording_sample_offset,
            p0_sample=(
                float(br_analysis_session.pattern_result.symbol_time_s[0])
                * recording.sample_rate_hz
                - 0.5 * recording.sample_rate_hz / 1_000_000.0
            ),
            drift_block_symbols=50,
        )
        metrics.extend(rf_metrics)
    elif detected_edr_sync_start is not None:
        eligibility = _rf_test_eligibility(
            profile=profile,
            whitening_enabled=bool(whitening_enabled),
            sample_rate_hz=recording.sample_rate_hz,
            minimum_sample_rate_hz=4_000_000.0,
        )
        try:
            br_packet_start = int(
                br_analysis_session.pattern_result.result_start_sample
            )
            header_end_sample = br_packet_start + int(
                round(126.0 * samples_per_br_symbol)
            )
            fm_trace = build_fm_measurement_trace(
                recording.iq,
                sample_rate_hz=recording.sample_rate_hz,
                symbol_rate_hz=1_000_000.0,
                p0_sample=(
                    float(br_analysis_session.pattern_result.symbol_time_s[0])
                    * recording.sample_rate_hz
                    - 0.5 * samples_per_br_symbol
                ),
                profile=BluetoothRFMeasurementFilterProfile.BR_1M,
            )
            initial = measure_initial_carrier_frequency(
                fm_trace,
                packet.raw_bits[72:126],
                nominal_frequency_hz=recording.center_frequency_hz,
                start_symbol=72,
                symbol_count=54,
            )
            fsk_power = measure_burst_power(
                recording.iq,
                full_scale=recording.full_scale,
                dbfs_to_dbm_offset_db=recording.dbfs_to_dbm_offset_db,
                start_sample=br_packet_start,
                stop_sample=header_end_sample,
                central_fraction=0.8,
            )
            psk_power = measure_burst_power(
                recording.iq,
                full_scale=recording.full_scale,
                dbfs_to_dbm_offset_db=recording.dbfs_to_dbm_offset_db,
                start_sample=detected_edr_sync_start,
                stop_sample=max(
                    detected_edr_sync_start + 1,
                    packet_stop_sample
                    - recording_sample_offset
                    - int(round(2.0 * samples_per_br_symbol)),
                ),
                central_fraction=0.8,
            )
            output_power = measure_burst_power(
                recording.iq,
                full_scale=recording.full_scale,
                dbfs_to_dbm_offset_db=recording.dbfs_to_dbm_offset_db,
                start_sample=br_packet_start,
                stop_sample=packet_stop_sample - recording_sample_offset,
                central_fraction=0.8,
            )
            # Differential demodulation needs the preceding absolute symbol;
            # Generic PatternSearchResult therefore reports its range one
            # symbol after the physical EDR reference-symbol boundary.
            edr_reference_start_sample = (
                detected_edr_sync_start - samples_per_psk_symbol
            )
            guard = measure_edr_guard_time(
                header_end_sample=header_end_sample,
                reference_symbol_start_sample=edr_reference_start_sample,
                sample_rate_hz=recording.sample_rate_hz,
            )
            first_psk_center = (
                analysis_sample_offset
                + float(pattern.symbol_time_s[0]) * recording.sample_rate_hz
            )
            devm = measure_edr_devm(
                recording.iq,
                sample_rate_hz=recording.sample_rate_hz,
                symbol_rate_hz=analysis_session.signal.symbol_rate_hz,
                first_symbol_center_sample=first_psk_center,
                decoded_symbols=pattern.decoded_symbols,
                modulation=analysis_session.signal.modulation,
                symbol_mapping=analysis_session.signal.symbol_mapping,
                initial_frequency_error_hz=initial.error_hz,
            )
            conformance = EDRConformanceResult(
                sync_symbol_errors=int(
                    np.count_nonzero(
                        pattern.decoded_symbols[
                            : expected_edr_decoded_sync_symbols.size
                        ]
                        != expected_edr_decoded_sync_symbols
                    )
                ),
                trailer_symbol_errors=int(
                    np.count_nonzero(pattern.decoded_symbols[-2:])
                ),
                evaluated_sync_symbols=int(expected_edr_sync_symbols.size),
                evaluated_trailer_symbols=min(2, pattern.decoded_symbols.size),
            )
            sync_symbol_count = min(
                pattern.decoded_symbols.size,
                expected_edr_decoded_sync_symbols.size,
            )
            sync_bit_errors = int(
                np.count_nonzero(
                    _symbols_to_air_bits(
                        pattern.decoded_symbols[:sync_symbol_count],
                        analysis_session.signal.modulation.order,
                    )
                    != _symbols_to_air_bits(
                        expected_edr_decoded_sync_symbols[:sync_symbol_count],
                        analysis_session.signal.modulation.order,
                    )
                )
            )
            trailer_bit_errors = int(
                np.count_nonzero(
                    _symbols_to_air_bits(
                        pattern.decoded_symbols[-2:],
                        analysis_session.signal.modulation.order,
                    )
                )
            )
            payload_body = _packet_field_by_id(packet.root_fields, "payload_body")
            payload_bit_errors: int | None = None
            payload_bit_count = 0
            payload_pattern: str | None = None
            if payload_body is not None and payload_body.raw_bits.size:
                measured_payload = np.asarray(payload_body.raw_bits, dtype=np.uint8)
                prbs9 = prbs9_period()
                expected_payload = prbs9[
                    np.arange(measured_payload.size, dtype=np.int64) % prbs9.size
                ]
                payload_bit_errors = int(
                    np.count_nonzero(measured_payload != expected_payload)
                )
                payload_bit_count = int(measured_payload.size)
                payload_pattern = "PRBS9" if payload_bit_errors == 0 else "Other"
            if payload_pattern != "PRBS9":
                eligibility = RFTestEligibility.from_reasons(
                    (
                        *eligibility.reasons,
                        "EDR RF test payload must be PRBS9",
                    )
                )
            relative_power_db = psk_power.average_dbm - fsk_power.average_dbm
            rf_metrics = (
                BluetoothMetric(
                    "sig_eligibility",
                    "SIG RF Test Eligibility",
                    "Eligible"
                    if eligibility.eligible
                    else "N/A - " + "; ".join(eligibility.reasons),
                ),
                BluetoothMetric("sig_edr_pgfsk", "SIG PGFSK", _display(fsk_power.average_dbm, "dBm")),
                BluetoothMetric("sig_edr_pdpsk", "SIG PDPSK", _display(psk_power.average_dbm, "dBm")),
                BluetoothMetric("sig_edr_relative_power", "SIG PDPSK - PGFSK", _display(relative_power_db, "dB")),
                BluetoothMetric("sig_edr_initial_frequency_error", "SIG Initial Frequency Error omega_i", _display(initial.error_hz, "kHz", 1e3)),
                BluetoothMetric("sig_edr_guard_time", "SIG Guard Time", _display(guard.guard_time_s, "us", 1e-6)),
                BluetoothMetric("sig_edr_rms_devm", "SIG Worst 50-symbol RMS DEVM", _display_evm(None if devm.rms_worst is None else 100.0 * devm.rms_worst)),
                BluetoothMetric("sig_edr_99_devm", "SIG 99% DEVM", _display_evm(None if devm.devm_99_percentile is None else 100.0 * devm.devm_99_percentile)),
                BluetoothMetric("sig_edr_peak_devm", "SIG Peak DEVM", _display_evm(None if devm.peak_worst is None else 100.0 * devm.peak_worst)),
                BluetoothMetric(
                    "sig_edr_omega0",
                    "SIG Worst Residual Frequency Error omega_0",
                    _display(
                        max(
                            (
                                abs(block.residual_frequency_error_hz)
                                for block in devm.blocks
                            ),
                            default=None,
                        ),
                        "kHz",
                        1e3,
                    ),
                ),
                BluetoothMetric("sig_edr_block_count", "SIG DEVM Blocks", f"{len(devm.blocks)} / 200 (verdict N/A)"),
                BluetoothMetric("sig_edr_sync_errors", "SIG Sync Symbol Errors", str(conformance.sync_symbol_errors)),
                BluetoothMetric("sig_edr_trailer_errors", "SIG Trailer Symbol Errors", str(conformance.trailer_symbol_errors)),
            )
            metrics.extend(rf_metrics)
            rf_measurements = (
                BluetoothRFMeasurementResult(
                    test_case_id="bluetooth.edr",
                    eligibility=eligibility,
                    verdict=RFTestVerdict.NOT_APPLICABLE,
                    metrics={
                        "output_power_dbm": output_power.average_dbm,
                        "pgfsk_dbm": fsk_power.average_dbm,
                        "pdpsk_dbm": psk_power.average_dbm,
                        "relative_power_db": relative_power_db,
                        "initial_frequency_error_hz": initial.error_hz,
                        "guard_time_s": guard.guard_time_s,
                        "rms_devm_worst": devm.rms_worst,
                        "devm_99_percentile": devm.devm_99_percentile,
                        "peak_devm_worst": devm.peak_worst,
                        "omega0_abs_worst_hz": max(
                            (
                                abs(block.residual_frequency_error_hz)
                                for block in devm.blocks
                            ),
                            default=None,
                        ),
                        "omega0_worst_hz": max(
                            (
                                block.residual_frequency_error_hz
                                for block in devm.blocks
                            ),
                            key=abs,
                            default=None,
                        ),
                        "block_count": len(devm.blocks),
                        "payload_bit_errors": payload_bit_errors,
                        "payload_bit_count": payload_bit_count,
                        "sync_symbol_errors": conformance.sync_symbol_errors,
                        "sync_bit_errors": sync_bit_errors,
                        "evaluated_sync_symbols": conformance.evaluated_sync_symbols,
                        "trailer_symbol_errors": conformance.trailer_symbol_errors,
                        "trailer_bit_errors": trailer_bit_errors,
                        "evaluated_trailer_symbols": (
                            conformance.evaluated_trailer_symbols
                        ),
                    },
                    arrays={
                        "block_rms_devm": np.asarray(
                            [block.rms_devm for block in devm.blocks]
                        ),
                        "block_peak_devm": np.asarray(
                            [block.peak_devm for block in devm.blocks]
                        ),
                        "symbol_devm": np.concatenate(
                            [block.symbol_devm for block in devm.blocks]
                        )
                        if devm.blocks
                        else np.empty(0, dtype=np.float64),
                        "omega0_hz": np.asarray(
                            [block.residual_frequency_error_hz for block in devm.blocks]
                        ),
                        "omega_i_plus_omega0_hz": np.asarray(
                            [
                                initial.error_hz
                                + block.residual_frequency_error_hz
                                for block in devm.blocks
                            ]
                        ),
                        "timing_offset_symbols": np.asarray(
                            [block.timing_offset_symbols for block in devm.blocks]
                        ),
                        "omega_i_selected_header_bit_indices": (
                            initial.selected_bit_indices
                        ),
                    },
                    metadata={
                        "trailer_excluded_from_devm": True,
                        "required_block_count": 200,
                        "modulation": analysis_session.signal.modulation.value,
                        "payload_pattern": payload_pattern,
                        "output_power_window_start_sample": output_power.start_sample,
                        "output_power_window_stop_sample": output_power.stop_sample,
                        "appendix_c_final_audit": False,
                    },
                ),
            )
        except (ValueError, RuntimeError) as error:
            unavailable = RFTestEligibility.from_reasons(
                (*eligibility.reasons, str(error))
            )
            metrics.append(
                BluetoothMetric(
                    "sig_eligibility",
                    "SIG RF Test Eligibility",
                    "N/A - " + "; ".join(unavailable.reasons),
                )
            )
            rf_measurements = (
                BluetoothRFMeasurementResult(
                    "bluetooth.edr",
                    unavailable,
                    metadata={"reason": str(error)},
                ),
            )
    # The shared semantic decoder must see the composite BR-header + EDR data
    # stream, not only the PSK result used by the generic VSA plot products.
    return BluetoothDedicatedResult(
        profile=BluetoothAnalysisProfile(profile),
        vsa_result=vsa_result,
        packet=packet,
        metrics=tuple(metrics),
        metadata={
            "source": recording.source,
            "sample_rate_hz": recording.sample_rate_hz,
            "center_frequency_hz": recording.center_frequency_hz,
            "classic_phy_auto_detected": True,
            "br_access_correlation": br_frontend.demodulation.access_correlation,
            "edr_candidate_error": edr_error,
            "analysis_session": analysis_session,
            "br_analysis_session": br_analysis_session,
            "recording_sample_offset": recording_sample_offset,
            "analysis_sample_offset": analysis_sample_offset_global,
            "packet_start_sample": packet_start_sample,
            "packet_stop_sample": packet_stop_sample,
            "selected_match_index": int(match_index),
            "eligible_match_count": int(
                br_analysis_session.pattern_result.metadata.get(
                    "eligible_match_count", 1
                )
            ),
            "rf_measurements": rf_measurements,
        },
    )


def analyze_bluetooth_le_recording(
    recording: IQRecording,
    *,
    profile: BluetoothAnalysisProfile,
    phy: BluetoothLEPhy | str,
    access_address: int = 0x8E89BED6,
    channel_index: int = 37,
    crc_init: int = 0x555555,
    whitening_enabled: bool = True,
    result_length: int = 4096,
    match_index: int = 1,
    iq_power_trigger: IQPowerTriggerSettings | None = None,
    _recording_sample_offset: int = 0,
) -> BluetoothDedicatedResult:
    """Synchronize and decode one uncoded LE 1M/2M packet from IQ."""

    phy = BluetoothLEPhy(phy)
    sync = _le_sync_bits(phy, int(access_address))
    session = _analyze_known_pattern(
        recording,
        _le_signal(phy),
        sync,
        result_length=result_length,
        minimum_correlation=0.60,
        match_index=match_index,
        iq_power_trigger=iq_power_trigger,
    )
    pattern = session.pattern_result
    bits = _trim_le_packet_bits(
        pattern.decoded_bits,
        phy=phy,
        whitening_enabled=bool(whitening_enabled),
        channel_index=int(channel_index),
    )
    context = {
        "phy": phy.value,
        "whitening_enabled": bool(whitening_enabled),
        "whitening_channel_index": int(channel_index),
        "crc_enabled": True,
        "crc_init": int(crc_init) & 0xFFFFFF,
    }
    packet = analyze_demodulated_packet_bits(
        bits,
        protocol_id="bluetooth.le",
        phy_name=phy.value,
        context=context,
        packet_index=0,
        center_frequency_hz=recording.center_frequency_hz,
        start_sample=pattern.result_start_sample,
        stop_sample=pattern.result_stop_sample,
    )
    vsa_result = (
        session.carrier_corrected_pattern_range_result
        or session.pattern_range_result
        or session.result
    )
    if vsa_result is None:
        raise RuntimeError("Bluetooth LE PHY analysis produced no VSA result")
    duration_ms = bits.size / float(_le_signal(phy).symbol_rate_hz) * 1e3
    recording_sample_offset = max(0, int(_recording_sample_offset))
    packet_start_sample = recording_sample_offset + int(pattern.result_start_sample)
    packet_stop_sample = packet_start_sample + int(
        round(bits.size * recording.sample_rate_hz / _le_signal(phy).symbol_rate_hz)
    )
    try:
        rate_error = float(vsa_result.metadata.get("symbol_rate_error_ppm"))
    except (TypeError, ValueError):
        rate_error = None
    metrics = (
        BluetoothMetric("detected_phy", "Detected PHY", phy.value),
        BluetoothMetric("access_address", "Access Address", f"0x{int(access_address) & 0xFFFFFFFF:08X}"),
        BluetoothMetric("profile", "Analysis Profile", BluetoothAnalysisProfile(profile).value),
        BluetoothMetric("packet_power", "Packet Average Power", _display(_mean_power_dbm(vsa_result.power_dbm), "dBm")),
        BluetoothMetric("peak_power", "Peak Power", _display(_finite_stat(vsa_result.power_dbm, peak=True), "dBm")),
        BluetoothMetric("cfo", "Carrier Frequency Offset", _display(float(pattern.carrier_frequency_offset_hz), "kHz", 1e3)),
        BluetoothMetric("symbol_rate_error", "Symbol Rate Error", _display(rate_error, "ppm")),
        BluetoothMetric("duration", "Packet Duration", f"{duration_ms:.3f} ms"),
        BluetoothMetric("correlation", "Synchronization Correlation", f"{100.0 * float(pattern.correlation):.2f} %"),
    )
    rf_profile = (
        BluetoothRFMeasurementFilterProfile.LE_2M
        if phy is BluetoothLEPhy.LE_2M
        else BluetoothRFMeasurementFilterProfile.LE_1M
    )
    rf_metrics, rf_measurements = _sig_fsk_measurements(
        recording,
        packet,
        profile=profile,
        whitening_enabled=bool(whitening_enabled),
        symbol_rate_hz=_le_signal(phy).symbol_rate_hz,
        filter_profile=rf_profile,
        packet_start_sample=packet_start_sample - recording_sample_offset,
        packet_stop_sample=packet_stop_sample - recording_sample_offset,
        p0_sample=(
            float(pattern.symbol_time_s[0]) * recording.sample_rate_hz
            - 0.5 * recording.sample_rate_hz / _le_signal(phy).symbol_rate_hz
        ),
        extra_eligibility_reasons=tuple(
            reason
            for reason in (
                None
                if int(access_address) == 0x71764129
                else "RF test Access Address must be 0x71764129",
                None
                if (int(crc_init) & 0xFFFFFF) == 0x555555
                else "RF test CRCInit must be 0x555555",
            )
            if reason is not None
        ),
        drift_block_symbols=(100 if phy is BluetoothLEPhy.LE_2M else 50),
    )
    metrics = (*metrics, *rf_metrics)
    return BluetoothDedicatedResult(
        profile=BluetoothAnalysisProfile(profile),
        vsa_result=vsa_result,
        packet=packet,
        metrics=metrics,
        metadata={
            "source": recording.source,
            "sample_rate_hz": recording.sample_rate_hz,
            "center_frequency_hz": recording.center_frequency_hz,
            "access_address": int(access_address) & 0xFFFFFFFF,
            "analysis_session": session,
            "recording_sample_offset": recording_sample_offset,
            "analysis_sample_offset": recording_sample_offset,
            "packet_start_sample": packet_start_sample,
            "packet_stop_sample": packet_stop_sample,
            "selected_match_index": int(pattern.metadata.get("selected_match_index", match_index)),
            "eligible_match_count": int(pattern.metadata.get("eligible_match_count", 1)),
            "rf_measurements": rf_measurements,
        },
    )


def analyze_bluetooth_classic_recordings(
    recording: IQRecording,
    *,
    cancelled: Callable[[], bool] | None = None,
    max_candidates: int = 64,
    **kwargs: object,
) -> tuple[BluetoothDedicatedResult, ...]:
    """Analyze every eligible Classic/EDR packet in chronological order."""

    first = analyze_bluetooth_classic_recording(recording, match_index=1, **kwargs)
    first_pattern = first.metadata["br_analysis_session"].pattern_result
    candidate_starts = tuple(
        int(value)
        for value in (
            first_pattern.metadata.get("eligible_match_start_samples", ())
            if first_pattern is not None
            else ()
        )
    )
    if not candidate_starts:
        candidate_starts = (int(first.metadata.get("packet_start_sample", 0)),)
    candidate_starts = candidate_starts[: max(1, int(max_candidates))]
    count = len(candidate_starts)
    results = [first]
    capture_result = first.metadata["br_analysis_session"].result
    margin_samples = max(1, int(round(recording.sample_rate_hz * 16.0e-6)))
    for index, candidate_start in enumerate(candidate_starts[1:], start=2):
        if cancelled is not None and cancelled():
            break
        try:
            crop_start = max(0, int(candidate_start) - margin_samples)
            crop_stop = (
                min(recording.sample_count, int(candidate_starts[index]))
                if index < count
                else recording.sample_count
            )
            local = replace(
                recording,
                iq=recording.iq[crop_start:crop_stop],
                start_sample_index=recording.start_sample_index + crop_start,
                trigger_sample_index=None,
            )
            item = analyze_bluetooth_classic_recording(
                local,
                match_index=1,
                _recording_sample_offset=crop_start,
                **kwargs,
            )
            metadata = dict(item.metadata)
            metadata.update(
                {
                    "selected_match_index": index,
                    "eligible_match_count": count,
                    "capture_result": capture_result,
                }
            )
            item = replace(item, metadata=metadata)
            results.append(item)
        except (RuntimeError, ValueError):
            # A BR access-code candidate can fail the PHY/header integrity
            # checks.  It must not hide later valid packets in the capture.
            continue
    first_metadata = dict(first.metadata)
    first_metadata.update(
        {"selected_match_index": 1, "eligible_match_count": count, "capture_result": capture_result}
    )
    results[0] = replace(first, metadata=first_metadata)
    return _attach_rf_capture_aggregates(results)


def analyze_bluetooth_le_recordings(
    recording: IQRecording,
    *,
    cancelled: Callable[[], bool] | None = None,
    max_candidates: int = 64,
    **kwargs: object,
) -> tuple[BluetoothDedicatedResult, ...]:
    """Analyze every eligible LE packet in chronological order."""

    first = analyze_bluetooth_le_recording(recording, match_index=1, **kwargs)
    first_pattern = first.metadata["analysis_session"].pattern_result
    candidate_starts = tuple(
        int(value)
        for value in (
            first_pattern.metadata.get("eligible_match_start_samples", ())
            if first_pattern is not None
            else ()
        )
    )
    if not candidate_starts:
        candidate_starts = (int(first.metadata.get("packet_start_sample", 0)),)
    candidate_starts = candidate_starts[: max(1, int(max_candidates))]
    count = len(candidate_starts)
    results = [first]
    capture_result = first.metadata["analysis_session"].result
    margin_samples = max(1, int(round(recording.sample_rate_hz * 16.0e-6)))
    for index, candidate_start in enumerate(candidate_starts[1:], start=2):
        if cancelled is not None and cancelled():
            break
        try:
            crop_start = max(0, int(candidate_start) - margin_samples)
            crop_stop = (
                min(recording.sample_count, int(candidate_starts[index]))
                if index < count
                else recording.sample_count
            )
            local = replace(
                recording,
                iq=recording.iq[crop_start:crop_stop],
                start_sample_index=recording.start_sample_index + crop_start,
                trigger_sample_index=None,
            )
            item = analyze_bluetooth_le_recording(
                local,
                match_index=1,
                _recording_sample_offset=crop_start,
                **kwargs,
            )
            metadata = dict(item.metadata)
            metadata.update(
                {
                    "selected_match_index": index,
                    "eligible_match_count": count,
                    "capture_result": capture_result,
                }
            )
            item = replace(item, metadata=metadata)
            results.append(item)
        except (RuntimeError, ValueError):
            continue
    first_metadata = dict(first.metadata)
    first_metadata.update(
        {"selected_match_index": 1, "eligible_match_count": count, "capture_result": capture_result}
    )
    results[0] = replace(first, metadata=first_metadata)
    return _attach_rf_capture_aggregates(results)
