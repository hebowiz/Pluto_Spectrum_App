"""Bluetooth dedicated-analyzer result assembled from Generic VSA products."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from collections.abc import Callable
from typing import Mapping

import numpy as np

from pluto_protocol.model import PacketAnalysisResult, PacketField
from pluto_protocol.bluetooth.common import le_whitening_sequence
from pluto_protocol.bluetooth.hdt import (
    HDT_DEFINITIONS,
    HDTRate,
    hdt_coded_payload_bit_count,
    hdt_definition,
    map_hdt_symbols,
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
from pluto_sa.vsa.profiles.bluetooth_br import BluetoothBRProfile, access_code_bits
from pluto_sa.vsa.profiles.bluetooth_edr import edr_sync_symbols
from pluto_sa.vsa.protocol import analyze_demodulated_packet_bits
from pluto_sa.vsa.session import VSASession


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


@dataclass(frozen=True)
class BluetoothDedicatedResult:
    profile: BluetoothAnalysisProfile
    vsa_result: VSAAnalysisResult
    packet: PacketAnalysisResult
    metrics: tuple[BluetoothMetric, ...]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile", BluetoothAnalysisProfile(self.profile))
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


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
        frequency_deviation_hz=250_000.0,
        tx_filter="Gaussian",
        filter_parameter=0.5,
        symbol_mapping="Natural",
    )


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
_HDT_PAYLOAD_START_SYMBOL = _HDT_TRAINING_SYMBOLS + _HDT_CONTROL_SYMBOLS


def _hdt_training_bits() -> np.ndarray:
    return np.resize(
        np.asarray([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8),
        _HDT_TRAINING_SYMBOLS * 2,
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
    reference = map_hdt_symbols(_hdt_training_bits(), HDTRate.HDT2).astype(
        np.complex128
    )
    reference_energy = float(np.sum(np.abs(reference) ** 2))
    candidates: list[tuple[float, int]] = []
    for phase in range(integer_sps):
        sampled = np.asarray(recording.iq[phase::integer_sps], dtype=np.complex128)
        if sampled.size < reference.size:
            continue
        correlation = np.correlate(sampled, reference, mode="valid")
        energy = np.convolve(
            np.abs(sampled) ** 2,
            np.ones(reference.size, dtype=np.float64),
            mode="valid",
        )
        scores = np.abs(correlation) / np.sqrt(
            np.maximum(energy * reference_energy, np.finfo(np.float64).tiny)
        )
        for index in np.flatnonzero(scores >= 0.72):
            candidates.append(
                (float(scores[index]), int(phase + index * integer_sps))
            )
    selected: list[tuple[float, int]] = []
    guard_samples = int(round(_HDT_TRAINING_SYMBOLS * samples_per_symbol))
    for score, center in sorted(candidates, reverse=True):
        if all(abs(center - other_center) >= guard_samples for _, other_center in selected):
            selected.append((score, center))
    return tuple((center, score) for score, center in sorted(selected, key=lambda item: item[1]))


def _viterbi_decode_hdt_control(encoded: np.ndarray) -> tuple[np.ndarray, int]:
    """Hard-decision K=6 Viterbi decode for one 40-bit HDT control word."""

    values = np.asarray(encoded, dtype=np.uint8)[:40]
    if values.size < 40:
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
    first_center, training_correlation = matches[int(match_index) - 1]
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
    header_observed = _hdt_sample_symbols(
        filtered_recording, first_center, _HDT_PAYLOAD_START_SYMBOL
    )
    if header_observed.size < _HDT_PAYLOAD_START_SYMBOL:
        raise RuntimeError("Bluetooth HDT control header is incomplete")
    training_reference = map_hdt_symbols(_hdt_training_bits(), HDTRate.HDT2)
    training_phase_error = np.unwrap(
        np.angle(
            header_observed[:_HDT_TRAINING_SYMBOLS]
            * np.conj(training_reference)
        )
    )
    phase_step, phase_intercept = np.polyfit(
        np.arange(_HDT_TRAINING_SYMBOLS, dtype=np.float64),
        training_phase_error,
        1,
    )
    header_axis = np.arange(_HDT_PAYLOAD_START_SYMBOL, dtype=np.float64)
    corrected_header = header_observed * np.exp(
        -1j * (phase_intercept + phase_step * header_axis)
    )
    header_rms = float(np.sqrt(np.mean(np.abs(corrected_header) ** 2)))
    corrected_header /= max(header_rms, np.finfo(np.float64).tiny)
    qpsk_alphabets = _hdt_qpsk_constellation(
        np.arange(_HDT_PAYLOAD_START_SYMBOL)
    )
    header_labels = np.argmin(
        np.abs(corrected_header[:, None] - qpsk_alphabets), axis=1
    ).astype(np.int16)
    header_reference = qpsk_alphabets[
        np.arange(_HDT_PAYLOAD_START_SYMBOL), header_labels
    ]
    header_evm = 100.0 * float(
        np.sqrt(
            np.sum(np.abs(corrected_header - header_reference) ** 2)
            / np.sum(np.abs(header_reference) ** 2)
        )
    )
    control_labels = header_labels[_HDT_TRAINING_SYMBOLS:]
    control_encoded = (
        (control_labels[:, None] >> np.asarray([1, 0], dtype=np.int16)) & 1
    ).astype(np.uint8).reshape(-1)
    control_data, control_path_errors = _viterbi_decode_hdt_control(
        control_encoded
    )
    rate_indicator = int(
        sum(int(control_data[index]) << (2 - index) for index in range(3))
    )
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
    payload_length = int(
        sum(int(control_data[3 + index]) << index for index in range(12))
    )
    definition = hdt_definition(rate)
    coded_payload_bits = hdt_coded_payload_bit_count(rate, payload_length)
    payload_symbol_count = int(
        np.ceil(coded_payload_bits / float(definition.bits_per_symbol))
    )
    payload_start = int(
        round(first_center - 0.5 * samples_per_symbol + _HDT_PAYLOAD_START_SYMBOL * samples_per_symbol)
    )
    payload_stop = int(round(payload_start + payload_symbol_count * samples_per_symbol))
    if payload_stop > recording.sample_count:
        raise RuntimeError(
            f"HDT header declares {payload_length} payload byte(s), but the capture is incomplete"
        )

    payload_observed = _hdt_sample_symbols(
        filtered_recording,
        first_center + _HDT_PAYLOAD_START_SYMBOL * samples_per_symbol,
        payload_symbol_count,
    )
    payload_axis = _HDT_PAYLOAD_START_SYMBOL + np.arange(payload_symbol_count)
    payload_corrected = payload_observed * np.exp(
        -1j * (phase_intercept + phase_step * payload_axis)
    )
    payload_rms = float(np.sqrt(np.mean(np.abs(payload_corrected) ** 2)))
    payload_corrected /= max(payload_rms, np.finfo(np.float64).tiny)
    payload_signal = _hdt_signal(rate)
    payload_session: VSASession | None = None
    if definition.modulation in {"8PSK", "16QAM"} and payload_symbol_count >= 4:
        alphabet = psk_constellation(
            payload_signal.modulation, BLUETOOTH_HDT_MAPPING
        )
        seed_count = min(24, payload_symbol_count)
        seed_labels = np.argmin(
            np.abs(payload_corrected[:seed_count, None] - alphabet[None, :]),
            axis=1,
        ).astype(np.int16)
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
        payload_labels = payload_pattern.decoded_symbols
        payload_air_bits = _symbols_to_air_bits(
            payload_labels, payload_signal.modulation.order
        )[:coded_payload_bits]
        payload_evm = payload_pattern.evm_rms_percent
        vsa_result = (
            payload_session.carrier_corrected_pattern_range_result
            or payload_session.pattern_range_result
            or payload_session.result
        )
        analysis_sample_offset = crop_start
    else:
        qpsk_alphabets = _hdt_qpsk_constellation(payload_axis)
        payload_labels = np.argmin(
            np.abs(payload_corrected[:, None] - qpsk_alphabets), axis=1
        ).astype(np.int16)
        payload_reference = qpsk_alphabets[
            np.arange(payload_symbol_count), payload_labels
        ]
        payload_evm = 100.0 * float(
            np.sqrt(
                np.sum(np.abs(payload_corrected - payload_reference) ** 2)
                / np.sum(np.abs(payload_reference) ** 2)
            )
        )
        payload_air_bits = (
            (payload_labels[:, None] >> np.asarray([1, 0], dtype=np.int16)) & 1
        ).astype(np.uint8).reshape(-1)[:coded_payload_bits]
        fallback_session = VSASession(name="Bluetooth HDT QPSK")
        fallback_session.set_recording(recording)
        fallback_session.set_signal(payload_signal)
        fallback_session.analyze()
        payload_session = fallback_session
        vsa_result = fallback_session.result
        analysis_sample_offset = 0
    if vsa_result is None:
        raise RuntimeError("Bluetooth HDT payload analysis produced no VSA result")

    packet_bits = np.concatenate(
        (_hdt_training_bits(), control_data, payload_air_bits)
    )
    packet_start = max(0, int(round(first_center - 0.5 * samples_per_symbol)))
    packet = analyze_demodulated_packet_bits(
        packet_bits,
        protocol_id="bluetooth.hdt",
        phy_name=rate.value,
        context={
            "training_bit_count": _HDT_TRAINING_SYMBOLS * 2,
            "control_bit_count": int(control_data.size),
            "expected_payload_bit_count": coded_payload_bits,
        },
        packet_index=int(match_index) - 1,
        center_frequency_hz=recording.center_frequency_hz,
        start_sample=packet_start,
        stop_sample=payload_stop,
    )
    cfo_hz = phase_step * _HDT_SYMBOL_RATE_HZ / (2.0 * np.pi)
    header_power_dbm = _recording_range_power_dbm(
        recording, packet_start, payload_start
    )
    payload_power_dbm = _recording_range_power_dbm(
        recording, payload_start, payload_stop
    )
    relative_power_db = (
        payload_power_dbm - header_power_dbm
        if payload_power_dbm is not None and header_power_dbm is not None
        else None
    )
    metrics = (
        BluetoothMetric("detected_phy", "Detected PHY", rate.value),
        BluetoothMetric("payload_modulation", "Payload Modulation", definition.modulation),
        BluetoothMetric("payload_code_rate", "Payload Code Rate", definition.payload_code_rate),
        BluetoothMetric("payload_length", "Payload Length", f"{payload_length} byte(s)"),
        BluetoothMetric("automatic_result_range", "Payload Result Range", f"{payload_symbol_count} symbol(s) (automatic)"),
        BluetoothMetric("qpsk_evm_rms", "QPSK Header EVM RMS", _display_evm(header_evm)),
        BluetoothMetric("payload_evm_rms", f"{definition.modulation} Payload EVM RMS", _display_evm(payload_evm)),
        BluetoothMetric("qpsk_average_power", "QPSK Header Average Power", _display(header_power_dbm, "dBm")),
        BluetoothMetric("payload_average_power", f"{definition.modulation} Payload Average Power", _display(payload_power_dbm, "dBm")),
        BluetoothMetric("payload_relative_power", "Relative Power (Payload - Header)", _display(relative_power_db, "dB")),
        BluetoothMetric("cfo", "Carrier Frequency Offset", _display(cfo_hz, "kHz", 1e3)),
        BluetoothMetric("correlation", "Training Correlation", f"{100.0 * training_correlation:.2f} %"),
    )
    display_header = corrected_header * np.exp(
        -1j * (np.arange(corrected_header.size) & 1) * np.pi / 4.0
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
            "hdt_header_symbols": np.asarray(display_header, dtype=np.complex64),
            "hdt_header_evm_rms_percent": header_evm,
            "hdt_payload_evm_rms_percent": payload_evm,
            "hdt_rate_indicator": rate_indicator,
            "hdt_control_path_errors": control_path_errors,
            "hdt_payload_symbol_count": payload_symbol_count,
            "hdt_payload_start_sample": payload_start,
            "hdt_payload_stop_sample": payload_stop,
            "recording_sample_offset": 0,
            "analysis_sample_offset": analysis_sample_offset,
            "packet_start_sample": packet_start,
            "packet_stop_sample": payload_stop,
            "selected_match_index": int(match_index),
            "eligible_match_count": len(matches),
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
    return tuple(results)


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
    if edr_candidate is not None:
        width = 2 if edr_candidate is BluetoothClassicPhy.EDR_2M else 3
        # ``edr_sync_symbols`` is expressed as physical differential phase
        # indices, while PatternAnalyzer consumes the logical symbol numbers
        # selected by SignalDescription.symbol_mapping.  Convert through the
        # Bluetooth mapping before applying the R&S-style LSB symbol display
        # order.  Treating phase indices as logical values made TYPE 0x4/0x8
        # packets fall back to BR (and consequently broke EDR Length/CRC).
        edr_signal = _classic_signal(edr_candidate)
        sync = reverse_symbol_bits(
            phase_indices_to_logical_symbols(
                edr_signal.modulation,
                BLUETOOTH_EDR_MAPPING,
                edr_sync_symbols(width),
            ),
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
    return tuple(results)


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
    return tuple(results)
