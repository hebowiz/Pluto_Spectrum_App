import numpy as np
import pytest

from pluto_sa.vsa.analysis import VSAAnalyzer
from pluto_sa.vsa.model import (
    CompositeSignalDescription,
    IQRecording,
    ModulationKind,
    ModulationSegment,
    SignalDescription,
    VSASettings,
)
from pluto_sa.vsa.session import VSASession
from pluto_sa.vsa.sources import FileIQSource, GeneratedIQSource


def test_recording_owns_read_only_iq() -> None:
    source = np.ones(16, dtype=np.complex64)
    recording = IQRecording(source, sample_rate_hz=1_000_000.0)
    source[:] = 0.0

    assert np.all(recording.iq == 1.0)
    assert recording.duration_s == pytest.approx(16e-6)
    with pytest.raises(ValueError):
        recording.iq[0] = 0.0


def test_zero_span_power_uses_common_dbm_correction_convention() -> None:
    recording = IQRecording(
        np.full(32, 256.0 + 0.0j, dtype=np.complex64),
        sample_rate_hz=1_000_000.0,
        full_scale=2048.0,
        calibration_offset_db=-62.0,
        frequency_dependent_offset_db=1.25,
        input_correction_db=-3.0,
        amplitude_calibrated=True,
    )
    signal = SignalDescription(ModulationKind.FSK2, symbol_rate_hz=100_000.0)

    result = VSAAnalyzer().analyze(recording, signal, VSASettings(remove_dc=False))

    expected_dbm = 20.0 * np.log10(256.0) - 62.0 + 1.25 - 3.0
    np.testing.assert_allclose(result.power_dbm, expected_dbm, atol=1e-10)
    np.testing.assert_allclose(result.power_dbfs, 20.0 * np.log10(256.0 / 2048.0))


def test_composite_signal_preserves_order_and_rejects_overlap() -> None:
    signal = SignalDescription(ModulationKind.GFSK, symbol_rate_hz=1_000_000.0)
    first = ModulationSegment(0, 100, signal, name="FSK")
    second = ModulationSegment(100, 200, signal, name="PSK")

    composite = CompositeSignalDescription((second, first))
    assert composite.segments == (first, second)

    with pytest.raises(ValueError, match="must not overlap"):
        CompositeSignalDescription((first, ModulationSegment(99, 150, signal)))


def test_generated_gfsk_is_decoded_on_symbol_timeline() -> None:
    recording, signal = GeneratedIQSource.fsk(symbol_count=128, seed=42)
    result = VSAAnalyzer().analyze(recording, signal, VSASettings(remove_dc=False))

    expected = recording.metadata["generated_symbols"]
    np.testing.assert_array_equal(result.decoded_bits, expected)
    assert result.symbol_time_s.size == expected.size
    assert result.frequency_error_hz == pytest.approx(0.0, abs=5_000.0)
    assert result.metadata["estimated_deviation_hz"] == pytest.approx(
        signal.frequency_deviation_hz, rel=0.01
    )


def test_generated_qpsk_has_near_zero_evm_and_decodes_symbols() -> None:
    recording, signal = GeneratedIQSource.psk(
        modulation=ModulationKind.QPSK,
        symbol_count=128,
        seed=5,
    )
    result = VSAAnalyzer().analyze(recording, signal, VSASettings(remove_dc=False))

    np.testing.assert_array_equal(
        result.decoded_symbols, recording.metadata["generated_symbols"]
    )
    assert result.evm_rms_percent == pytest.approx(0.0, abs=1e-5)
    assert result.measured_symbols.flags.writeable is False


def test_generated_differential_psk_decodes_phase_changes() -> None:
    recording, signal = GeneratedIQSource.psk(
        modulation=ModulationKind.PI4_DQPSK,
        symbol_count=128,
        seed=9,
    )
    result = VSAAnalyzer().analyze(recording, signal, VSASettings(remove_dc=False))

    np.testing.assert_array_equal(
        result.decoded_symbols, recording.metadata["generated_symbols"][1:]
    )
    assert result.evm_rms_percent == pytest.approx(0.0, abs=1e-5)


def test_session_invalidates_result_when_signal_changes() -> None:
    recording, signal = GeneratedIQSource.psk(symbol_count=32)
    session = VSASession(recording=recording, signal=signal)
    assert session.analyze().decoded_symbols.size == 32

    session.set_signal(
        SignalDescription(ModulationKind.BPSK, symbol_rate_hz=signal.symbol_rate_hz)
    )
    assert session.result is None
    assert session.revision == 1


def test_npz_file_source_round_trip_preserves_capture_metadata(tmp_path) -> None:
    recording, _ = GeneratedIQSource.fsk(symbol_count=16)
    path = tmp_path / "capture.npz"

    FileIQSource.save_npz(path, recording)
    loaded = FileIQSource.load(path)

    np.testing.assert_array_equal(loaded.iq, recording.iq)
    assert loaded.sample_rate_hz == recording.sample_rate_hz
    assert loaded.center_frequency_hz == recording.center_frequency_hz


def test_spectrum_peak_uses_relative_frequency_axis() -> None:
    sample_rate_hz = 1_000_000.0
    frequency_hz = 125_000.0
    n = np.arange(4096)
    recording = IQRecording(
        np.exp(2j * np.pi * frequency_hz * n / sample_rate_hz),
        sample_rate_hz=sample_rate_hz,
    )
    signal = SignalDescription(ModulationKind.FSK2, symbol_rate_hz=10_000.0)

    result = VSAAnalyzer().analyze(recording, signal, VSASettings(remove_dc=False))
    peak_frequency = result.spectrum_frequency_hz[np.argmax(result.spectrum_dbfs)]
    assert peak_frequency == pytest.approx(frequency_hz, abs=sample_rate_hz / 4096)


def test_composite_fsk_psk_capture_is_analyzed_on_one_timeline() -> None:
    fsk_recording, fsk_signal = GeneratedIQSource.fsk(
        symbol_count=64, gaussian_bt=None, seed=11
    )
    psk_recording, psk_signal = GeneratedIQSource.psk(
        modulation=ModulationKind.QPSK, symbol_count=64, seed=12
    )
    boundary = fsk_recording.sample_count
    recording = IQRecording(
        np.concatenate((fsk_recording.iq, psk_recording.iq)),
        sample_rate_hz=fsk_recording.sample_rate_hz,
        source="Generated mixed packet",
    )
    description = CompositeSignalDescription(
        (
            ModulationSegment(0, boundary, fsk_signal, name="FSK block"),
            ModulationSegment(
                boundary,
                recording.sample_count,
                psk_signal,
                name="PSK block",
            ),
        ),
        profile_name="Mixed test packet",
    )

    result = VSAAnalyzer().analyze_composite(
        recording, description, VSASettings(remove_dc=False)
    )

    assert [item.segment.name for item in result.segments] == ["FSK block", "PSK block"]
    np.testing.assert_array_equal(
        result.segments[0].result.decoded_bits,
        fsk_recording.metadata["generated_symbols"],
    )
    np.testing.assert_array_equal(
        result.segments[1].result.decoded_symbols,
        psk_recording.metadata["generated_symbols"],
    )
    assert result.segments[0].result.time_s[0] == 0.0
    assert result.segments[1].result.time_s[0] == pytest.approx(
        boundary / recording.sample_rate_hz
    )
    assert result.decoded_bits.size == 64 + 64 * 2
