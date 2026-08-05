from __future__ import annotations

import numpy as np
import pytest

from pluto_sa.config.input_frontend import InputPowerCorrection
from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.sdr.iq_stream import IQBlock
from pluto_sa.vsa.analysis import VSAAnalyzer
from pluto_sa.vsa.model import ModulationKind, SignalDescription, VSASettings
from pluto_sa.vsa.pluto_source import PlutoCaptureSettings, PlutoLiveSource


class _FakeReceiver:
    instances: list["_FakeReceiver"] = []

    def __init__(self, config: SpectrumConfig) -> None:
        self.config = config
        self.closed = False
        self.reconfigured: list[SpectrumConfig] = []
        self.__class__.instances.append(self)

    def reconfigure(self, config: SpectrumConfig) -> None:
        self.config = config
        self.reconfigured.append(config)

    def capture_iq_block(self, count: int, *, source: str) -> IQBlock:
        iq = np.full(count, 100.0 + 0.0j, dtype=np.complex64)
        return IQBlock(
            sequence=0,
            stream_id=1,
            block_index=0,
            start_sample_index=0,
            iq=iq,
            timestamp_s=1.0,
            discontinuity_before=True,
            source=source,
            capture_elapsed_s=0.003,
        )

    def get_current_sample_rate_hz(self) -> int:
        return self.config.sample_rate_hz

    def get_current_rf_bandwidth_hz(self) -> int:
        return self.config.rx_bandwidth_hz

    def close(self) -> None:
        self.closed = True


def test_input_power_correction_is_shared_with_spectrum_config() -> None:
    config = SpectrumConfig(
        calibration_offset_db=-61.5,
        rx_gain_db=12,
        ext_att_db=30.0,
        ext_gain_db=4.0,
    )

    assert config.input_power_correction == InputPowerCorrection(
        calibration_offset_db=-61.5,
        internal_gain_db=12,
        external_attenuation_db=30.0,
        external_gain_db=4.0,
    )
    assert config.input_correction_db == pytest.approx(14.0)


def test_pluto_single_capture_defaults_to_eight_samples_per_symbol() -> None:
    _FakeReceiver.instances.clear()
    source = PlutoLiveSource(receiver_factory=_FakeReceiver)
    settings = PlutoCaptureSettings()

    recording = source.capture_single(settings)

    assert settings.samples_per_symbol == 8
    assert settings.requested_sample_rate_hz == 8_000_000
    assert settings.capture_samples == 24_000
    assert settings.nominal_usable_bandwidth_hz == 6_400_000.0
    assert recording.sample_count == 24_000
    assert recording.sample_rate_hz == 8_000_000.0
    assert recording.center_frequency_hz == 2_441_000_000.0
    assert recording.usable_bandwidth_hz == 6_400_000.0
    assert recording.metadata["capture_oversampling"] == 8
    assert recording.metadata["internal_gain_db"] == 30.0
    assert recording.metadata["external_attenuation_db"] == 30.0
    assert recording.metadata["external_gain_db"] == 0.0
    receiver = _FakeReceiver.instances[0]
    assert receiver.config.rx_gain_db == 30
    assert receiver.config.time_analyzer_sample_rate_hz == 8_000_000
    assert receiver.config.time_analyzer_rf_bandwidth_hz == 8_000_000

    result = VSAAnalyzer().analyze(
        recording,
        SignalDescription(ModulationKind.FSK2, symbol_rate_hz=1_000_000.0),
        VSASettings(remove_dc=False),
    )
    # Common Pluto SA convention: 20log10(100) - 62 dB, while the default
    # external 30 dB attenuator and internal 30 dB gain cancel.
    assert np.median(result.power_dbm) == pytest.approx(-22.0, abs=1e-6)

    source.close()
    assert receiver.closed


def test_pluto_capture_applies_external_path_and_swap_iq() -> None:
    _FakeReceiver.instances.clear()
    source = PlutoLiveSource(receiver_factory=_FakeReceiver)
    settings = PlutoCaptureSettings(
        capture_length_s=100e-6,
        swap_iq=True,
        power_correction=InputPowerCorrection(
            calibration_offset_db=-62.0,
            internal_gain_db=10.0,
            external_attenuation_db=20.0,
            external_gain_db=3.0,
        ),
    )

    recording = source.capture_single(settings)

    assert recording.sample_count == 800
    np.testing.assert_array_equal(recording.iq, 100j)
    assert recording.input_correction_db == pytest.approx(7.0)
    assert recording.dbfs_to_dbm_offset_db == pytest.approx(
        20.0 * np.log10(2048.0) - 62.0 + 7.0
    )
