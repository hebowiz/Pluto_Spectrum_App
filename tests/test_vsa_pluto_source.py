from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pluto_sa.config.input_frontend import InputPowerCorrection
from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.sdr.iq_stream import IQBlock, IQStreamBuffer
from pluto_sa.sdr.trigger import TriggerKind, TriggerSlope
from pluto_sa.vsa.analysis import VSAAnalyzer
from pluto_sa.vsa.model import ModulationKind, SignalDescription, VSASettings
from pluto_sa.vsa.pluto_source import (
    CaptureCancelledError,
    PlutoCaptureSettings,
    PlutoLiveSource,
)


class _FakeReceiver:
    instances: list["_FakeReceiver"] = []

    def __init__(self, config: SpectrumConfig) -> None:
        self.config = config
        self.closed = False
        self.reconfigured: list[SpectrumConfig] = []
        self.capture_fresh: list[bool] = []
        self.stream_fresh: list[bool] = []
        self.stream_block_sizes: list[int] = []
        self.stop_calls = 0
        self.iq_stream = IQStreamBuffer(capacity_blocks=8)
        self.__class__.instances.append(self)

    def reconfigure(self, config: SpectrumConfig) -> None:
        self.config = config
        self.reconfigured.append(config)

    def capture_iq_block(
        self, count: int, *, source: str, fresh: bool = False
    ) -> IQBlock:
        self.capture_fresh.append(bool(fresh))
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

    def start(
        self,
        *,
        block_size: int | None = None,
        source: str = "continuous",
        max_blocks: int | None = None,
        fresh: bool = False,
    ):
        del max_blocks
        count = max(1, int(block_size or self.config.rx_buffer_size))
        self.stream_fresh.append(bool(fresh))
        self.stream_block_sizes.append(count)
        self.iq_stream.begin_stream(clear=True)
        cursor = self.iq_stream.create_cursor(start="latest")
        self.iq_stream.publish(
            np.full(count, 100.0 + 0.0j, dtype=np.complex64),
            source=source,
        )
        self.iq_stream.publish(
            np.full(count, 200.0 + 0.0j, dtype=np.complex64),
            source=source,
        )
        return cursor

    def read_iq_stream(self, cursor, *, max_blocks: int | None = None):
        return self.iq_stream.read(cursor, max_blocks=max_blocks)

    def stop(self) -> bool:
        self.stop_calls += 1
        return True

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
    assert receiver.capture_fresh == [True]

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


def test_pluto_single_capture_reuses_unchanged_receiver_configuration() -> None:
    _FakeReceiver.instances.clear()
    source = PlutoLiveSource(receiver_factory=_FakeReceiver)
    settings = PlutoCaptureSettings(capture_length_s=0.010)

    first = source.capture_single(settings)
    second = source.capture_single(settings)

    receiver = _FakeReceiver.instances[0]
    assert len(_FakeReceiver.instances) == 1
    assert receiver.reconfigured == []
    assert first.sample_count == second.sample_count == 80_000

    changed = replace(settings, center_frequency_hz=2_402_000_000.0)
    source.capture_single(changed)

    assert len(receiver.reconfigured) == 1
    assert receiver.reconfigured[0].center_freq_hz == 2_402_000_000


def test_pluto_capture_can_preserve_rx_buffer_between_continuous_blocks() -> None:
    _FakeReceiver.instances.clear()
    source = PlutoLiveSource(receiver_factory=_FakeReceiver)
    settings = PlutoCaptureSettings(capture_length_s=0.010)

    source.capture_single(settings, fresh=True)
    source.capture_single(settings, fresh=False)

    assert _FakeReceiver.instances[0].capture_fresh == [True, False]


def test_pluto_experimental_lo_offset_preserves_requested_analysis_center() -> None:
    _FakeReceiver.instances.clear()
    source = PlutoLiveSource(receiver_factory=_FakeReceiver)
    settings = PlutoCaptureSettings(
        center_frequency_hz=2_441_000_000.0,
        lo_offset_hz=1_500_000.0,
        analysis_bandwidth_hz=1_250_000.0,
    )

    recording = source.capture_single(settings)

    receiver = _FakeReceiver.instances[0]
    assert settings.hardware_lo_frequency_hz == 2_442_500_000.0
    assert receiver.config.center_freq_hz == 2_442_500_000
    assert recording.center_frequency_hz == 2_442_500_000.0
    assert recording.metadata["requested_center_frequency_hz"] == pytest.approx(
        2_441_000_000.0
    )
    assert recording.metadata["hardware_lo_frequency_hz"] == pytest.approx(
        2_442_500_000.0
    )
    assert recording.metadata["lo_offset_hz"] == pytest.approx(1_500_000.0)
    assert recording.metadata["experimental_lo_offset"] is True
    assert recording.metadata["requested_analysis_bandwidth_hz"] == pytest.approx(
        1_250_000.0
    )


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


@pytest.mark.parametrize(
    ("offset_symbols", "expected_trigger_offset"),
    ((-10.0, 10), (10.0, None)),
)
def test_pluto_power_trigger_applies_signed_trigger_offset(
    offset_symbols: float,
    expected_trigger_offset: int | None,
) -> None:
    _FakeReceiver.instances.clear()
    source = PlutoLiveSource(receiver_factory=_FakeReceiver)
    settings = PlutoCaptureSettings(
        symbol_rate_hz=1_000_000.0,
        samples_per_symbol=8,
        capture_length_s=100 / 8_000_000.0,
        trigger_source=TriggerKind.POWER_LEVEL,
        trigger_level_dbm=-30.0,
        trigger_slope=TriggerSlope.RISING,
        trigger_offset_s=offset_symbols / 8_000_000.0,
    )

    recording = source.capture_single(settings)

    assert recording.sample_count == 100
    assert recording.metadata["acquisition_trigger_source"] == "power_level"
    assert recording.metadata["acquisition_trigger_offset_s"] == pytest.approx(
        offset_symbols / 8_000_000.0
    )
    if expected_trigger_offset is None:
        assert recording.trigger_sample_index is None
    else:
        assert recording.trigger_sample_index is not None
        assert (
            recording.trigger_sample_index - recording.start_sample_index
            == expected_trigger_offset
        )


def test_pluto_power_trigger_zero_offset_retains_default_prestore() -> None:
    _FakeReceiver.instances.clear()
    source = PlutoLiveSource(receiver_factory=_FakeReceiver)
    settings = PlutoCaptureSettings(
        symbol_rate_hz=1_000_000.0,
        samples_per_symbol=8,
        capture_length_s=400 / 8_000_000.0,
        trigger_source=TriggerKind.POWER_LEVEL,
        trigger_level_dbm=-30.0,
        trigger_offset_s=0.0,
    )

    recording = source.capture_single(settings)

    assert settings.default_trigger_prestore_samples == 128
    assert recording.sample_count == 400
    assert recording.trigger_sample_index is not None
    assert recording.trigger_sample_index - recording.start_sample_index == 128
    assert recording.metadata["acquisition_trigger_offset_s"] == 0.0
    assert recording.metadata["acquisition_default_prestore_samples"] == 128
    receiver = _FakeReceiver.instances[0]
    assert receiver.stream_block_sizes == [65_536]
    assert receiver.stream_fresh == [True]
    assert receiver.stop_calls == 1


def test_pluto_power_trigger_record_crosses_continuous_stream_blocks() -> None:
    _FakeReceiver.instances.clear()
    source = PlutoLiveSource(receiver_factory=_FakeReceiver)
    settings = PlutoCaptureSettings(
        symbol_rate_hz=1_000_000.0,
        samples_per_symbol=8,
        capture_length_s=0.010,
        trigger_source=TriggerKind.POWER_LEVEL,
        trigger_level_dbm=-30.0,
    )

    recording = source.capture_single(settings)

    assert recording.sample_count == 80_000
    assert recording.trigger_sample_index - recording.start_sample_index == 128
    np.testing.assert_array_equal(recording.iq[:65_536], 100.0 + 0.0j)
    np.testing.assert_array_equal(recording.iq[65_536:], 200.0 + 0.0j)


def test_pluto_power_trigger_wait_can_be_cancelled() -> None:
    _FakeReceiver.instances.clear()
    source = PlutoLiveSource(receiver_factory=_FakeReceiver)
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 3

    with pytest.raises(CaptureCancelledError, match="cancelled"):
        source.capture_single(
            PlutoCaptureSettings(
                trigger_source=TriggerKind.POWER_LEVEL,
                trigger_level_dbm=-30.0,
            ),
            cancelled=cancelled,
        )
