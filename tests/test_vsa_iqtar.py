from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tarfile

import numpy as np
import pytest

from pluto_sa.vsa.model import ModulationKind, SignalDescription
from pluto_sa.vsa.pattern import (
    KnownPattern,
    PatternSearchMode,
    PatternSearchSettings,
    ResultRangeSettings,
)
from pluto_sa.vsa.session import VSASession
from pluto_sa.vsa.sources import FileIQSource


def _write_iq_tar(
    path: Path,
    values: np.ndarray,
    *,
    samples: int,
    clock_hz: float = 4_000_000.0,
    data_format: str = "complex",
    data_type: str = "float32",
    scaling_factor: float = 1.0,
    channels: int = 1,
    user_data: str = "",
    extra_xml: bool = False,
) -> None:
    data_name = f"capture.{data_format}.{channels}ch.{data_type}"
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<RS_IQ_TAR_FileFormat fileFormatVersion="2">
  <Name>pytest R&amp;S capture</Name>
  <Comment>round trip</Comment>
  <DateTime>2026-08-04T12:00:00</DateTime>
  <Samples>{samples}</Samples>
  <Clock unit="Hz">{clock_hz}</Clock>
  <Format>{data_format}</Format>
  <DataType>{data_type}</DataType>
  <ScalingFactor unit="V">{scaling_factor}</ScalingFactor>
  <NumberOfChannels>{channels}</NumberOfChannels>
  <DataFilename>{data_name}</DataFilename>
  <UserData>{user_data}</UserData>
</RS_IQ_TAR_FileFormat>
""".encode("utf-8")
    with tarfile.open(path, "w") as archive:
        _add_bytes(archive, "capture.xml", xml)
        _add_bytes(archive, data_name, values.tobytes())
        if extra_xml:
            _add_bytes(archive, "unexpected.xml", b"<unexpected/>")


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, BytesIO(payload))


def test_load_iq_tar_complex_float32_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "capture.iq.tar"
    interleaved = np.array([1.0, -2.0, 0.5, 0.25], dtype="<f4")
    _write_iq_tar(
        path,
        interleaved,
        samples=2,
        scaling_factor=0.5,
        user_data="""
          <RohdeSchwarz><DataImportExport_MandatoryData>
            <ChannelNames><ChannelName>IQ Analyzer</ChannelName></ChannelNames>
            <CenterFrequency unit="MHz">2441</CenterFrequency>
          </DataImportExport_MandatoryData></RohdeSchwarz>
        """,
    )

    recording = FileIQSource.load(path)

    np.testing.assert_allclose(recording.iq, [0.5 - 1j, 0.25 + 0.125j])
    assert recording.sample_rate_hz == 4_000_000.0
    assert recording.center_frequency_hz == 2_441_000_000.0
    assert recording.usable_bandwidth_hz == recording.sample_rate_hz
    assert recording.source == "R&S iq-tar: capture.iq.tar"
    assert recording.amplitude_calibrated is False
    assert recording.metadata["iq_tar_scaling_factor_v"] == 0.5
    assert recording.metadata["iq_tar_selected_channel_name"] == "IQ Analyzer"


def test_load_iq_tar_selects_interleaved_multichannel_int16(tmp_path: Path) -> None:
    path = tmp_path / "mimo.iq.tar"
    # time 0: ch0=(1,2), ch1=(10,20); time 1 follows in the same order.
    values = np.array([1, 2, 10, 20, 3, 4, 30, 40], dtype="<i2")
    _write_iq_tar(
        path,
        values,
        samples=2,
        data_type="int16",
        scaling_factor=0.1,
        channels=2,
    )

    recording = FileIQSource.load(path, channel_index=1)

    np.testing.assert_allclose(recording.iq, [1 + 2j, 3 + 4j])
    assert recording.metadata["iq_tar_channel_count"] == 2
    assert recording.metadata["iq_tar_channel_index"] == 1


def test_load_iq_tar_converts_polar_and_real_data(tmp_path: Path) -> None:
    polar_path = tmp_path / "polar.iq.tar"
    polar = np.array([2.0, 0.0, 3.0, np.pi / 2.0], dtype="<f8")
    _write_iq_tar(
        polar_path,
        polar,
        samples=2,
        data_format="polar",
        data_type="float64",
        scaling_factor=0.25,
    )
    real_path = tmp_path / "real.iq.tar"
    real = np.array([-2, 4], dtype="i1")
    _write_iq_tar(
        real_path,
        real,
        samples=2,
        data_format="real",
        data_type="int8",
        scaling_factor=0.5,
    )

    np.testing.assert_allclose(FileIQSource.load(polar_path).iq, [0.5, 0.75j], atol=1e-7)
    np.testing.assert_allclose(FileIQSource.load(real_path).iq, [-1 + 0j, 2 + 0j])


def test_load_iq_tar_rejects_invalid_archive_layout_and_size(tmp_path: Path) -> None:
    multiple_xml = tmp_path / "multiple.iq.tar"
    _write_iq_tar(
        multiple_xml,
        np.zeros(4, dtype="<f4"),
        samples=2,
        extra_xml=True,
    )
    with pytest.raises(ValueError, match="exactly one parameter XML"):
        FileIQSource.load(multiple_xml)

    wrong_size = tmp_path / "wrong-size.iq.tar"
    _write_iq_tar(wrong_size, np.zeros(2, dtype="<f4"), samples=2)
    with pytest.raises(ValueError, match="binary size does not match"):
        FileIQSource.load(wrong_size)


def test_load_iq_tar_rejects_out_of_range_channel(tmp_path: Path) -> None:
    path = tmp_path / "capture.iq.tar"
    _write_iq_tar(path, np.zeros(4, dtype="<f4"), samples=2)

    with pytest.raises(ValueError, match="channel_index"):
        FileIQSource.load(path, channel_index=1)


def test_committed_rs_iq_tar_sample_is_loadable() -> None:
    path = Path(__file__).with_name("fixtures") / "rs_sample_gfsk_8msps.iq.tar"

    recording = FileIQSource.load(path)

    assert recording.sample_count == 2048
    assert recording.sample_rate_hz == 8_000_000.0
    assert recording.center_frequency_hz == 2_441_000_000.0
    assert recording.metadata["iq_tar_format"] == "complex"
    assert recording.metadata["iq_tar_data_type"] == "float32"
    np.testing.assert_allclose(np.abs(recording.iq), 0.1, atol=1e-6)


def test_analysis_bandwidth_preserves_gfsk_symbol_timing() -> None:
    path = Path(__file__).with_name("fixtures") / "rs_sample_gfsk_8msps.iq.tar"
    recording = FileIQSource.load(path)
    expected = np.concatenate(
        (np.array([1, 0] * 8, dtype=np.uint8), _prbs9_bits(240))
    )
    session = VSASession(
        recording=recording,
        signal=SignalDescription(
            modulation=ModulationKind.GFSK,
            symbol_rate_hz=1_000_000.0,
            frequency_deviation_hz=250_000.0,
            tx_filter="Gaussian",
            filter_parameter=0.5,
        ),
    )
    session.update_settings(
        remove_dc=True,
        analysis_center_frequency_hz=2_441_000_000.0,
        analysis_bandwidth_hz=2_000_000.0,
    )
    session.configure_pattern_analysis(
        PatternSearchSettings(
            pattern=KnownPattern(tuple(map(int, expected[:16]))),
            mode=PatternSearchMode.ON,
        ),
        ResultRangeSettings(result_length=200),
    )

    session.analyze()

    assert session.pattern_result is not None
    assert session.pattern_result.timing_phase_samples == 0
    np.testing.assert_array_equal(
        session.pattern_result.decoded_symbols,
        expected[:200],
    )


def _prbs9_bits(count: int) -> np.ndarray:
    state = 0x1FF
    bits = np.empty(count, dtype=np.uint8)
    for index in range(count):
        bits[index] = state & 1
        feedback = ((state >> 0) ^ (state >> 4)) & 1
        state = (state >> 1) | (feedback << 8)
    return bits
