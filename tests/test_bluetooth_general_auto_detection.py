from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pluto_vsa.model import IQRecording
from pluto_vsa.sources import FileIQSource
from pluto_vsa.channel import extract_requested_analysis_channel
from pluto_vsa.profiles.bluetooth_br import (
    access_code_bits,
    recover_lap_from_access_code_bits,
)
from pluto_vsa.protocol_modes.bluetooth.model import (
    BluetoothAnalysisProfile,
    analyze_bluetooth_classic_recordings,
    analyze_bluetooth_hdt_recordings,
    analyze_bluetooth_le_recordings,
)
from pluto_vsg.engine import (
    BluetoothBRWaveformEngine,
    BluetoothHDTWaveformEngine,
    BluetoothLEWaveformEngine,
)
from pluto_protocol.bluetooth.hdt import HDTRate
from pluto_vsg.model import BluetoothPacketKind, BluetoothLEPhy
from pluto_vsg.profiles import (
    bluetooth_br_edr_project,
    bluetooth_br_fields,
    bluetooth_le_fields,
    bluetooth_le_project,
    bluetooth_le_test_project,
    bluetooth_hdt_fields,
    bluetooth_hdt_project,
)


def _le_air_bits(value: int, count: int) -> str:
    return "".join(str((int(value) >> index) & 1) for index in range(count))


def _le_project_with_access_address(access_address: int):
    base = bluetooth_le_project(BluetoothLEPhy.LE_1M)
    settings = replace(
        base.bluetooth_le,
        sync_word_bits=_le_air_bits(access_address, 32),
        preamble_bits=(
            "10101010" if (int(access_address) & 1) == 0 else "01010101"
        ),
    )
    return replace(base, bluetooth_le=settings, fields=bluetooth_le_fields(settings))


def test_classic_access_code_recovers_lap_with_four_air_bit_errors() -> None:
    expected_lap = 0x5A1234
    damaged = access_code_bits(expected_lap).copy()
    damaged[[10, 31, 50, 70]] ^= 1

    assert recover_lap_from_access_code_bits(damaged) == (expected_lap, 4)


def test_general_classic_auto_detects_mixed_lap_uap_without_matching_hints() -> None:
    base = bluetooth_br_edr_project()
    settings = (
        replace(
            base.bluetooth_br,
            packet_kind=BluetoothPacketKind.DH1,
            payload_length_bytes=12,
            lap=0x5A1234,
            uap=0x47,
            clock_6_1=0x15,
            whitening_enabled=False,
        ),
        replace(
            base.bluetooth_br,
            packet_kind=BluetoothPacketKind.DH1,
            payload_length_bytes=9,
            lap=0x13579B,
            uap=0xA2,
            clock_6_1=0x31,
            whitening_enabled=False,
        ),
    )
    generated = [
        BluetoothBRWaveformEngine().generate(
            replace(base, bluetooth_br=item, fields=bluetooth_br_fields(item))
        )
        for item in settings
    ]
    spacer = np.zeros(256, dtype=np.complex64)
    recording = IQRecording(
        np.concatenate((spacer, generated[0].iq, spacer, generated[1].iq, spacer)),
        sample_rate_hz=generated[0].sample_rate_hz,
        center_frequency_hz=base.center_frequency_hz,
    )

    results = analyze_bluetooth_classic_recordings(
        recording,
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        lap=0x9E8B33,
        uap=0,
        clock_6_1=0,
        whitening_enabled=False,
        result_length=1024,
    )

    assert [item.metadata["detected_lap"] for item in results] == [
        0x5A1234,
        0x13579B,
    ]
    assert [item.metadata["detected_uap"] for item in results] == [0x47, 0xA2]
    assert all(item.metadata["detected_clock_6_1"] is None for item in results)
    assert all(item.packet.integrity.crc_valid is True for item in results)


def test_general_classic_auto_recovers_whitened_uap_and_clock_from_crc() -> None:
    base = bluetooth_br_edr_project()
    settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH1,
        payload_length_bytes=12,
        lap=0x5A1234,
        uap=0x47,
        clock_6_1=0x15,
        whitening_enabled=True,
    )
    generated = BluetoothBRWaveformEngine().generate(
        replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))
    )

    result = analyze_bluetooth_classic_recordings(
        IQRecording(
            generated.iq,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=base.center_frequency_hz,
        ),
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        lap=0x9E8B33,
        uap=0,
        clock_6_1=0,
        whitening_enabled=False,
        result_length=1024,
    )[0]

    assert result.metadata["detected_lap"] == settings.lap
    assert result.metadata["detected_uap"] == settings.uap
    assert result.metadata["detected_clock_6_1"] == settings.clock_6_1
    assert result.metadata["detected_whitening_enabled"] is True
    assert result.metadata["uap_clock_source"] == "hec_and_payload_crc"
    assert result.packet.integrity.hec_valid is True
    assert result.packet.integrity.crc_valid is True


@pytest.mark.parametrize(
    "packet_kind, expected_phy",
    (
        (BluetoothPacketKind.DH1_2, "EDR 2M"),
        (BluetoothPacketKind.DH1_3, "EDR 3M"),
    ),
)
def test_general_classic_auto_detects_edr_with_unknown_identity(
    packet_kind: BluetoothPacketKind,
    expected_phy: str,
) -> None:
    base = bluetooth_br_edr_project()
    settings = replace(
        base.bluetooth_br,
        packet_kind=packet_kind,
        payload_length_bytes=12,
        lap=0x5A1234,
        uap=0x47,
        whitening_enabled=False,
    )
    generated = BluetoothBRWaveformEngine().generate(
        replace(base, bluetooth_br=settings, fields=bluetooth_br_fields(settings))
    )
    recording = IQRecording(
        generated.iq,
        sample_rate_hz=generated.sample_rate_hz,
        center_frequency_hz=base.center_frequency_hz,
    )

    known = analyze_bluetooth_classic_recordings(
        recording,
        profile=BluetoothAnalysisProfile.RF_PHY_TEST,
        phy_search=expected_phy,
        lap=settings.lap,
        uap=settings.uap,
        clock_6_1=settings.clock_6_1,
        whitening_enabled=False,
        result_length=1024,
    )[0]
    automatic = analyze_bluetooth_classic_recordings(
        recording,
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        lap=0x9E8B33,
        uap=0,
        clock_6_1=0,
        whitening_enabled=False,
        result_length=1024,
    )[0]

    assert automatic.packet.phy_name == expected_phy
    assert automatic.packet.integrity.crc_valid is True
    assert automatic.metadata["detected_lap"] == settings.lap
    assert automatic.metadata["detected_uap"] == settings.uap
    np.testing.assert_array_equal(automatic.packet.raw_bits, known.packet.raw_bits)
    auto_pattern = automatic.metadata["analysis_session"].pattern_result
    known_pattern = known.metadata["analysis_session"].pattern_result
    assert auto_pattern.pattern_start_sample == known_pattern.pattern_start_sample
    assert auto_pattern.carrier_frequency_offset_hz == pytest.approx(
        known_pattern.carrier_frequency_offset_hz,
        abs=1e-9,
    )


def test_general_classic_detects_real_2dhx_with_explicit_phy() -> None:
    recording = FileIQSource.load(
        Path(__file__).with_name("fixtures") / "2-DHx.npz"
    )

    result = analyze_bluetooth_classic_recordings(
        recording,
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        lap=None,
        uap=None,
        clock_6_1=None,
        whitening_enabled=True,
        phy_search="EDR 2M",
        result_length=5_000,
    )[0]

    assert result.packet.phy_name == "EDR 2M"
    assert result.packet.packet_type == "2-DH3"
    assert result.metadata["detected_lap"] == 0xFCF255
    assert result.metadata["detected_uap"] == 0xC1
    assert result.metadata["detected_clock_6_1"] == 28
    assert result.metadata["uap_clock_source"] == "hec_and_payload_length"
    assert result.packet.integrity.hec_valid is True
    # This capture contains payload symbol errors, so identity selection must
    # remain possible even though its complete payload CRC does not validate.
    assert result.packet.integrity.crc_valid is False


def test_general_classic_rejects_wrong_explicit_edr_phy_quick_path() -> None:
    recording = FileIQSource.load(
        Path(__file__).with_name("fixtures") / "2-DHx.npz"
    )

    with pytest.raises(RuntimeError):
        analyze_bluetooth_classic_recordings(
            recording,
            profile=BluetoothAnalysisProfile.GENERAL_PACKET,
            lap=None,
            uap=None,
            clock_6_1=None,
            whitening_enabled=True,
            phy_search="EDR 3M",
            result_length=5_000,
        )


def test_general_classic_restores_offset_lo_and_matches_real_ldac_capture() -> None:
    raw_recording = FileIQSource.load(
        Path(__file__).with_name("fixtures") / "LDAC.npz"
    )
    recording = extract_requested_analysis_channel(raw_recording)

    result = analyze_bluetooth_classic_recordings(
        recording,
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        lap=None,
        uap=None,
        clock_6_1=None,
        whitening_enabled=True,
        phy_search=None,
        result_length=10_000,
    )[0]

    assert recording.center_frequency_hz == pytest.approx(2_404_000_000.0)
    assert result.packet.phy_name == "EDR 2M"
    assert result.packet.packet_type == "2-DH5"
    assert result.metadata["detected_lap"] == 0xFCF255
    assert result.metadata["edr_sync_correlation"] > 0.99
    assert result.metadata["packet_stop_source"] == (
        "rf_burst_end_unconfirmed_length"
    )
    assert result.metadata["physical_packet_stop_sample"] == pytest.approx(
        23211, abs=4
    )
    assert result.metadata["physical_packet_stop_sample"] > (
        result.metadata["decoded_packet_stop_sample"]
    )
    assert result.packet.source.stop_sample == pytest.approx(23211, abs=4)


def test_general_le_auto_detects_arbitrary_access_address_and_leaves_crc_unknown() -> None:
    access_address = 0xD2A4C68E
    project = _le_project_with_access_address(access_address)
    generated = BluetoothLEWaveformEngine().generate(project)

    results = analyze_bluetooth_le_recordings(
        IQRecording(
            generated.iq,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=project.center_frequency_hz,
        ),
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        phy="LE 1M",
        access_address=0x8E89BED6,
        channel_index=37,
        crc_init=0x123456,
        whitening_enabled=True,
        result_length=512,
    )

    assert len(results) == 1
    assert results[0].metadata["detected_access_address"] == access_address
    assert results[0].metadata["crc_init_source"] == "unknown"
    assert results[0].packet.integrity.crc_valid is None


def test_general_le_auto_detects_rf_test_access_address_not_its_complement() -> None:
    project = bluetooth_le_test_project(BluetoothLEPhy.LE_1M)
    generated = BluetoothLEWaveformEngine().generate(project)

    result = analyze_bluetooth_le_recordings(
        IQRecording(
            generated.iq,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=project.center_frequency_hz,
        ),
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        phy="LE 1M",
        access_address=0x8E89BED6,
        channel_index=project.bluetooth_le.whitening_channel_index,
        crc_init=0,
        whitening_enabled=project.bluetooth_le.whitening_enabled,
        result_length=512,
    )[0]

    assert result.metadata["detected_access_address"] == 0x71764129
    assert result.metadata["crc_init_source"] == "rf_test_packet_default"
    assert result.packet.integrity.crc_valid is True


def test_general_le_uses_observed_preamble_for_real_advertising_packet() -> None:
    recording = FileIQSource.load(
        Path(__file__).with_name("fixtures") / "Adv_test.npz"
    )

    result = analyze_bluetooth_le_recordings(
        recording,
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        phy="LE 1M",
        access_address=None,
        channel_index=0,
        crc_init=0,
        whitening_enabled=True,
        result_length=10_000,
    )[0]
    pattern = result.metadata["analysis_session"].pattern_result

    assert result.metadata["detected_access_address"] == 0x8E89BED6
    assert result.metadata["detected_preamble_first_bit"] == 0
    assert pattern.correlation > 0.99
    assert pattern.metadata["timing_confidence"] > 0.01
    assert result.packet.integrity.crc_valid is True
    assert result.packet.raw_bits.size == 336
    assert pattern.decoded_bits.size == result.packet.raw_bits.size
    assert result.metadata["packet_length_refined"] is True


def test_general_le_refines_each_real_advertising_packet_to_its_pdu_length() -> None:
    recording = FileIQSource.load(
        Path(__file__).with_name("fixtures") / "Adv_test0.npz"
    )

    results = analyze_bluetooth_le_recordings(
        recording,
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        phy="LE 1M",
        access_address=None,
        channel_index=0,
        crc_init=0,
        whitening_enabled=True,
        result_length=10_000,
    )

    assert [item.packet.raw_bits.size for item in results] == [336, 128, 136]
    assert all(item.packet.integrity.crc_valid is True for item in results)
    assert all(
        item.metadata["channel_index_source"] == "requested_center_frequency"
        for item in results
    )
    assert all(item.metadata["detected_channel_index"] == 39 for item in results)
    assert all(item.metadata["packet_length_refined"] is True for item in results)
    assert all(
        item.metadata["analysis_session"].pattern_result.decoded_bits.size
        == item.packet.raw_bits.size
        for item in results
    )
    assert all(
        item.metadata["packet_stop_sample"]
        == item.metadata["recording_sample_offset"]
        + item.metadata["analysis_session"].pattern_result.result_stop_sample
        for item in results
    )


def test_general_le_auto_detects_mixed_access_addresses() -> None:
    addresses = (0x8E89BED6, 0xD2A4C68E)
    projects = [_le_project_with_access_address(value) for value in addresses]
    generated = [BluetoothLEWaveformEngine().generate(item) for item in projects]
    spacer = np.zeros(256, dtype=np.complex64)

    results = analyze_bluetooth_le_recordings(
        IQRecording(
            np.concatenate((spacer, generated[0].iq, spacer, generated[1].iq, spacer)),
            sample_rate_hz=generated[0].sample_rate_hz,
            center_frequency_hz=projects[0].center_frequency_hz,
        ),
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
        phy="LE 1M",
        access_address=0,
        channel_index=37,
        crc_init=0,
        whitening_enabled=True,
        result_length=512,
    )

    assert [item.metadata["detected_access_address"] for item in results] == list(addresses)
    assert results[0].packet.integrity.crc_valid is True
    assert results[1].packet.integrity.crc_valid is None


def test_general_hdt_identifies_rf_test_training_without_claiming_normal_context() -> None:
    project = bluetooth_hdt_project(HDTRate.HDT7_5)
    generated = BluetoothHDTWaveformEngine().generate(
        replace(project, fields=bluetooth_hdt_fields(project.bluetooth_hdt))
    )

    result = analyze_bluetooth_hdt_recordings(
        IQRecording(
            generated.iq,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=project.center_frequency_hz,
        ),
        profile=BluetoothAnalysisProfile.GENERAL_PACKET,
    )[0]

    assert result.metadata["acquisition_mode"] == "auto_detect"
    assert result.metadata["hdt_training_source"] == "rf_test_training"
    assert result.metadata["hdt_general_training_supported"] is False
