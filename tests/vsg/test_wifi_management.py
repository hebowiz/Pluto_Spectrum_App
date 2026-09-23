"""Management frame vectors, active-field validation and IQ verification."""
from dataclasses import replace

import numpy as np
import pytest

from pluto_protocol import PacketDecodeInput, analyze_packet
from pluto_vsg.engine.wifi_legacy_ofdm import WiFiLegacyOFDMWaveformEngine
from pluto_vsg.model import WiFiPSDUSource as Source, WiFiSettings, validate_project
from pluto_vsg.persistence import project_from_dict, project_to_dict
from pluto_vsg.profiles.wifi import management_defaults, wifi_project
from pluto_vsg.protocol import analyze_generation_result
from pluto_vsg.wifi.mac import append_fcs, build_psdu
from pluto_vsg.wifi.validation import validate_wifi_settings


SOURCES = (Source.BEACON, Source.PROBE_REQUEST, Source.PROBE_RESPONSE)
# Literal IEEE 802.11 fields; trailing CRCs independently calculated by the
# reflected 0xEDB88320 bit recurrence, not by the builder or MAC decoder.
VECTORS = {
    Source.BEACON: '80003412ffffffffffff02060708090a020102030405c3abf0debc9a78563412c8000104'
                   '00075465737420524601038c98b003010b0504000200002a0102e12b9162',
    Source.PROBE_REQUEST: '40003412ffffffffffff02060708090a020102030405c3ab'
                          '00075465737420524601038c98b0320212188dd2a8bd',
    Source.PROBE_RESPONSE: '50003412ffffffffffff02060708090a020102030405c3abf0debc9a78563412c8000104'
                           '00075465737420524601038c98b003010b2a0102320212187af50bc4',
}


def decode(data):
    return analyze_packet(PacketDecodeInput(np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder='little'),
                                           protocol_hint='wifi.non_ht'))


def fields(packet):
    def walk(items):
        for item in items:
            yield item.field_id, item
            yield from walk(item.children)
    return dict(walk(packet.root_fields))


@pytest.mark.parametrize('source', SOURCES)
def test_exact_management_octets_and_independent_fcs(source):
    settings = WiFiSettings(psdu_source=source, ssid='Test RF', channel=11, bssid='02:01:02:03:04:05',
        source_address='02:06:07:08:09:0A', destination_address='FF:FF:FF:FF:FF:FF', duration_id=0x1234,
        sequence_number=0xABC, fragment_number=3, timestamp=0x123456789ABCDEF0, beacon_interval_tu=200,
        capability_information=0x401, supported_rates_hex='8C98B0', tim_hex='00020000', erp_information=2,
        extended_supported_rates_hex='' if source == Source.BEACON else '1218')
    expected = bytes.fromhex(VECTORS[source])
    assert build_psdu(settings) == expected
    packet = decode(expected)
    assert packet.packet_type == source.value
    assert packet.integrity.crc_valid and packet.integrity.complete
    f = fields(packet)
    if source == Source.PROBE_REQUEST:
        assert 'wifi.timestamp' not in f and 'wifi.ie.5' not in f and 'wifi.ie.3' not in f
        assert f['wifi.probe_request.ies'].start_bit == 192
    else:
        assert f['wifi.timestamp'].value == settings.timestamp
        assert ('wifi.ie.5' in f) == (source == Source.BEACON)
    if source != Source.BEACON:
        assert f['wifi.ie.50'].name == 'Extended Supported Rates'


@pytest.mark.parametrize('source', SOURCES)
@pytest.mark.parametrize('factor', [1, 2])
def test_management_generated_iq_to_mac_and_fcs(source, factor):
    settings = management_defaults(source, WiFiSettings(oversample_factor=factor, packet_period_us=1000))
    result = WiFiLegacyOFDMWaveformEngine().generate(wifi_project(settings))
    # Verify must recover everything from IQ even with unusable generation hints.
    result = replace(result, metadata={'phy_format': 'Non-HT OFDM', 'psdu': b'wrong', 'legacy_rate_mbps': 54})
    packet = analyze_generation_result(result)
    assert packet.packet_type == source.value
    assert packet.integrity.crc_valid and packet.integrity.complete
    assert not packet.issues
    assert packet.decode_context['psdu_hex'] == build_psdu(settings).hex()
    assert packet.decode_context['length'] == len(build_psdu(settings))
    assert packet.decode_context['rate_mbps'] == 6


@pytest.mark.parametrize('source', SOURCES)
def test_manual_fc_and_fcs_preserved(source):
    s = WiFiSettings(psdu_source=source, frame_control_auto=False, frame_control=0xA234,
                     fcs_auto=False, manual_fcs_hex='11223344')
    assert not validate_wifi_settings(s)
    assert build_psdu(s)[:2] == bytes.fromhex('34a2')
    assert build_psdu(s)[-4:] == bytes.fromhex('11223344')
    assert decode(build_psdu(replace(s, frame_control_auto=True))).integrity.crc_valid is False


def test_wildcard_and_directed_request_own_information_and_unknown_ie():
    s = management_defaults(Source.PROBE_REQUEST, WiFiSettings())
    s = replace(s, additional_ies_hex='DD050011220102')
    p = decode(build_psdu(s))
    summary = {item.key: item for item in p.summary}
    assert summary['ssid'].value == '' and 'Wildcard' in summary['ssid'].display
    assert summary['source_address'].value == '02:11:22:33:44:66'
    assert summary['destination_address'].value == summary['bssid'].value == 'FF:FF:FF:FF:FF:FF'
    unknown = fields(p)['wifi.ie.221']
    assert unknown.name == 'Unknown IE'
    assert [child.value for child in unknown.children] == [221, 5, '00 11 22 01 02']
    directed = decode(build_psdu(replace(s, ssid='Target AP', destination_address='02:AA:BB:CC:DD:EE',
                                        bssid='02:AA:BB:CC:DD:EE', source_address='02:01:02:03:04:05')))
    assert {item.key: item.value for item in directed.summary}['ssid'] == 'Target AP'
    assert fields(directed)['wifi.address2'].value == '02:01:02:03:04:05'


@pytest.mark.parametrize('source', SOURCES)
def test_new_settings_persistence_name_and_legacy_manual_fc(source):
    s = replace(management_defaults(source, WiFiSettings()), additional_ies_hex='DD03001122',
                frame_control_auto=False, frame_control=0x0880)
    assert wifi_project(s).name == f'Wi-Fi {source.value}'
    project = replace(wifi_project(s), name='My custom project')
    assert project_from_dict(project_to_dict(project)) == project
    payload = project_to_dict(wifi_project())
    for key in ('frame_control_auto', 'extended_supported_rates_hex', 'additional_ies_hex'):
        payload['project']['wifi'].pop(key)
    payload['project']['wifi']['frame_control'] = 0x0880
    restored = project_from_dict(payload)
    assert not restored.wifi.frame_control_auto
    assert build_psdu(restored.wifi)[:2] == bytes.fromhex('8008')
    payload['project']['wifi'].pop('frame_control')
    assert project_from_dict(payload).wifi.frame_control_auto


@pytest.mark.parametrize('source', [Source.PROBE_REQUEST, Source.RAW_HEX, Source.PATTERN, Source.PRBS9])
def test_inactive_beacon_fields_do_not_block_validation(source):
    base = wifi_project()
    s = replace(base.wifi, psdu_source=source, timestamp=-1, tim_hex='invalid', beacon_interval_tu=0,
                capability_information=-1, ds_channel=99, erp_information=-1)
    if source != Source.PROBE_REQUEST:
        s = replace(s, ssid='x'*100, bssid='invalid', destination_address='invalid', sequence_number=9999,
                    supported_rates_hex='invalid', extended_supported_rates_hex='invalid', additional_ies_hex='invalid')
    assert not validate_project(replace(base, wifi=s))


@pytest.mark.parametrize('change', [
    {'ssid': '\u3042'*11}, {'supported_rates_hex': '0C'*9}, {'extended_supported_rates_hex': '0C'*256},
    {'extended_supported_rates_hex': '00'}, {'additional_ies_hex': 'DD05AA'}, {'additional_ies_hex': 'DD'},
    {'packet_period_us': 1}, {'source_address': 'invalid'}, {'fragment_number': 16},
])
def test_probe_request_active_field_validation(change):
    s = replace(management_defaults(Source.PROBE_REQUEST, WiFiSettings()), **change)
    assert validate_wifi_settings(s)


@pytest.mark.parametrize('source', SOURCES)
def test_common_ie_truncation_and_source_specific_required_ies(source):
    s = management_defaults(source, WiFiSettings())
    damaged = decode(append_fcs(build_psdu(s)[:-4] + b'\xdd\x05\xaa'))
    assert not damaged.integrity.complete and damaged.integrity.crc_valid
    assert any(issue.code.endswith('ie_truncated') for issue in damaged.issues)
    start = 24 if source == Source.PROBE_REQUEST else 36
    missing = decode(append_fcs(build_psdu(s)[:start]))
    assert any('SSID' in issue.message for issue in missing.issues)
    if source == Source.PROBE_REQUEST:
        assert not any('TIM' in issue.message or 'DS Parameter' in issue.message for issue in missing.issues)
