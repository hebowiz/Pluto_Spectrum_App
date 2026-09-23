from dataclasses import replace
import binascii
import numpy as np
import pytest

from pluto_protocol import PacketDecodeInput, analyze_packet
from pluto_vsg.model import WiFiSettings, WiFiPSDUSource, validate_project
from pluto_vsg.profiles.wifi import wifi_project
from pluto_vsg.wifi.mac import build_psdu
from pluto_vsg.persistence import project_to_dict, project_from_dict


def decode(psdu):
    return analyze_packet(PacketDecodeInput(np.unpackbits(np.frombuffer(psdu,dtype=np.uint8),bitorder='little'),protocol_hint='wifi.non_ht'))


def fields(packet):
    def walk(items):
        for item in items:
            yield item.field_id,item
            yield from walk(item.children)
    return dict(walk(packet.root_fields))


def test_beacon_all_editable_fields_and_ie_byte_order():
    s = WiFiSettings(ssid='Test RF',channel=11,bssid='02:01:02:03:04:05',source_address='02:06:07:08:09:0A',
        destination_address='FF:FF:FF:FF:FF:FF',duration_id=0x1234,sequence_number=0xABC,fragment_number=3,
        timestamp=0x123456789ABCDEF0,beacon_interval_tu=200,capability_information=0x401,
        supported_rates_hex='8C98B0',tim_hex='00020000',erp_information=2)
    psdu = build_psdu(s)
    assert psdu[2:4] == bytes.fromhex('3412')
    assert psdu[22:24] == bytes.fromhex('C3AB')
    assert psdu[24:32] == bytes.fromhex('F0DEBC9A78563412')
    p = decode(psdu)
    f = fields(p)
    assert p.integrity.crc_valid is True
    assert p.packet_type == 'Beacon'
    assert f['wifi.frame_type'].value == 'Management'
    assert f['wifi.frame_subtype'].value == 'Beacon (8)'
    assert [child.field_id for child in f['payload_body'].children] == ['wifi.beacon.fixed', 'wifi.beacon.ies']
    assert f['wifi.beacon.ies'].children[0].field_id == 'wifi.ie.0'
    assert f['wifi.address2'].value == s.source_address
    assert f['wifi.address3'].value == s.bssid
    assert f['wifi.timestamp'].value == s.timestamp
    assert f['wifi.beacon_interval'].value == 200
    assert f['wifi.ie.0'].value == 'Test RF'
    assert f['wifi.ie.3'].value == '11'
    assert f['wifi.ie.5'].value == '00 02 00 00'
    assert f['wifi.ie.42'].value == '02'


def test_raw_complete_and_without_fcs_are_distinct():
    beacon = build_psdu(WiFiSettings())
    s = WiFiSettings(psdu_source=WiFiPSDUSource.RAW_HEX,raw_psdu_hex=beacon.hex())
    assert build_psdu(s) == beacon  # No silent double-FCS on legacy raw projects.
    assert build_psdu(replace(s,raw_includes_fcs=False,raw_psdu_hex=beacon[:-4].hex())) == beacon
    bad = build_psdu(replace(s,raw_includes_fcs=False,raw_psdu_hex=beacon[:-4].hex(),fcs_auto=False,manual_fcs_hex='11223344'))
    assert bad[-4:] == bytes.fromhex('11223344')
    assert decode(bad).integrity.crc_valid is False


def test_manual_beacon_fcs_and_manual_ds_channel():
    psdu = build_psdu(WiFiSettings(fcs_auto=False,manual_fcs_hex='01234567',ds_channel_auto=False,ds_channel=2))
    assert psdu[-4:] == bytes.fromhex('01234567')
    p = decode(psdu)
    assert fields(p)['wifi.ie.3'].value == '2'
    assert not p.integrity.crc_valid


@pytest.mark.parametrize('change',[
    {'timestamp':-1},{'timestamp':2**64},{'fragment_number':16},{'duration_id':65536},
    {'destination_address':'bad'},{'source_address':'01:02'},{'manual_fcs_hex':'11','fcs_auto':False},
    {'tim_hex':'0000'},{'supported_rates_hex':''},{'ds_channel':14},
])
def test_invalid_fields_rejected(change):
    base = wifi_project()
    assert validate_project(replace(base,wifi=replace(base.wifi,**change)))


def test_mac_truncated_ie_and_fcs_failure_are_visible():
    frame = build_psdu(WiFiSettings())[:-4] + b'\xDD\x20\x01'
    p = decode(frame+binascii.crc32(frame).to_bytes(4,'little'))
    assert p.integrity.crc_valid
    assert not p.integrity.complete
    assert any(i.code == 'wifi.beacon.ie_truncated' for i in p.issues)


def test_new_fields_persist_and_old_projects_get_defaults():
    p = wifi_project(WiFiSettings(timestamp=12,duration_id=34,manual_fcs_hex='11223344',ds_channel_auto=False,ds_channel=3))
    assert project_from_dict(project_to_dict(p)) == p
    payload = project_to_dict(wifi_project())
    for key in ('raw_includes_fcs','timestamp','frame_control','source_address','manual_fcs_hex'):
        payload['project']['wifi'].pop(key)
    restored = project_from_dict(payload)
    assert restored.wifi.raw_includes_fcs is True
    assert restored.wifi.timestamp == 0
    assert restored.wifi.frame_control == 0x80


def test_prbs9_has_full_511_bit_period_and_balanced_population():
    raw = build_psdu(WiFiSettings(psdu_source=WiFiPSDUSource.PRBS9,payload_length_bytes=128))
    bits = np.unpackbits(np.frombuffer(raw,dtype=np.uint8),bitorder='little')
    np.testing.assert_array_equal(bits[:511],bits[511:1022])
    assert np.sum(bits[:511])==256
    assert not np.array_equal(bits[:490],bits[21:511])
