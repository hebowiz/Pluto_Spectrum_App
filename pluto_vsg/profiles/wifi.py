"""Wi-Fi Non-HT OFDM project profile."""

from __future__ import annotations

import math

from pluto_vsg.model import (
    DataSourceKind, FieldDefinition, FilterKind, ModulationDefinition,
    ModulationKind, PowerEnvelopeDefinition, StandardProfile, WaveformProject,
    WiFiPSDUSource, WiFiSettings,
)
from pluto_vsg.wifi.common import LEGACY_RATES
from pluto_vsg.wifi.mac import build_psdu


def _modulation(kind: ModulationKind) -> ModulationDefinition:
    return ModulationDefinition(kind=kind, symbol_rate_hz=250_000.0, filter_kind=FilterKind.NONE, filter_parameter=0.0)


def wifi_fields(settings: WiFiSettings) -> tuple[FieldDefinition, ...]:
    psdu_length = len(build_psdu(settings))
    rate = LEGACY_RATES[int(settings.legacy_rate_mbps)]
    n_sym = math.ceil((16 + 8 * psdu_length + 6) / rate.n_dbps)
    data_modulation = {
        "BPSK": ModulationKind.OFDM_BPSK, "QPSK": ModulationKind.OFDM_QPSK,
        "16QAM": ModulationKind.OFDM_QAM16, "64QAM": ModulationKind.OFDM_QAM64,
    }[rate.modulation]
    training = _modulation(ModulationKind.TRAINING)
    data_children = tuple(
        FieldDefinition(
            name=f"OFDM #{index}", symbol_count=1, data_source=DataSourceKind.COMPUTED,
            data=f"48 data + 4 pilot subcarriers", modulation=_modulation(data_modulation),
        )
        for index in range(n_sym)
    )
    return (
        FieldDefinition("L-STF", 2, data_source=DataSourceKind.COMPUTED, data="10 short training repetitions", modulation=training),
        FieldDefinition("L-LTF", 2, data_source=DataSourceKind.COMPUTED, data="GI2 + 2 long training symbols", modulation=training),
        FieldDefinition("L-SIG", 1, logical_bit_count=24, data_source=DataSourceKind.COMPUTED, data=f"{settings.legacy_rate_mbps} Mbps / {psdu_length} byte", modulation=_modulation(ModulationKind.OFDM_BPSK)),
        FieldDefinition("DATA", n_sym, data_source=DataSourceKind.COMPUTED, data=f"SERVICE + {psdu_length}-byte PSDU + TAIL + PAD", modulation=_modulation(data_modulation), children=data_children),
    )


def wifi_project(settings: WiFiSettings | None = None) -> WaveformProject:
    settings = settings or WiFiSettings()
    oversample = int(settings.oversample_factor)
    return WaveformProject(
        name=("Wi-Fi Beacon" if WiFiPSDUSource(settings.psdu_source) == WiFiPSDUSource.BEACON else "Wi-Fi Non-HT OFDM Packet"),
        standard=StandardProfile.WIFI,
        sample_rate_hz=20_000_000.0 * oversample,
        samples_per_symbol=80 * oversample,
        period_symbols=float(settings.packet_period_us) / 4.0,
        center_frequency_hz=(2407 + 5 * int(settings.channel)) * 1e6,
        fields=wifi_fields(settings),
        power_envelope=PowerEnvelopeDefinition(enabled=False),
        wifi=settings,
    )


def wifi_beacon_project() -> WaveformProject:
    return wifi_project(WiFiSettings(psdu_source=WiFiPSDUSource.BEACON, packet_period_us=102_400.0))


__all__ = ["wifi_beacon_project", "wifi_fields", "wifi_project"]
