"""Protocol decoder registry and shared analysis entry point."""

from __future__ import annotations

from typing import Protocol
from dataclasses import replace

from pluto_protocol.model import DecodeProbeResult, PacketAnalysisResult, PacketDecodeInput


class ProtocolDecoder(Protocol):
    protocol_id: str
    protocol_name: str
    def probe(self, packet: PacketDecodeInput) -> DecodeProbeResult: ...
    def decode(self, packet: PacketDecodeInput) -> PacketAnalysisResult: ...


class ProtocolRegistry:
    def __init__(self) -> None:
        self._decoders: dict[str, ProtocolDecoder] = {}

    def register(self, decoder: ProtocolDecoder) -> None:
        if decoder.protocol_id in self._decoders:
            raise ValueError(f"decoder already registered: {decoder.protocol_id}")
        self._decoders[decoder.protocol_id] = decoder

    def get(self, protocol_id: str) -> ProtocolDecoder:
        try:
            return self._decoders[protocol_id]
        except KeyError as error:
            raise ValueError(f"unknown protocol decoder: {protocol_id}") from error

    def probe(self, packet: PacketDecodeInput) -> tuple[DecodeProbeResult, ...]:
        return tuple(sorted((decoder.probe(packet) for decoder in self._decoders.values()), key=lambda item: item.confidence, reverse=True))

    def decode(self, packet: PacketDecodeInput) -> PacketAnalysisResult:
        protocol_id = packet.protocol_hint
        if protocol_id is None:
            probes = self.probe(packet)
            if not probes:
                raise ValueError("no protocol decoders are registered")
            protocol_id = probes[0].protocol_id
        return replace(self.get(protocol_id).decode(packet), decode_context=packet.context)


def default_registry() -> ProtocolRegistry:
    from pluto_protocol.bluetooth.br_edr import BluetoothBREDRDecoder
    from pluto_protocol.bluetooth.hdt import BluetoothHDTDecoder
    from pluto_protocol.bluetooth.le import BluetoothLEDecoder
    from pluto_protocol.dect.classic import DectClassicDecoder
    from pluto_protocol.wifi.mac import WiFiMACDecoder
    registry = ProtocolRegistry()
    registry.register(BluetoothBREDRDecoder())
    registry.register(BluetoothHDTDecoder())
    registry.register(BluetoothLEDecoder())
    registry.register(DectClassicDecoder())
    registry.register(WiFiMACDecoder())
    return registry


def analyze_packet(packet: PacketDecodeInput) -> PacketAnalysisResult:
    return default_registry().decode(packet)
