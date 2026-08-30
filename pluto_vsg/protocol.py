"""Adapter from generated waveforms to the shared packet analyzer."""

from __future__ import annotations

from pluto_protocol import PacketDecodeInput, PacketSourceInfo, analyze_packet
from pluto_protocol.model import PacketAnalysisResult
from pluto_vsg.engine.base import GenerationResult


def analyze_generation_result(result: GenerationResult) -> PacketAnalysisResult:
    """Decode the exact air bits emitted by a protocol waveform engine."""

    artifact = result.packet_bits
    if artifact is None:
        raise ValueError("generation result does not contain protocol packet bits")
    return analyze_packet(
        PacketDecodeInput(
            bits=artifact.bits,
            representation=artifact.representation,
            protocol_hint=artifact.protocol_id,
            phy_hint=artifact.phy_name,
            source=PacketSourceInfo(
                source_kind="vsg_generated",
                packet_index=0,
                center_frequency_hz=(
                    float(result.metadata["center_frequency_hz"])
                    if "center_frequency_hz" in result.metadata else None
                ),
            ),
            context=artifact.context,
        )
    )
