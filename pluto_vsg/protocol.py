"""Adapter from generated waveforms to the shared packet analyzer."""

from __future__ import annotations

from pluto_protocol import PacketDecodeInput, PacketSourceInfo, analyze_packet
from pluto_protocol.model import PacketAnalysisResult
from pluto_vsg.engine.base import GenerationResult


def supports_packet_verification(result: GenerationResult | None) -> bool:
    return result is not None and (result.packet_bits is not None or is_wifi_iq(result))


def is_wifi_iq(result: GenerationResult) -> bool:
    return result.metadata.get("phy_format") == "Non-HT OFDM"


def analyze_generation_result(result: GenerationResult) -> PacketAnalysisResult:
    """Verify Wi-Fi from generated IQ; decode other protocols' emitted air bits."""

    if is_wifi_iq(result):
        from pluto_protocol.wifi import analyze_iq
        # Metadata selects the decoder only. RATE/LENGTH/seed/PSDU, timing and
        # boundaries are all recovered from IQ, even if metadata is corrupted.
        return analyze_iq(result.iq, result.sample_rate_hz,
                          source=PacketSourceInfo(source_kind="vsg_generated_iq",packet_index=0))
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
