"""Adapter from VSA-demodulated air bits to the shared packet analyzer."""

from __future__ import annotations

import numpy as np

from pluto_protocol import PacketDecodeInput, PacketSourceInfo, analyze_packet
from pluto_protocol.model import PacketAnalysisResult


def analyze_demodulated_packet_bits(
    bits: np.ndarray,
    *,
    protocol_id: str,
    phy_name: str,
    context: dict[str, object] | None = None,
    packet_index: int | None = None,
    center_frequency_hz: float | None = None,
    start_sample: int | None = None,
    stop_sample: int | None = None,
) -> PacketAnalysisResult:
    """Decode one packet after VSA synchronization/demodulation.

    ``bits`` must be canonical over-the-air order.  Symbol mapping and packet
    boundary detection remain VSA responsibilities; field semantics and
    integrity checks are shared with VSG.
    """

    return analyze_packet(
        PacketDecodeInput(
            bits=np.asarray(bits, dtype=np.uint8),
            protocol_hint=protocol_id,
            phy_hint=phy_name,
            source=PacketSourceInfo(
                source_kind="vsa_demodulated",
                packet_index=packet_index,
                center_frequency_hz=center_frequency_hz,
                start_sample=start_sample,
                stop_sample=stop_sample,
            ),
            context={} if context is None else context,
        )
    )
