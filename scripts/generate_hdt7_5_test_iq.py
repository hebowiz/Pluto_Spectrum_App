"""Generate the deterministic HDT7.5 IQ regression fixture.

This fixture exercises the Bluetooth HDT RF PHY test packet format-0 path.
It is generated from the same project/engine used by Pluto VSG so that VSG,
Generic VSA, and dedicated VSA tests share one reproducible reference waveform.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pluto_protocol.bluetooth.hdt import HDTRate, map_hdt_symbols
from pluto_vsg.engine import BluetoothHDTWaveformEngine
from pluto_vsg.profiles import bluetooth_hdt_fields, bluetooth_hdt_project


DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "tests"
    / "fixtures"
    / "bluetooth_hdt7_5_prbs9_16msps.npz"
)


def generate(output: Path) -> Path:
    project = bluetooth_hdt_project(HDTRate.HDT7_5)
    settings = replace(project.bluetooth_hdt, payload_length_bytes=255)
    project = replace(
        project,
        name="Bluetooth HDT7.5 PRBS-9 PHY Test",
        repeat_count=1,
        bluetooth_hdt=settings,
        fields=bluetooth_hdt_fields(settings),
    )
    result = BluetoothHDTWaveformEngine().generate(project)

    payload_bits = np.asarray(result.metadata["payload_bits"], dtype=np.uint8)
    coded_bits = np.asarray(result.metadata["coded_payload_bits"], dtype=np.uint8)
    payload_symbols = map_hdt_symbols(coded_bits, HDTRate.HDT7_5).astype(
        np.complex64
    )
    field_names = np.asarray(
        [boundary.name for boundary in result.field_boundaries], dtype="U64"
    )
    field_start_samples = np.asarray(
        [boundary.start_sample for boundary in result.field_boundaries],
        dtype=np.int64,
    )
    field_stop_samples = np.asarray(
        [boundary.stop_sample for boundary in result.field_boundaries],
        dtype=np.int64,
    )
    payload_index = next(
        index
        for index, name in enumerate(field_names)
        if name == "Coded PDU Header / Payload / CRC"
    )
    payload_start_sample = int(field_start_samples[payload_index])
    payload_stop_sample = int(field_stop_samples[payload_index])

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        iq=np.asarray(result.iq, dtype=np.complex64),
        sample_rate_hz=np.float64(result.sample_rate_hz),
        center_frequency_hz=np.float64(project.center_frequency_hz),
        usable_bandwidth_hz=np.float64(result.sample_rate_hz),
        full_scale=np.float64(1.0),
        amplitude_calibrated=np.bool_(False),
        source=np.asarray("Pluto VSG deterministic HDT PHY fixture"),
        project_name=np.asarray(project.name),
        phy=np.asarray(HDTRate.HDT7_5.value),
        modulation=np.asarray("16QAM"),
        symbol_mapping=np.asarray("Bluetooth HDT"),
        symbol_rate_hz=np.float64(result.metadata["symbol_rate_hz"]),
        samples_per_symbol=np.int64(result.metadata["samples_per_symbol"]),
        payload_source=np.asarray("PRBS-9"),
        payload_length_bytes=np.int64(settings.payload_length_bytes),
        payload_code_rate=np.asarray(result.metadata["payload_code_rate"]),
        payload_bits=payload_bits,
        coded_payload_bits=coded_bits,
        expected_payload_symbols=payload_symbols,
        packet_start_sample=np.int64(result.metadata["data_start_sample"]),
        packet_stop_sample=np.int64(result.metadata["data_stop_sample"]),
        payload_start_sample=np.int64(payload_start_sample),
        payload_stop_sample=np.int64(payload_stop_sample),
        field_names=field_names,
        field_start_samples=field_start_samples,
        field_stop_samples=field_stop_samples,
    )

    sidecar = output.with_suffix(output.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {
                "description": "Deterministic Pluto VSG HDT7.5 PHY test IQ",
                "implementation_scope": (
                    "Bluetooth HDT RF PHY test packet format 0 with standard "
                    "training, Control Header/HEC-C, PRBS-9 payload, and CRC-32"
                ),
                "sample_rate_hz": result.sample_rate_hz,
                "center_frequency_hz": project.center_frequency_hz,
                "symbol_rate_hz": result.metadata["symbol_rate_hz"],
                "samples_per_symbol": result.metadata["samples_per_symbol"],
                "phy": HDTRate.HDT7_5.value,
                "modulation": "16QAM",
                "symbol_mapping": "Bluetooth HDT",
                "payload_source": "PRBS-9",
                "payload_length_bytes": settings.payload_length_bytes,
                "payload_code_rate": result.metadata["payload_code_rate"],
                "packet_sample_range": [
                    int(result.metadata["data_start_sample"]),
                    int(result.metadata["data_stop_sample"]),
                ],
                "payload_sample_range": [payload_start_sample, payload_stop_sample],
                "generic_vsa_setup": {
                    "modulation": "16QAM",
                    "symbol_rate_hz": 2_000_000,
                    "tx_filter": "Root Raised Cosine",
                    "filter_parameter": settings.rrc_rolloff,
                    "symbol_mapping": "Bluetooth HDT",
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generated = generate(args.output.resolve())
    print(generated)


if __name__ == "__main__":
    main()
