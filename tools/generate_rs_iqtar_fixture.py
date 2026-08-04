"""Generate the committed Rohde & Schwarz iq-tar sample capture."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import tarfile

import numpy as np
from scipy.ndimage import gaussian_filter1d


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "rs_sample_gfsk_8msps.iq.tar"
)


def _prbs9_bits(count: int) -> np.ndarray:
    """Return a deterministic x^9 + x^5 + 1 PRBS sequence."""

    state = 0x1FF
    bits = np.empty(count, dtype=np.uint8)
    for index in range(count):
        bits[index] = state & 1
        feedback = ((state >> 0) ^ (state >> 4)) & 1
        state = (state >> 1) | (feedback << 8)
    return bits


def _gfsk_iq() -> np.ndarray:
    sample_rate_hz = 8_000_000.0
    samples_per_symbol = 8
    deviation_hz = 250_000.0
    symbols = np.concatenate(
        (
            np.array([1, 0] * 8, dtype=np.uint8),
            _prbs9_bits(240),
        )
    )
    levels = np.repeat(2.0 * symbols.astype(np.float64) - 1.0, samples_per_symbol)
    sigma_samples = samples_per_symbol / (2.0 * np.pi * 0.5)
    shaped = gaussian_filter1d(levels, sigma=sigma_samples, mode="nearest")
    phase = 2.0 * np.pi * np.cumsum(deviation_hz * shaped) / sample_rate_hz
    # A moderate voltage leaves headroom while producing an easy-to-see capture.
    return (0.1 * np.exp(1j * phase)).astype(np.complex64)


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def generate(output: Path) -> None:
    iq = _gfsk_iq()
    interleaved = np.empty(iq.size * 2, dtype="<f4")
    interleaved[0::2] = iq.real
    interleaved[1::2] = iq.imag
    binary = interleaved.tobytes()
    data_filename = "rs_sample_gfsk.complex.1ch.float32"
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<RS_IQ_TAR_FileFormat fileFormatVersion="2"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
 xsi:noNamespaceSchemaLocation="RsIqTar.xsd">
  <Name>Pluto Spectrum App iq-tar Sample</Name>
  <Comment>Deterministic 1 Msym/s GFSK, BT 0.5, PRBS9 payload</Comment>
  <DateTime>2026-08-04T00:00:00</DateTime>
  <Samples>{iq.size}</Samples>
  <Clock unit="Hz">8000000</Clock>
  <Format>complex</Format>
  <DataType>float32</DataType>
  <ScalingFactor unit="V">1</ScalingFactor>
  <NumberOfChannels>1</NumberOfChannels>
  <DataFilename>{data_filename}</DataFilename>
  <UserData>
    <RohdeSchwarz>
      <DataImportExport_MandatoryData>
        <ChannelNames><ChannelName>IQ Analyzer</ChannelName></ChannelNames>
        <CenterFrequency unit="Hz">2441000000</CenterFrequency>
      </DataImportExport_MandatoryData>
      <PlutoSpectrumApp>
        <Modulation>GFSK</Modulation>
        <SymbolRate unit="Hz">1000000</SymbolRate>
        <FrequencyDeviation unit="Hz">250000</FrequencyDeviation>
        <GaussianBT>0.5</GaussianBT>
      </PlutoSpectrumApp>
    </RohdeSchwarz>
  </UserData>
</RS_IQ_TAR_FileFormat>
""".encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        archive.addfile(_tar_info("rs_sample_gfsk.xml", len(xml)), BytesIO(xml))
        archive.addfile(_tar_info(data_filename, len(binary)), BytesIO(binary))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
