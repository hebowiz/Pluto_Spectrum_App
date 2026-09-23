"""Reader for Rohde & Schwarz ``.iq.tar`` I/Q archives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import tarfile
import xml.etree.ElementTree as ET

import numpy as np


_MAX_XML_BYTES = 4 * 1024 * 1024
_DTYPES = {
    "int8": np.dtype("i1"),
    "int16": np.dtype("<i2"),
    "int32": np.dtype("<i4"),
    "float32": np.dtype("<f4"),
    "float64": np.dtype("<f8"),
}


@dataclass(frozen=True)
class IQTarData:
    """One selected channel plus metadata decoded from an iq-tar archive."""

    iq: np.ndarray
    sample_rate_hz: float
    center_frequency_hz: float
    scaling_factor_v: float
    channel_count: int
    channel_index: int
    data_format: str
    data_type: str
    metadata: dict[str, object]


def load_iq_tar(path: str | Path, *, channel_index: int = 0) -> IQTarData:
    """Read one channel from an R&S iq-tar file without extracting it."""

    archive_path = Path(path)
    try:
        archive = tarfile.open(archive_path, mode="r:*")
    except (tarfile.TarError, OSError) as error:
        raise ValueError(f"Invalid R&S iq-tar archive: {error}") from error

    with archive:
        regular_members = [member for member in archive.getmembers() if member.isfile()]
        xml_members = [
            member for member in regular_members if member.name.lower().endswith(".xml")
        ]
        if len(xml_members) != 1:
            raise ValueError("R&S iq-tar must contain exactly one parameter XML file")
        xml_member = xml_members[0]
        if xml_member.size > _MAX_XML_BYTES:
            raise ValueError("R&S iq-tar parameter XML is too large")
        xml_stream = archive.extractfile(xml_member)
        if xml_stream is None:
            raise ValueError("Cannot read R&S iq-tar parameter XML")
        xml_bytes = xml_stream.read(_MAX_XML_BYTES + 1)
        if b"<!doctype" in xml_bytes.lower() or b"<!entity" in xml_bytes.lower():
            raise ValueError("DTD and entity declarations are not allowed in iq-tar XML")
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as error:
            raise ValueError(f"Invalid R&S iq-tar parameter XML: {error}") from error

        if _local_name(root.tag) != "RS_IQ_TAR_FileFormat":
            raise ValueError("Unexpected R&S iq-tar XML root element")

        samples = _required_int(root, "Samples")
        sample_rate_hz = _required_float(root, "Clock")
        data_format = _required_text(root, "Format").lower()
        data_type = _required_text(root, "DataType").lower()
        data_filename = _required_text(root, "DataFilename")
        scaling_factor_v = _optional_float(root, "ScalingFactor", 1.0)
        channel_count = _optional_int(root, "NumberOfChannels", 1)

        if samples <= 0:
            raise ValueError("R&S iq-tar Samples must be positive")
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
            raise ValueError("R&S iq-tar Clock must be positive")
        if not np.isfinite(scaling_factor_v) or scaling_factor_v <= 0.0:
            raise ValueError("R&S iq-tar ScalingFactor must be positive")
        if channel_count <= 0:
            raise ValueError("R&S iq-tar NumberOfChannels must be positive")
        if not 0 <= int(channel_index) < channel_count:
            raise ValueError(
                f"channel_index must be between 0 and {channel_count - 1}"
            )
        if data_format not in {"complex", "real", "polar"}:
            raise ValueError(f"Unsupported R&S iq-tar Format: {data_format}")
        if data_type not in _DTYPES:
            raise ValueError(f"Unsupported R&S iq-tar DataType: {data_type}")
        if data_format == "polar" and data_type not in {"float32", "float64"}:
            raise ValueError("R&S iq-tar polar data requires float32 or float64")

        normalized_data_name = _safe_member_name(data_filename)
        matching_members = [
            member
            for member in regular_members
            if _normalized_member_name(member.name) == normalized_data_name
        ]
        if len(matching_members) != 1:
            raise ValueError(
                "R&S iq-tar DataFilename must identify exactly one regular file"
            )
        data_member = matching_members[0]
        components = 1 if data_format == "real" else 2
        value_count = samples * channel_count * components
        dtype = _DTYPES[data_type]
        expected_bytes = value_count * dtype.itemsize
        if data_member.size != expected_bytes:
            raise ValueError(
                "R&S iq-tar binary size does not match Samples, Format, "
                "DataType, and NumberOfChannels"
            )
        data_stream = archive.extractfile(data_member)
        if data_stream is None:
            raise ValueError("Cannot read R&S iq-tar binary data")
        raw = data_stream.read(expected_bytes + 1)
        if len(raw) != expected_bytes:
            raise ValueError("R&S iq-tar binary data is truncated")

    values = np.frombuffer(raw, dtype=dtype, count=value_count)
    selected = values.reshape(samples, channel_count, components)[:, int(channel_index), :]
    if data_format == "complex":
        iq = (selected[:, 0] + 1j * selected[:, 1]) * scaling_factor_v
    elif data_format == "polar":
        iq = selected[:, 0] * scaling_factor_v * np.exp(1j * selected[:, 1])
    else:
        iq = selected[:, 0] * scaling_factor_v + 0j

    center_element = _find_any(root, "CenterFrequency")
    center_frequency_hz = (
        _frequency_hz(center_element) if center_element is not None else 0.0
    )
    channel_names = [
        (element.text or "").strip()
        for element in root.iter()
        if _local_name(element.tag) == "ChannelName" and (element.text or "").strip()
    ]
    metadata: dict[str, object] = {
        "iq_tar_file_format_version": root.attrib.get("fileFormatVersion", ""),
        "iq_tar_format": data_format,
        "iq_tar_data_type": data_type,
        "iq_tar_scaling_factor_v": scaling_factor_v,
        "iq_tar_channel_count": channel_count,
        "iq_tar_channel_index": int(channel_index),
        "iq_tar_data_filename": data_filename,
        "iq_tar_amplitude_unit": "V",
    }
    for xml_name, metadata_name in (
        ("Name", "iq_tar_name"),
        ("Comment", "iq_tar_comment"),
        ("DateTime", "iq_tar_datetime"),
    ):
        element = _find_direct(root, xml_name)
        if element is not None and (element.text or "").strip():
            metadata[metadata_name] = (element.text or "").strip()
    if channel_names:
        metadata["iq_tar_channel_names"] = tuple(channel_names)
        if int(channel_index) < len(channel_names):
            metadata["iq_tar_selected_channel_name"] = channel_names[int(channel_index)]
    if center_element is not None:
        metadata["iq_tar_center_frequency_present"] = True

    return IQTarData(
        iq=np.asarray(iq, dtype=np.complex64),
        sample_rate_hz=sample_rate_hz,
        center_frequency_hz=center_frequency_hz,
        scaling_factor_v=scaling_factor_v,
        channel_count=channel_count,
        channel_index=int(channel_index),
        data_format=data_format,
        data_type=data_type,
        metadata=metadata,
    )


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_direct(root: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in root if _local_name(child.tag) == name), None)


def _find_any(root: ET.Element, name: str) -> ET.Element | None:
    return next((element for element in root.iter() if _local_name(element.tag) == name), None)


def _required_text(root: ET.Element, name: str) -> str:
    element = _find_direct(root, name)
    value = "" if element is None else (element.text or "").strip()
    if not value:
        raise ValueError(f"R&S iq-tar XML is missing {name}")
    return value


def _required_float(root: ET.Element, name: str) -> float:
    try:
        return float(_required_text(root, name))
    except ValueError as error:
        raise ValueError(f"R&S iq-tar {name} is not a valid number") from error


def _required_int(root: ET.Element, name: str) -> int:
    try:
        return int(_required_text(root, name))
    except ValueError as error:
        raise ValueError(f"R&S iq-tar {name} is not a valid integer") from error


def _optional_float(root: ET.Element, name: str, default: float) -> float:
    element = _find_direct(root, name)
    if element is None or not (element.text or "").strip():
        return default
    try:
        return float(element.text)
    except ValueError as error:
        raise ValueError(f"R&S iq-tar {name} is not a valid number") from error


def _optional_int(root: ET.Element, name: str, default: int) -> int:
    element = _find_direct(root, name)
    if element is None or not (element.text or "").strip():
        return default
    try:
        return int(element.text)
    except ValueError as error:
        raise ValueError(f"R&S iq-tar {name} is not a valid integer") from error


def _frequency_hz(element: ET.Element) -> float:
    try:
        value = float((element.text or "").strip())
    except ValueError as error:
        raise ValueError("R&S iq-tar CenterFrequency is not a valid number") from error
    unit = element.attrib.get("unit", "Hz").strip().lower()
    factors = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
    if unit not in factors:
        raise ValueError(f"Unsupported CenterFrequency unit: {unit}")
    result = value * factors[unit]
    if not np.isfinite(result):
        raise ValueError("R&S iq-tar CenterFrequency must be finite")
    return result


def _normalized_member_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _safe_member_name(name: str) -> str:
    normalized = _normalized_member_name(name)
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe R&S iq-tar DataFilename")
    return normalized
