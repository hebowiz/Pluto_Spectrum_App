"""JSON project persistence for Pluto VSG."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from pluto_vsg.model import (
    BluetoothBRSettings,
    BluetoothLEPayloadType,
    BluetoothLEPayloadSourceKind,
    BluetoothLEPhy,
    BluetoothLESettings,
    BluetoothPacketKind,
    DataSourceKind,
    FieldDefinition,
    FilterKind,
    ModulationDefinition,
    ModulationKind,
    PayloadSourceKind,
    PowerEnvelopeDefinition,
    StandardProfile,
    WaveformProject,
    validate_project,
)


PROJECT_FORMAT = "pluto-vsg-project"
PROJECT_VERSION = 1


def _field_to_dict(packet_field: FieldDefinition) -> dict[str, object]:
    return {
        "name": packet_field.name,
        "symbol_count": packet_field.symbol_count,
        "logical_bit_count": packet_field.logical_bit_count,
        "data_source": packet_field.data_source.value,
        "data": packet_field.data,
        "relative_power_db": packet_field.relative_power_db,
        "modulation": {
            **asdict(packet_field.modulation),
            "kind": packet_field.modulation.kind.value,
            "filter_kind": packet_field.modulation.filter_kind.value,
        },
        "children": [_field_to_dict(child) for child in packet_field.children],
    }


def _field_from_dict(item: object) -> FieldDefinition:
    if not isinstance(item, dict) or not isinstance(item.get("modulation"), dict):
        raise ValueError("Invalid project field")
    modulation_payload = item["modulation"]
    children_payload = item.get("children", [])
    if not isinstance(children_payload, list):
        raise ValueError("Project field children must be a list")
    logical_count = item.get("logical_bit_count")
    return FieldDefinition(
        name=str(item["name"]),
        symbol_count=int(item["symbol_count"]),
        logical_bit_count=(None if logical_count is None else int(logical_count)),
        data_source=DataSourceKind(str(item["data_source"])),
        data=str(item.get("data", "")),
        relative_power_db=float(item.get("relative_power_db", 0.0)),
        modulation=ModulationDefinition(
            kind=ModulationKind(str(modulation_payload["kind"])),
            symbol_rate_hz=float(modulation_payload["symbol_rate_hz"]),
            filter_kind=FilterKind(str(modulation_payload["filter_kind"])),
            filter_parameter=float(modulation_payload["filter_parameter"]),
        ),
        children=tuple(_field_from_dict(child) for child in children_payload),
    )


def project_to_dict(project: WaveformProject) -> dict[str, object]:
    payload = asdict(project)
    payload["standard"] = project.standard.value
    payload["fields"] = [_field_to_dict(packet_field) for packet_field in project.fields]
    if project.bluetooth_br is not None:
        payload["bluetooth_br"] = {
            **asdict(project.bluetooth_br),
            "packet_kind": BluetoothPacketKind(project.bluetooth_br.packet_kind).value,
            "payload_source": project.bluetooth_br.payload_source.value,
        }
    if project.bluetooth_le is not None:
        payload["bluetooth_le"] = {
            **asdict(project.bluetooth_le),
            "phy": BluetoothLEPhy(project.bluetooth_le.phy).value,
            "payload_type": BluetoothLEPayloadType(
                project.bluetooth_le.payload_type
            ).value,
            "payload_source": BluetoothLEPayloadSourceKind(
                project.bluetooth_le.payload_source
            ).value,
        }
    return {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "project": payload,
    }


def project_from_dict(document: dict[str, object]) -> WaveformProject:
    if document.get("format") != PROJECT_FORMAT:
        raise ValueError("Not a Pluto VSG project")
    if int(document.get("version", 0)) != PROJECT_VERSION:
        raise ValueError("Unsupported Pluto VSG project version")
    payload = document.get("project")
    if not isinstance(payload, dict):
        raise ValueError("Project payload is missing")
    field_payloads = payload.get("fields", [])
    if not isinstance(field_payloads, list):
        raise ValueError("Project fields must be a list")
    fields = [_field_from_dict(item) for item in field_payloads]
    envelope_payload = payload.get("power_envelope", {})
    if not isinstance(envelope_payload, dict):
        raise ValueError("Invalid power envelope")
    bluetooth_payload = payload.get("bluetooth_br")
    bluetooth = None
    if bluetooth_payload is not None:
        if not isinstance(bluetooth_payload, dict):
            raise ValueError("Invalid Bluetooth BR settings")
        bluetooth_values = {
                **bluetooth_payload,
                "payload_source": PayloadSourceKind(
                    str(bluetooth_payload["payload_source"])
                ),
            }
        if "packet_kind" in bluetooth_payload:
            bluetooth_values["packet_kind"] = BluetoothPacketKind(
                str(bluetooth_payload["packet_kind"])
            )
        bluetooth = BluetoothBRSettings(**bluetooth_values)
    bluetooth_le_payload = payload.get("bluetooth_le")
    bluetooth_le = None
    if bluetooth_le_payload is not None:
        if not isinstance(bluetooth_le_payload, dict):
            raise ValueError("Invalid Bluetooth LE settings")
        bluetooth_le = BluetoothLESettings(
            **{
                **bluetooth_le_payload,
                "phy": BluetoothLEPhy(str(bluetooth_le_payload["phy"])),
                "payload_type": BluetoothLEPayloadType(
                    str(bluetooth_le_payload["payload_type"])
                ),
                "payload_source": BluetoothLEPayloadSourceKind(
                    str(
                        bluetooth_le_payload.get(
                            "payload_source", BluetoothLEPayloadSourceKind.PATTERN.value
                        )
                    )
                ),
            }
        )
    standard = StandardProfile(str(payload["standard"]))
    if (
        standard == StandardProfile.BLUETOOTH_BR_EDR
        and bluetooth is not None
        and fields
        and not any(packet_field.children for packet_field in fields)
    ):
        # Version-1 projects created before hierarchical fields remain readable.
        from pluto_vsg.profiles.bluetooth import bluetooth_br_fields

        fields = list(bluetooth_br_fields(bluetooth))
    project = WaveformProject(
        name=str(payload["name"]),
        standard=standard,
        sample_rate_hz=float(payload["sample_rate_hz"]),
        samples_per_symbol=int(payload["samples_per_symbol"]),
        repeat_count=int(payload["repeat_count"]),
        center_frequency_hz=float(payload.get("center_frequency_hz", 0.0)),
        fields=tuple(fields),
        power_envelope=PowerEnvelopeDefinition(**envelope_payload),
        bluetooth_br=bluetooth,
        bluetooth_le=bluetooth_le,
    )
    issues = validate_project(project)
    if issues:
        details = "; ".join(f"{issue.path}: {issue.message}" for issue in issues)
        raise ValueError(f"Invalid Pluto VSG project: {details}")
    return project


def save_project(path: str | Path, project: WaveformProject) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(project_to_dict(project), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_project(path: str | Path) -> WaveformProject:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read Pluto VSG project: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("Pluto VSG project root must be an object")
    return project_from_dict(document)
