"""Local aircraft metadata index populated from an external OpenSky CSV."""

from __future__ import annotations

import csv
import os
import sqlite3
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


OPENSKY_AIRCRAFT_DATABASE_URL = (
    "https://opensky-network.org/datasets/metadata/aircraftDatabase.csv"
)


@dataclass(frozen=True)
class AircraftMetadata:
    icao_address: str
    registration: str = ""
    manufacturer: str = ""
    model: str = ""
    type_code: str = ""
    serial_number: str = ""
    operator: str = ""
    operator_callsign: str = ""
    owner: str = ""
    country: str = ""


_FIELD_ALIASES = {
    "icao_address": ("icao24", "icao", "icao_address", "mode_s"),
    "registration": ("registration", "reg"),
    "manufacturer": ("manufacturername", "manufacturer", "maker"),
    "model": ("model", "modelname"),
    "type_code": ("typecode", "type_code", "icaoaircrafttype"),
    "serial_number": ("serialnumber", "serial_number", "serial"),
    "operator": ("operator", "operatorname"),
    "operator_callsign": ("operatorcallsign", "operator_callsign"),
    "owner": ("owner",),
    "country": ("registered", "country", "registrationcountry"),
}


def _normalized_row(row: dict[str, str | None]) -> dict[str, str]:
    return {
        str(key).strip().lower().replace(" ", "").replace("_", ""): str(
            value or ""
        ).strip()
        for key, value in row.items()
        if key is not None
    }


def _field(row: dict[str, str], name: str) -> str:
    for alias in _FIELD_ALIASES[name]:
        key = alias.lower().replace(" ", "").replace("_", "")
        value = row.get(key, "").strip()
        if value:
            return value
    return ""


class AircraftMetadataDatabase:
    """SQLite-backed ICAO metadata lookup safe for large CSV snapshots."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    def available(self) -> bool:
        return self.path.is_file()

    def lookup(self, icao_address: str) -> AircraftMetadata | None:
        if not self.available:
            return None
        connection = sqlite3.connect(self.path)
        try:
            row = connection.execute(
                """
                SELECT icao_address, registration, manufacturer, model,
                       type_code, serial_number, operator, operator_callsign,
                       owner, country
                  FROM aircraft
                 WHERE icao_address = ?
                """,
                (icao_address.strip().upper(),),
            ).fetchone()
        finally:
            connection.close()
        return AircraftMetadata(*row) if row is not None else None

    def import_opensky_csv(self, csv_path: str | Path) -> int:
        """Atomically rebuild the local index from an OpenSky-compatible CSV."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="aircraft-metadata-", suffix=".sqlite", dir=self.path.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        count = 0
        try:
            connection = sqlite3.connect(temporary_path)
            try:
                connection.execute(
                    """
                    CREATE TABLE aircraft (
                        icao_address TEXT PRIMARY KEY,
                        registration TEXT NOT NULL,
                        manufacturer TEXT NOT NULL,
                        model TEXT NOT NULL,
                        type_code TEXT NOT NULL,
                        serial_number TEXT NOT NULL,
                        operator TEXT NOT NULL,
                        operator_callsign TEXT NOT NULL,
                        owner TEXT NOT NULL,
                        country TEXT NOT NULL
                    )
                    """
                )
                with open(csv_path, "r", encoding="utf-8-sig", newline="") as stream:
                    reader = csv.DictReader(stream)
                    batch: list[tuple[str, ...]] = []
                    for source_row in reader:
                        row = _normalized_row(source_row)
                        icao = _field(row, "icao_address").upper().removeprefix("0X")
                        if not icao or len(icao) > 6:
                            continue
                        try:
                            icao = f"{int(icao, 16):06X}"
                        except ValueError:
                            continue
                        batch.append(
                            (
                                icao,
                                _field(row, "registration"),
                                _field(row, "manufacturer"),
                                _field(row, "model"),
                                _field(row, "type_code"),
                                _field(row, "serial_number"),
                                _field(row, "operator"),
                                _field(row, "operator_callsign"),
                                _field(row, "owner"),
                                _field(row, "country"),
                            )
                        )
                        if len(batch) >= 5_000:
                            connection.executemany(
                                "INSERT OR REPLACE INTO aircraft VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                batch,
                            )
                            count += len(batch)
                            batch.clear()
                    if batch:
                        connection.executemany(
                            "INSERT OR REPLACE INTO aircraft VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            batch,
                        )
                        count += len(batch)
                connection.execute(
                    "CREATE INDEX aircraft_registration ON aircraft(registration)"
                )
                connection.commit()
            finally:
                connection.close()
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return count

    def download_and_import(
        self, url: str = OPENSKY_AIRCRAFT_DATABASE_URL
    ) -> int:
        """Download an external CSV snapshot, then atomically index it."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="opensky-aircraft-", suffix=".csv", dir=self.path.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Pluto-VSA-ADSB1090/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                with open(temporary_path, "wb") as destination:
                    while chunk := response.read(1024 * 1024):
                        destination.write(chunk)
            return self.import_opensky_csv(temporary_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
