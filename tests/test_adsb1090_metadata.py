from pathlib import Path

from pluto_sa.standards.adsb1090.metadata import AircraftMetadataDatabase


def test_opensky_csv_import_builds_persistent_icao_lookup(tmp_path: Path) -> None:
    csv_path = tmp_path / "aircraft.csv"
    csv_path.write_text(
        "icao24,registration,manufacturerName,model,typecode,serialNumber,"
        "operator,operatorCallsign,owner,registered\n"
        "40621d,G-EUUE,Airbus,A320-232,A320,1234,Example Air,EXAMPLE,"
        "Example Owner,United Kingdom\n",
        encoding="utf-8",
    )
    database = AircraftMetadataDatabase(tmp_path / "aircraft.sqlite")

    count = database.import_opensky_csv(csv_path)
    metadata = database.lookup("40621D")

    assert count == 1
    assert metadata is not None
    assert metadata.registration == "G-EUUE"
    assert metadata.manufacturer == "Airbus"
    assert metadata.model == "A320-232"
    assert metadata.type_code == "A320"
    assert metadata.operator == "Example Air"
    assert metadata.country == "United Kingdom"


def test_aircraft_metadata_lookup_returns_none_without_database(tmp_path: Path) -> None:
    database = AircraftMetadataDatabase(tmp_path / "missing.sqlite")

    assert database.lookup("40621D") is None
