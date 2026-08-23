import pyqtgraph as pg

from pluto_sa.standards.adsb1090.leaflet_map import (
    LeafletAircraftMap,
    _LEAFLET_HTML,
)


def test_leaflet_html_uses_openstreetmap_tiles_with_attribution() -> None:
    assert "leaflet@1.9.4" in _LEAFLET_HTML
    assert (
        "sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
        in _LEAFLET_HTML
    )
    assert "tile.openstreetmap.org" in _LEAFLET_HTML
    assert "OpenStreetMap" in _LEAFLET_HTML
    assert "updateAircraftTrack" in _LEAFLET_HTML
    assert "ResizeObserver" in _LEAFLET_HTML
    assert "map.invalidateSize" in _LEAFLET_HTML
    assert "#e000ff" in _LEAFLET_HTML
    assert "aircraftIcon" in _LEAFLET_HTML
    assert "bearingBetween" in _LEAFLET_HTML
    assert "payload.track_deg === null" in _LEAFLET_HTML
    assert "QWebChannel" in _LEAFLET_HTML
    assert "updateReceiverLocation" in _LEAFLET_HTML
    assert "receiverLocationSelected" in _LEAFLET_HTML
    assert "if (map.hasLayer(receiver)) map.removeLayer(receiver)" in _LEAFLET_HTML


def test_leaflet_map_caches_track_before_webengine_is_created() -> None:
    pg.mkQApp("ADS-B Leaflet map test")
    map_widget = LeafletAircraftMap()
    try:
        map_widget.set_aircraft_track(
            icao="40621D",
            callsign="TEST123",
            track_deg=123.5,
            points=[
                {
                    "elapsed_s": 1.0,
                    "latitude": 52.25,
                    "longitude": 3.91,
                    "altitude_ft": 38_000,
                }
            ],
        )

        assert map_widget.web_view_created is False
        assert map_widget.last_track_payload["icao"] == "40621D"
        assert map_widget.last_track_payload["track_deg"] == 123.5
        assert map_widget.last_track_payload["points"][0]["altitude_ft"] == 38_000
        map_widget.set_receiver_location(35.681236, 139.767125)
        assert map_widget.last_receiver_payload == {
            "latitude": 35.681236,
            "longitude": 139.767125,
        }
    finally:
        map_widget.shutdown()
        map_widget.close()
