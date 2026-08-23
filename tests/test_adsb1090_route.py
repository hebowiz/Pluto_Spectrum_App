import io
import json
import urllib.error

from pluto_sa.standards.adsb1090.route import (
    ADSBDBRouteClient,
    parse_adsbdb_route,
)


_RESPONSE = {
    "response": {
        "flightroute": {
            "callsign": "ANA123",
            "callsign_icao": "ANA123",
            "callsign_iata": "NH123",
            "airline": {"name": "All Nippon Airways"},
            "origin": {
                "name": "Tokyo Haneda Airport",
                "municipality": "Tokyo",
                "country_name": "Japan",
                "iata_code": "HND",
                "icao_code": "RJTT",
                "latitude": 35.5523,
                "longitude": 139.7798,
            },
            "destination": {
                "name": "Fukuoka Airport",
                "municipality": "Fukuoka",
                "country_name": "Japan",
                "iata_code": "FUK",
                "icao_code": "RJFF",
            },
        }
    }
}


def test_parse_adsbdb_route_extracts_airports() -> None:
    route = parse_adsbdb_route(_RESPONSE)
    assert route is not None
    assert route.callsign == "ANA123"
    assert route.airline_name == "All Nippon Airways"
    assert route.origin.compact_name == "HND"
    assert route.origin.icao_code == "RJTT"
    assert route.destination.compact_name == "FUK"


def test_adsbdb_client_treats_unknown_callsign_as_no_route(monkeypatch) -> None:
    def missing(*_args, **_kwargs):
        raise urllib.error.HTTPError("url", 404, "Not Found", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", missing)
    assert ADSBDBRouteClient().lookup("UNKNOWN") is None


def test_adsbdb_client_normalizes_callsign_and_parses_json(monkeypatch) -> None:
    requested: list[str] = []

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def open_request(request, *, timeout):
        requested.append(request.full_url)
        assert timeout == 3.0
        return _Response(json.dumps(_RESPONSE).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", open_request)
    route = ADSBDBRouteClient().lookup(" ana123 ")
    assert requested == ["https://api.adsbdb.com/v0/callsign/ANA123"]
    assert route is not None
    assert route.destination.icao_code == "RJFF"
