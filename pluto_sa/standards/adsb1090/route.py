"""Best-effort live flight-route lookup through the public ADSBDB API."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


ADSBDB_CALLSIGN_URL = "https://api.adsbdb.com/v0/callsign/{callsign}"


@dataclass(frozen=True)
class RouteAirport:
    name: str = ""
    municipality: str = ""
    country: str = ""
    iata_code: str = ""
    icao_code: str = ""
    latitude: float | None = None
    longitude: float | None = None

    @property
    def compact_name(self) -> str:
        return self.iata_code or self.icao_code or self.name or "-"


@dataclass(frozen=True)
class FlightRoute:
    callsign: str
    callsign_icao: str = ""
    callsign_iata: str = ""
    airline_name: str = ""
    origin: RouteAirport = RouteAirport()
    destination: RouteAirport = RouteAirport()


def normalize_callsign(value: str) -> str:
    return "".join(str(value).strip().upper().split())


def _optional_float(value: object) -> float | None:
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None


def _airport(payload: object) -> RouteAirport:
    data = payload if isinstance(payload, dict) else {}
    return RouteAirport(
        name=str(data.get("name") or "").strip(),
        municipality=str(data.get("municipality") or "").strip(),
        country=str(data.get("country_name") or "").strip(),
        iata_code=str(data.get("iata_code") or "").strip().upper(),
        icao_code=str(data.get("icao_code") or "").strip().upper(),
        latitude=_optional_float(data.get("latitude")),
        longitude=_optional_float(data.get("longitude")),
    )


def parse_adsbdb_route(payload: object) -> FlightRoute | None:
    root = payload if isinstance(payload, dict) else {}
    response = root.get("response")
    if not isinstance(response, dict):
        return None
    route = response.get("flightroute")
    if not isinstance(route, dict):
        return None
    callsign = normalize_callsign(str(route.get("callsign") or ""))
    if not callsign:
        return None
    airline = route.get("airline")
    airline_data = airline if isinstance(airline, dict) else {}
    return FlightRoute(
        callsign=callsign,
        callsign_icao=normalize_callsign(str(route.get("callsign_icao") or "")),
        callsign_iata=normalize_callsign(str(route.get("callsign_iata") or "")),
        airline_name=str(airline_data.get("name") or "").strip(),
        origin=_airport(route.get("origin")),
        destination=_airport(route.get("destination")),
    )


class ADSBDBRouteClient:
    """Perform one bounded ADSBDB lookup; scheduling and caching live in the UI."""

    def lookup(self, callsign: str, *, timeout_s: float = 3.0) -> FlightRoute | None:
        normalized = normalize_callsign(callsign)
        if not normalized:
            return None
        url = ADSBDB_CALLSIGN_URL.format(
            callsign=urllib.parse.quote(normalized, safe="")
        )
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Pluto-VSA-ADSB1090/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise
        return parse_adsbdb_route(payload)
