"""Lazy Qt WebEngine host for an OpenStreetMap/Leaflet aircraft track."""

from __future__ import annotations

import json
from typing import Any

from pyqtgraph.Qt import QtCore, QtWidgets


_LEAFLET_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
          integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
  <style>
    html, body, #map { width: 100%; height: 100%; margin: 0; background: #101214; }
    .leaflet-container { background: #101214; font-family: sans-serif; }
    .leaflet-control-attribution { font-size: 10px; }
    .aircraft-div-icon { background: transparent; border: 0; }
    .aircraft-symbol {
      transform-origin: 18px 21px;
      filter: drop-shadow(0 1px 2px rgba(0, 0, 0, 0.9));
    }
  </style>
</head>
<body>
<div id="map"></div>
<script>
  const map = L.map('map', {worldCopyJump: true, preferCanvas: true})
    .setView([20, 0], 2);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }).addTo(map);
  // A dark casing keeps the magenta track legible over streets and railways.
  const trackOutline = L.polyline([], {
    color: '#20102b', weight: 8, opacity: 0.82, lineJoin: 'round'
  }).addTo(map);
  const track = L.polyline([], {
    color: '#e000ff', weight: 4, opacity: 0.96, lineJoin: 'round'
  }).addTo(map);

  function aircraftIcon(trackDegree) {
    const rotation = Number.isFinite(trackDegree) ? trackDegree : 0;
    return L.divIcon({
      className: 'aircraft-div-icon',
      iconSize: [36, 42],
      iconAnchor: [18, 21],
      popupAnchor: [0, -22],
      html: `<svg class="aircraft-symbol" width="36" height="42" viewBox="0 0 36 42"
                  style="transform: rotate(${rotation}deg)" aria-label="Aircraft">
        <path d="M18 2 L22 14 L34 21 L34 25 L22 22 L21 34 L26 38 L26 40
                 L18 37 L10 40 L10 38 L15 34 L14 22 L2 25 L2 21 L14 14 Z"
              fill="#00d9ff" stroke="#ffffff" stroke-width="5" stroke-linejoin="round"/>
        <path d="M18 2 L22 14 L34 21 L34 25 L22 22 L21 34 L26 38 L26 40
                 L18 37 L10 40 L10 38 L15 34 L14 22 L2 25 L2 21 L14 14 Z"
              fill="#00d9ff" stroke="#10232b" stroke-width="2" stroke-linejoin="round"/>
      </svg>`
    });
  }

  function bearingBetween(first, second) {
    const lat1 = Number(first.latitude) * Math.PI / 180;
    const lat2 = Number(second.latitude) * Math.PI / 180;
    const deltaLongitude = (Number(second.longitude) - Number(first.longitude))
      * Math.PI / 180;
    const y = Math.sin(deltaLongitude) * Math.cos(lat2);
    const x = Math.cos(lat1) * Math.sin(lat2)
      - Math.sin(lat1) * Math.cos(lat2) * Math.cos(deltaLongitude);
    return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
  }

  const current = L.marker([0, 0], {icon: aircraftIcon(0), zIndexOffset: 1000});
  let currentIcao = null;

  // A QWebEngineView can be resized when its dock or tab becomes visible.
  // Leaflet otherwise keeps the tile viewport size from initial construction.
  new ResizeObserver(() => map.invalidateSize({pan: false})).observe(
    document.getElementById('map')
  );

  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>'"]/g, character => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
    })[character]);
  }

  window.updateAircraftTrack = function(payload) {
    map.invalidateSize({pan: false});
    const points = payload.points || [];
    const latLngs = points.map(point => [point.latitude, point.longitude]);
    trackOutline.setLatLngs(latLngs);
    track.setLatLngs(latLngs);
    if (!points.length) {
      if (map.hasLayer(current)) map.removeLayer(current);
      currentIcao = payload.icao || null;
      map.setView([20, 0], 2);
      return;
    }
    const latest = points[points.length - 1];
    let trackDegree = payload.track_deg === null || payload.track_deg === undefined
      ? NaN : Number(payload.track_deg);
    if (!Number.isFinite(trackDegree) && points.length > 1) {
      trackDegree = bearingBetween(points[points.length - 2], latest);
    }
    current.setLatLng([latest.latitude, latest.longitude]);
    current.setIcon(aircraftIcon(trackDegree));
    if (!map.hasLayer(current)) current.addTo(map);
    let popup = '<b>' + escapeHtml(payload.icao || '-') + '</b>';
    if (payload.callsign) popup += ' / ' + escapeHtml(payload.callsign);
    popup += '<br>Lat: ' + Number(latest.latitude).toFixed(6);
    popup += '<br>Lon: ' + Number(latest.longitude).toFixed(6);
    if (latest.altitude_ft !== null && latest.altitude_ft !== undefined) {
      popup += '<br>Altitude: ' + Number(latest.altitude_ft).toFixed(0) + ' ft / '
        + (Number(latest.altitude_ft) * 0.3048).toFixed(0) + ' m';
    }
    if (Number.isFinite(trackDegree)) {
      popup += '<br>Track: ' + trackDegree.toFixed(1) + '&deg;';
    }
    current.bindPopup(popup);
    if (currentIcao !== payload.icao) {
      if (latLngs.length > 1) {
        map.fitBounds(track.getBounds(), {padding: [24, 24], maxZoom: 13});
      } else {
        map.setView(latLngs[0], 10);
      }
    } else {
      map.panInside([latest.latitude, latest.longitude], {padding: [30, 30]});
    }
    currentIcao = payload.icao;
  };
</script>
</body>
</html>
"""


class LeafletAircraftMap(QtWidgets.QWidget):
    """Create WebEngine only when the map tab is first opened."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._view: QtWidgets.QWidget | None = None
        self._loaded = False
        self._payload: dict[str, Any] = {"icao": None, "points": []}
        self._layout = QtWidgets.QStackedLayout(self)
        self._placeholder = QtWidgets.QLabel(
            "Open Position History to load Leaflet / OpenStreetMap tiles."
        )
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(self._placeholder)

    @property
    def last_track_payload(self) -> dict[str, Any]:
        return self._payload

    @property
    def web_view_created(self) -> bool:
        return self._view is not None

    def activate(self) -> None:
        if self._view is not None:
            return
        from PySide6.QtWebEngineCore import QWebEngineSettings
        from PySide6.QtWebEngineWidgets import QWebEngineView

        view = QWebEngineView(self)
        view.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls,
            True,
        )
        # Leaflet only needs normal DOM/CSS/tile rendering. Avoid Chromium GPU
        # surfaces here: on some Windows/Qt combinations they can lose their
        # context during window teardown and raise a native 0x80000003 exception.
        view.settings().setAttribute(
            QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled,
            False,
        )
        view.settings().setAttribute(
            QWebEngineSettings.WebAttribute.WebGLEnabled,
            False,
        )
        view.loadFinished.connect(self._load_finished)
        self._layout.addWidget(view)
        self._layout.setCurrentWidget(view)
        self._view = view
        view.setHtml(_LEAFLET_HTML, QtCore.QUrl("https://unpkg.com/"))

    def set_aircraft_track(
        self,
        *,
        icao: str | None,
        callsign: str | None,
        track_deg: float | None,
        points: list[dict[str, float | int | None]],
    ) -> None:
        self._payload = {
            "icao": icao,
            "callsign": callsign,
            "track_deg": track_deg,
            "points": points,
        }
        self._send_payload()

    @QtCore.Slot(bool)
    def _load_finished(self, successful: bool) -> None:
        self._loaded = bool(successful)
        if successful:
            self._send_payload()
        else:
            self._placeholder.setText(
                "Leaflet could not be loaded. Check the network connection."
            )

    def _send_payload(self) -> None:
        if not self._loaded or self._view is None:
            return
        payload_json = json.dumps(
            self._payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).replace("</", "<\\/")
        self._view.page().runJavaScript(
            f"window.updateAircraftTrack({payload_json});"
        )

    def shutdown(self) -> None:
        """Stop Chromium callbacks before Qt destroys the surrounding docks."""

        view = self._view
        if view is None:
            return
        self._loaded = False
        try:
            view.loadFinished.disconnect(self._load_finished)
        except (RuntimeError, TypeError):
            pass
        try:
            view.stop()
            view.setVisible(False)
        except RuntimeError:
            pass
        self._layout.setCurrentWidget(self._placeholder)
        self._layout.removeWidget(view)
        self._view = None
        view.deleteLater()
