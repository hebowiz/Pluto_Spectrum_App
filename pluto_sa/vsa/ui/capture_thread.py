"""Shared Qt worker for one finite Pluto VSA capture."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore

from pluto_sa.vsa.pluto_source import (
    CaptureCancelledError,
    PlutoCaptureSettings,
    PlutoLiveSource,
)


class PlutoSingleCaptureThread(QtCore.QThread):
    """Run ``PlutoLiveSource.capture_single`` without blocking a workspace."""

    capture_armed = QtCore.Signal(str)
    capture_ready = QtCore.Signal(object)
    capture_failed = QtCore.Signal(str)
    capture_cancelled = QtCore.Signal()

    def __init__(
        self,
        source: PlutoLiveSource,
        settings: PlutoCaptureSettings,
        armed_message: str,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._settings = settings
        self._armed_message = str(armed_message)

    def run(self) -> None:
        try:
            recording = self._source.capture_single(
                self._settings,
                cancelled=self.isInterruptionRequested,
                armed=lambda: self.capture_armed.emit(self._armed_message),
            )
            if self.isInterruptionRequested():
                self.capture_cancelled.emit()
            else:
                self.capture_ready.emit(recording)
        except CaptureCancelledError:
            self.capture_cancelled.emit()
        except Exception as error:
            self.capture_failed.emit(str(error))

    def cancel(self) -> None:
        self.requestInterruption()

    @property
    def sample_rate_hz(self) -> float:
        return float(self._settings.requested_sample_rate_hz)
