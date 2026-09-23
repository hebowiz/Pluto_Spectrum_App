"""Wi-Fi analysis worker, sharing the existing Pluto acquisition contract."""
from pyqtgraph.Qt import QtCore
from .analysis import analyze_wifi_recording


class WiFiAnalysisThread(QtCore.QThread):
    analysis_ready = QtCore.Signal(object)
    analysis_failed = QtCore.Signal(str)

    def __init__(self, recording, parent=None):
        super().__init__(parent)
        self.recording = recording

    def run(self):
        try:
            result = analyze_wifi_recording(self.recording, cancelled=self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.analysis_ready.emit(result)
        except Exception as error:
            self.analysis_failed.emit(str(error))
