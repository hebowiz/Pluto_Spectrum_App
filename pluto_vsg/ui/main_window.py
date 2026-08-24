"""Initial Visual Composer shell for Pluto VSG."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_vsg.model import WaveformProject, create_default_project, validate_project
from pluto_vsg.profiles import bluetooth_br_edr_project
from pluto_vsg.ui.style import panel_title_font


class _Panel(QtWidgets.QGroupBox):
    def __init__(self, title: str, child: QtWidgets.QWidget) -> None:
        super().__init__(title)
        self.setFont(panel_title_font(self.font()))
        child_font = QtGui.QFont(child.font())
        child_font.setBold(False)
        child.setFont(child_font)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 4)
        layout.addWidget(child)


class PlutoVSGWindow(QtWidgets.QMainWindow):
    """Own project state and the first-stage composer layout."""

    def __init__(self, project: WaveformProject | None = None) -> None:
        super().__init__()
        self.project = project or create_default_project()
        self.setWindowTitle("Pluto VSG - IQ Waveform Generator")
        self.resize(1500, 900)
        self._build_actions()
        self._build_menus()
        self._build_workspace()
        self._refresh_project_view()

    def _build_actions(self) -> None:
        self.new_action = QtGui.QAction("New Project", self)
        self.new_action.triggered.connect(self._new_project)
        self.bluetooth_action = QtGui.QAction("Bluetooth BR / EDR", self)
        self.bluetooth_action.triggered.connect(self._new_bluetooth_project)
        self.validate_action = QtGui.QAction("Validate Project", self)
        self.validate_action.triggered.connect(self._show_validation)
        self.exit_action = QtGui.QAction("Exit", self)
        self.exit_action.triggered.connect(self.close)

    def _build_menus(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        file_menu.addAction(self.new_action)
        file_menu.addSeparator()
        file_menu.addAction("Open...")
        file_menu.addAction("Save")
        file_menu.addAction("Save As...")
        file_menu.addSeparator()
        file_menu.addAction("Export IQ...")
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        edit_menu = menu_bar.addMenu("Edit")
        edit_menu.addActions(
            [QtGui.QAction("Undo", self), QtGui.QAction("Redo", self)]
        )

        waveform_menu = menu_bar.addMenu("Waveform")
        profile_menu = waveform_menu.addMenu("Profile / Standard")
        profile_menu.addAction(self.bluetooth_action)
        for label in (
            "Packet Composer",
            "Data Sources and Lists",
            "Modulation Profiles",
            "Filters",
            "Power Envelope and Control Tracks",
            "Impairments / Dirty Transmitter",
            "Recording Layout / Sequence",
        ):
            waveform_menu.addAction(label)

        graphics_menu = menu_bar.addMenu("Graphics")
        graphics_menu.addAction("Add Graphic")
        graphics_menu.addAction("Save Layout")
        graphics_menu.addAction("Restore Layout")

        output_menu = menu_bar.addMenu("Output")
        output_menu.addAction("Device Manager")
        output_menu.addAction("RF Frequency / Level / Calibration")
        output_menu.addSeparator()
        output_menu.addAction("Generate / Transfer")
        output_menu.addAction("Start")
        output_menu.addAction("Stop")

        tools_menu = menu_bar.addMenu("Tools")
        tools_menu.addAction(self.validate_action)
        tools_menu.addAction("Inspect Generated IQ")
        tools_menu.addAction("Device Capabilities")
        tools_menu.addAction("Calibration")
        menu_bar.addMenu("Help")

    def _build_workspace(self) -> None:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        upper = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        self.block_library = QtWidgets.QListWidget()
        self.block_library.addItems(
            [
                "Fixed Data",
                "Pattern",
                "PRBS",
                "Computed Field",
                "Guard / Idle",
                "Power Ramp",
            ]
        )

        self.field_table = QtWidgets.QTableWidget(0, 4)
        self.field_table.setHorizontalHeaderLabels(
            ["Field", "Symbols", "Data Source", "Modulation"]
        )
        self.field_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.field_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )

        self.inspector = QtWidgets.QTableWidget(0, 2)
        self.inspector.setHorizontalHeaderLabels(["Parameter", "Current"])
        self.inspector.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )

        upper.addWidget(_Panel("Block Library", self.block_library))
        upper.addWidget(_Panel("Packet Composer", self.field_table))
        upper.addWidget(_Panel("Inspector", self.inspector))
        upper.setStretchFactor(0, 1)
        upper.setStretchFactor(1, 3)
        upper.setStretchFactor(2, 2)

        previews = QtWidgets.QTabWidget()
        for title in (
            "IQ / Power",
            "Spectrum",
            "Instantaneous Frequency",
            "Constellation",
        ):
            placeholder = QtWidgets.QLabel(
                "Preview will be calculated from generated IQ.",
                alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
            )
            previews.addTab(placeholder, title)

        splitter.addWidget(upper)
        splitter.addWidget(_Panel("Generated IQ Preview", previews))
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)

    def _new_project(self) -> None:
        self.project = create_default_project()
        self._refresh_project_view()

    def _new_bluetooth_project(self) -> None:
        self.project = bluetooth_br_edr_project()
        self._refresh_project_view()

    def _refresh_project_view(self) -> None:
        self.field_table.setRowCount(len(self.project.fields))
        for row, packet_field in enumerate(self.project.fields):
            values = (
                packet_field.name,
                str(packet_field.symbol_count),
                packet_field.data_source.value,
                packet_field.modulation.kind.value,
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.field_table.setItem(row, column, item)

        parameters = (
            ("Project", self.project.name),
            ("Standard", self.project.standard.value),
            ("Sample Rate", f"{self.project.sample_rate_hz / 1e6:.3f} MS/s"),
            ("Samples / Symbol", str(self.project.samples_per_symbol)),
            ("Repeat Count", str(self.project.repeat_count)),
        )
        self.inspector.setRowCount(len(parameters))
        for row, values in enumerate(parameters):
            for column, value in enumerate(values):
                self.inspector.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        issues = validate_project(self.project)
        status = "Ready" if not issues else f"Validation: {len(issues)} issue(s)"
        self.statusBar().showMessage(status)

    def _show_validation(self) -> None:
        issues = validate_project(self.project)
        if not issues:
            text = "Project settings are valid."
        else:
            text = "\n".join(f"{issue.path}: {issue.message}" for issue in issues)
        QtWidgets.QMessageBox.information(self, "Project Validation", text)
