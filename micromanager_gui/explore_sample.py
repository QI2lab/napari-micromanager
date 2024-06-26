from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import useq
from napari.experimental import link_layers
from qtpy import QtWidgets as QtW
from qtpy import uic
from useq import MDASequence

from ._saving import save_sequence
from .multid_widget import SequenceMeta

if TYPE_CHECKING:
    import napari.viewer
    from pymmcore_plus import RemoteMMCore


UI_FILE = str(Path(__file__).parent / "_ui" / "explore_sample.ui")


def _channel_key(name: str, index: int) -> str:
    return f"[{name}_idx{index}]"


class ExploreSample(QtW.QWidget):
    # The UI_FILE above contains these objects:
    scan_explorer_groupBox: QtW.QGroupBox
    scan_size_label: QtW.QLabel
    scan_size_spinBox_r: QtW.QSpinBox
    scan_size_spinBox_c: QtW.QSpinBox
    channel_explorer_groupBox: QtW.QGroupBox
    channel_explorer_tableWidget: QtW.QTableWidget
    add_ch_explorer_Button: QtW.QPushButton
    clear_ch_explorer_Button: QtW.QPushButton
    remove_ch_explorer_Button: QtW.QPushButton
    save_explorer_groupBox: QtW.QGroupBox
    dir_explorer_lineEdit: QtW.QLineEdit
    fname_explorer_lineEdit: QtW.QLineEdit
    browse_save_explorer_Button: QtW.QPushButton
    start_scan_Button: QtW.QPushButton
    stop_scan_Button: QtW.QPushButton
    move_to_position_groupBox: QtW.QGroupBox
    move_to_Button: QtW.QPushButton
    x_lineEdit: QtW.QLineEdit
    y_lineEdit: QtW.QLineEdit
    ovelap_spinBox: QtW.QSpinBox

    # metadata associated with a given experiment
    SEQUENCE_META: dict[MDASequence, SequenceMeta] = {}

    def __init__(self, viewer: napari.viewer.Viewer, mmcore: RemoteMMCore, parent=None):

        self._mmc = mmcore
        self.viewer = viewer
        super().__init__(parent)
        uic.loadUi(UI_FILE, self)

        self.pixel_size = 0

        # connect buttons
        self.add_ch_explorer_Button.clicked.connect(self.add_channel)
        self.remove_ch_explorer_Button.clicked.connect(self.remove_channel)
        self.clear_ch_explorer_Button.clicked.connect(self.clear_channel)

        self.start_scan_Button.clicked.connect(self.start_scan)
        self.move_to_Button.clicked.connect(self.move_to)
        self.browse_save_explorer_Button.clicked.connect(self.set_explorer_dir)

        self.stop_scan_Button.clicked.connect(lambda e: self._mmc.cancel())

        self.x_lineEdit.setText(str(None))
        self.y_lineEdit.setText(str(None))

        # connect mmcore signals
        mmcore.events.systemConfigurationLoaded.connect(self.clear_channel)

        mmcore.events.sequenceStarted.connect(self._on_mda_started)
        mmcore.events.frameReady.connect(self._on_explorer_frame)
        mmcore.events.sequenceFinished.connect(self._on_mda_finished)
        mmcore.events.sequenceFinished.connect(self._refresh_positions)

        @self.viewer.mouse_drag_callbacks.append
        def get_event(viewer, event):
            if not self.isVisible():
                return
            if mmcore.getPixelSizeUm() > 0:
                width = mmcore.getROI(mmcore.getCameraDevice())[2]
                height = mmcore.getROI(mmcore.getCameraDevice())[3]

                x = viewer.cursor.position[-1] * mmcore.getPixelSizeUm()
                y = viewer.cursor.position[-2] * mmcore.getPixelSizeUm() * (-1)

                # to match position coordinates with center of the image
                x = f"{x - ((width / 2) * mmcore.getPixelSizeUm()):.1f}"
                y = f"{y - ((height / 2) * mmcore.getPixelSizeUm() * (-1)):.1f}"

            else:
                x, y = "None", "None"

            self.x_lineEdit.setText(x)
            self.y_lineEdit.setText(y)

    def _on_mda_started(self, sequence: useq.MDASequence):
        """Block gui when mda starts."""
        self._set_enabled(False)

    def _on_explorer_frame(self, image: np.ndarray, event: useq.MDAEvent):
        seq = event.sequence
        meta = self.SEQUENCE_META.get(event.sequence) or SequenceMeta()
        if meta.mode != "explorer":
            return

        x = event.x_pos / self.pixel_size
        y = event.y_pos / self.pixel_size * (-1)

        pos_idx = event.index["p"]
        file_name = meta.file_name if meta.should_save else "Exp"
        ch_name = event.channel.config
        ch_id = event.index["c"]
        layer_name = f"Pos{pos_idx:03d}_{file_name}_{_channel_key(ch_name, ch_id)}"

        meta = dict(
            useq_sequence=seq,
            uid=seq.uid,
            scan_coord=(y, x),
            scan_position=f"Pos{pos_idx:03d}",
            ch_name=ch_name,
            ch_id=ch_id,
        )
        self.viewer.add_image(
            image, name=layer_name, blending="additive", translate=(y, x), metadata=meta
        )
        self.viewer.reset_view()

    def _on_mda_finished(self, sequence: useq.MDASequence):
        meta = self.SEQUENCE_META.get(sequence) or SequenceMeta()
        seq_uid = sequence.uid

        if meta.mode == "explorer":
            layergroups = defaultdict(set)
            for lay in self.viewer.layers:
                if lay.metadata.get("uid") == seq_uid:
                    key = _channel_key(lay.metadata["ch_name"], lay.metadata["ch_id"])
                    layergroups[key].add(lay)
            for group in layergroups.values():
                link_layers(group)

        meta = self.SEQUENCE_META.pop(sequence, SequenceMeta())
        save_sequence(sequence, self.viewer.layers, meta)
        self._set_enabled(True)

    def _set_enabled(self, enabled):
        self.scan_size_spinBox_r.setEnabled(enabled)
        self.scan_size_spinBox_c.setEnabled(enabled)
        self.ovelap_spinBox.setEnabled(enabled)
        self.channel_explorer_groupBox.setEnabled(enabled)
        self.move_to_Button.setEnabled(enabled)
        self.start_scan_Button.setEnabled(enabled)
        self.save_explorer_groupBox.setEnabled(enabled)

    def _refresh_positions(self):
        if self._mmc.getXYStageDevice():
            x, y = f"{self._mmc.getXPosition():.1f}", f"{self._mmc.getYPosition():.1f}"
        else:
            x, y = "None", "None"

        self.x_lineEdit.setText(x)
        self.y_lineEdit.setText(y)

    # add, remove, clear channel table
    def add_channel(self):
        dev_loaded = list(self._mmc.getLoadedDevices())
        if len(dev_loaded) > 1:

            idx = self.channel_explorer_tableWidget.rowCount()
            self.channel_explorer_tableWidget.insertRow(idx)

            # create a combo_box for channels in the table
            self.channel_explorer_comboBox = QtW.QComboBox(self)
            self.channel_explorer_exp_spinBox = QtW.QSpinBox(self)
            self.channel_explorer_exp_spinBox.setRange(0, 10000)
            self.channel_explorer_exp_spinBox.setValue(100)

            if "Channel" not in self._mmc.getAvailableConfigGroups():
                raise ValueError("Could not find 'Channel' in the ConfigGroups")
            channel_list = list(self._mmc.getAvailableConfigs("Channel"))
            self.channel_explorer_comboBox.addItems(channel_list)

            self.channel_explorer_tableWidget.setCellWidget(
                idx, 0, self.channel_explorer_comboBox
            )
            self.channel_explorer_tableWidget.setCellWidget(
                idx, 1, self.channel_explorer_exp_spinBox
            )

    def remove_channel(self):
        # remove selected position
        rows = {r.row() for r in self.channel_explorer_tableWidget.selectedIndexes()}
        for idx in sorted(rows, reverse=True):
            self.channel_explorer_tableWidget.removeRow(idx)

    def clear_channel(self):
        # clear all positions
        self.channel_explorer_tableWidget.clearContents()
        self.channel_explorer_tableWidget.setRowCount(0)

    def _get_state_dict(self) -> dict:
        # position settings
        table = self.channel_explorer_tableWidget
        return {
            "axis_order": "tpzc",
            "stage_positions": [dict(zip("xyz", g)) for g in self.set_grid()],
            "z_plan": None,
            "time_plan": None,
            "channels": [
                {
                    "config": table.cellWidget(c, 0).currentText(),
                    "group": self._mmc.getChannelGroup() or "Channel",
                    "exposure": table.cellWidget(c, 1).value(),
                }
                for c in range(table.rowCount())
            ],
        }

    def set_grid(self) -> list[tuple[float, float, float]]:

        self.scan_size_r = self.scan_size_spinBox_r.value()
        self.scan_size_c = self.scan_size_spinBox_c.value()

        # get current position
        x_pos = float(self._mmc.getXPosition())
        y_pos = float(self._mmc.getYPosition())
        z_pos = float(self._mmc.getZPosition())

        # calculate initial scan position
        _, _, width, height = self._mmc.getROI(self._mmc.getCameraDevice())

        overlap_percentage = self.ovelap_spinBox.value()
        overlap_px_w = width - (width * overlap_percentage) / 100
        overlap_px_h = height - (height * overlap_percentage) / 100

        if self.scan_size_r == 1 and self.scan_size_c == 1:
            raise Exception("RxC -> 1x1. Use MDA to acquire a single position image.")

        move_x = (width / 2) * (self.scan_size_c - 1) - overlap_px_w
        move_y = (height / 2) * (self.scan_size_r - 1) - overlap_px_h

        # to match position coordinates with center of the image
        x_pos -= self.pixel_size * (move_x + width)
        y_pos += self.pixel_size * (move_y + height)

        # calculate position increments depending on pixle size
        if overlap_percentage > 0:
            increment_x = overlap_px_w * self.pixel_size
            increment_y = overlap_px_h * self.pixel_size
        else:
            increment_x = width * self.pixel_size
            increment_y = height * self.pixel_size

        list_pos_order = []
        for r in range(self.scan_size_r):
            if r % 2:  # for odd rows
                col = self.scan_size_c - 1
                for c in range(self.scan_size_c):
                    if c == 0:
                        y_pos -= increment_y
                    list_pos_order.append([x_pos, y_pos, z_pos])
                    if col > 0:
                        col -= 1
                        x_pos -= increment_x
            else:  # for even rows
                for c in range(self.scan_size_c):
                    if r > 0 and c == 0:
                        y_pos -= increment_y
                    list_pos_order.append([x_pos, y_pos, z_pos])
                    if c < self.scan_size_c - 1:
                        x_pos += increment_x

        return list_pos_order

    def set_explorer_dir(self):
        # set the directory
        self.dir = QtW.QFileDialog(self)
        self.dir.setFileMode(QtW.QFileDialog.DirectoryOnly)
        self.save_dir = QtW.QFileDialog.getExistingDirectory(self.dir)
        self.dir_explorer_lineEdit.setText(self.save_dir)
        self.parent_path = Path(self.save_dir)

    def start_scan(self):

        self.pixel_size = self._mmc.getPixelSizeUm()

        if len(self._mmc.getLoadedDevices()) < 2:
            raise ValueError("Load a cfg file first.")

        if self.pixel_size <= 0:
            raise ValueError("PIXEL SIZE NOT SET.")

        if self.channel_explorer_tableWidget.rowCount() <= 0:
            raise ValueError("Select at least one channel.")

        if self.save_explorer_groupBox.isChecked() and (
            self.fname_explorer_lineEdit.text() == ""
            or (
                self.dir_explorer_lineEdit.text() == ""
                or not Path.is_dir(Path(self.dir_explorer_lineEdit.text()))
            )
        ):
            raise ValueError("select a filename and a valid directory.")

        explore_sample = MDASequence(**self._get_state_dict())

        self.SEQUENCE_META[explore_sample] = SequenceMeta(
            mode="explorer",
            split_channels=True,
            should_save=self.save_explorer_groupBox.isChecked(),
            file_name=self.fname_explorer_lineEdit.text(),
            save_dir=self.dir_explorer_lineEdit.text(),
        )

        self._mmc.run_mda(explore_sample)  # run the MDA experiment asynchronously
        return

    def move_to(self):

        move_to_x = self.x_lineEdit.text()
        move_to_y = self.y_lineEdit.text()

        if move_to_x == "None" and move_to_y == "None":
            warnings.warn("PIXEL SIZE NOT SET.")
        else:
            move_to_x = float(move_to_x)
            move_to_y = float(move_to_y)
            self._mmc.setXYPosition(float(move_to_x), float(move_to_y))


if __name__ == "__main__":
    from qtpy.QtWidgets import QApplication

    app = QApplication([])
    window = ExploreSample()
    window.show()
    app.exec_()
