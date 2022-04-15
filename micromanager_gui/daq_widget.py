from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy import QtWidgets as QtW
from qtpy import uic
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QIcon
from typing_extensions import Literal
from useq import MDASequence

if TYPE_CHECKING:
    from pymmcore_plus import RemoteMMCore

import numpy as np
import mcsim.expt_ctrl.dlp6500
import mcsim.expt_ctrl.daq
ICONS = Path(__file__).parent / "icons"

class _MultiDUI:
    UI_FILE = str(Path(__file__).parent / "_ui" / "daq_gui.ui")

    channel_groupBox: QtW.QGroupBox
    channel_tableWidget: QtW.QTableWidget  # TODO: extract
    add_ch_Button: QtW.QPushButton
    clear_ch_Button: QtW.QPushButton
    remove_ch_Button: QtW.QPushButton
    update_immediately_checkBox: QtW.QCheckBox

    run_Button: QtW.QPushButton

    def setup_ui(self):
        uic.loadUi(self.UI_FILE, self)  # load QtDesigner .ui file
        # button icon
        self.run_Button.setIcon(QIcon(str(ICONS / "play-button_1.svg")))
        self.run_Button.setIconSize(QSize(20, 0))


class DaqWidget(QtW.QWidget, _MultiDUI):

    def __init__(self, mmcores: list[RemoteMMCore], daq: mcsim.expt_ctrl.daq.daq, dmd: mcsim.expt_ctrl.dlp6500.dlp6500,
                 viewer, parent=None):

        mmcore = mmcores[0]
        self._mmcores = mmcores
        self._mmc = self._mmcores[0]

        self.daq = daq
        self.dmd = dmd
        self.viewer = viewer
        super().__init__(parent)
        self.setup_ui()

        # connect buttons
        self.add_ch_Button.clicked.connect(self.add_channel)
        self.remove_ch_Button.clicked.connect(self.remove_channel)
        self.clear_ch_Button.clicked.connect(self.clear_channel)
        self.run_Button.clicked.connect(self._on_setting_change)
        self.update_immediately_checkBox.clicked.connect(self._on_channel_changed)

    # add, remove, clear channel table
    def add_channel(self):
        digital_channels = list(self.daq.digital_line_names.keys())
        analog_channels = list(self.daq.analog_line_names.keys())

        # add channel
        idx = self.channel_tableWidget.rowCount()
        self.channel_tableWidget.insertRow(idx)

        # create a combo_box for channels in the table
        self.channel_comboBox = QtW.QComboBox(self)
        self.value_spinBox = QtW.QDoubleSpinBox(self)

        pks = digital_channels + analog_channels
        self.channel_comboBox.addItems(pks)

        self.channel_tableWidget.setCellWidget(idx, 0, self.channel_comboBox)
        self.channel_tableWidget.setCellWidget(idx, 1, self.value_spinBox)

        self.channel_comboBox.currentTextChanged.connect(self._on_channel_changed)

        # call function to make sure updated
        self._on_channel_changed()

    def _on_channel_changed(self):
        digital_channels = list(self.daq.digital_line_names.keys())
        analog_channels = list(self.daq.analog_line_names.keys())

        for ii in range(self.channel_tableWidget.rowCount()):
            ch = self.channel_tableWidget.cellWidget(ii, 0).currentText()

            if ch in digital_channels:
                self.channel_tableWidget.cellWidget(ii, 1).setDecimals(0)
                self.channel_tableWidget.cellWidget(ii, 1).setSingleStep(1)
                self.channel_tableWidget.cellWidget(ii, 1).setMinimum(0)
                self.channel_tableWidget.cellWidget(ii, 1).setMaximum(1)

                index = self.daq.digital_line_names[ch]
                last_val_known = self.daq.last_known_digital_val[index]
                self.channel_tableWidget.cellWidget(ii, 1).setValue(last_val_known)

            elif ch in analog_channels:
                self.channel_tableWidget.cellWidget(ii, 1).setDecimals(3)
                self.channel_tableWidget.cellWidget(ii, 1).setSingleStep(0.01)
                self.channel_tableWidget.cellWidget(ii, 1).setMinimum(-10.)
                self.channel_tableWidget.cellWidget(ii, 1).setMaximum(10.)

                index = self.daq.analog_line_names[ch]
                last_val_known = self.daq.last_known_analog_val[index]
                self.channel_tableWidget.cellWidget(ii, 1).setValue(last_val_known)
            else:
                raise ValueError(f"channel '{ch:s}' was not present in analog or digital channels")

            # if update immediately, connect
            if self.update_immediately_checkBox.isChecked():
                self.channel_tableWidget.cellWidget(ii, 1).valueChanged.connect(self._on_setting_change)
                self.channel_tableWidget.cellWidget(ii, 1).setKeyboardTracking(False)
            else:
                # disconnect will fail if not connected to anything
                try:
                    self.channel_tableWidget.cellWidget(ii, 1).valueChanged.disconnect()
                except TypeError:
                    pass

    def remove_channel(self):
        # remove selected position
        rows = {r.row() for r in self.channel_tableWidget.selectedIndexes()}
        for idx in sorted(rows, reverse=True):
            self.channel_tableWidget.removeRow(idx)

    def clear_channel(self):
        # clear all positions
        self.channel_tableWidget.clearContents()
        self.channel_tableWidget.setRowCount(0)


    def _on_setting_change(self):
        # grab analog/digital lines from channels
        digital_channels = list(self.daq.digital_line_names.keys())
        analog_channels = list(self.daq.analog_line_names.keys())

        dig_ch_now = {}
        an_ch_now = {}
        for ii in range(self.channel_tableWidget.rowCount()):
            ch_name = self.channel_tableWidget.cellWidget(ii, 0).currentText()
            val = self.channel_tableWidget.cellWidget(ii, 1).value()

            if ch_name in digital_channels:
                dig_ch_now.update({ch_name: val})
            elif ch_name in analog_channels:
                an_ch_now.update({ch_name: val})
            else:
                raise ValueError(f"channel `{ch_name}` was not present in digital or analog channels")


        if dig_ch_now != {}:
            # unpack dictionaries to lists
            dig_ch_to_set, dig_vals = zip(*list(dig_ch_now.items()))
            dig_vals = np.array(dig_vals, dtype=np.uint8)
            self.daq.set_digital_lines_by_name(dig_vals, dig_ch_to_set)

        if an_ch_now != {}:
            an_ch_to_set, an_vals = zip(*list(an_ch_now.items()))
            an_vals = np.array(an_vals, dtype=float)
            self.daq.set_analog_lines_by_name(an_vals, an_ch_to_set)





if __name__ == "__main__":
    from qtpy.QtWidgets import QApplication

    app = QApplication([])
    window = DaqWidget()
    window.show()
    app.exec_()
