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

# daq
import mcsim.expt_ctrl.expt_map
import mcsim.expt_ctrl.daq
from mcsim.expt_ctrl.program_sim_odt import build_odt_sim_sequence
# dmd
import mcsim.expt_ctrl.dlp6500
import mcsim.expt_ctrl.set_dmd_sim
import numpy as np
import time

ICONS = Path(__file__).parent / "icons"
OBJECTIVE_DEVICE = "Objective"
# Once the PR #43 is merged, we pass the objective device to this variable


@dataclass
class SequenceMeta:
    mode: Literal["mda"] | Literal["explorer"] = ""
    split_channels: bool = False
    should_save: bool = False
    file_name: str = ""
    save_dir: str = ""
    save_pos: bool = False


class _MultiDUI:
    UI_FILE = str(Path(__file__).parent / "_ui" / "sim_odt_gui.ui")

    # The UI_FILE above contains these objects:
    save_groupBox: QtW.QGroupBox
    fname_lineEdit: QtW.QLineEdit
    dir_lineEdit: QtW.QLineEdit
    browse_save_Button: QtW.QPushButton

    channel_groupBox: QtW.QGroupBox
    channel_tableWidget: QtW.QTableWidget  # TODO: extract
    add_ch_Button: QtW.QPushButton
    clear_ch_Button: QtW.QPushButton
    remove_ch_Button: QtW.QPushButton

    time_groupBox: QtW.QGroupBox
    timepoints_spinBox: QtW.QSpinBox
    interval_spinBox: QtW.QSpinBox
    time_comboBox: QtW.QComboBox

    stack_groupBox: QtW.QGroupBox
    z_tabWidget: QtW.QTabWidget
    step_size_doubleSpinBox: QtW.QDoubleSpinBox
    n_images_label: QtW.QLabel
    # TopBottom
    set_top_Button: QtW.QPushButton
    set_bottom_Button: QtW.QPushButton
    z_top_doubleSpinBox: QtW.QDoubleSpinBox
    z_bottom_doubleSpinBox: QtW.QDoubleSpinBox
    z_range_topbottom_doubleSpinBox: QtW.QDoubleSpinBox
    # RangeAround
    zrange_spinBox: QtW.QSpinBox
    range_around_label: QtW.QLabel
    # AboveBelow
    above_doubleSpinBox: QtW.QDoubleSpinBox
    below_doubleSpinBox: QtW.QDoubleSpinBox
    z_range_abovebelow_doubleSpinBox: QtW.QDoubleSpinBox

    run_Button: QtW.QPushButton
    pause_Button: QtW.QPushButton
    cancel_Button: QtW.QPushButton

    sim_exposure_SpinBox: QtW.QDoubleSpinBox
    odt_exposure_SpinBox: QtW.QDoubleSpinBox

    def setup_ui(self):
        uic.loadUi(self.UI_FILE, self)  # load QtDesigner .ui file
        self.pause_Button.hide()
        self.cancel_Button.hide()
        # button icon
        self.run_Button.setIcon(QIcon(str(ICONS / "play-button_1.svg")))
        self.run_Button.setIconSize(QSize(20, 0))


class SimOdtWidget(QtW.QWidget, _MultiDUI):

    # metadata associated with a given experiment
    SEQUENCE_META: dict[MDASequence, SequenceMeta] = {}

    def __init__(self, mmcore: RemoteMMCore, daq: mcsim.expt_ctrl.daq.daq, dmd: mcsim.expt_ctrl.dlp6500.dlp6500, parent=None):
        self._mmc = mmcore
        self.daq = daq
        self.dmd = dmd
        super().__init__(parent)
        self.setup_ui()

        self.pause_Button.released.connect(self._mmc.toggle_pause)
        self.cancel_Button.released.connect(self._mmc.cancel)

        # connect buttons
        self.add_ch_Button.clicked.connect(self.add_channel)
        self.remove_ch_Button.clicked.connect(self.remove_channel)
        self.clear_ch_Button.clicked.connect(self.clear_channel)

        self.browse_save_Button.clicked.connect(self.set_multi_d_acq_dir)
        self.run_Button.clicked.connect(self._on_run_clicked)

        # connect for z stack
        self.set_top_Button.clicked.connect(self._set_top)
        self.set_bottom_Button.clicked.connect(self._set_bottom)
        self.z_top_doubleSpinBox.valueChanged.connect(self._update_topbottom_range)
        self.z_bottom_doubleSpinBox.valueChanged.connect(self._update_topbottom_range)

        self.zrange_spinBox.valueChanged.connect(self._update_rangearound_label)

        self.above_doubleSpinBox.valueChanged.connect(self._update_abovebelow_range)
        self.below_doubleSpinBox.valueChanged.connect(self._update_abovebelow_range)

        self.z_range_abovebelow_doubleSpinBox.valueChanged.connect(
            self._update_n_images
        )
        self.zrange_spinBox.valueChanged.connect(self._update_n_images)
        self.z_range_topbottom_doubleSpinBox.valueChanged.connect(self._update_n_images)
        self.step_size_doubleSpinBox.valueChanged.connect(self._update_n_images)
        self.z_tabWidget.currentChanged.connect(self._update_n_images)
        self.stack_groupBox.toggled.connect(self._update_n_images)

        # events
        mmcore.events.sequenceStarted.connect(self._on_mda_started)
        mmcore.events.sequenceFinished.connect(self._on_mda_finished)
        mmcore.events.sequencePauseToggled.connect(self._on_mda_paused)

    def _set_enabled(self, enabled: bool):
        self.save_groupBox.setEnabled(enabled)
        self.channel_groupBox.setEnabled(enabled)
        self.time_groupBox.setEnabled(enabled)
        self.stack_groupBox.setEnabled(enabled)

    def _set_top(self):
        self.z_top_doubleSpinBox.setValue(self._mmc.getZPosition())

    def _set_bottom(self):
        self.z_bottom_doubleSpinBox.setValue(self._mmc.getZPosition())

    def _update_topbottom_range(self):
        self.z_range_topbottom_doubleSpinBox.setValue(
            abs(self.z_top_doubleSpinBox.value() - self.z_bottom_doubleSpinBox.value())
        )

    def _update_rangearound_label(self, value):
        self.range_around_label.setText(f"-{value/2} µm <- z -> +{value/2} µm")

    def _update_abovebelow_range(self):
        self.z_range_abovebelow_doubleSpinBox.setValue(
            self.above_doubleSpinBox.value() + self.below_doubleSpinBox.value()
        )

    def _update_n_images(self):
        step = self.step_size_doubleSpinBox.value()
        # set what is the range to consider depending on the z_stack mode
        if self.z_tabWidget.currentIndex() == 0:
            range = self.z_range_topbottom_doubleSpinBox.value()
        if self.z_tabWidget.currentIndex() == 1:
            range = self.zrange_spinBox.value()
        if self.z_tabWidget.currentIndex() == 2:
            range = self.z_range_abovebelow_doubleSpinBox.value()

        self.n_images_label.setText(f"{round((range / step) + 1)}")

    def _on_mda_started(self, sequence):
        self._set_enabled(False)
        self.pause_Button.show()
        self.cancel_Button.show()
        self.run_Button.hide()

    def _on_mda_finished(self, sequence):
        self._set_enabled(True)
        self.pause_Button.hide()
        self.cancel_Button.hide()
        self.run_Button.show()

    def _on_mda_paused(self, paused):
        self.pause_Button.setText("GO" if paused else "PAUSE")

    # add, remove, clear channel table
    def add_channel(self):
        presets = mcsim.expt_ctrl.expt_map.presets

        if len(presets) > 0:
            idx = self.channel_tableWidget.rowCount()
            self.channel_tableWidget.insertRow(idx)

            # create a combo_box for channels in the table
            self.channel_comboBox = QtW.QComboBox(self)

            pks = list(presets.keys())
            self.channel_comboBox.addItems(pks)

            self.channel_tableWidget.setCellWidget(idx, 0, self.channel_comboBox)

    def remove_channel(self):
        # remove selected position
        rows = {r.row() for r in self.channel_tableWidget.selectedIndexes()}
        for idx in sorted(rows, reverse=True):
            self.channel_tableWidget.removeRow(idx)

    def clear_channel(self):
        # clear all positions
        self.channel_tableWidget.clearContents()
        self.channel_tableWidget.setRowCount(0)


    def set_multi_d_acq_dir(self):
        # set the directory
        self.dir = QtW.QFileDialog(self)
        self.dir.setFileMode(QtW.QFileDialog.DirectoryOnly)
        self.save_dir = QtW.QFileDialog.getExistingDirectory(self.dir)
        self.dir_lineEdit.setText(self.save_dir)
        self.parent_path = Path(self.save_dir)


    def _on_run_clicked(self):

        if len(self._mmc.getLoadedDevices()) < 2:
            raise ValueError("Load a cfg file first.")

        # grab information
        # channels = self.channel_tableWidget

        exposure_t_sim = self.sim_exposure_SpinBox.value()
        exposure_t_odt = self.odt_exposure_SpinBox.value()

        # time lapse
        do_time_lapse = self.time_groupBox.isChecked()
        ntimes = self.timepoints_spinBox.value()
        interval_ms = self.interval_spinBox.value()

        # zstack
        do_zstack = self.stack_groupBox.isChecked()

        # line info
        daq_do_map = mcsim.expt_ctrl.expt_map.daq_do_map
        daq_ao_map = mcsim.expt_ctrl.expt_map.daq_ao_map

        # ##################################
        # program DMD
        # ##################################
        digital_start = np.zeros(self.daq.n_digital_lines, dtype=np.uint8)
        digital_start[daq_do_map["dmd_enable"]] = 0
        digital_start[daq_do_map["dmd_advance"]] = 0
        self.daq.set_digital_once(digital_start)

        modes = []
        channels = []
        # mcsim.expt_ctrl.set_dmd_sim.program_dmd_seq(self.dmd, modes, channels, nrepeats=1, ndarkframes=0,
        #                                             blank=False, mode_pattern_indices=None, triggered=True,
        #                                             verbose=True)

        # ##################################
        # program DMD
        # ##################################
        # todo ... add more arguments to this.
        # digital_program, analog_program = build_odt_sim_sequence(daq_do_map, daq_ao_map, None)

        # ##################################
        # get cameras
        # ##################################
        # todo: maybe make selectable on channel table?
        odt_cam = "HamamatsuHam_DCAM"
        sim_cam = "HamamatsuHam_DCAM-1"

        nsim_pics = 5
        nodt_pics = 5

        # ##################################
        # set odt camera properties
        # ##################################
        # set camera properties
        self._mmc.setProperty(odt_cam, "ScanMode", "2")
        self._mmc.setProperty(odt_cam, "Exposure", exposure_t_odt)
        # set external triggering
        self._mmc.setProperty(odt_cam, "TRIGGER ACTIVE", "EDGE")
        self._mmc.setProperty(odt_cam, "TRIGGER DELAY", "0.000")
        # mmc.setProperty(odt_cam, "TRIGGER GLOBAL EXPOSURE", "DELAYED")
        self._mmc.setProperty(odt_cam, "TRIGGER GLOBAL EXPOSURE", "GLOBAL RESET")
        # self._mmc.setProperty(odt_cam, "TRIGGER SOURCE", "EXTERNAL")
        self._mmc.setProperty(odt_cam, "TriggerPolarity", "POSITIVE")

        # set output signal
        # line 1 trigger ready
        # mmc.setProperty(odt_cam, "OUTPUT TRIGGER KIND[0]", "TRIGGER READY")
        self._mmc.setProperty(odt_cam, "OUTPUT TRIGGER KIND[0]", "EXPOSURE")
        self._mmc.setProperty(odt_cam, "OUTPUT TRIGGER POLARITY[0]", "POSITIVE")
        # line 2 at end of readout
        self._mmc.setProperty(odt_cam, "OUTPUT TRIGGER DELAY[1]", "0.0000")
        self._mmc.setProperty(odt_cam, "OUTPUT TRIGGER KIND[1]", "PROGRAMABLE")
        self._mmc.setProperty(odt_cam, "OUTPUT TRIGGER PERIOD[1]", "0.001")
        self._mmc.setProperty(odt_cam, "OUTPUT TRIGGER POLARITY[1]", "POSITIVE")
        self._mmc.setProperty(odt_cam, "OUTPUT TRIGGER SOURCE[1]", "READOUT END")
        # line 3 at start of readout
        self._mmc.setProperty(odt_cam, "OUTPUT TRIGGER DELAY[2]", "0.0000")
        self._mmc.setProperty(odt_cam, "OUTPUT TRIGGER KIND[2]", "PROGRAMABLE")
        self._mmc.setProperty(odt_cam, "OUTPUT TRIGGER PERIOD[2]", "0.001")
        self._mmc.setProperty(odt_cam, "OUTPUT TRIGGER POLARITY[2]", "POSITIVE")
        self._mmc.setProperty(odt_cam, "OUTPUT TRIGGER SOURCE[2]", "VSYNC")

        # ##################################
        # set SIM camera properties
        # ##################################
        # set camera properties
        self._mmc.setProperty(sim_cam, "ScanMode", "2")
        self._mmc.setProperty(sim_cam, "Exposure", exposure_t_odt)
        # set external triggering
        # self._mmc.setProperty(sim_cam, "TRIGGER ACTIVE", "EDGE")
        # self._mmc.setProperty(sim_cam, "TRIGGER DELAY", "0.000")
        ## mmc.setProperty(odt_cam, "TRIGGER GLOBAL EXPOSURE", "DELAYED")
        ## self._mmc.setProperty(odt_cam, "TRIGGER GLOBAL EXPOSURE", "GLOBAL RESET")
        # self._mmc.setProperty(sim_cam, "TRIGGER SOURCE", "EXTERNAL")
        # self._mmc.setProperty(sim_cam, "TriggerPolarity", "POSITIVE")
        self._mmc.setProperty(sim_cam, "TRIGGER SOURCE", "INTERNAL")

        # set output signal
        # line 1 trigger ready
        ## mmc.setProperty(odt_cam, "OUTPUT TRIGGER KIND[0]", "TRIGGER READY")
        self._mmc.setProperty(sim_cam, "OUTPUT TRIGGER KIND[0]", "EXPOSURE")
        self._mmc.setProperty(odt_cam, "OUTPUT TRIGGER POLARITY[0]", "POSITIVE")
        # line 2 at end of readout
        self._mmc.setProperty(sim_cam, "OUTPUT TRIGGER DELAY[1]", "0.0000")
        self._mmc.setProperty(sim_cam, "OUTPUT TRIGGER KIND[1]", "PROGRAMABLE")
        self._mmc.setProperty(sim_cam, "OUTPUT TRIGGER PERIOD[1]", "0.001")
        self._mmc.setProperty(sim_cam, "OUTPUT TRIGGER POLARITY[1]", "POSITIVE")
        self._mmc.setProperty(sim_cam, "OUTPUT TRIGGER SOURCE[1]", "READOUT END")
        # line 3 at start of readout
        self._mmc.setProperty(sim_cam, "OUTPUT TRIGGER DELAY[2]", "0.0000")
        self._mmc.setProperty(sim_cam, "OUTPUT TRIGGER KIND[2]", "PROGRAMABLE")
        self._mmc.setProperty(sim_cam, "OUTPUT TRIGGER PERIOD[2]", "0.001")
        self._mmc.setProperty(sim_cam, "OUTPUT TRIGGER POLARITY[2]", "POSITIVE")
        self._mmc.setProperty(sim_cam, "OUTPUT TRIGGER SOURCE[2]", "VSYNC")


        # ##################################
        # burst acquisition
        # ##################################
        sx = 801
        cx = 1024
        sy = 511
        # cy = 1024
        cy = 1120
        self._mmc.setROI(odt_cam, cx - sx // 2, cy - sy // 2, sx, sy)

        # startSequenceAcquisition() with first argument a string does not initialize the circular buffer
        self._mmc.clearCircularBuffer()
        self._mmc.initializeCircularBuffer()
        # todo: test if I can run two sequence acquisitions like this
        self._mmc.startSequenceAcquisition(sim_cam, nsim_pics, 0, True)
        self._mmc.startSequenceAcquisition(odt_cam, nodt_pics, 0, True)

        time.sleep(10)
        imgs = []
        while (self._mmc.getRemainingImageCount() != 0):
            print("images remaining")
            imgs.append(self._mmc.popNextImage())

        print("found %d images" % (len(imgs)))

        for ii in imgs:
            print(len(ii))

        return


if __name__ == "__main__":
    from qtpy.QtWidgets import QApplication

    app = QApplication([])
    window = SimOdtWidget()
    window.show()
    app.exec_()
