from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import dask.array
from qtpy import QtWidgets as QtW
from qtpy import uic
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QIcon
from typing_extensions import Literal
from useq import MDASequence

if TYPE_CHECKING:
    from pymmcore_plus import RemoteMMCore

# daq
import mcsim.expt_ctrl.daq
from mcsim.expt_ctrl.program_sim_odt import get_sim_odt_sequence
# dmd
from mcsim.expt_ctrl import dlp6500
from localize_psf import affine
import numpy as np
import time
import datetime
import zarr
import dask.array as da
import threading

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

    cx_spinBox: QtW.QSpinBox
    sx_spinBox: QtW.QSpinBox
    cy_spinBox: QtW.QSpinBox
    sy_spinBox: QtW.QSpinBox

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

    show_dataset_checkBox: QtW.QCheckBox
    run_Button: QtW.QPushButton
    pause_Button: QtW.QPushButton
    cancel_Button: QtW.QPushButton

    sim_exposure_SpinBox: QtW.QDoubleSpinBox
    odt_exposure_SpinBox: QtW.QDoubleSpinBox
    odt_frametime_SpinBox: QtW.QDoubleSpinBox
    odt_circbuff_SpinBox: QtW.QDoubleSpinBox
    sim_circbuf_doubleSpinBox : QtW.QDoubleSpinBox
    daq_dt_doubleSpinBox: QtW.QDoubleSpinBox
    shutter_delay_doubleSpinBox: QtW.QDoubleSpinBox
    odt_warmup_doubleSpinBox: QtW.QDoubleSpinBox
    sim_warmup_doubleSpinBox: QtW.QDoubleSpinBox

    #
    stage_groupBox: QtW.QGroupBox
    stage_tableWidget: QtW.QTableWidget
    add_pos_Button: QtW.QPushButton
    remove_pos_Button: QtW.QPushButton
    clear_pos_Button: QtW.QPushButton

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

    def __init__(self, mmcores: list[RemoteMMCore], daq: mcsim.expt_ctrl.daq.daq, dmd: dlp6500,
                 viewer, parent=None, configuration=None):

        mmcore = mmcores[0]
        self._mmcores = mmcores
        self._mmc = self._mmcores[0]

        # todo: would it be better to pass through the main frame instead of these various attributes?
        # todo: or maybe create a python microscope object which contains mmc, daq, DMD?
        self.daq = daq
        self.dmd = dmd

        self.configuration = configuration

        self.viewer = viewer
        super().__init__(parent)
        self.setup_ui()

        self.pause_Button.released.connect(self._mmc.toggle_pause)
        self.cancel_Button.released.connect(self._mmc.cancel)

        self.odt_circbuff_SpinBox.setValue(3.)
        self.sim_circbuf_doubleSpinBox.setValue(3.)
        if self.configuration is not None:
            # initial value for ROI
            self.sx_spinBox.setValue(int(self.configuration["sim_odt_program_defaults"]["sx"]))
            self.cx_spinBox.setValue(int(self.configuration["sim_odt_program_defaults"]["cx"]))
            self.sy_spinBox.setValue(int(self.configuration["sim_odt_program_defaults"]["sy"]))
            self.cy_spinBox.setValue(int(self.configuration["sim_odt_program_defaults"]["cy"]))

            # default value for exposure times
            self.odt_exposure_SpinBox.setValue(float(self.configuration["sim_odt_program_defaults"]["odt_exposure_ms"]))
            self.sim_exposure_SpinBox.setValue(float(self.configuration["sim_odt_program_defaults"]["sim_exposure_ms"]))
            self.odt_frametime_SpinBox.setValue(float(self.configuration["sim_odt_program_defaults"]["odt_frametime_ms"]))
            self.daq_dt_doubleSpinBox.setValue(int(self.configuration["sim_odt_program_defaults"]["daq_dt_us"]))
            self.sim_warmup_doubleSpinBox.setValue(float(self.configuration["sim_odt_program_defaults"]["sim_warmup_time_ms"]))
            self.odt_warmup_doubleSpinBox.setValue(float(self.configuration["sim_odt_program_defaults"]["odt_warmup_time_ms"]))
            self.shutter_delay_doubleSpinBox.setValue(float(self.configuration["sim_odt_program_defaults"]["shutter_delay_ms"]))

        # connect buttons
        self.add_pos_Button.clicked.connect(self.add_position)
        self.remove_pos_Button.clicked.connect(self.remove_position)
        self.clear_pos_Button.clicked.connect(self.clear_positions)
        self.add_ch_Button.clicked.connect(self.add_channel)
        self.remove_ch_Button.clicked.connect(self.remove_channel)
        self.clear_ch_Button.clicked.connect(self.clear_channel)

        self.browse_save_Button.clicked.connect(self.set_multi_d_acq_dir)
        self.run_Button.clicked.connect(self._on_run_clicked)
        # self.run_Button.clicked.connect(self.on_run_clicked)

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
        presets = self.daq.presets

        if len(presets) > 0:
            idx = self.channel_tableWidget.rowCount()
            self.channel_tableWidget.insertRow(idx)

            # create a combo_box for channels in the table
            self.channel_comboBox = QtW.QComboBox(self)
            self.mode_comboBox = QtW.QComboBox(self)
            self.camera_select_comboBox = QtW.QComboBox(self)

            # populate channel options
            pks = list(presets.keys())
            self.channel_comboBox.addItems(pks)

            # populate camera options
            self.camera_select_comboBox.addItems(["default", "both"])

            self.channel_tableWidget.setCellWidget(idx, 0, self.channel_comboBox)
            self.channel_tableWidget.setCellWidget(idx, 1, self.mode_comboBox)
            self.channel_tableWidget.setCellWidget(idx, 2, self.camera_select_comboBox)

            self.channel_comboBox.currentTextChanged.connect(self._on_channel_changed)

    def _on_channel_changed(self):
        dmd_cmap = self.dmd.presets

        for ii in range(self.channel_tableWidget.rowCount()):
            ch = self.channel_tableWidget.cellWidget(ii, 0).currentText()

            # clear old modes
            self.channel_tableWidget.cellWidget(ii, 1).clear()

            # add new modes
            modes = list(dmd_cmap[ch].keys())
            self.channel_tableWidget.cellWidget(ii, 1).addItems(modes)

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

    def _get_zstack_params(self):
        znow = self._mmc.getZPosition()

        if self.stack_groupBox.isChecked():

            step = self.step_size_doubleSpinBox.value()
            if self.z_tabWidget.currentIndex() == 0:
                top = self.z_top_doubleSpinBox.value()
                bottom = self.z_bottom_doubleSpinBox.value()

            elif self.z_tabWidget.currentIndex() == 1:
                range = self.zrange_spinBox.value()
                top = znow + range / 2
                bottom = znow - range / 2

            elif self.z_tabWidget.currentIndex() == 2:
                above = self.above_doubleSpinBox.value()
                below = self.below_doubleSpinBox.value()

                top = znow + above
                bottom = znow - below

        nz = (top - bottom) // step
        zpos = bottom + np.arange(nz) * step

        return zpos

    # add, remove, clear, move_to positions table
    def add_position(self):
        dev_loaded = list(self._mmc.getLoadedDevices())
        if len(dev_loaded) > 1:
            x = self._mmc.getXPosition()
            y = self._mmc.getYPosition()

            x_txt = QtW.QTableWidgetItem(str(x))
            y_txt = QtW.QTableWidgetItem(str(y))
            x_txt.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            y_txt.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

            idx = self.stage_tableWidget.rowCount()
            self.stage_tableWidget.insertRow(idx)

            self.stage_tableWidget.setItem(idx, 0, QtW.QTableWidgetItem(x_txt))
            self.stage_tableWidget.setItem(idx, 1, QtW.QTableWidgetItem(y_txt))


    def remove_position(self):
        # remove selected position
        rows = {r.row() for r in self.stage_tableWidget.selectedIndexes()}
        for idx in sorted(rows, reverse=True):
            self.stage_tableWidget.removeRow(idx)

    def clear_positions(self):
        # clear all positions
        self.stage_tableWidget.clearContents()
        self.stage_tableWidget.setRowCount(0)

    # todo: this naive approach caused problems .. particularly complaints like
    #16:29:39 WARNING QBasicTimer::start: QBasicTimer can only be used with threads started with QThread
    #WARNING: QBasicTimer::start: QBasicTimer can only be used with threads started with QThread
    #16:29:39 WARNING QBasicTimer::start: QBasicTimer can only be used with threads started with QThread
    #WARNING: QObject::setParent: Cannot set parent, new parent is in a different thread
    #16:29:39 WARNING QObject::setParent: Cannot set parent, new parent is in a different thread
    # def on_run_clicked(self):
    #     th = threading.Thread(target=self._on_run_clicked)
    #     th.start()

    def _on_run_clicked(self):

        mmc1 = self._mmcores[0]
        mmc2 = self._mmcores[1]

        if len(self._mmc.getLoadedDevices()) < 2:
            raise ValueError("Load a cfg file first.")


        # ##############################
        # turn off live mode if on
        # ##############################
        mmc1.stopSequenceAcquisition()
        mmc2.stopSequenceAcquisition()

        print("##############################################################")
        print(f"starting acquisition for new dataset")

        # todo: next thing is to allow different "channels" to have different numbers of pictures
        # this would enable also doing things like widefield or etc...along with SIM

        # saving
        if self.save_groupBox.isChecked() and not (
                self.fname_lineEdit.text() and Path(self.dir_lineEdit.text()).is_dir()):
            raise ValueError("Select a filename and a valid directory.")

        if self.save_groupBox.isChecked():
            subdir = self.fname_lineEdit.text()
            save_path = Path(self.dir_lineEdit.text()) / subdir / "sim_odt.zarr"

            # make sure save path is unique
            if save_path.exists():
                ii = 1
                while save_path.exists():
                    save_path = Path(self.dir_lineEdit.text()) / Path(f"{subdir:s}_{ii:d}") / "sim_odt.zarr"
                    ii += 1

        else:
            save_path = None
            subdir = None

        # ##############################
        # grab timing information from GUI
        # ##############################
        exposure_tms_sim = self.sim_exposure_SpinBox.value()
        exposure_tms_odt = self.odt_exposure_SpinBox.value()
        min_odt_frame_time_ms = self.odt_frametime_SpinBox.value()
        odt_circ_buffer_mb = int(np.round(self.odt_circbuff_SpinBox.value() * 1e3))
        sim_circ_buffer_mb = int(np.round(self.sim_circbuf_doubleSpinBox.value() * 1e3))
        dt = int(np.round(self.daq_dt_doubleSpinBox.value())) * 1e-6

        sim_warmup_time_ms = self.sim_warmup_doubleSpinBox.value()
        odt_warmup_time_ms = self.odt_warmup_doubleSpinBox.value()
        shutter_delay_time_ms = self.shutter_delay_doubleSpinBox.value()

        # ##############################
        # time lapse
        # ##############################
        do_time_lapse = self.time_groupBox.isChecked()
        if do_time_lapse:
            ntimes = self.timepoints_spinBox.value()
            interval_ms = self.interval_spinBox.value()
        else:
            ntimes = 1
            interval_ms = 0.

        # ##############################
        # parameter scan
        # ##############################
        nparams = 1

        # ##############################
        # xy-positions
        # ##############################
        do_position_scan = self.stage_groupBox.isChecked() and self.stage_tableWidget.rowCount() > 0
        xy_positions = []
        xy_positions_real = []
        if do_position_scan:
            for r in range(self.stage_tableWidget.rowCount()):
                xy_positions.append([float(self.stage_tableWidget.item(r, 0).text()),
                                  float(self.stage_tableWidget.item(r, 1).text())])
        else:
            xy_positions.append([float(mmc1.getXPosition()),
                              float(mmc1.getYPosition())])
        npositions = len(xy_positions)

        # ##############################
        # zstack
        # ##############################
        # get current z-position info
        z_now = mmc1.getZPosition()
        z_volts_start = self.daq.last_known_analog_val[self.daq.analog_line_names["z_stage"]]

        do_zstack = self.stack_groupBox.isChecked()
        if do_zstack:
            zpositions = self._get_zstack_params()
            nz = len(zpositions)

            # get focus device info
            focus_dev = mmc1.getFocusDevice()
            focus_dev_props = mmc1.getDeviceProperties(focus_dev)
            guess_calibration_um_per_v = (float(focus_dev_props["Upper Limit"]) - float(focus_dev_props["Lower Limit"])) / 10

            # guess voltages to reach desired positions
            dzs = zpositions - z_now
            z_volts_guesses = z_volts_start + dzs / guess_calibration_um_per_v

            print("z-start position was %0.3fV" % z_volts_start)
            print("z guess calibration = %0.3fum/V" % guess_calibration_um_per_v)
            print("z volts guesses= ", end="")
            print(z_volts_guesses)

            if np.any(z_volts_guesses < -5) or np.any(z_volts_guesses > 5):
                print("z_volts_guesses were outside allowed range")
                return

            # go to guess voltage positions
            z_check = np.zeros(nz)
            for ii, v in enumerate(z_volts_guesses):
                self.daq.set_analog_lines_by_name([v], ["z_stage"])
                time.sleep(0.1)
                z_check[ii] = mmc1.getZPosition()

            # get better calibration
            calibration_um_per_v = np.mean((z_check[1:] - z_check[:-1]) / (z_volts_guesses[1:] - z_volts_guesses[:-1]))
            z_volts = z_volts_start + dzs / calibration_um_per_v

            print("z-positions= ", end="")
            print(z_check)
            print("z-calibration was %0.3f um/V" % calibration_um_per_v)
            print("new z-volts= ", end="")
            print(z_volts)

            if np.any(z_volts < -5) or np.any(z_volts > 5):
                print("z_volts were outside allowed range")
                return

            z_real = np.zeros(nz)
            for ii, v in enumerate(z_volts):
                self.daq.set_analog_lines_by_name([v], ["z_stage"])
                time.sleep(0.1)
                z_real[ii] = mmc1.getZPosition()

            dz = np.mean(z_real[1:] - z_real[:-1])

            print("real z values= ", end="")
            print(z_real)
            print("voltages= ", end="")
            print(z_volts)
            print("dz = %0.3fum" % dz)
        else:
            calibration_um_per_v = 0
            zpositions = [0]
            nz = len(zpositions)
            z_volts = np.array([z_volts_start])
            z_real = np.atleast_1d(mmc1.getZPosition())
            dz = 0

        # ##############################
        # grab channels information from GUI
        # ##############################
        nrows = self.channel_tableWidget.rowCount()

        channels = [self.channel_tableWidget.cellWidget(c, 0).currentText() for c in range(nrows)]
        channels_pattern_mode = [self.channel_tableWidget.cellWidget(c, 1).currentText() for c in range(nrows)]
        acquisition_mode = [self.channel_tableWidget.cellWidget(c, 2).currentText() for c in range(nrows)]

        # channels for camera 1
        cam1_channels = [ch for ch in channels if ch != "odt"]
        ncam1_channels = len(cam1_channels)

        # channels for camera 2
        cam2_channels = [ch for ch, cam in zip(channels, acquisition_mode) if ch == "odt" or cam == "both"]
        ncam2_channels = len(cam2_channels)
        n_odt_sim_channels = len([ch for ch in cam2_channels if ch != "odt"])

        if ncam2_channels - n_odt_sim_channels > 1:
            raise NotImplementedError("currently only one ODT channel is allowed")

        print(f'channels = {channels}')
        print(f"channel_pattern_modes = {channels_pattern_mode}")
        print(f"cams = {acquisition_mode}")
        print(f"sim_channels = {cam1_channels}")
        print(f"odt_cam_channels = {cam2_channels}")

        # ##################################
        # get odt camera and set up
        # ##################################
        odt_cam = mmc2.getCameraDevice()

        # set camera properties
        mmc2.setProperty(odt_cam, "Exposure", exposure_tms_odt)
        # set external triggering
        mmc2.setProperty(odt_cam, "TriggerMode", "Edge Trigger")
        mmc2.setProperty(odt_cam, 'PP  1   ENABLED', 'No')
        mmc2.setProperty(odt_cam, 'PP  2   ENABLED', 'No')
        mmc2.setProperty(odt_cam, 'PP  3   ENABLED', 'No')
        mmc2.setProperty(odt_cam, 'PP  4   ENABLED', 'No')
        mmc2.setProperty(odt_cam, 'PP  5   ENABLED', 'No')

        # set ROI
        # todo: add check in bounds...
        cx = self.cx_spinBox.value()
        sx = self.sx_spinBox.value()
        cy = self.cy_spinBox.value()
        sy = self.sy_spinBox.value()

        odt_cam_roi = [cy - sy // 2, cy - sy // 2 + sy,
                       cx - sx // 2, cx - sx // 2 + sx]

        mmc2.setROI(cx - sx // 2, cy - sy // 2, sx, sy)

        nx_odt = mmc2.getImageWidth()
        ny_odt = mmc2.getImageHeight()

        # ##################################
        # get SIM camera and set properties
        # ##################################
        sim_cam = mmc1.getCameraDevice()

        #set camera properties
        mmc1.setProperty(sim_cam, "ScanMode", "2")
        mmc1.setProperty(sim_cam, "Exposure", exposure_tms_sim)
        # set external triggering
        mmc1.setProperty(sim_cam, "TRIGGER SOURCE", "EXTERNAL")
        mmc1.setProperty(sim_cam, "TriggerPolarity", "POSITIVE")

        # set output signal
        # line 1 trigger ready
        ## mmc.setProperty(odt_cam, "OUTPUT TRIGGER KIND[0]", "TRIGGER READY")
        mmc1.setProperty(sim_cam, "OUTPUT TRIGGER KIND[0]", "EXPOSURE")
        mmc1.setProperty(sim_cam, "OUTPUT TRIGGER POLARITY[0]", "POSITIVE")
        # line 2 at end of readout
        mmc1.setProperty(sim_cam, "OUTPUT TRIGGER DELAY[1]", "0.0000")
        mmc1.setProperty(sim_cam, "OUTPUT TRIGGER KIND[1]", "EXPOSURE")
        # mmc1.setProperty(sim_cam, "OUTPUT TRIGGER PERIOD[1]", "0.001")
        mmc1.setProperty(sim_cam, "OUTPUT TRIGGER POLARITY[1]", "POSITIVE")
        # mmc1.setProperty(sim_cam, "OUTPUT TRIGGER SOURCE[1]", "READOUT END")
        # line 3 at start of readout
        mmc1.setProperty(sim_cam, "OUTPUT TRIGGER DELAY[2]", "0.0000")
        mmc1.setProperty(sim_cam, "OUTPUT TRIGGER KIND[2]", "PROGRAMABLE")
        mmc1.setProperty(sim_cam, "OUTPUT TRIGGER PERIOD[2]", "0.001")
        mmc1.setProperty(sim_cam, "OUTPUT TRIGGER POLARITY[2]", "POSITIVE")
        mmc1.setProperty(sim_cam, "OUTPUT TRIGGER SOURCE[2]", "VSYNC")

        nx_sim = mmc1.getImageWidth()
        ny_sim = mmc1.getImageHeight()

        # ##################################
        # prepare daq program, but don't program device yet
        # ##################################
        # number of patterns for single channel
        if "odt" in channels:
            odt_mode = [m for m, ch in zip(channels_pattern_mode, channels) if ch == "odt"][0]
            n_odt_patterns = len(self.dmd.presets["odt"][odt_mode]["picture_indices"])
        else:
            n_odt_patterns = 0

        n_sim_patterns_channel = len(self.dmd.presets["blue"]["sim"]["picture_indices"])
        n_trig_width = np.max([int(np.floor(min_odt_frame_time_ms * 1e-3 / 2 / dt)), 1])

        # odt stabilize time
        if (len(channels) == 1 or ntimes == 1) and channels[0] == "odt" and nz == 1:
            odt_warmup_time_ms = 0
            print("set odt_warmup_time_ms=0")

        npics_odt_cam_per_channels = [n_odt_patterns if ch == "odt" else n_sim_patterns_channel for ch in cam2_channels]

        # line info
        daq_do_map = self.daq.digital_line_names
        daq_ao_map = self.daq.analog_line_names
        daq_presets = self.daq.presets

        digital_program, analog_program, daq_programming_info = get_sim_odt_sequence(daq_do_map, daq_ao_map, daq_presets, channels,
                                                                     exposure_tms_odt * 1e-3,
                                                                     exposure_tms_sim * 1e-3,
                                                                     n_odt_patterns,
                                                                     n_sim_patterns_channel,
                                                                     dt=dt,
                                                                     interval=interval_ms * 1e-3,
                                                                     n_odt_per_sim=1,
                                                                     n_trig_width=n_trig_width,
                                                                     odt_stabilize_t=odt_warmup_time_ms * 1e-3,
                                                                     min_odt_frame_time=min_odt_frame_time_ms * 1e-3,
                                                                     sim_stabilize_t=sim_warmup_time_ms * 1e-3,
                                                                     shutter_delay_time=shutter_delay_time_ms * 1e-3,
                                                                     z_voltages=z_volts,
                                                                     use_dmd_as_odt_shutter=False,
                                                                     n_digital_ch=self.daq.n_digital_lines,
                                                                     n_analog_ch=self.daq.n_analog_lines,
                                                                     cameras=[c if c == "both" else "sim" for c in acquisition_mode])
        # ##################################
        # create zarr
        # ##################################
        if save_path is not None:
            img_data = zarr.open(save_path, mode="w")
            img_data.attrs["save_directory"] = str(save_path)
        else:
            img_data = zarr.open(mode="w")

        # other metadata
        img_data.attrs["timestamp"] = datetime.datetime.now().strftime('%Y_%d_%m_%H;%M;%S')
        img_data.attrs["channels"] = channels
        img_data.attrs["pattern_modes"] = channels_pattern_mode
        img_data.attrs["acquisition_modes"] = acquisition_mode
        img_data.attrs["xy_position_um_set"] = xy_positions
        img_data.attrs["z_position_um"] = list(z_real)
        img_data.attrs["dz_um"] = dz
        img_data.attrs["z_calibration_um_per_v"] = calibration_um_per_v
        img_data.attrs["dt"] = dt

        # micromanager configuration
        img_data.attrs["micromanager_core1_state"] = mmc1.getSystemState().dict()
        img_data.attrs["micromanager_core2_state"] = mmc2.getSystemState().dict()

        if self.configuration is not None:
            img_data.attrs["configuration"] = self.configuration

            # affine transformation from ODT ROI to SIM full image
            xform = np.array(self.configuration["camera_affine_transforms"]["xform"])
            # todo: check this is correct
            xform_real_roi2full = affine.params2xform([1, 0, odt_cam_roi[2], 1, 0, odt_cam_roi[0]])
            xform_cam2_roi_to_cam1 = np.linalg.inv(xform).dot(xform_real_roi2full)

            img_data.attrs["affine_cam2_roi_to_cam1"] = xform_cam2_roi_to_cam1.tolist()
        else:
            img_data.attrs["configuration"] = None
            img_data.attrs["affine_cam2_roi_to_cam1"] = None

        # ###################################
        # datasets for camera # 1
        # ###################################
        # sim dataset
        img_data.create_dataset("cam1/sim", shape=(npositions, ntimes, nz, nparams, ncam1_channels, n_sim_patterns_channel, ny_sim, nx_sim),
                                chunks=(1, 1, 1, 1, 1, 1, ny_sim, nx_sim), dtype='uint16', compressor="none")
        img_data.cam1.sim.attrs["dimensions"] = ["position", "time", "z", "parameters", "channel", "pattern", "y", "x"]
        img_data.cam1.attrs["channels"] = cam1_channels
        img_data.cam1.attrs["exposure_time_ms"] = exposure_tms_sim
        try:
            img_data.cam1.attrs["dx_um"] = self.configuration["camera_settings_1"]["dxy"]
            img_data.cam1.attrs["dy_um"] = self.configuration["camera_settings_1"]["dxy"]
            img_data.cam1.attrs["na_detection"] = self.configuration["camera_settings_1"]["na_detection"]
        except (ValueError, TypeError):
            img_data.cam1.attrs["dx_um"] = None
            img_data.cam1.attrs["dy_um"] = None
            img_data.cam1.attrs["na_detection"] = None

        # sim pattern information for specific channels we are using
        sim_pattern_dat = [dlp6500.get_preset_info(self.dmd.presets[ch]["default"], self.dmd.firmware_pattern_info)[0] for ch in cam1_channels]
        img_data.cam1.sim.attrs["nangles"] = np.array([spd["nangles"][0] for spd in sim_pattern_dat]).tolist()
        img_data.cam1.sim.attrs["nphases"] = np.array([spd["nphases"][0] for spd in sim_pattern_dat]).tolist()
        img_data.cam1.sim.attrs["lattice_vects1"] = np.array([spd["a1"] for spd in sim_pattern_dat]).tolist()
        img_data.cam1.sim.attrs["lattice_vects2"] = np.array([spd["a2"] for spd in sim_pattern_dat]).tolist()
        img_data.cam1.sim.attrs["phases"] = np.array([spd["phase"] for spd in sim_pattern_dat]).tolist()
        img_data.cam1.sim.attrs["frqs"] = np.array([spd["frq"] for spd in sim_pattern_dat]).tolist()

        # OTF
        try:
            img_data.cam1.attrs["otf_model_parameters"] = self.configuration["camera_settings_1"]["otf_calibration"]["fit_params"]
        except (KeyError, TypeError) as e:
            print(e)
            img_data.cam1.attrs["affine_transformations"] = [[]] * ncam1_channels

        # affine transformation information for specific channels we are using
        try:
            img_data.cam1.attrs["affine_transformations"] = [self.configuration["camera_settings_1"]["dmd_affine_transforms"][ch] for ch in cam1_channels]
        except (KeyError, TypeError) as e:
            print(e)
            img_data.cam1.attrs["otf_model_parameters"] = None

        # ###################################
        # datasets for camera # 2
        # ###################################
        # odt dataset
        img_data.create_dataset("cam2/odt", shape=(npositions, ntimes, nz, nparams, 1, n_odt_patterns, ny_odt, nx_odt),
                                chunks=(1, 1, 1, 1, 1, 1, ny_odt, nx_odt), dtype='uint16', compressor="none")
        # only add "channel" so compatible shape with SIM for display
        img_data.cam2.odt.attrs["dimensions"] = ["position", "time", "z", "parameters", "channel", "pattern", "y", "x"]
        img_data.cam2.attrs["exposure_time_ms"] = exposure_tms_odt
        img_data.cam2.odt.attrs["frame_time_ms"] = min_odt_frame_time_ms
        img_data.cam2.odt.attrs["volume_time_ms"] = min_odt_frame_time_ms * n_odt_patterns # todo: correct this

        try:
            img_data.cam2.attrs["dx_um"] = self.configuration["camera_settings_2"]["dxy"]
            img_data.cam2.attrs["dy_um"] = self.configuration["camera_settings_2"]["dxy"]
            img_data.cam2.attrs["na_excitation"] = self.configuration["camera_settings_1"]["na_excitation"]
            img_data.cam2.attrs["na_detection"] = self.configuration["camera_settings_1"]["na_detection"]
        except (KeyError, TypeError) as e:
            print(e)
            img_data.cam2.attrs["dx_um"] = None
            img_data.cam2.attrs["dy_um"] = None
            img_data.cam2.attrs["na_excitation"] = None
            img_data.cam2.attrs["na_detection"] = None

        # get odt pattern data
        if "odt" in channels:
            odt_firmware_data = self.dmd.presets["odt"][odt_mode]
            odt_pic_inds = odt_firmware_data["picture_indices"]
            odt_bit_inds = odt_firmware_data["bit_indices"]
            odt_firmware_inds = dlp6500.pic_bit_ind_2firmware_ind(odt_pic_inds, odt_bit_inds)
            dmd_pattern_data = self.dmd.firmware_pattern_info

            xyoffsets = [(dmd_pattern_data[ii]["xoffset"], dmd_pattern_data[ii]["yoffset"]) for ii in odt_firmware_inds]
            xoffsets, yoffsets = zip(*xyoffsets)

            # set odt dataset metadata
            img_data.cam2.odt.attrs["camera_roi"] = odt_cam_roi
            img_data.cam2.odt.attrs["x_offsets"] = xoffsets
            img_data.cam2.odt.attrs["y_offsets"] = yoffsets
            img_data.cam2.odt.attrs["carrier_frq"] = list(dmd_pattern_data[odt_firmware_inds[0]]["frequency"])
            img_data.cam2.odt.attrs["angle"] = dmd_pattern_data[odt_firmware_inds[0]]["angle"]
            img_data.cam2.odt.attrs["radius"] = dmd_pattern_data[odt_firmware_inds[0]]["radius"]

        # SIM data from other objective
        img_data.create_dataset("cam2/sim", shape=(npositions, ntimes, nz, nparams, n_odt_sim_channels, n_sim_patterns_channel, ny_odt, nx_odt),
                                chunks=(1, 1, 1, 1, 1, 1, ny_odt, nx_odt), dtype='uint16', compressor="none")
        img_data.cam2.sim.attrs["dimensions"] = ["position", "time", "z", "parameters", "channel", "pattern", "y", "x"]

        # daq program
        img_data.create_dataset("daq/digital_program", shape=digital_program.shape, dtype='int8', compressor="none")
        img_data.daq.digital_program.attrs["dimensions"] = ["time", "channel"]
        img_data.daq.digital_program[:] = digital_program
        img_data.daq.digital_program.attrs["channel_map"] = daq_do_map

        img_data.create_dataset("daq/analog_program", shape=analog_program.shape, dtype='float32', compressor="none")
        img_data.daq.analog_program.attrs["dimensions"] = ["time", "channel"]
        img_data.daq.analog_program[:] = analog_program
        img_data.daq.analog_program.attrs["channel_map"] = daq_ao_map


        # ##################################
        # loop over positions and collect data
        # ##################################
        # set circular buffer
        mmc1.setCircularBufferMemoryFootprint(sim_circ_buffer_mb)
        mmc2.setCircularBufferMemoryFootprint(odt_circ_buffer_mb)

        # start timer
        tstart_full_sequence = time.perf_counter()

        if save_path is not None:
            print(f"saving to {str(save_path):s}")

        print(daq_programming_info)

        # time estimate
        position_time_s = dt * digital_program.shape[0] * ntimes * nz
        timeout = 10 + position_time_s

        pgm_time_s = position_time_s * npositions
        pgm_time_mins = int(pgm_time_s // 60)

        print(f"program expected time = {pgm_time_mins:02d}m:{(pgm_time_s - 60 * pgm_time_mins):.3f}s, timeout per position = {timeout:.1f}s")

        for pp in range(npositions):
            # move to new position
            if do_position_scan:
                mmc1.setXYPosition(xy_positions[pp][0], xy_positions[pp][1])

                xy_positions_real.append([float(self._mmc.getXPosition()), float(self._mmc.getYPosition())])

            # ##################################
            # set DAQ to initial state
            # ##################################

            # make sure DMD advance/enable trigger lines are low before we program the DMD
            self.daq.set_digital_lines_by_name(np.array([0, 0, 1, 1, 0, 0], dtype=np.uint8),
                                               ["dmd_enable",
                                                "dmd_advance",
                                                "odt_laser",
                                                "odt_shutter",
                                                "odt_cam_sync",
                                                "camera_trigger_monitor"])

            # ##################################
            # program DMD
            # ##################################
            blank = [False if ch == "odt" else True for ch in channels]
            dmd_modes = [chm if ch == "odt" else "default" for ch, chm in zip(channels, channels_pattern_mode)]

            pic_inds, bit_inds = self.dmd.program_dmd_seq(dmd_modes, channels, nrepeats=1, ndarkframes=0, blank=blank,
                                                          mode_pattern_indices=None, triggered=True, verbose=False)
            dmd_data = np.vstack((pic_inds, bit_inds))

            if pp == 0:
                # store DMD information in zarr
                img_data.create_dataset("dmd_data", shape=dmd_data.shape, dtype='int16', compressor='none')
                img_data.dmd_data[:] = dmd_data
                img_data.dmd_data.attrs["dimensions"] = ["pattern", "time"]
                img_data.dmd_data.attrs["dmd_nx"] = self.dmd.width
                img_data.dmd_data.attrs["dmd_ny"] = self.dmd.height
                img_data.dmd_data.attrs["dmd_pitch_um"] = self.dmd.pitch

            # ##################################
            # trigger camera twice, required for Phantom camera
            # ##################################
            # self.daq.set_digital_lines_by_name(np.array([1], dtype=np.uint8), ["odt_cam"])
            # time.sleep(0.1)
            # self.daq.set_digital_lines_by_name(np.array([0], dtype=np.uint8), ["odt_cam"])
            # time.sleep(0.1)
            # self.daq.set_digital_lines_by_name(np.array([1], dtype=np.uint8), ["odt_cam"])
            # time.sleep(0.1)
            # self.daq.set_digital_lines_by_name(np.array([0], dtype=np.uint8), ["odt_cam"])
            # time.sleep(0.1)

            # total number of pictures for camera 1
            nsim_pics = n_sim_patterns_channel * ncam1_channels * ntimes * nz
            n_cam1_pics = nsim_pics

            # total number of pictures for camera 2
            nodt_pics = n_odt_patterns * ntimes * nz * (ncam2_channels - n_odt_sim_channels)
            nodt_sim_pics = n_sim_patterns_channel * n_odt_sim_channels * ntimes * nz
            n_cam2_pics = nodt_pics + nodt_sim_pics

            # lock to use for printing
            lock = threading.Lock()

            def read_cam1():
                ii_sim_cam = 0
                iz_sim = 0
                ic_sim = 0
                ip_sim = 0
                it_sim = 0
                for icount in range(n_cam1_pics):
                    while mmc1.getRemainingImageCount() == 0:
                        tnow = time.perf_counter() - tstart_acq

                        if tnow > timeout:
                            print("timeout reached......................")
                            break

                    # if we timed out, break out of loop
                    npics = mmc1.getRemainingImageCount()
                    if npics == 0:
                        break

                    img_data.cam1.sim[pp, it_sim, iz_sim, 0, ic_sim, ip_sim] = mmc1.popNextImage()

                    # indexing logic. We acquire images (from slow to fast) time, z-position, channel, pattern
                    ii_sim_cam += 1
                    if ip_sim != (n_sim_patterns_channel - 1):
                        # increment pattern everytime
                        ip_sim += 1
                    else:
                        ip_sim = 0

                        elapsed_time = time.perf_counter() - tstart_full_sequence
                        elapsed_time_min = int(elapsed_time // 60)

                        # print in threadsafe way
                        with lock:
                            print(
                                f"Camera #1 position {pp + 1:d}/{npositions:d},"
                                f" channels {ic_sim + 1:d}/{ncam1_channels:d},"
                                f" z-step {iz_sim + 1:d}/{nz:d},"
                                f" time {it_sim + 1:d}/{ntimes:d},"                                
                                f" images left in buffer = {npics - 1:d},"
                                f" elapsed time = {elapsed_time_min:02d}m:{elapsed_time - elapsed_time_min * 60:.1f}s")

                        # increment channel after pattern
                        if ic_sim != (ncam1_channels - 1):
                            ic_sim += 1
                        else:
                            ic_sim = 0

                            # increment z after channels
                            if iz_sim != (nz - 1):
                                iz_sim += 1
                            else:
                                iz_sim = 0
                                if it_sim != (ntimes - 1):
                                    it_sim += 1

                return ii_sim_cam

            def read_cam2():
                # cam #2 counters
                ii_odt_cam = 0
                iz_odt = 0
                ic_odt = 0
                ip_odt = 0
                it_odt = 0

                for icount in range(n_cam2_pics):
                    while mmc2.getRemainingImageCount() == 0:
                        tnow = time.perf_counter() - tstart_acq

                        if tnow > timeout:
                            print("timeout reached......................")
                            break

                    # if we timed out, break out of loop
                    npics = mmc2.getRemainingImageCount()
                    if npics == 0:
                        break

                    if cam2_channels[ic_odt] == "odt":
                        img_data.cam2.odt[pp, it_odt, iz_odt, 0, 0, ip_odt] = mmc2.popNextImage()
                    else:
                        ic_odt_sim_now = ic_odt - len([ch for ch in cam2_channels[:ic_odt] if ch == "odt"])
                        img_data.cam2.sim[pp, it_odt, iz_odt, 0, ic_odt_sim_now, ip_odt] = mmc2.popNextImage()

                    # indexing logic. We acquire images (from slow to fast) time, z-position, pattern
                    ii_odt_cam += 1
                    # if ip_odt != (n_odt_patterns - 1):
                    if ip_odt != (npics_odt_cam_per_channels[ic_odt] - 1):
                        # increment pattern everytime
                        ip_odt += 1
                    else:
                        ip_odt = 0

                        elapsed_time = time.perf_counter() - tstart_full_sequence
                        elapsed_time_min = int(elapsed_time // 60)
                        # thread safe print
                        with lock:
                            print(
                                f"Camera #2 position {pp + 1:d}/{npositions:d},"
                                f" channels {ic_odt + 1:d}/{ncam2_channels:d},"
                                f" z-step {iz_odt + 1:d}/{nz:d},"
                                f" time {it_odt + 1:d}/{ntimes:d},"                                
                                f" images left in buffer = {npics - 1:d},"
                                f" elapsed time = {elapsed_time_min:02d}m:{elapsed_time - elapsed_time_min * 60:.1f}s")

                        # increment channel
                        if ic_odt != (ncam2_channels - 1):
                            ic_odt += 1
                        else:
                            ic_odt = 0

                            # increment z after channels
                            if iz_odt != (nz - 1):
                                iz_odt += 1
                            else:
                                iz_odt = 0
                                if it_odt != (ntimes - 1):
                                    it_odt += 1

                return ii_odt_cam

            # ##################################
            # burst acquisition
            # ##################################

            # enable DMD, otherwise can have timing problems at the start
            self.daq.set_digital_lines_by_name(np.array([1], dtype=np.uint8), ["dmd_enable"])

            # let lasers and etc warmup
            time.sleep(5)

            # program DAQ
            self.daq.set_sequence(digital_program, analog_program, 1/dt,
                                  analog_clock_source="/Dev1/PFI2",
                                  digital_input_source="/Dev1/PFI1",
                                  di_export_line="/Dev1/PFI2",
                                  continuous=True)

            # start camera
            mmc1.startSequenceAcquisition(n_cam1_pics, 0, True)
            mmc2.startSequenceAcquisition(n_cam2_pics, 0, True)

            # start daq
            tstart_acq = time.perf_counter()
            self.daq.start_sequence()
            thread_save_cam1 = threading.Thread(target=read_cam1)
            thread_save_cam2 = threading.Thread(target=read_cam2)

            thread_save_cam1.start()
            thread_save_cam2.start()

            # wait until program is over, then stop daq
            t_elapsed_now = time.perf_counter() - tstart_acq
            time.sleep(position_time_s - (t_elapsed_now) + 0.1)

            mmc1.stopSequenceAcquisition()
            mmc2.stopSequenceAcquisition()

            # reset DAQ
            self.daq.stop_sequence()
            self.daq.set_preset("off")
            self.daq.set_digital_lines_by_name(np.array([0], dtype=np.uint8), ["odt_cam_enable"])

            # wait for pictures to be stored to disk
            thread_save_cam1.join()
            thread_save_cam2.join()

        # after all positions have run, set z-position back to start
        self.daq.set_analog_lines_by_name([z_volts_start], ["z_stage"])

        # store real xy-positions
        img_data.attrs["xy_position_um_real"] = xy_positions_real

        # ##################################
        # reset cameras to internal triggering...
        # ##################################
        mmc1.setProperty(sim_cam, "TRIGGER SOURCE", "INTERNAL")

        mmc2.setProperty(odt_cam, "TriggerMode", "Internal Trigger")
        mmc2.clearROI()

        # ##################################
        # optionally create viewer layer...
        # ##################################
        # todo: if we were saving, then reload data with dask because datasets may be large

        if self.show_dataset_checkBox.isChecked():

            # show odt
            if not np.any(np.array(img_data.cam2.odt.shape) == 0):
                if subdir == "" or subdir is None:
                    layer_name = "odt preview"
                else:
                    layer_name = subdir + " odt"

                # odt_img_to_show = da.from_zarr()

                try:
                    preview_layer = self.viewer.layers[layer_name]
                    preview_layer.data = img_data.cam2.odt
                except KeyError:
                    self.viewer.add_image(img_data.cam2.odt, name=layer_name, channel_axis=4)
                self.viewer.dims.axis_labels = img_data.cam2.odt.attrs["dimensions"]

            if not np.any(np.array(img_data.cam2.sim.shape) == 0):
                if subdir == "" or subdir is None:
                    layer_name = "sim cam2 preview"
                else:
                    layer_name = subdir + " sim cam2"

                try:
                    preview_layer = self.viewer.layers[layer_name]
                    preview_layer.data = img_data.cam2.sim
                except KeyError:
                    self.viewer.add_image(img_data.cam2.sim, name=layer_name, channel_axis=4)
                self.viewer.dims.axis_labels = img_data.cam2.sim.attrs["dimensions"]

            # show SIM
            if not np.any(np.array(img_data.cam1.sim.shape) == 0):
                if subdir == "" or subdir is None:
                    layer_name = "sim preview"
                else:
                    layer_name = subdir + " sim"

                # todo: debug
                # clims_low = [np.percentile(im, 1) for im in img_data.sim[0, 0, 0, 0, :, 0]]
                # clims_high = [np.percentile(im, 99) for im in img_data.sim[0, 0, 0, 0, :, 0]]

                # sim_img_to_show = da.from_array()

                try:
                    preview_layer = self.viewer.layers[layer_name]
                    preview_layer.data = img_data.cam1.sim
                    #preview_layer.contrast_limits = [clims_low, clims_high]
                except KeyError:
                    names = [layer_name + f" ({ch:s})" for ch in cam1_channels]
                    cmap_dict = {"red": "magenta", "blue": "cyan", "green": "yellow"}
                    cmaps = [cmap_dict[ch] for ch in cam1_channels]

                    self.viewer.add_image(img_data.cam1.sim, name=names, channel_axis=4, colormap=cmaps) #, contrast_limits=[clims_low, clims_high])
                self.viewer.dims.axis_labels = img_data.cam1.sim.attrs["dimensions"]
        return


if __name__ == "__main__":
    from qtpy.QtWidgets import QApplication

    app = QApplication([])
    window = SimOdtWidget()
    window.show()
    app.exec_()
