from __future__ import annotations

import itertools
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

import re
import numpy as np
import time
import datetime
import zarr
import numcodecs
import dask.array as da
from dask_image.imread import imread
from dask.diagnostics import ProgressBar
import threading
# custom code
from localize_psf import affine
# daq and dmd
import mcsim.expt_ctrl.daq
from mcsim.expt_ctrl.program_sim_odt import get_sim_odt_sequence
from mcsim.expt_ctrl import dlp6500
import mcsim.expt_ctrl.phantom_cam as phc


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

    show_Button: QtW.QPushButton
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
    sim_readout_doubleSpinBox: QtW.QDoubleSpinBox

    # stage group
    stage_groupBox: QtW.QGroupBox
    stage_tableWidget: QtW.QTableWidget
    add_pos_Button: QtW.QPushButton
    remove_pos_Button: QtW.QPushButton
    clear_pos_Button: QtW.QPushButton

    # parameter group
    parameter_groupBox: QtW.QGroupBox
    scan_together_checkBox: QtW.QCheckBox
    parameter_tableWidget: QtW.QGroupBox
    add_parameter_pushButton: QtW.QPushButton
    remove_parameter_pushButton: QtW.QPushButton
    clear_parameter_pushButton: QtW.QPushButton


    def setup_ui(self):
        uic.loadUi(self.UI_FILE, self)  # load QtDesigner .ui file


class SimOdtWidget(QtW.QWidget, _MultiDUI):

    # metadata associated with a given experiment
    SEQUENCE_META: dict[MDASequence, SequenceMeta] = {}

    def __init__(self, mmcores: list[RemoteMMCore], daq: mcsim.expt_ctrl.daq.daq, dmd: dlp6500,
                 viewer, phcam, parent=None, configuration=None):

        self._mmcores = mmcores
        self._mmc = self._mmcores[0]

        self.daq = daq
        self.dmd = dmd
        self.phcam = phcam
        self.configuration = configuration
        self.img_data = None
        self.pattern_modes = ["default", "average"]
        self.camera_modes = ["default", "cam1", "cam2", "both"]

        self.viewer = viewer
        super().__init__(parent)
        self.setup_ui()

        self.odt_circbuff_SpinBox.setValue(3.)
        self.sim_circbuf_doubleSpinBox.setValue(3.)

        # save dialog
        self.browse_save_Button.clicked.connect(self._set_save_dir)

        # channel widget
        self.add_ch_Button.clicked.connect(self.add_channel)
        self.remove_ch_Button.clicked.connect(self.remove_channel)
        self.clear_ch_Button.clicked.connect(self.clear_channel)

        # run/show
        self.run_Button.clicked.connect(self._on_run_clicked)
        self.show_Button.clicked.connect(self.show_dataset)

        # z-stack widget
        self.set_top_Button.clicked.connect(self._set_top)
        self.set_bottom_Button.clicked.connect(self._set_bottom)
        self.z_top_doubleSpinBox.valueChanged.connect(self._update_topbottom_range)
        self.z_bottom_doubleSpinBox.valueChanged.connect(self._update_topbottom_range)
        self.zrange_spinBox.valueChanged.connect(self._update_rangearound_label)
        self.above_doubleSpinBox.valueChanged.connect(self._update_abovebelow_range)
        self.below_doubleSpinBox.valueChanged.connect(self._update_abovebelow_range)

        self.z_range_abovebelow_doubleSpinBox.valueChanged.connect(self._update_n_images)
        self.zrange_spinBox.valueChanged.connect(self._update_n_images)
        self.z_range_topbottom_doubleSpinBox.valueChanged.connect(self._update_n_images)
        self.step_size_doubleSpinBox.valueChanged.connect(self._update_n_images)
        self.z_tabWidget.currentChanged.connect(self._update_n_images)
        self.stack_groupBox.toggled.connect(self._update_n_images)

        # stage widget
        self.add_pos_Button.clicked.connect(self.add_position)
        self.remove_pos_Button.clicked.connect(self.remove_position)
        self.clear_pos_Button.clicked.connect(self.clear_positions)

        # parameter widget
        parameter_tableWidget: QtW.QGroupBox
        self.add_parameter_pushButton.clicked.connect(self._add_parameter)
        self.remove_parameter_pushButton.clicked.connect(self._remove_parameter)
        self.clear_parameter_pushButton.clicked.connect(self._clear_parameter)

    def set_cfg(self):
        defaults = self.configuration["sim_odt_program_defaults"]

        # default value for exposure times
        self.odt_exposure_SpinBox.setValue(float(defaults["odt_exposure_ms"]))
        self.sim_exposure_SpinBox.setValue(float(defaults["sim_exposure_ms"]))
        self.odt_frametime_SpinBox.setValue(float(defaults["odt_frametime_ms"]))
        self.daq_dt_doubleSpinBox.setValue(int(defaults["daq_dt_us"]))
        self.sim_warmup_doubleSpinBox.setValue(float(defaults["sim_warmup_time_ms"]))
        self.odt_warmup_doubleSpinBox.setValue(float(defaults["odt_warmup_time_ms"]))
        self.shutter_delay_doubleSpinBox.setValue(float(defaults["shutter_delay_ms"]))
        self.sim_readout_doubleSpinBox.setValue(float(defaults["sim_readout_time_ms"]))

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


    # add, remove, clear channel table
    def add_channel(self):
        presets = self.daq.presets

        if len(presets) > 0:
            idx = self.channel_tableWidget.rowCount()
            self.channel_tableWidget.insertRow(idx)

            # create a combo_box for channels in the table
            channel_comboBox = QtW.QComboBox(self)
            patterns_comboBox = QtW.QComboBox(self)
            mode_comboBox = QtW.QComboBox(self)
            camera_comboBox = QtW.QComboBox(self)

            # populate channel options
            pks = list(presets.keys())
            channel_comboBox.addItems(pks)

            # create combo_boxes in table
            self.channel_tableWidget.setCellWidget(idx, 0, channel_comboBox)
            self.channel_tableWidget.setCellWidget(idx, 1, patterns_comboBox)
            self.channel_tableWidget.setCellWidget(idx, 2, mode_comboBox)
            self.channel_tableWidget.setCellWidget(idx, 3, camera_comboBox)

            # connect to on channel changes
            channel_comboBox.currentTextChanged.connect(lambda: self._on_channel_changed(channel_comboBox))

            # call once to ensure all channel/mode options populated
            self._on_channel_changed(channel_comboBox)

    def _on_channel_changed(self, channel_comboBox):
        # loop over rows until find the right one
        for ii in range(self.channel_tableWidget.rowCount()):
            if channel_comboBox == self.channel_tableWidget.cellWidget(ii, 0):
                # get current channel name
                ch = self.channel_tableWidget.cellWidget(ii, 0).currentText()

                # populate pattern options and set to "default"
                self.channel_tableWidget.cellWidget(ii, 1).clear()
                self.channel_tableWidget.cellWidget(ii, 1).addItems(list(self.dmd.presets[ch].keys()))
                self.channel_tableWidget.cellWidget(ii, 1).setCurrentText("default")

                # populate mode options and set to "default"
                self.channel_tableWidget.cellWidget(ii, 2).clear()
                self.channel_tableWidget.cellWidget(ii, 2).addItems(self.pattern_modes)
                self.channel_tableWidget.cellWidget(ii, 2).setCurrentText("default")

                # average patterns
                self.channel_tableWidget.cellWidget(ii, 3).clear()
                self.channel_tableWidget.cellWidget(ii, 3).addItems(self.camera_modes)
                self.channel_tableWidget.cellWidget(ii, 3).setCurrentText("default")

    def remove_channel(self):
        # remove selected position
        rows = {r.row() for r in self.channel_tableWidget.selectedIndexes()}
        for idx in sorted(rows, reverse=True):
            self.channel_tableWidget.removeRow(idx)

    def clear_channel(self):
        # clear all positions
        self.channel_tableWidget.clearContents()
        self.channel_tableWidget.setRowCount(0)

    def _set_save_dir(self):
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

            else:
                raise NotImplementedError()

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

    def _add_parameter(self):
        analog_lines = list(self.daq.analog_line_names.keys())

        idx = self.parameter_tableWidget.rowCount()
        self.parameter_tableWidget.insertRow(idx)

        channel_comboBox = QtW.QComboBox(self)
        channel_comboBox.addItems(analog_lines)
        self.parameter_tableWidget.setCellWidget(idx, 0, channel_comboBox)

        start_SpinBox = QtW.QDoubleSpinBox(self)
        start_SpinBox.setDecimals(3)
        start_SpinBox.setSingleStep(0.01)
        start_SpinBox.setMinimum(-10.)
        start_SpinBox.setMaximum(10.)
        self.parameter_tableWidget.setCellWidget(idx, 1, start_SpinBox)

        stop_SpinBox = QtW.QDoubleSpinBox(self)
        stop_SpinBox.setDecimals(3)
        stop_SpinBox.setSingleStep(0.01)
        stop_SpinBox.setMinimum(-10.)
        stop_SpinBox.setMaximum(10.)
        self.parameter_tableWidget.setCellWidget(idx, 2, stop_SpinBox)

        step_SpinBox = QtW.QDoubleSpinBox(self)
        step_SpinBox.setDecimals(3)
        step_SpinBox.setSingleStep(0.01)
        step_SpinBox.setMinimum(-10.)
        step_SpinBox.setMaximum(10.)
        self.parameter_tableWidget.setCellWidget(idx, 3, step_SpinBox)

    def _remove_parameter(self):
        # remove selected position
        rows = {r.row() for r in self.parameter_tableWidget.selectedIndexes()}
        for idx in sorted(rows, reverse=True):
            self.parameter_tableWidget.removeRow(idx)

    def _clear_parameter(self):
        self.parameter_tableWidget.clearContents()
        self.parameter_tableWidget.setRowCount(0)

    def _on_run_clicked(self):
        """
        Run hardware triggered SIM/DMD sequence
        """

        mmc1 = self._mmcores[0]
        mmc2 = self._mmcores[1]

        if len(self._mmc.getLoadedDevices()) < 2:
            raise ValueError("Load a cfg file first.")

        # ##############################
        # turn off live mode if on
        # ##############################
        if mmc1.isSequenceRunning():
            mmc1.stopSequenceAcquisition()

        if mmc2.isSequenceRunning():
            mmc2.stopSequenceAcquisition()

        saving = self.save_groupBox.isChecked()

        # saving
        if saving and not (self.fname_lineEdit.text() and Path(self.dir_lineEdit.text()).is_dir()):
            raise ValueError("Select a filename and a valid directory.")

        if saving:
            # grab values form GUI
            main_dir = Path(self.dir_lineEdit.text())
            subdir = Path(self.fname_lineEdit.text())

            # ##############################
            # ensure subdirs are of the form and numbes are correctly ordered 000_...
            # ##############################

            # test other subdirs and get their numbers
            path_exp = "(\d{3})_.*"
            other_nums = [int(re.match(path_exp, n.name).group(1)) for n in main_dir.glob("*") if re.match(path_exp, n.name)]

            # number for new subdir should be larger than all old ones
            if other_nums != []:
                new_num = int(np.max(other_nums)) + 1
            else:
                new_num = 1

            # test if subdir matches pattern ... if so trim the number off the beginning
            m = re.match("(\d+_).*", subdir.name)

            if m:
                subdir_final = Path(f"{new_num:03d}_{subdir.name[len(m.group(1)):]}")
            else:
                subdir_final = Path(f"{new_num:03d}_{subdir.name}")

            # reset the GUI with the correct name
            self.fname_lineEdit.setText(subdir_final.name)

            # ensure save path does not already exist
            save_path = main_dir / subdir_final / "sim_odt.zarr"

            if save_path.exists():
                raise ValueError(f"save path {str(save_path):s} already exists")

        else:
            save_path = None

        print("##############################################################")
        start_str = f"starting acquisition"
        if saving:
            start_str += f", saving to {str(save_path):s}"
        print(start_str)

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
        sim_readout_time_ms = self.sim_readout_doubleSpinBox.value()

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
        do_param_scan = self.parameter_groupBox.isChecked()
        if do_param_scan:
            scan_params = []
            scan_param_vals = []
            for rr in range(self.parameter_tableWidget.rowCount()):
                scan_params.append(self.parameter_tableWidget.cellWidget(rr, 0).currentText())

                start = self.parameter_tableWidget.cellWidget(rr, 1).value()
                stop = self.parameter_tableWidget.cellWidget(rr, 2).value()
                step = self.parameter_tableWidget.cellWidget(rr, 3).value()
                scan_param_vals.append(np.arange(start, stop, step))

            n_per_param = np.array([len(v) for v in scan_param_vals])

            # set scanning mode
            if self.scan_together_checkBox.isChecked():
                # dot product like
                if not np.all(n_per_param == n_per_param[0]):
                    raise ValueError(f"when scan together is selected all parameter scans must have the same length, but they had lengths {n_per_param}")

                param_dict = dict(zip(scan_params, scan_param_vals))
                nparams = len(scan_param_vals[0])

            else:
                # outer product like
                scan_param_vals_full = np.array(list(itertools.product(*scan_param_vals))).transpose()
                param_dict = dict(zip(scan_params, scan_param_vals_full))
                nparams = scan_param_vals_full.shape[1]

        else:
            nparams = 1
            param_dict = None

        # ##############################
        # xy-positions
        # ##############################
        do_xy_scan = self.stage_groupBox.isChecked() and self.stage_tableWidget.rowCount() > 0

        xy_positions = []
        xy_positions_real = []
        if do_xy_scan:
            for r in range(self.stage_tableWidget.rowCount()):
                xy_positions.append([float(self.stage_tableWidget.item(r, 0).text()),
                                     float(self.stage_tableWidget.item(r, 1).text())
                                     ])
        else:
            xy_positions.append([float(mmc1.getXPosition()),
                                 float(mmc1.getYPosition())
                                 ])

        nxy_positions = len(xy_positions)

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
            ul = float(mmc1.getProperty(focus_dev, "Upper Limit (um)"))
            ll = float(mmc1.getProperty(focus_dev, "Lower Limit (um)"))
            guess_calibration_um_per_v = (ul - ll) / 10

            # guess voltages to reach desired positions
            dzs = zpositions - z_now
            z_volts_guesses = z_volts_start + dzs / guess_calibration_um_per_v

            print(f"z-start position was {z_volts_start:.3f}V")
            print(f"z guess calibration = {guess_calibration_um_per_v:.3f}um/V")
            print(f"z volts guesses = {z_volts_guesses}")

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

            print(f"z-positions= {z_check}")
            print(f"z-calibration was {calibration_um_per_v:.3f}f um/V")
            print(f"new z-volts = {z_volts}")

            if np.any(z_volts < -5) or np.any(z_volts > 5):
                print("z_volts were outside allowed range")
                return

            z_real = np.zeros(nz)
            for ii, v in enumerate(z_volts):
                self.daq.set_analog_lines_by_name([v], ["z_stage"])
                time.sleep(0.1)
                z_real[ii] = mmc1.getZPosition()

            dz = np.mean(z_real[1:] - z_real[:-1])

            print(f"real z values = {z_real}")
            print(f"voltages = {z_volts}")
            print(f"dz = {dz:.3f}um")
        else:
            calibration_um_per_v = 0
            zpositions = [0]
            nz = len(zpositions)
            z_volts = np.array([z_volts_start])
            z_real = np.atleast_1d(mmc1.getZPosition())
            dz = 0

        # ##############################
        # grab channel information from GUI
        # ##############################
        nrows = self.channel_tableWidget.rowCount()

        if nrows == 0:
            print("no channels/modes selected")
            return

        # (dmd channel, pattern mode, acquisition mode, camera, number of patterns)
        acq_modes = list(zip([self.channel_tableWidget.cellWidget(c, 0).currentText() for c in range(nrows)],
                             [self.channel_tableWidget.cellWidget(c, 1).currentText() for c in range(nrows)],
                             [self.channel_tableWidget.cellWidget(c, 2).currentText() for c in range(nrows)],
                             [self.channel_tableWidget.cellWidget(c, 3).currentText() for c in range(nrows)]
                             ))
        # convert to lists
        # acq_modes = [list(am) for am in acq_modes]

        acq_modes = [{"channel": am[0],
                      "patterns": am[1],
                      "pattern_mode": am[2],
                      "camera": am[3],
                      "npatterns": 0,
                      "nimages": 0}
                     for am in acq_modes]



        # ##################################
        # get odt camera and set up
        # ##################################
        # cam2 = mmc2.getCameraDevice()
        # cam_is_phantom = cam2 == ""
        cam_is_phantom = True

        if not cam_is_phantom:
            cam2_name = mmc2.getCameraDevice()
            # set camera properties
            mmc2.setProperty(cam2_name, "Exposure", exposure_tms_odt)
            # set external triggering
            mmc2.setProperty(cam2_name, "TriggerMode", "Edge Trigger")
            # set circular buffer
            mmc2.setCircularBufferMemoryFootprint(odt_circ_buffer_mb)
        else:
            mmc2 = self.phcam

            # set up cine: only need one
            mmc2.set_cines(1)
            # this also seems to clear CSR, so do CSR
            try:
                mmc2.set_black_reference()
            except Exception as e:
                print(e)

            # get current parameters
            cine_no = 1 # cine indexing starts at 1
            params = mmc2.get_params(cine_no)
            try:
                params_out = mmc2.setAcqParams(cine_no=cine_no,
                                               Exposure=int(np.round(exposure_tms_odt * 1e6)), # in ns
                                               dFrameRate=50., # need fps to be a little bit slower ... can set to 50
                                               SyncImaging=phc.SYNC_EXTERNAL,
                                               PTFrames=params.ImCount # post-trigger frames
                                               )
            except Exception as e:
                print(e)
                return

        # get size and ROI
        nx_cam2 = mmc2.getImageWidth()
        ny_cam2 = mmc2.getImageHeight()

        # get ROI
        nx_start, ny_start, nx_size, ny_size = mmc2.getROI()
        cam2_roi = [ny_start, ny_start + ny_size, nx_start, nx_start + nx_size]

        # ##################################
        # get SIM camera and set properties
        # ##################################
        cam1 = mmc1.getCameraDevice()

        #set camera properties
        mmc1.setProperty(cam1, "ScanMode", "2")
        mmc1.setProperty(cam1, "Exposure", exposure_tms_sim)
        # set external triggering
        mmc1.setProperty(cam1, "TRIGGER SOURCE", "EXTERNAL")
        mmc1.setProperty(cam1, "TriggerPolarity", "POSITIVE")

        # set output signal
        # line 1 trigger ready
        ## mmc.setProperty(odt_cam, "OUTPUT TRIGGER KIND[0]", "TRIGGER READY")
        mmc1.setProperty(cam1, "OUTPUT TRIGGER KIND[0]", "EXPOSURE")
        mmc1.setProperty(cam1, "OUTPUT TRIGGER POLARITY[0]", "POSITIVE")
        # line 2 at end of readout
        mmc1.setProperty(cam1, "OUTPUT TRIGGER DELAY[1]", "0.0000")
        mmc1.setProperty(cam1, "OUTPUT TRIGGER KIND[1]", "EXPOSURE")
        # mmc1.setProperty(sim_cam, "OUTPUT TRIGGER PERIOD[1]", "0.001")
        mmc1.setProperty(cam1, "OUTPUT TRIGGER POLARITY[1]", "POSITIVE")
        # mmc1.setProperty(sim_cam, "OUTPUT TRIGGER SOURCE[1]", "READOUT END")
        # line 3 at start of readout
        mmc1.setProperty(cam1, "OUTPUT TRIGGER DELAY[2]", "0.0000")
        mmc1.setProperty(cam1, "OUTPUT TRIGGER KIND[2]", "PROGRAMABLE")
        mmc1.setProperty(cam1, "OUTPUT TRIGGER PERIOD[2]", "0.001")
        mmc1.setProperty(cam1, "OUTPUT TRIGGER POLARITY[2]", "POSITIVE")
        mmc1.setProperty(cam1, "OUTPUT TRIGGER SOURCE[2]", "VSYNC")

        nx_cam1 = mmc1.getImageWidth()
        ny_cam1 = mmc1.getImageHeight()

        nx1_start, ny1_start, nx1_size, ny1_size = mmc1.getROI()
        cam1_roi = [ny1_start, ny1_start + ny1_size, nx1_start, nx1_start + nx1_size]

        # set circular buffer
        mmc1.setCircularBufferMemoryFootprint(sim_circ_buffer_mb)

        # ##################################
        # prepare daq program, but don't program device yet
        # ##################################
        for am in acq_modes:
            # reset number of pictures if mode is average
            am["npatterns"] = len(self.dmd.presets[am["channel"]][am["patterns"]]["picture_indices"])

            if am["pattern_mode"] == "average":
                am["nimages"] = 1
            else:
                am["nimages"] = am["npatterns"]

            # set default camera modes
            if am["camera"] == "default":
                if am["channel"] == "odt":
                    am["camera"] = "cam2"
                else:
                    am["camera"] = "cam1"

        cam1_acq_modes = [am for am in acq_modes if am["camera"] in ["cam1", "both"]]
        cam2_acq_modes = [am for am in acq_modes if am["camera"] in ["cam2", "both"]]
        print(f'     acquisition modes = {acq_modes}')
        print(f"cam1 acquisition modes = {cam1_acq_modes}")
        print(f"cam2 acquisition modes = {cam2_acq_modes}")

        # set trigger width
        # todo: calculate this based on modes ... need to account for average mode
        # n_trig_width = np.max([int(np.floor(min_odt_frame_time_ms * 1e-3 / 2 / dt)), 1])
        n_trig_width = 1

        # odt stabilize time
        if (len(acq_modes) == 1 or ntimes == 1) and acq_modes[0]["channel"] == "odt" and nz == 1:
            odt_warmup_time_ms = 0
            print("set odt_warmup_time_ms=0")

        # line info
        daq_do_map = self.daq.digital_line_names
        daq_ao_map = self.daq.analog_line_names
        daq_presets = self.daq.presets

        # pgm_channels = [am[0] for am in acq_modes]
        # pgm_acq_modes = [pmode if pmode == "both" or pmode == "average" else "sim" for (chan, patt, pmode, cam, _) in acq_modes]
        # pgm_npatterns = [nps if am != "average" else len(self.dmd.presets[dm][pm]["picture_indices"])
        #                  for dm, pm, am, cam, nps in acq_modes]

        try:
            digital_program, analog_program, daq_programming_info = \
                get_sim_odt_sequence(daq_do_map,
                                     daq_ao_map,
                                     daq_presets,
                                     acq_modes,
                                     exposure_tms_odt * 1e-3,
                                     exposure_tms_sim * 1e-3,
                                     dt=dt,
                                     interval=interval_ms * 1e-3,
                                     n_odt_per_sim=1,
                                     n_trig_width=n_trig_width,
                                     odt_stabilize_t=odt_warmup_time_ms * 1e-3,
                                     min_odt_frame_time=min_odt_frame_time_ms * 1e-3,
                                     sim_stabilize_t=sim_warmup_time_ms * 1e-3,
                                     shutter_delay_time=shutter_delay_time_ms * 1e-3,
                                     sim_readout_time=sim_readout_time_ms * 1e-3,
                                     z_voltages=z_volts,
                                     use_dmd_as_odt_shutter=False,
                                     n_digital_ch=self.daq.n_digital_lines,
                                     n_analog_ch=self.daq.n_analog_lines,
                                     parameter_scan=param_dict)
        except NotImplementedError:
            return

        # ##################################
        # create zarr
        # ##################################
        if save_path is not None:
            img_data = zarr.open(save_path, mode="w")
            img_data.attrs["save_directory"] = str(save_path)

            self.img_data = zarr.open(save_path, mode="r")
        else:
            img_data = zarr.open(mode="w")

            self.img_data = img_data

        # other metadata
        img_data.attrs["timestamp"] = datetime.datetime.now().strftime('%Y_%d_%m_%H;%M;%S')
        img_data.attrs["channels"] = acq_modes
        img_data.attrs["xy_position_um_set"] = xy_positions
        img_data.attrs["z_position_um"] = list(z_real)
        img_data.attrs["dz_um"] = dz
        img_data.attrs["z_calibration_um_per_v"] = calibration_um_per_v
        img_data.attrs["interval_ms"] = interval_ms

        # micromanager configuration
        for aaaa, core in enumerate(self._mmcores):
            img_data.attrs[f"micromanager_core{aaaa + 1 :d}_state"] = core.getSystemState().dict()

        if self.configuration is not None:
            img_data.attrs["configuration"] = self.configuration

            # affine transformation from ODT ROI to SIM full image
            xform = np.array(self.configuration["camera_affine_transforms"]["xform"])
            # todo: check this is correct
            xform_real_roi2full = affine.params2xform([1, 0, cam2_roi[2], 1, 0, cam2_roi[0]])
            xform_cam2_roi_to_cam1 = np.linalg.inv(xform).dot(xform_real_roi2full)

            img_data.attrs["affine_cam2_roi_to_cam1"] = xform_cam2_roi_to_cam1.tolist()
        else:
            img_data.attrs["configuration"] = None
            img_data.attrs["affine_cam2_roi_to_cam1"] = None

        axis_list = ["position", "time", "z", "parameters", "pattern", "y", "x"]

        # ###################################
        # group for camera # 1
        # ###################################
        g1 = img_data.create_group("cam1")
        g1.attrs["channels"] = [c["channel"] for c in cam1_acq_modes]
        g1.attrs["acquisition_modes"] = cam1_acq_modes
        g1.attrs["exposure_time_ms"] = exposure_tms_sim
        g1.attrs["camera_roi"] = cam1_roi
        g1.attrs["na_detection"] = self.configuration["camera_settings_1"]["na_detection"]

        try:
            g1.attrs["dx_um"] = self.configuration["camera_settings_1"]["dxy"]
            g1.attrs["dy_um"] = self.configuration["camera_settings_1"]["dxy"]
        except (ValueError, TypeError) as e:
            print(e)
            g1.attrs["dx_um"] = None
            g1.attrs["dy_um"] = None

        # OTF
        try:
            g1.attrs["otf_model_parameters"] = self.configuration["camera_settings_1"]["otf_calibration"]["fit_params"]
        except (KeyError, TypeError) as e:
            print(e)
            g1.attrs["otf_model_parameters"] = None

        # affine transformation information for specific channels we are using
        try:
            dmd_affine_transforms = self.configuration["camera_settings_1"]["dmd_affine_transforms"]
            g1.attrs["affine_transformations"] = [dmd_affine_transforms[am["channel"]] for am in cam1_acq_modes]
        except (KeyError, TypeError) as e:
            print(e)
            g1.attrs["affine_transformations"] = [[]] * len(cam1_acq_modes)

        # ###################################
        # create datasets for camera #1
        # ###################################
        cam1_dsets = []
        for am in cam1_acq_modes:
            # chan, patt, pmode, np_now = mode

            name = f"{am['channel']:s}"

            if am["patterns"] == "default":
                name = f"sim_{am['channel']:s}"

            if am["pattern_mode"] == "average":
                name = f"widefield_{am['channel']:s}"

            # ensure name does not already exist
            name_final = name + ""
            icount = 1
            while hasattr(g1, name_final):
                name_final = name + f"_{icount:d}"
                icount += 1

            # create dataset and add attributes
            ds = g1.create_dataset(name_final, shape=(nxy_positions, ntimes, nz, nparams, am["nimages"], ny_cam1, nx_cam1),
                                   chunks=(1, 1, 1, 1, 1, ny_cam1, nx_cam1), dtype="uint16", compressor="none")
            ds.attrs["dimensions"] = axis_list
            ds.attrs["channels"] = [am["channel"]]

            # sim pattern information for specific channels we are using
            sim_pattern_dat = dlp6500.get_preset_info(self.dmd.presets[am["channel"]][am["patterns"]], self.dmd.firmware_pattern_info)[0]

            try:
                ds.attrs["nangles"] = sim_pattern_dat["nangles"][0]
                ds.attrs["nphases"] = sim_pattern_dat["nphases"][0]
                ds.attrs["lattice_vects1"] = np.array(sim_pattern_dat["a1"]).tolist()
                ds.attrs["lattice_vects2"] = np.array(sim_pattern_dat["a2"]).tolist()
                ds.attrs["phases"] = np.array(sim_pattern_dat["phase"]).tolist()
                ds.attrs["frqs"] = np.array(sim_pattern_dat["frq"]).tolist()
            except KeyError as e:
                print(e)

            cam1_dsets.append(ds)

        # ###################################
        # group for camera #2
        # ###################################
        if cam_is_phantom:
            cam2_settings_name = "camera_settings_phantom"
        else:
            cam2_settings_name = "camera_settings_2"

        g2 = img_data.create_group("cam2")
        g2.attrs["channels"] = cam2_acq_modes
        g2.attrs["camera_roi"] = cam2_roi
        g2.attrs["exposure_time_ms"] = exposure_tms_odt

        try:
            g2.attrs["dx_um"] = self.configuration[cam2_settings_name]["dxy"]
            g2.attrs["dy_um"] = self.configuration[cam2_settings_name]["dxy"]
            g2.attrs["na_excitation"] = self.configuration[cam2_settings_name]["na_excitation"]
            g2.attrs["na_detection"] = self.configuration[cam2_settings_name]["na_detection"]
        except (KeyError, TypeError) as e:
            print(e)
            g2.attrs["dx_um"] = None
            g2.attrs["dy_um"] = None
            g2.attrs["na_excitation"] = None
            g2.attrs["na_detection"] = None

        if cam_is_phantom:
            params_save = phc.struct2dict(params_out)

            # convert byte fields to string to serialize in zarr
            for k, v in params_save.items():
                if isinstance(v, bytes):
                    params_save[k] = str(v, "ascii")

            g2.attrs["phantom_cam_acquisition_properties"] = params_save
            # todo: would love to add tagSETUP structure here too, but can't find a function in the SDK to populate this

        # ###################################
        # datasets for camera #2
        # ###################################
        cam2_dsets = []
        for am in cam2_acq_modes:
            # chan, patt, pmode, np_now = mode

            if am["channel"] == "odt":
                name = am["channel"]
            else:
                name = f"sim_{am['channel']:s}"

            # ensure name does not already exist
            name_final = name + ""
            icount = 1
            while hasattr(g2, name_final):
                name_final = name + f"_{icount:d}"
                icount += 1

            # create dataset and add attributes
            ds = g2.create_dataset(name_final,
                                   shape=(nxy_positions, ntimes, nz, nparams, am["nimages"], ny_cam2, nx_cam2),
                                   chunks=(1, 1, 1, 1, 1, ny_cam2, nx_cam2),
                                   dtype="uint16",
                                   compressor="none")
            ds.attrs["dimensions"] = axis_list
            ds.attrs["channels"] = [am["channel"]]

            if am["channel"] == "odt":
                ds.attrs["wavelength_um"] = 0.785
                ds.attrs["volume_time_ms"] = min_odt_frame_time_ms * am["npatterns"]  # todo: correct this

                odt_firmware_data = self.dmd.presets[am["channel"]][am["patterns"]]
                odt_pic_inds = odt_firmware_data["picture_indices"]
                odt_bit_inds = odt_firmware_data["bit_indices"]
                odt_firmware_inds = dlp6500.pic_bit_ind_2firmware_ind(odt_pic_inds, odt_bit_inds)
                dmd_pattern_data = self.dmd.firmware_pattern_info

                # set odt dataset metadata
                ds.attrs["offsets"] = [np.array(dmd_pattern_data[ii]["offsets"]).tolist() for ii in odt_firmware_inds]
                ds.attrs["drs"] = np.array(dmd_pattern_data[odt_firmware_inds[0]]["drs"]).tolist()
                ds.attrs["spot_frqs_mirrors"] = np.array(dmd_pattern_data[odt_firmware_inds[0]]["spot_frqs_mirrors"]).tolist()
                ds.attrs["carrier_frq"] = np.array(dmd_pattern_data[odt_firmware_inds[0]]["carrier frequency"]).tolist()
                ds.attrs["radius"] = dmd_pattern_data[odt_firmware_inds[0]]["radius"]

            else:
                # sim pattern information for specific channels we are using
                sim_pattern_dat = dlp6500.get_preset_info(self.dmd.presets[am["channel"]][am["patterns"]], self.dmd.firmware_pattern_info)[0]

                try:
                    ds.attrs["nangles"] = sim_pattern_dat["nangles"][0]
                    ds.attrs["nphases"] = sim_pattern_dat["nphases"][0]
                    ds.attrs["lattice_vects1"] = np.array(sim_pattern_dat["a1"]).tolist()
                    ds.attrs["lattice_vects2"] = np.array(sim_pattern_dat["a2"]).tolist()
                    ds.attrs["phases"] = np.array(sim_pattern_dat["phase"]).tolist()
                    ds.attrs["frqs"] = np.array(sim_pattern_dat["frq"]).tolist()
                except KeyError as e:
                    print(e)

            cam2_dsets.append(ds)

        # ###################################
        # DAQ data
        # ###################################
        # daq program
        img_data.create_dataset("daq/digital_program", shape=digital_program.shape, dtype='int8', compressor="none")
        img_data.daq.digital_program.attrs["dimensions"] = ["time", "channel"]
        img_data.daq.digital_program[:] = digital_program
        img_data.daq.digital_program.attrs["channel_map"] = daq_do_map

        img_data.create_dataset("daq/analog_program", shape=analog_program.shape, dtype='float32', compressor="none")
        img_data.daq.analog_program.attrs["dimensions"] = ["time", "channel"]
        img_data.daq.analog_program[:] = analog_program
        img_data.daq.analog_program.attrs["channel_map"] = daq_ao_map

        img_data.create_dataset("daq/analog_input",
                                shape=(nxy_positions, digital_program.shape[0] * ntimes * nz, self.daq.n_analog_inputs),
                                dtype="float",
                                compressor="none")
        img_data.daq.analog_input.attrs["dimensions"] = ["position", "time/z", "analog channel"]

        img_data.daq.attrs["dt"] = dt


        # ##################################
        # loop over positions and collect data
        # ##################################
        def run():
            # start timer
            tstart_full_sequence = time.perf_counter()

            print(daq_programming_info)

            # time estimate
            position_time_s = dt * digital_program.shape[0] * ntimes * nz * nparams
            timeout = 10 + position_time_s

            pgm_time_s = position_time_s * nxy_positions
            pgm_time_mins = int(pgm_time_s // 60)

            print(f"program expected time = {pgm_time_mins:02d}m:{(pgm_time_s - 60 * pgm_time_mins):.3f}s,"
                  f" timeout per position = {timeout:.1f}s")

            for pp in range(nxy_positions):
                # move to new position
                if do_xy_scan:
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
                # blank = [False if ch == "odt" or am == "average" else True for ch, _, am, _ in acq_modes]
                # noff_after = [1 if am == "average" else 0 for ch, _, am, _ in acq_modes]
                # dmd_modes = [chm if ch == "odt" else "default" for ch, chm, _, _ in acq_modes]
                # dmd_channels = [ch for ch, _, _, _, in acq_modes]

                blank = [False if am["channel"] == "odt" or am["pattern_mode"] == "average" else True for am in acq_modes]
                noff_after = [1 if am["pattern_mode"] == "average" else 0 for am in acq_modes]
                dmd_modes = [am["patterns"] if am["channel"] == "odt" else "default" for am in acq_modes]
                dmd_channels = [am["channel"] for am in acq_modes]

                pic_inds, bit_inds = self.dmd.program_dmd_seq(dmd_modes,
                                                              dmd_channels,
                                                              nrepeats=1,
                                                              noff_before=0,
                                                              noff_after=noff_after,
                                                              blank=blank,
                                                              mode_pattern_indices=None,
                                                              triggered=True,
                                                              verbose=False)
                dmd_data = np.vstack((pic_inds, bit_inds))

                if pp == 0:
                    # store DMD information in zarr
                    g_dmd = img_data.create_group("dmd_data")
                    g_dmd.attrs["dmd_nx"] = self.dmd.width
                    g_dmd.attrs["dmd_ny"] = self.dmd.height
                    g_dmd.attrs["dmd_pitch_um"] = self.dmd.pitch
                    fware_info = self.dmd.get_firmware_type()
                    g_dmd.attrs["dmd_type"] = fware_info["dmd type"]
                    g_dmd.attrs["firmware_tag"] = fware_info["firmware tag"]

                    ds = g_dmd.create_dataset("dmd_program", shape=dmd_data.shape, dtype='int16', compressor='none')
                    ds[:] = dmd_data
                    ds.attrs["dimensions"] = ["pattern", "time"]

                    if self.dmd.firmware_patterns is not None:
                        ds = g_dmd.array("firmware_patterns", self.dmd.firmware_patterns.astype(bool),
                                         compressor=numcodecs.packbits.PackBits(), dtype=bool,
                                         chunks=(1, self.dmd.height, self.dmd.width))
                        ds.attrs["picture_indices"] = self.dmd.picture_indices.tolist()
                        ds.attrs["bit_indices"] = self.dmd.bit_indices.tolist()

                # ##################################
                # trigger camera twice, required for Phantom camera
                # ##################################
                if cam_is_phantom:
                    if cam2_acq_modes != []:
                        mmc2.quiet_fan(True) # quiet fan to avoid vibrations

                    mmc2.record_cine(1)

                    self.daq.set_digital_lines_by_name(np.array([1], dtype=np.uint8), ["odt_cam_sync"])
                    time.sleep(0.1)
                    self.daq.set_digital_lines_by_name(np.array([0], dtype=np.uint8), ["odt_cam_sync"])
                    time.sleep(0.1)
                    self.daq.set_digital_lines_by_name(np.array([1], dtype=np.uint8), ["odt_cam_sync"])
                    time.sleep(0.1)
                    self.daq.set_digital_lines_by_name(np.array([0], dtype=np.uint8), ["odt_cam_sync"])
                    time.sleep(0.1)

                # total number of pictures for cameras per position
                n_cam1_pics = ntimes * nz * nparams * int(np.sum([am["nimages"] for am in cam1_acq_modes]))
                n_cam2_pics = ntimes * nz * nparams * int(np.sum([am["nimages"] for am in cam2_acq_modes]))

                # lock to use for printing
                lock = threading.Lock()

                # todo: test
                def single_index2multi(ii, npatterns_channel):
                    # order from slowest to fastest
                    # ['time', 'parameters', 'z', 'channel', 'pattern']
                    npatterns_all_channels = np.sum(npatterns_channel)
                    npatterns_cumsum = np.cumsum(npatterns_channel)

                    # channel index
                    cinds = np.arange(len(npatterns_channel))
                    ic = cinds[ii % npatterns_all_channels < npatterns_cumsum][0]
                    # pattern index
                    ip = int((ii % npatterns_all_channels - np.sum(npatterns_channel[:ic])) % npatterns_channel[ic])
                    # other indices are easy
                    iz = (ii // (npatterns_all_channels)) % nz
                    iparam = (ii // (npatterns_all_channels * nz)) % nparams
                    it = (ii // (npatterns_all_channels * nparams * nz)) % ntimes

                    return (it, iz, iparam, ic, ip)


                def read_cam(mmc, dsets, cam_acq_modes, ncam_pics, desc=""):
                    # order from slowest to fastest
                    # ['time', 'parameters', 'z', 'channel', 'pattern']

                    ncam_channels = len(cam_acq_modes)
                    # npatterns_channel = [cam_acq_modes[ic][3] for ic in range(ncam_channels)]
                    npatterns_channel = [am["nimages"] for am in cam_acq_modes]

                    ii_acquired = 0
                    iz = 0
                    ic = 0
                    iparam = 0
                    ipatt = 0
                    it = 0
                    for icount in range(ncam_pics):
                        while mmc.getRemainingImageCount() == 0:
                            tnow = time.perf_counter() - tstart_acq

                            if tnow > timeout:
                                print("timeout reached......................")
                                break

                        # if we timed out, break out of loop
                        npics = mmc.getRemainingImageCount()
                        if npics == 0:
                            break

                        dsets[ic][pp, it, iz, iparam, ipatt] = mmc.popNextImage()

                        # indexing logic. We acquire images (from slow to fast) time, parameters, z-position, channel, pattern
                        ii_acquired += 1

                        # if ipatt != (cam_acq_modes[ic][3] - 1):
                        if ipatt != (cam_acq_modes[ic]["nimages"] - 1):
                            # increment pattern everytime
                            ipatt += 1
                        else:
                            ipatt = 0

                            elapsed_time = time.perf_counter() - tstart_full_sequence
                            elapsed_time_min = int(elapsed_time // 60)

                            # print in threadsafe way
                            with lock:
                                print(
                                    f"{desc:s} image {ii_acquired:d}/{ncam_pics:d}, "
                                    f" position {pp + 1:d}/{nxy_positions:d},"                                    
                                    f" time {it + 1:d}/{ntimes:d},"
                                    f" param {iparam + 1:d}/{nparams:d}"
                                    f" z-step {iz + 1:d}/{nz:d},"
                                    f" channels {ic + 1:d}/{ncam_channels:d},"                                                                                                                                            
                                    f" images in buffer = {npics - 1:d},"
                                    f" multi index = {single_index2multi(ii_acquired, npatterns_channel)},"
                                    f" elapsed time = {elapsed_time_min:02d}m:{elapsed_time - elapsed_time_min * 60:.1f}s")

                            # increment channel after pattern
                            if ic != (ncam_channels - 1):
                                ic += 1
                            else:
                                ic = 0

                                # increment z after channels
                                if iz != (nz - 1):
                                    iz += 1
                                else:
                                    iz = 0
                                    # increment parameters
                                    if iparam != (nparams - 1):
                                        iparam += 1
                                    else:
                                        iparam = 0

                                        # increment time
                                        if it != (ntimes - 1):
                                            it += 1

                    return ii_acquired

                # ##################################
                # burst acquisition
                # ##################################

                # enable DMD, otherwise can have timing problems at the start
                self.daq.set_digital_lines_by_name(np.array([1], dtype=np.uint8), ["dmd_enable"])

                if cam_is_phantom:
                    # todo: testing this
                    self.daq.set_digital_lines_by_name(np.array([1], dtype=np.uint8), ["odt_cam_enable"])

                # let lasers and etc. warmup
                time.sleep(5)

                # program DAQ
                self.daq.set_sequence(digital_program,
                                      analog_program,
                                      1/dt,
                                      analog_clock_source="/Dev1/PFI2",
                                      digital_input_source="/Dev1/PFI1",
                                      di_export_line="/Dev1/PFI2",
                                      continuous=True,
                                      nrepeats=ntimes * nz)

                # start camera
                mmc1.startSequenceAcquisition(n_cam1_pics, 0, True)

                if not cam_is_phantom:
                    mmc2.startSequenceAcquisition(n_cam2_pics, 0, True)

                # start daq
                tstart_acq = time.perf_counter()
                self.daq.start_sequence()
                thread_save_cam1 = threading.Thread(target=read_cam,
                                                    args=(mmc1, cam1_dsets, cam1_acq_modes, n_cam1_pics, "cam1"))
                if not cam_is_phantom:
                    thread_save_cam2 = threading.Thread(target=read_cam,
                                                        args=(mmc2, cam2_dsets, cam2_acq_modes, n_cam2_pics, "cam2"))
                else:
                    # otherwise thread prints elapsed time
                    def print_time(tstart, timeout, sleeptime=0.1):
                        t_elapsed_now = time.perf_counter() - tstart

                        timeout_min = int(timeout // 60)

                        while t_elapsed_now < timeout:
                            elapsed_time_min = int(t_elapsed_now // 60)

                            print(f"elapsed time = "
                                  f"{elapsed_time_min:02d}m:{t_elapsed_now - elapsed_time_min * 60.:.1f}s/"
                                  f"{timeout_min:02d}m:{timeout - timeout_min * 60:.1f}s",
                                  end="\r")
                            time.sleep(sleeptime)
                            t_elapsed_now = time.perf_counter() - tstart

                        return

                    thread_save_cam2 = threading.Thread(target=print_time,
                                                        args=(tstart_acq, position_time_s))

                # start threads
                thread_save_cam1.start()
                thread_save_cam2.start()

                # wait until program is over, then stop daq. Meanwhile, print timing information
                t_elapsed_now = time.perf_counter() - tstart_acq
                time.sleep(position_time_s - (t_elapsed_now) + 0.1) # need extra margin or lose frames at fast frame rates

                # reset DAQ
                self.daq.stop_sequence()
                self.daq.set_preset("off")
                self.daq.set_digital_lines_by_name(np.array([0], dtype=np.uint8), ["odt_cam_enable"])
                self.daq.set_digital_lines_by_name(np.array([0], dtype=np.uint8), ["odt_cam_sync"])

                try:
                    img_data.daq.analog_input[pp] = self.daq.read_ai(img_data.daq.analog_input.shape[1])
                except Exception as e:
                    print(e)

                if mmc1.isSequenceRunning():
                    mmc1.stopSequenceAcquisition()

                if mmc2.isSequenceRunning():
                    mmc2.stopSequenceAcquisition()

                # wait for pictures to be stored to disk
                thread_save_cam1.join()
                thread_save_cam2.join()

                if cam_is_phantom:
                    mmc2.stopSequenceAcquisition() # stop recording
                    mmc2.quiet_fan(False) # turn fan back on

                    if cam2_acq_modes != []:
                        # todo: is it possible to grab pictures as they come in on the phantom?
                        #  So far wait until done recording to grab them
                        tstart_ph_save = time.perf_counter()

                        # todo: save using phantom format instead ... need to understand how to convert to zarr
                        print("saving cine to tif")
                        try:
                            mmc2.save_cine(1,
                                           save_path.parent / f"ph_position={pp:d}_odt.tif",
                                           first_image=0,
                                           img_count=n_cam2_pics)
                        except Exception as e:
                            print(e)

                        print(f"saved position {pp + 1:d}/{nxy_positions:d} to disk in {time.perf_counter() - tstart_ph_save:.2f}s")

                        # delete cine
                        try:
                            mmc2.destroy_cine(1)
                        except Exception as e:
                            print(e)

                        # print(f"saved position {pp:d} to disk in {time.perf_counter() - tstart_ph_save:.2f}s")

            # after all positions have run, set z-position back to start
            self.daq.set_analog_lines_by_name([z_volts_start], ["z_stage"])

            # store real xy-positions
            img_data.attrs["xy_position_um_real"] = xy_positions_real

            if cam_is_phantom and cam2_acq_modes != [] and saving:
                # write images to zarr
                npatterns_ch_2 = np.array([am["nimages"] for am in cam2_acq_modes], dtype=int)
                npatterns2_all = np.sum(npatterns_ch_2) # number of combined patterns for all channels

                imgs_tif = []
                for pp in range(nxy_positions):
                    tif_pattern = f"ph_position={pp:d}_odt*.tif"
                    temp = imread(save_path.parent / tif_pattern)
                    if len(temp) > n_cam2_pics:
                        # in case extra pictures bc DAQ didn't stop immediately (it is stopped in software so this can happen)
                        temp2 = temp[:n_cam2_pics]
                    elif len(temp) < n_cam2_pics:
                        # this should not happen, but want to debug it if it does
                        temp2 = da.concatenate((temp,
                                                da.zeros((n_cam2_pics - len(temp), temp.shape[1], temp.shape[2]))),
                                               axis=0)
                    else:
                        temp2 = temp

                    imgs_tif.append(temp2.reshape([ntimes, nz, nparams, npatterns2_all, ny_cam2, nx_cam2]).rechunk((1, 1, 1, 1, nx_cam2, nx_cam2)))

                ims = da.stack(imgs_tif, axis=0)

                # deal with multiple channels
                print("writing to zarr...")
                for ic, ds in enumerate(cam2_dsets):
                    istart = np.sum(npatterns_ch_2[:ic])

                    with ProgressBar():
                        da.to_zarr(ims[..., istart:istart + npatterns_ch_2[ic], :, :], ds)

                # # delete tif files
                fname_tifs = list(save_path.parent.glob("ph*.tif"))
                for f in fname_tifs:
                    f.unlink()


            # ##################################
            # reset cameras to internal triggering
            # ##################################
            mmc1.setProperty(cam1, "TRIGGER SOURCE", "INTERNAL")

            if not cam_is_phantom:
                mmc2.setProperty(cam2_name, "TriggerMode", "Internal Trigger")
            else:
                params_out = mmc2.setAcqParams(cine_no=0,
                                               SyncImaging=phc.SYNC_INTERNAL,
                                               )
            print("finished!")

        thread_run = threading.Thread(target=run)
        thread_run.start()

        return

    def show_dataset(self):
        """
        display acquired dataset

        img_data is a zarr object
        """
        img_data = self.img_data
        suffix = ""

        cmap_dict = {"red": "magenta",
                     "blue": "cyan",
                     "green": "yellow",
                     "odt": "bone"}

        if img_data is not None:
            groups = []
            if hasattr(img_data, "cam1"):
                groups.append(img_data["cam1"])
            if hasattr(img_data, "cam2"):
                groups.append(img_data["cam2"])

            # loop over camera groups
            for g in groups:

                # loop over arrays
                for name, arr in g.arrays():

                    # skip if arr does not contain data
                    if np.any(np.array(arr.shape) == 0):
                        continue

                    layer_name = f"{g.name:s} {name:s} {suffix:s}"
                    img_to_show = da.from_zarr(arr)

                    dxy = g.attrs["dx_um"]
                    dz = self.img_data.attrs["dz_um"]
                    if dz != 0:
                        z_scale = dz / dxy
                    else:
                        z_scale = 1.

                    # try to update layer. If it doesn't exist, add it
                    try:
                        preview_layer = self.viewer.layers[layer_name]
                        preview_layer.data = img_to_show
                        preview_layer.scale[-5] = z_scale


                    except KeyError:
                        # clims_low = [np.percentile(im, 1) for im in img_data.sim[0, 0, 0, 0, :, 0]]
                        # clims_high = [np.percentile(im, 99) for im in img_data.sim[0, 0, 0, 0, :, 0]]

                        # catch errors in case zarr attr does not exist yet
                        try:
                            cmap = cmap_dict[arr.attrs["channels"][0]]
                            self.viewer.add_image(img_to_show, name=layer_name, colormap=cmap,
                                                  scale=(1., 1., z_scale, 1., 1., 1., 1.))
                            self.viewer.dims.axis_labels = arr.attrs["dimensions"]

                        except Exception as e:
                            print(e)




if __name__ == "__main__":
    from qtpy.QtWidgets import QApplication

    app = QApplication([])
    window = SimOdtWidget()
    window.show()
    app.exec_()
