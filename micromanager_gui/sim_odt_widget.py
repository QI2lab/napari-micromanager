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

from copy import deepcopy
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
from mcsim.expt_ctrl.daq import nidaq
from mcsim.expt_ctrl.program_sim_odt import get_sim_odt_sequence
from mcsim.expt_ctrl import dlp6500
import mcsim.expt_ctrl.phantom_cam as phc


ICONS = Path(__file__).parent / "icons"
OBJECTIVE_DEVICE = "Objective"
# Once the PR #43 is merged, we pass the objective device to this variable


def parse_time(dt: float,
               print_days: bool = False,
               print_hours: bool = False):
    """
    Helper function for printing timing info
    """
    days = int(dt // (24 * 60 * 60))
    hours = int(dt // (60 * 60) - days * 24)
    mins = int(dt // 60 - hours * 60 - days * 24 * 60)
    secs = dt - mins * 60 - hours * 60 * 60 - days * 24 * 60 * 60
    str = ""
    if print_days:
        str += f"{days:02d} days, "
    if print_hours:
        str += f"{hours:02d}h:"
    str += f"{mins:02d}m:{secs:04.1f}s"

    return (days, hours, mins, secs), str


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
    notes_textEdit: QtW.QTextEdit

    channel_groupBox: QtW.QGroupBox
    channel_tableWidget: QtW.QTableWidget  # TODO: extract
    add_ch_Button: QtW.QPushButton
    clear_ch_Button: QtW.QPushButton
    remove_ch_Button: QtW.QPushButton

    time_groupBox: QtW.QGroupBox
    timepoints_spinBox: QtW.QSpinBox
    interval_spinBox: QtW.QSpinBox
    time_comboBox: QtW.QComboBox

    fast_time_groupBox: QtW.QGroupBox
    fast_time_spinBox: QtW.QSpinBox

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
    sim_circbuf_doubleSpinBox: QtW.QDoubleSpinBox
    daq_dt_doubleSpinBox: QtW.QDoubleSpinBox
    shutter_delay_doubleSpinBox: QtW.QDoubleSpinBox
    odt_warmup_doubleSpinBox: QtW.QDoubleSpinBox
    sim_warmup_doubleSpinBox: QtW.QDoubleSpinBox
    sim_readout_doubleSpinBox: QtW.QDoubleSpinBox
    stage_delay_doubleSpinBox: QtW.QDoubleSpinBox
    fan_checkBox: QtW.QCheckBox
    fan_wait_doubleSpinBox: QtW.QDoubleSpinBox
    cine2zarr_checkBox: QtW.QCheckBox

    return_checkBox: QtW.QCheckBox

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

    def __init__(self,
                 mmcores: list[RemoteMMCore],
                 daq: nidaq,
                 dmd: dlp6500,
                 viewer,
                 phcam: phc.phantom_cam,
                 parent=None,
                 configuration: dict = None):

        self._mmcores = mmcores
        self._mmc = self._mmcores[0]

        self.daq = daq
        self.dmd = dmd
        self.phcam = phcam
        self.configuration = configuration
        self.img_data = None
        self.pattern_modes = ["default", "average"]
        self.camera_modes = ["default", "cam1", "cam2", "both"]

        # flag to cancel sequence if necessary
        self.cancel_sequence = False

        self.viewer = viewer
        super().__init__(parent)
        self.setup_ui()

        self.odt_circbuff_SpinBox.setValue(3.)
        self.sim_circbuf_doubleSpinBox.setValue(3.)

        # save dialog
        self.browse_save_Button.clicked.connect(self._set_save_dir)

        #
        self.cine2zarr_checkBox.setChecked(True)
        self.return_checkBox.setChecked(True)

        # channel widget
        self.add_ch_Button.clicked.connect(self.add_channel)
        self.remove_ch_Button.clicked.connect(self.remove_channel)
        self.clear_ch_Button.clicked.connect(self.clear_channel)

        # run/show
        self.run_Button.clicked.connect(self._on_run_clicked)
        self.cancel_Button.clicked.connect(self._on_cancel_clicked)
        self.pause_Button.clicked.connect(self._on_paused_clicked)
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
        self.step_size_doubleSpinBox.setValue(0.25)

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

        # lock for printing while multithreading
        self.print_lock = threading.Lock()

    def set_cfg(self):
        defaults = self.configuration["sim_odt_program_defaults"]

        names = {"odt_exposure_ms": self.odt_exposure_SpinBox,
                 "sim_exposure_ms": self.sim_exposure_SpinBox,
                 "odt_frametime_ms": self.odt_frametime_SpinBox,
                 "daq_dt_us": self.daq_dt_doubleSpinBox,
                 "sim_warmup_time_ms": self.sim_warmup_doubleSpinBox,
                 "odt_warmup_time_ms": self.odt_warmup_doubleSpinBox,
                 "shutter_delay_ms": self.shutter_delay_doubleSpinBox,
                 "sim_readout_time_ms": self.sim_readout_doubleSpinBox,
                 "stage_delay_ms": self.stage_delay_doubleSpinBox
                 }

        # set default values
        for k, v in names.items():
            try:
                v.setValue(float(defaults[k]))
            except KeyError as e:
                print(f"while loading default ODT/SIM parameter information: {e}")

    def _set_enabled(self, enabled: bool):
        self.save_groupBox.setEnabled(enabled)
        self.channel_groupBox.setEnabled(enabled)
        self.time_groupBox.setEnabled(enabled)
        self.fast_time_groupBox.setEnabled(enabled)
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

            dmd_on_time_spinBox = QtW.QDoubleSpinBox(self)
            dmd_on_time_spinBox.setDecimals(1)
            dmd_on_time_spinBox.setSingleStep(10)
            dmd_on_time_spinBox.setMinimum(0.)
            dmd_on_time_spinBox.setMaximum(10000.)

            # populate channel options
            pks = list(presets.keys())
            channel_comboBox.addItems(pks)

            # create combo_boxes in table
            self.channel_tableWidget.setCellWidget(idx, 0, channel_comboBox)
            self.channel_tableWidget.setCellWidget(idx, 1, patterns_comboBox)
            self.channel_tableWidget.setCellWidget(idx, 2, mode_comboBox)
            self.channel_tableWidget.setCellWidget(idx, 3, camera_comboBox)
            self.channel_tableWidget.setCellWidget(idx, 4, dmd_on_time_spinBox)

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

                # populate pattern mode options and set to "default"
                self.channel_tableWidget.cellWidget(ii, 2).clear()
                self.channel_tableWidget.cellWidget(ii, 2).addItems(self.pattern_modes)
                self.channel_tableWidget.cellWidget(ii, 2).setCurrentText("default")

                # populate camer mode options camera modes patterns
                self.channel_tableWidget.cellWidget(ii, 3).clear()
                self.channel_tableWidget.cellWidget(ii, 3).addItems(self.camera_modes)
                self.channel_tableWidget.cellWidget(ii, 3).setCurrentText("default")

                # set DMD on time
                self.channel_tableWidget.cellWidget(ii, 4).setValue(0.)

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

    def _get_next_data_dir(self,
                           main_dir: Path,
                           subdir: Path):
        """
        ensure subdirs are of the form and numbes are correctly ordered 000_...
        """
        # test other subdirs and get their numbers
        path_exp = "(\d{3})_.*"
        other_nums = [int(re.match(path_exp, n.name).group(1)) for n in main_dir.glob("*") if
                      re.match(path_exp, n.name)]

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

        return subdir_final

    def _on_run_clicked(self):
        """
        Run hardware triggered SIM/DMD sequence
        """
        self.cancel_sequence = False
        mmc1 = self._mmcores[0]
        mmc2 = self._mmcores[1]

        if len(self._mmc.getLoadedDevices()) < 2:
            raise ValueError("Load a cfg file first.")

        # ###########################################
        # grab information from GUI
        # ###########################################
        # todo: use more flexible datatype to store this info?
        axis_list = ["time (fast)", "position", "time", "z", "parameters", "pattern", "y", "x"]
        axis_acquisition_order = ['time', 'position', 'time (fast)', 'parameter', 'z', 'channel', 'pattern']
        # axis_data = [{"name": "time",
        #               "save_order": 2},
        #              {"name": "position",
        #               "save_order": 1},
        #              {"name": "time (fast)",
        #               "save_order": 0},
        #              {"name": "parameters",
        #               "save_order": 4},
        #              {"name": "z",
        #               "save_order": 3},
        #              {"name": "channel",
        #               "save_order": None},
        #              {"name": "pattern",
        #               "save_order": 5},
        #              ]

        gui_settings = {"quiet_fan": self.fan_checkBox.isChecked(),
                        "fan_delay_s": self.fan_wait_doubleSpinBox.value(),
                        "saving": self.save_groupBox.isChecked(),
                        "convert_cine_to_zarr_live": self.cine2zarr_checkBox.isChecked(),
                        "return_to_start_position": self.return_checkBox.isChecked(),
                        "exposure_tms_sim": self.sim_exposure_SpinBox.value(),
                        "exposure_tms_odt": self.odt_exposure_SpinBox.value(),
                        "min_odt_frame_time_ms": self.odt_frametime_SpinBox.value(),
                        "odt_circ_buffer_mb": int(np.round(self.odt_circbuff_SpinBox.value() * 1e3)),
                        "sim_circ_buffer_mb": int(np.round(self.sim_circbuf_doubleSpinBox.value() * 1e3)),
                        "dt": int(np.round(self.daq_dt_doubleSpinBox.value())) * 1e-6,
                        "sim_warmup_time_ms": self.sim_warmup_doubleSpinBox.value(),
                        "odt_warmup_time_ms": self.odt_warmup_doubleSpinBox.value(),
                        "shutter_delay_time_ms": self.shutter_delay_doubleSpinBox.value(),
                        "sim_readout_time_ms": self.sim_readout_doubleSpinBox.value(),
                        "stage_delay_ms": self.stage_delay_doubleSpinBox.value()
                        }

        # ##############################
        # turn off cameras/lasers, if running
        # ##############################
        if mmc1.isSequenceRunning():
            mmc1.stopSequenceAcquisition()
        if mmc2.isSequenceRunning():
            mmc2.stopSequenceAcquisition()

        self.daq.set_preset("off")

        # ##############################
        # saving
        # ##############################
        if gui_settings["saving"] and not (self.fname_lineEdit.text() and
                                           Path(self.dir_lineEdit.text()).is_dir()):
            print("Select a filename and a valid directory.")
            return

        if gui_settings["saving"]:
            main_dir = Path(self.dir_lineEdit.text())

            # get next subdirectory
            subdir = self._get_next_data_dir(main_dir,
                                             Path(self.fname_lineEdit.text()))
            # reset the GUI with the correct name
            self.fname_lineEdit.setText(subdir.name)

            # ensure save path does not already exist
            save_path = main_dir / subdir / "sim_odt.zarr"

            if save_path.exists():
                raise ValueError(f"save path {str(save_path):s} already exists")

        else:
            save_path = None

        print("##############################################################")
        start_str = f"starting acquisition"
        if gui_settings["saving"]:
            start_str += f", saving to {str(save_path):s}"
        print(start_str)

        # ##############################
        # slow time lapse
        # ##############################
        if self.time_groupBox.isChecked():
            ntimes = self.timepoints_spinBox.value()
            time_unit = self.time_comboBox.currentText()

            if time_unit == "ms":
                requested_interval_ms = self.interval_spinBox.value()
            elif time_unit == "sec":
                requested_interval_ms = self.interval_spinBox.value() * 1e3
            elif time_unit == "min":
                requested_interval_ms = self.interval_spinBox.value() * 1e3 * 60
            else:
                raise ValueError(f"time_unit={time_unit:s} is not supported")

            interval_ms = (requested_interval_ms * 1e-3 // gui_settings["dt"]) * gui_settings["dt"] * 1e3
            print(f"rounded requested interval {requested_interval_ms:.3f}ms to "
                  f"{interval_ms:.3f}ms to be commensurate with sampling rate")
        else:
            ntimes = 1
            interval_ms = 0.

        # ##############################
        # fast time lapse
        # ##############################
        if self.fast_time_groupBox.isChecked():
            ntimes_fast = self.fast_time_spinBox.value()
        else:
            ntimes_fast = 1

        # ##############################
        # parameter scan
        # ##############################
        if self.parameter_groupBox.isChecked():
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
                    raise ValueError(f"when scan together is selected all parameter scans must have the same length, "
                                     f"but they had lengths {n_per_param}")

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
        # todo: note this will not work if you have restarted the program ... because the daq only remembers
        #  the values you have set since it was instantiated
        # ##############################
        z_now = mmc1.getZPosition()
        z_volts_start = self.daq.last_known_analog_val[self.daq.analog_line_names["z_stage"]]

        if self.stack_groupBox.isChecked():
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

            print(f"z-start position was {z_volts_start:.3f}V with guess calibration {guess_calibration_um_per_v:.3f}um/V")
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

            print(f"z-positions= {z_check} with calibration {calibration_um_per_v:.3f} um/V")
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
        # (dmd channel, pattern mode, acquisition mode, camera, number of patterns)
        acq_modes = [{"channel": self.channel_tableWidget.cellWidget(c, 0).currentText(),
                      "patterns": self.channel_tableWidget.cellWidget(c, 1).currentText(),
                      "pattern_mode": self.channel_tableWidget.cellWidget(c, 2).currentText(),
                      "camera": self.channel_tableWidget.cellWidget(c, 3).currentText(),
                      "dmd_on_time": self.channel_tableWidget.cellWidget(c, 4).value() * 1e-3,  # convert to s
                      "npatterns": 0,
                      "nimages": 0}
                     for c in range(self.channel_tableWidget.rowCount())]

        if acq_modes == []:
            print("no channels/modes selected")
            return

        # get npatterns and nimages information
        for am in acq_modes:
            # if dmd_on_time = 0, interpret this as always on
            if am["dmd_on_time"] == 0:
                am["dmd_on_time"] = None

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

        any_mode_odt = np.any([am["channel"] == "odt" for am in acq_modes])
        cam1_acq_modes = [am for am in acq_modes if am["camera"] in ["cam1", "both"]]
        cam2_acq_modes = [am for am in acq_modes if am["camera"] in ["cam2", "both"]]

        # ##################################
        # get odt camera and set up
        # ##################################
        cam_is_phantom = True

        if not cam_is_phantom:
            cam2_name = mmc2.getCameraDevice()
            # set camera properties
            mmc2.setProperty(cam2_name, "Exposure", gui_settings["exposure_tms_odt"])
            # set external triggering
            mmc2.setProperty(cam2_name, "TriggerMode", "Edge Trigger")
            # set circular buffer
            mmc2.setCircularBufferMemoryFootprint(gui_settings["odt_circ_buffer_mb"])
        else:
            mmc2 = self.phcam

            try:
                cine_no = 1  # cine indexing starts at 1
                mmc2.set_cines(1)
                # setting cine also seems to clear CSR, so do CSR
                try:
                    mmc2.set_black_reference()
                except Exception as e:
                    print(f"during camera CSR:{e}")

                params = mmc2.get_params(cine_no)
                params_out = mmc2.setAcqParams(cine_no=cine_no,
                                               Exposure=int(np.round(gui_settings["exposure_tms_odt"] * 1e6)), # in ns
                                               dFrameRate=50., # need fps to be a bit slower ... can set to 50
                                               SyncImaging=phc.SYNC_EXTERNAL,
                                               PTFrames=params.ImCount # post-trigger frames
                                               )
            except Exception as e:
                print(f"During cine preparation: {e}")
                return

        # get size and ROI
        nx_start, ny_start, nx_cam2, ny_cam2 = mmc2.getROI()
        cam2_roi = [ny_start, ny_start + ny_cam2, nx_start, nx_start + nx_cam2]

        # ##################################
        # get SIM camera and set properties
        # ##################################
        cam1 = mmc1.getCameraDevice()

        #set camera properties
        mmc1.setProperty(cam1, "ScanMode", "2")
        mmc1.setProperty(cam1, "Exposure", gui_settings["exposure_tms_sim"])
        # set external triggering
        mmc1.setProperty(cam1, "TRIGGER SOURCE", "EXTERNAL")
        mmc1.setProperty(cam1, "TriggerPolarity", "POSITIVE")
        # line 1 trigger ready
        mmc1.setProperty(cam1, "OUTPUT TRIGGER KIND[0]", "EXPOSURE")
        mmc1.setProperty(cam1, "OUTPUT TRIGGER POLARITY[0]", "POSITIVE")
        # line 2 at end of readout
        mmc1.setProperty(cam1, "OUTPUT TRIGGER DELAY[1]", "0.0000")
        mmc1.setProperty(cam1, "OUTPUT TRIGGER KIND[1]", "EXPOSURE")
        mmc1.setProperty(cam1, "OUTPUT TRIGGER POLARITY[1]", "POSITIVE")
        # line 3 at start of readout
        mmc1.setProperty(cam1, "OUTPUT TRIGGER DELAY[2]", "0.0000")
        mmc1.setProperty(cam1, "OUTPUT TRIGGER KIND[2]", "PROGRAMABLE")
        mmc1.setProperty(cam1, "OUTPUT TRIGGER PERIOD[2]", "0.001")
        mmc1.setProperty(cam1, "OUTPUT TRIGGER POLARITY[2]", "POSITIVE")
        mmc1.setProperty(cam1, "OUTPUT TRIGGER SOURCE[2]", "VSYNC")

        nx1_start, ny1_start, nx_cam1, ny_cam1 = mmc1.getROI()
        cam1_roi = [ny1_start, ny1_start + ny_cam1, nx1_start, nx1_start + nx_cam1]

        # set circular buffer
        mmc1.setCircularBufferMemoryFootprint(gui_settings["sim_circ_buffer_mb"])

        # ##################################
        # prepare daq program, but don't program device yet
        # ##################################
        # odt stabilize time
        all_modes_odt = np.all([am["channel"] == "odt" for am in acq_modes])
        if not do_xy_scan:
            gui_settings["stage_delay_ms"] = 0.
            print(f"Only one position present, so set stage delay = {gui_settings['stage_delay_ms']:.3f}ms")

        if all_modes_odt:
            gui_settings["odt_warmup_time_ms"] = 0
            print(f"set odt warmup time={gui_settings['odt_warmup_time_ms']:.3f}ms")

        try:
            digital_program, analog_program, daq_programming_info = \
                get_sim_odt_sequence(self.daq.digital_line_names,
                                     self.daq.analog_line_names,
                                     self.daq.presets,
                                     acq_modes,
                                     gui_settings["exposure_tms_odt"] * 1e-3,
                                     gui_settings["exposure_tms_sim"] * 1e-3,
                                     dt=gui_settings["dt"],
                                     interval=0., # now handled by daq
                                     n_odt_per_sim=1,
                                     n_trig_width=1,
                                     odt_stabilize_t=gui_settings["odt_warmup_time_ms"] * 1e-3,
                                     min_odt_frame_time=gui_settings["min_odt_frame_time_ms"] * 1e-3,
                                     sim_stabilize_t=gui_settings["sim_warmup_time_ms"] * 1e-3,
                                     shutter_delay_time=gui_settings["shutter_delay_time_ms"] * 1e-3,
                                     sim_readout_time=gui_settings["sim_readout_time_ms"] * 1e-3,
                                     n_xy_positions=nxy_positions,
                                     n_times_fast=ntimes_fast,
                                     stage_delay_time=gui_settings["stage_delay_ms"] * 1e-3,
                                     z_voltages=z_volts,
                                     use_dmd_as_odt_shutter=False,
                                     n_digital_ch=self.daq.n_digital_lines,
                                     n_analog_ch=self.daq.n_analog_lines,
                                     parameter_scan=param_dict,
                                     turn_lasers_off_interval=interval_ms != 0)

            if any_mode_odt:
                digital_program[:, self.daq.digital_line_names["odt_laser"]] = 1

            if all_modes_odt:
                digital_program[:, self.daq.digital_line_names["odt_shutter"]] = 1

        except Exception as e:
            print(f"exception while generating DAQ program: {e}")
            return

        print(daq_programming_info)

        # time estimate
        # program includes looping over patterns, channels, z, params, and positions
        # last time-step only needs to last as long as program
        pgm_time_s = gui_settings["dt"] * digital_program.shape[0]
        position_time_s = pgm_time_s / nxy_positions
        iteration_time = np.max([interval_ms * 1e-3, pgm_time_s])
        acquisition_time_s = iteration_time * (ntimes - 1) + pgm_time_s

        print(f"DAQ program time = {parse_time(pgm_time_s)[1]:s}")
        print(f"acquisition expected time = {parse_time(acquisition_time_s)[1]:s}")
        # ##################################
        # create zarr
        # ##################################
        img_data = zarr.open(save_path, mode="w")
        img_data.attrs["save_directory"] = str(save_path)
        if save_path is not None:
            self.img_data = zarr.open(save_path, mode="r")
        else:
            self.img_data = img_data

        # other metadata
        img_data.attrs["notes"] = self.notes_textEdit.toPlainText()
        img_data.attrs["timestamp"] = datetime.datetime.now().strftime('%Y_%d_%m_%H;%M;%S')
        img_data.attrs["channels"] = acq_modes
        img_data.attrs["xy_position_um_set"] = xy_positions
        img_data.attrs["xy_position_um_real"] = xy_positions  # updated later with correct values
        img_data.attrs["z_position_um"] = list(z_real)
        img_data.attrs["dz_um"] = dz
        img_data.attrs["z_calibration_um_per_v"] = calibration_um_per_v
        img_data.attrs["interval_ms"] = interval_ms
        img_data.attrs["axis_acquisition_order"] = axis_acquisition_order

        # make parameter dictionary json serializable
        try:
            param_dict_list = deepcopy(param_dict)
            for k, v in param_dict_list.items():
                param_dict_list[k] = param_dict_list[k].tolist()
            img_data.attrs["parameter_scan_dictionary"] = param_dict_list
        except Exception as e:
            print(f"while writing parameter scan information: {e}")
            img_data.attrs["parameter_scan_dictionary"] = None

        img_data.attrs["gui_settings"] = gui_settings

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

        # ###################################
        # group for camera # 1
        # ###################################
        g1 = img_data.create_group("cam1")
        g1.attrs["channels"] = [c["channel"] for c in cam1_acq_modes]
        g1.attrs["acquisition_modes"] = cam1_acq_modes
        g1.attrs["exposure_time_ms"] = gui_settings["exposure_tms_sim"]
        g1.attrs["camera_roi"] = cam1_roi
        g1.attrs["na_detection"] = self.configuration["camera_settings_1"]["na_detection"]

        g1_params = {"dx_um": "dxy",
                     "dy_um": "dxy",
                     "otf_model_parameters": "otf_calibration"}
        for k, v in g1_params.items():
            try:
                g1.attrs[k] = self.configuration["camera_settings_1"][v]
            except (KeyError, TypeError) as e:
                print(f"while writing cam1 settings: {e}")
                g1.attrs[k] = None

        # affine transformation information for specific channels we are using
        try:
            dmd_affine_transforms = self.configuration["camera_settings_1"]["dmd_affine_transforms"]
            g1.attrs["affine_transformations"] = [dmd_affine_transforms[am["channel"]] for am in cam1_acq_modes]
        except (KeyError, TypeError) as e:
            print(f"while writing cam1 affines: {e}")
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
            ds = g1.create_dataset(name_final,
                                   shape=(ntimes_fast, nxy_positions, ntimes, nz, nparams, am["nimages"], ny_cam1, nx_cam1),
                                   chunks=(1, 1, 1, 1, 1, 1, ny_cam1, nx_cam1),
                                   dtype="uint16",
                                   compressor="none")
            ds.attrs["acquisition_mode"] = am
            ds.attrs["dimensions"] = axis_list
            ds.attrs["channels"] = [am["channel"]]

            # sim pattern information for specific channels we are using
            sim_pattern_dat = dlp6500.get_preset_info(self.dmd.presets[am["channel"]][am["patterns"]],
                                                      self.dmd.firmware_pattern_info)[0]

            try:
                ds.attrs["nangles"] = sim_pattern_dat["nangles"][0]
                ds.attrs["nphases"] = sim_pattern_dat["nphases"][0]
                ds.attrs["lattice_vects1"] = np.array(sim_pattern_dat["a1"]).tolist()
                ds.attrs["lattice_vects2"] = np.array(sim_pattern_dat["a2"]).tolist()
                ds.attrs["phases"] = np.array(sim_pattern_dat["phase"]).tolist()
                ds.attrs["frqs"] = np.array(sim_pattern_dat["frq"]).tolist()
            except KeyError as e:
                print(f"while writing cam1 pattern data: {e}")

            cam1_dsets.append(ds)

        # ###################################
        # group for camera #2
        # ###################################
        g2 = img_data.create_group("cam2")
        g2.attrs["channels"] = cam2_acq_modes
        g2.attrs["camera_roi"] = cam2_roi
        g2.attrs["exposure_time_ms"] = gui_settings["exposure_tms_odt"]

        cam2_settings_name = "camera_settings_phantom" if cam_is_phantom else "camera_settings_2"
        g2_params = {"dx_um": "dxy",
                     "dy_um": "dxy",
                     "na_excitation": "na_excitation",
                     "na_detection": "na_detection"}

        for k, v in g2_params.items():
            try:
                g2.attrs[k] = self.configuration[cam2_settings_name][v]
            except (KeyError, TypeError) as e:
                print(f"while writing cam2 parameters: {e}")
                g2.attrs[k] = None

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
                                   shape=(ntimes_fast, nxy_positions, ntimes, nz, nparams, am["nimages"], ny_cam2, nx_cam2),
                                   chunks=(1, 1, 1, 1, 1, 1, ny_cam2, nx_cam2),
                                   dtype="uint16",
                                   compressor=numcodecs.Zlib()
                                   )

            ds.attrs["dimensions"] = axis_list
            ds.attrs["channels"] = [am["channel"]]
            if am["channel"] == "odt":
                ds.attrs["wavelength_um"] = 0.785  # todo: put in config file
                ds.attrs["volume_time_ms"] = gui_settings["min_odt_frame_time_ms"] * am["npatterns"]  # todo: correct this

                odt_firmware_data = self.dmd.presets[am["channel"]][am["patterns"]]
                odt_pic_inds = odt_firmware_data["picture_indices"]
                odt_bit_inds = odt_firmware_data["bit_indices"]
                odt_firmware_inds = dlp6500.pic_bit_ind_2firmware_ind(odt_pic_inds, odt_bit_inds)
                dmd_pattern_data = self.dmd.firmware_pattern_info

                # set odt dataset metadata
                try:
                    ds.attrs["offsets"] = [np.array(dmd_pattern_data[ii]["offsets"]).tolist() for ii in odt_firmware_inds]
                    ds.attrs["drs"] = np.array(dmd_pattern_data[odt_firmware_inds[0]]["drs"]).tolist()
                    ds.attrs["spot_frqs_mirrors"] = np.array(dmd_pattern_data[odt_firmware_inds[0]]["spot_frqs_mirrors"]).tolist()
                    ds.attrs["carrier_frq"] = np.array(dmd_pattern_data[odt_firmware_inds[0]]["carrier frequency"]).tolist()
                    ds.attrs["radius"] = dmd_pattern_data[odt_firmware_inds[0]]["radius"]
                except KeyError as e:
                    print(f"while writing cam2 pattern parameters: {e}")
            else:
                # sim pattern information for specific channels we are using
                sim_pattern_dat = dlp6500.get_preset_info(self.dmd.presets[am["channel"]][am["patterns"]],
                                                          self.dmd.firmware_pattern_info)[0]
                try:
                    ds.attrs["nangles"] = sim_pattern_dat["nangles"][0]
                    ds.attrs["nphases"] = sim_pattern_dat["nphases"][0]
                    ds.attrs["lattice_vects1"] = np.array(sim_pattern_dat["a1"]).tolist()
                    ds.attrs["lattice_vects2"] = np.array(sim_pattern_dat["a2"]).tolist()
                    ds.attrs["phases"] = np.array(sim_pattern_dat["phase"]).tolist()
                    ds.attrs["frqs"] = np.array(sim_pattern_dat["frq"]).tolist()
                except KeyError as e:
                    print(f"while writing cam2 pattern parameters: {e}")

            cam2_dsets.append(ds)

        # ###################################
        # DAQ data
        # ###################################
        g_daq = img_data.create_group("daq")
        g_daq.attrs["dt"] = gui_settings["dt"]

        # todo: check this
        g_daq.array("digital_program",
                    digital_program.astype(bool),
                    dtype=bool,
                    chunks=digital_program.shape,
                    compressor=numcodecs.packbits.PackBits(),
                    )

        g_daq.digital_program.attrs["dimensions"] = ["time", "channel"]
        g_daq.digital_program.attrs["channel_map"] = self.daq.digital_line_names

        g_daq.array("analog_program",
                    analog_program,
                    dtype=analog_program.dtype)
        g_daq.analog_program.attrs["dimensions"] = ["time", "channel"]
        g_daq.analog_program.attrs["channel_map"] = self.daq.analog_line_names

        # g_daq.create_dataset("analog_input",
        #                         shape=(nxy_positions, digital_program.shape[0] * ntimes * nz, self.daq.n_analog_inputs),
        #                         dtype="float",
        #                         compressor=numcodecs.Zlib())
        # g_daq.analog_input.attrs["dimensions"] = ["position", "time/z", "analog channel"]

        # total number of pictures for cameras per position
        n_cam1_pics = ntimes * nxy_positions * ntimes_fast * nparams * nz * int(np.sum([am["nimages"] for am in cam1_acq_modes]))
        n_cam2_pics = ntimes * nxy_positions * ntimes_fast * nparams * nz * int(np.sum([am["nimages"] for am in cam2_acq_modes]))

        def single_index2multi(ii, npatterns_channel):
            """
            Convert single-index into multiindex
            acuisition order from slowest to fastest:
            ['time', 'position', 'time (fast)', 'parameters', 'z', 'channel', 'pattern']
            """

            npatterns_all_channels = np.sum(npatterns_channel)
            npatterns_cumsum = np.cumsum(npatterns_channel)

            # channel index
            cinds = np.arange(len(npatterns_channel))
            ic = cinds[ii % npatterns_all_channels < npatterns_cumsum][0]

            # pattern index
            ipattern = int((ii % npatterns_all_channels - np.sum(npatterns_channel[:ic])) % npatterns_channel[ic])
            # other indices are easy
            iz = (ii // (npatterns_all_channels)) % nz
            iparam = (ii // (npatterns_all_channels * nz)) % nparams
            it_fast = (ii // (npatterns_all_channels * nz * nparams)) % ntimes_fast
            iposition = (ii // (npatterns_all_channels * nz * nparams * ntimes_fast)) % nxy_positions
            it = (ii // (npatterns_all_channels * nz * nparams * ntimes_fast * nxy_positions)) % ntimes

            return (it_fast, it, iposition, iparam, iz, ic, ipattern)

        def read_cam(tstart, timeout, mmc, dsets, cam_acq_modes, ncam_pics, desc=""):
            # order from slowest to fastest
            ncam_channels = len(cam_acq_modes)
            npatterns_channel = [am["nimages"] for am in cam_acq_modes]

            ii_acquired = None
            for ii_acquired in range(ncam_pics):
                # wait for camera to get picture
                while mmc.getRemainingImageCount() == 0:
                    if (time.perf_counter() - tstart) > timeout:
                        print("timeout reached......................")
                        break

                    if self.cancel_sequence:
                        print("sequence cancelled......................")
                        break

                # if we timed out or sequence was cancelled, break out of loop once all images are read
                npics = mmc.getRemainingImageCount()
                if npics == 0:
                    break

                # get index to assign picture
                inds = single_index2multi(ii_acquired, npatterns_channel)
                it_fast, it, ixy, iparam, iz, ic, ipatt = inds
                dsets[ic][it_fast, ixy, it, iz, iparam, ipatt] = mmc.popNextImage()

                # print in threadsafe way
                if ipatt == 0:
                    with self.print_lock:
                        print(f"{desc:s} image {ii_acquired + 1:05d}/{ncam_pics:05d}, "
                              f" time (slow), position, time (fast), param, z, chann, patt) = "
                              f"{(it, ixy, it_fast, iparam, iz, ic, ipatt)}/"
                              f"{(ntimes, nxy_positions, ntimes_fast, nparams, nz, ncam_channels, 0)},"
                              f" elapsed time = {parse_time(time.perf_counter() - tstart)[1]}")

            return ii_acquired

        def print_time(tstart, timeout, sleeptime=0.1):
            t_elapsed_now = time.perf_counter() - tstart

            while t_elapsed_now < timeout:
                if self.cancel_sequence:
                    break

                with self.print_lock:
                    print(f"elapsed time = "
                          f"{parse_time(t_elapsed_now)[1]:s}/"
                          f"{parse_time(timeout)[1]:s}",
                          end="\r")

                time.sleep(sleeptime)
                t_elapsed_now = time.perf_counter() - tstart

            with self.print_lock:
                print("")

            return

        def run():
            # ##################################
            # program DMD
            # ##################################
            # make sure DMD advance/enable trigger lines are low before we program the DMD
            self.daq.set_digital_lines_by_name(np.array([0, 0, 0, 0], dtype=np.uint8),
                                               ["dmd_enable",
                                                "dmd_advance",
                                                "odt_cam_sync",
                                                "camera_trigger_monitor"])

            # pre-warm the ODT laser/shutter if using ODT
            if any_mode_odt:
                self.daq.set_digital_lines_by_name(np.array([1, 1], dtype=np.uint8),
                                                   ["odt_laser",
                                                    "odt_shutter"])

            # program DMD
            blank = [False if am["channel"] == "odt" or am["pattern_mode"] == "average" else True for am in acq_modes]
            noff_after = [1 if am["pattern_mode"] == "average" else 0 for am in acq_modes]
            dmd_modes = [am["patterns"] if am["channel"] == "odt" else "default" for am in acq_modes]
            dmd_channels = [am["channel"] for am in acq_modes]

            with self.print_lock:
                tstart_program_dmd = time.perf_counter()
                print("programming DMD ...", end="\r")
            pic_inds, bit_inds = self.dmd.program_dmd_seq(dmd_modes,
                                                          dmd_channels,
                                                          nrepeats=1,
                                                          noff_before=0,
                                                          noff_after=noff_after,
                                                          blank=blank,
                                                          mode_pattern_indices=None,
                                                          triggered=True,
                                                          verbose=False)
            with self.print_lock:
                print(f"programmed DMD in {time.perf_counter() - tstart_program_dmd:.2f}s")

            # store DMD program information
            g_dmd = img_data.create_group("dmd_data")
            g_dmd.attrs["dmd_nx"] = self.dmd.width
            g_dmd.attrs["dmd_ny"] = self.dmd.height
            g_dmd.attrs["dmd_pitch_um"] = self.dmd.pitch
            g_dmd.attrs["firmware_info"] = self.dmd.get_firmware_type()
            ds = g_dmd.array("dmd_program",
                             np.vstack((pic_inds, bit_inds)),
                             dtype='int16',
                             compressor='none')
            ds.attrs["dimensions"] = ["pattern", "time"]

            if self.dmd.firmware_patterns is not None:
                ds = g_dmd.array("firmware_patterns",
                                 self.dmd.firmware_patterns.astype(bool),
                                 compressor=numcodecs.packbits.PackBits(),
                                 dtype=bool,
                                 chunks=(1, self.dmd.height, self.dmd.width))
                ds.attrs["picture_indices"] = self.dmd.picture_indices.tolist()
                ds.attrs["bit_indices"] = self.dmd.bit_indices.tolist()

            # ##################################
            # trigger camera twice, required for Phantom camera
            # ##################################
            with self.print_lock:
                print("warmup and prepare sequence")

            if cam_is_phantom:
                if gui_settings["quiet_fan"]:
                    mmc2.quiet_fan(True)

                    tstart_quiet = time.perf_counter()
                    tquiet_now = time.perf_counter() - tstart_quiet
                    while tquiet_now < gui_settings["fan_delay_s"]:
                        print(f"quieting fan and waiting {tquiet_now:.0f}s/{gui_settings['fan_delay_s']:.0f}:s",
                              end="\r")
                        time.sleep(0.5)
                        tquiet_now = time.perf_counter() - tstart_quiet

                mmc2.record_cine(cine_no)

                # seems to be necessary but not clear why
                self.daq.set_digital_lines_by_name(np.array([1], dtype=np.uint8), ["odt_cam_sync"])
                time.sleep(0.1)
                self.daq.set_digital_lines_by_name(np.array([0], dtype=np.uint8), ["odt_cam_sync"])
                time.sleep(0.1)
                self.daq.set_digital_lines_by_name(np.array([1], dtype=np.uint8), ["odt_cam_sync"])
                time.sleep(0.1)
                self.daq.set_digital_lines_by_name(np.array([0], dtype=np.uint8), ["odt_cam_sync"])
                time.sleep(0.1)

            # ##################################
            # prepare acquisition
            # ##################################
            # enable DMD, otherwise can have timing problems at the start
            self.daq.set_digital_lines_by_name(np.array([1], dtype=np.uint8), ["dmd_enable"])
            if cam_is_phantom:
                self.daq.set_digital_lines_by_name(np.array([1], dtype=np.uint8), ["odt_cam_enable"])

            # todo: make this settable somewhere
            # let lasers warmup
            time.sleep(5)

            # program DAQ
            self.daq.set_sequence(digital_program,
                                  analog_program,
                                  1 / gui_settings["dt"],
                                  analog_clock_source="/Dev1/PFI2",
                                  digital_input_source="/Dev1/PFI1",
                                  di_export_line="/Dev1/PFI2",
                                  continuous=True,
                                  nrepeats=ntimes,
                                  pause_trigger_line="/Dev1/PFI12",
                                  interval=interval_ms * 1e-3,
                                  pause_every_n=1
                                  )

            # start cameras
            mmc1.startSequenceAcquisition(n_cam1_pics, 0, True)
            if not cam_is_phantom:
                mmc2.startSequenceAcquisition(n_cam2_pics, 0, True)

            # start daq
            tstart_acq = time.perf_counter()
            self.daq.start_sequence()

            # prepare image saving threads
            thread_save_cam1 = threading.Thread(target=read_cam,
                                                args=(tstart_acq, acquisition_time_s + 10, mmc1,
                                                      cam1_dsets, cam1_acq_modes, n_cam1_pics, "cam1"))
            if not cam_is_phantom:
                thread_save_cam2 = threading.Thread(target=read_cam,
                                                    args=(tstart_acq, acquisition_time_s + 10, mmc2,
                                                          cam2_dsets, cam2_acq_modes, n_cam2_pics, "cam2"))
            else:
                # otherwise thread prints elapsed time
                thread_save_cam2 = threading.Thread(target=print_time,
                                                    args=(tstart_acq, acquisition_time_s))

            # start threads
            thread_save_cam1.start()
            thread_save_cam2.start()

            # ##############################
            # stage movement logic
            # todo: how precise is this timing for long time-lapses?
            # ##############################
            for tt in range(ntimes):
                tstart_time = time.perf_counter()
                for pp in range(nxy_positions):
                    tstart_pos = time.perf_counter()
                    # move to new position
                    if do_xy_scan:
                        mmc1.setXYPosition(xy_positions[pp][0], xy_positions[pp][1])

                        # todo: maybe want to do this every time and save all?
                        # todo: should I wait some time here for stage to settle before measuring?
                        # todo: or be measuring in separate thread?
                        if tt == 0:
                            # todo: does this actually store?
                            img_data.attrs["xy_position_um_real"][pp] = [float(self._mmc.getXPosition()),
                                                                         float(self._mmc.getYPosition())]

                    # wait until current position portion of sequence finishes
                    t_remaining_pos = position_time_s - (time.perf_counter() - tstart_pos)
                    while t_remaining_pos > 0:
                        if self.cancel_sequence:
                            break
                        time.sleep(np.min([1, t_remaining_pos]))
                        t_remaining_pos = position_time_s - (time.perf_counter() - tstart_pos)

                # wait until current time-lapse portion finishes
                if tt < (ntimes - 1):
                    t_remaining_lapse = iteration_time - (time.perf_counter() - tstart_time)
                    while t_remaining_lapse > 0:
                        if self.cancel_sequence:
                            break
                        time.sleep(np.min([1, t_remaining_lapse]))
                        t_remaining_lapse = iteration_time - (time.perf_counter() - tstart_time)

            # optionally return stage to start position
            if gui_settings["return_to_start_position"] and do_xy_scan:
                mmc1.setXYPosition(xy_positions[0][0], xy_positions[0][1])

            # ##############################
            # clean-up after sequence finishes
            # ##############################
            self.daq.stop_sequence()
            self.daq.set_preset("off")
            self.daq.set_digital_lines_by_name(np.array([0], dtype=np.uint8), ["odt_cam_enable"])
            self.daq.set_digital_lines_by_name(np.array([0], dtype=np.uint8), ["odt_cam_sync"])

            # try:
            #     img_data.daq.analog_input[pp] = self.daq.read_ai(img_data.daq.analog_input.shape[1])
            # except Exception as e:
            #     print(f"while storing analog in voltages: {e}")

            # stop recording
            if mmc1.isSequenceRunning():
                mmc1.stopSequenceAcquisition()
            if mmc2.isSequenceRunning() or cam_is_phantom:
                mmc2.stopSequenceAcquisition()

            # wait for pictures to be stored to disk
            # threads should stop themselves if self.cancel() = True
            thread_save_cam1.join()
            thread_save_cam2.join()

            if cam_is_phantom and not self.cancel_sequence:
                if gui_settings["quiet_fan"]:
                    mmc2.quiet_fan(False)  # turn fan back on

                if cam2_acq_modes != [] and gui_settings["saving"]:
                    # todo: is it possible to grab pictures as they come in on the phantom?
                    fname_cine = save_path.parent / "ph_odt.cine"

                    tstart_ph_save = time.perf_counter()
                    print("saving cine to disk...", end="\r")
                    try:
                        mmc2.save_cine(cine_no,
                                       fname_cine,
                                       first_image=0,
                                       img_count=n_cam2_pics,
                                       file_type="cine raw")
                    except Exception as e:
                        print(f"while saving cine: {e}")

                    print(f"saved ODT to disk in {time.perf_counter() - tstart_ph_save:.2f}s")

                    # delete cine
                    try:
                        mmc2.destroy_cine(cine_no)
                    except Exception as e:
                        print(f"while destroying cine {e}")

                    # convert cine to zarr
                    try:
                        imgs_cine, md = phc.imread_cine(fname_cine, read_setup_info=True)

                        # reshape must use acquisition order
                        npatterns_ch_2 = np.array([am["nimages"] for am in cam2_acq_modes], dtype=int)
                        npatterns2_all = np.sum(npatterns_ch_2)  # number of combined patterns for all channels
                        # acquisition order
                        dask_shape = (ntimes, nxy_positions, ntimes_fast, nparams, nz, npatterns2_all, ny_cam2, nx_cam2)
                        chunk_size = (1, 1, 1, 1, 1, 1, nx_cam2, nx_cam2)
                        imgs_cine = imgs_cine.reshape(dask_shape).rechunk(chunk_size)
                        # convert from acquisition order to ("time (fast)", "position", "time", "z", "parameters", "pattern", "y", "x")
                        imgs_cine = da.transpose(imgs_cine, axes=(2, 1, 0, 4, 3, 5, 6, 7))

                        # save header data
                        g2.attrs["cine_metadata"] = md["header_data"]
                        if gui_settings["convert_cine_to_zarr_live"]:
                            for ic, ds in enumerate(cam2_dsets):
                                istart = np.sum(npatterns_ch_2[:ic])
                                with ProgressBar():
                                    da.to_zarr(imgs_cine[..., istart:istart + npatterns_ch_2[ic], :, :], ds)

                            # delete cine files
                            fname_cines = list(save_path.parent.glob("ph*.cine"))
                            for f in fname_cines:
                                f.unlink()

                    except Exception as e:
                        print(f"while converting cine to zarr: {e}")

            # ##################################
            # reset DAQ to initial state and cameras to internal triggering
            # ##################################
            # after all positions have run, set z-position back to start
            self.daq.set_analog_lines_by_name([z_volts_start], ["z_stage"])

            mmc1.setProperty(cam1, "TRIGGER SOURCE", "INTERNAL")

            if not cam_is_phantom:
                mmc2.setProperty(cam2_name, "TriggerMode", "Internal Trigger")
            else:
                params_out = mmc2.setAcqParams(cine_no=0,
                                               SyncImaging=phc.SYNC_INTERNAL,
                                               )

            with self.print_lock:
                print("sequence finished!")

            return

        thread_run = threading.Thread(target=run)
        thread_run.start()

        return

    def _on_cancel_clicked(self):
        self.cancel_sequence = True

    def _on_paused_clicked(self):
        print("Pause button not implemented...")

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
                    dz = img_data.attrs["dz_um"]
                    if dz != 0:
                        z_scale = dz / dxy
                    else:
                        z_scale = 1.

                    dim_names = arr.attrs["dimensions"]
                    scale = [z_scale if d == "z" else 1. for d in dim_names]

                    # try to update layer. If it doesn't exist, add it
                    try:
                        preview_layer = self.viewer.layers[layer_name]
                        preview_layer.data = img_to_show
                        preview_layer.scale[-5] = z_scale

                    except KeyError:
                        # catch errors in case zarr attr does not exist yet
                        try:
                            cmap = cmap_dict[arr.attrs["channels"][0]]

                            self.viewer.add_image(img_to_show,
                                                  name=layer_name,
                                                  colormap=cmap,
                                                  scale=scale,
                                                  blending="additive")

                            self.viewer.dims.axis_labels = dim_names

                        except Exception as e:
                            print(f"while plotting channels: {e}")


if __name__ == "__main__":
    from qtpy.QtWidgets import QApplication

    app = QApplication([])
    window = SimOdtWidget()
    window.show()
    app.exec_()
