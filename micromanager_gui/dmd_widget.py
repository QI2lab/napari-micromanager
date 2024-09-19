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
    from pymmcore_plus import CMMCorePlus

import threading
from numcodecs import packbits
import time
import datetime
import zarr
import numpy as np
import tifffile
from PIL import Image
from warnings import warn
import mcsim.expt_ctrl.daq
from mcsim.expt_ctrl import dlp6500

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
    UI_FILE = str(Path(__file__).parent / "_ui" / "dmd_gui.ui")

    # The UI_FILE above contains these objects:
    save_groupBox: QtW.QGroupBox
    fname_lineEdit: QtW.QLineEdit
    dir_lineEdit: QtW.QLineEdit
    browse_save_Button: QtW.QPushButton

    # camera
    camera_comboBox: QtW.QComboBox

    # patterns
    patterns_lineEdit: QtW.QLineEdit
    browse_patterns_button: QtW.QPushButton
    trigger_dmd_checkBox: QtW.QCheckBox

    # exposure time
    exposure_SpinBox: QtW.QDoubleSpinBox

    # time lapse
    time_groupBox: QtW.QGroupBox
    timepoints_spinBox: QtW.QSpinBox
    interval_spinBox: QtW.QSpinBox
    time_comboBox: QtW.QComboBox

    notes_textEdit: QtW.QTextEdit

    # run sequence
    run_Button: QtW.QPushButton
    show_pushButton: QtW.QPushButton

    def setup_ui(self):
        uic.loadUi(self.UI_FILE, self)  # load QtDesigner .ui file
        # button icon
        self.run_Button.setIcon(QIcon(str(ICONS / "play-button_1.svg")))
        self.run_Button.setIconSize(QSize(20, 0))


class DmdWidget(QtW.QWidget, _MultiDUI):

    # metadata associated with a given experiment
    SEQUENCE_META: dict[MDASequence, SequenceMeta] = {}

    def __init__(self, mmcores: list[CMMCorePlus], daq: mcsim.expt_ctrl.daq.daq, dmd: dlp6500.dlpc900_dmd,
                 viewer, parent=None):

        mmcore = mmcores[0]
        self._mmcores = mmcores
        self._mmc = self._mmcores[0]

        self.daq = daq
        self.dmd = dmd
        self.viewer = viewer
        super().__init__(parent)
        self.setup_ui()

        self.img_data = None
        self.run_thread = None

        self._refresh_camera_list()

        # default values
        self.exposure_SpinBox.setValue(10.)
        self.trigger_dmd_checkBox.setChecked(True)

        # connect buttons
        self.browse_save_Button.clicked.connect(self.set_multi_d_acq_dir)
        self.browse_patterns_button.clicked.connect(self.load_dmd_patterns)
        self.run_Button.clicked.connect(self._on_run_clicked)
        self.show_pushButton.clicked.connect(self._show_dataset)

    def _refresh_camera_list(self):
        ncores = len(self._mmcores)

        core_inds = list(range(ncores))
        core_inds = [str(ind) for ind in core_inds]
        core_inds += ["phantom"]

        self.camera_comboBox.clear()
        self.camera_comboBox.addItems(core_inds)
        self.camera_comboBox.setCurrentText(core_inds[0])

    def _set_enabled(self, enabled: bool):
        self.save_groupBox.setEnabled(enabled)
        self.time_groupBox.setEnabled(enabled)

    def set_multi_d_acq_dir(self):
        # set the directory
        self.dir = QtW.QFileDialog(self)
        self.dir.setFileMode(QtW.QFileDialog.DirectoryOnly)
        self.save_dir = QtW.QFileDialog.getExistingDirectory(self.dir)
        self.dir_lineEdit.setText(self.save_dir)
        self.parent_path = Path(self.save_dir)

    def load_dmd_patterns(self):
        self.pdir = QtW.QFileDialog(self)
        self.pdir.setFileMode(QtW.QFileDialog.DirectoryOnly)
        self.psave_dir = QtW.QFileDialog.getExistingDirectory(self.pdir)
        self.patterns_lineEdit.setText(self.psave_dir)


    def _on_run_clicked(self):

        if self.run_thread is not None:
            self.run_thread.join()

        # ##############################
        # grab save information
        # ##############################

        if self.save_groupBox.isChecked() and not (
                self.fname_lineEdit.text() and Path(self.dir_lineEdit.text()).is_dir()):
            raise ValueError("Select a filename and a valid directory.")

        if self.save_groupBox.isChecked():
            subdir = self.fname_lineEdit.text()
            save_path = Path(self.dir_lineEdit.text()) / subdir / "sim_odt.zarr"

            if save_path.exists():
                warn(f"Path {str(save_path)} already exists. Choose a different path")
                return

        else:
            save_path = None
            subdir = None

        # ##############################
        # grab pattern directory and load patterns
        # ##############################
        pattern_dir = Path(self.patterns_lineEdit.text())

        patterns_png = []
        for p in pattern_dir.glob("*.png"):
            patterns_png.append(np.asarray(Image.open(str(p))))
        patterns_png = np.asarray(patterns_png)

        patterns_tif = []
        for p in pattern_dir.glob("*.tif"):
            patterns_tif.append(tifffile.imread(str(p)))

        if len(patterns_tif) > 0:
            patterns_tif = np.vstack(patterns_tif)

        if len(patterns_tif) > 0 and len(patterns_png) > 0:
            raise ValueError("both tif and png files were present. Do not know what to do")
        elif len(patterns_tif) > 0:
            patterns = patterns_tif
        elif len(patterns_png) > 0:
            patterns = patterns_png
        else:
            raise Exception("no *.png or *.tif images in directory")

        npatterns, ny_patterns, nx_patterns = patterns.shape
        print(f"loaded {npatterns:d} patterns")

        # ##############################
        # grab sequence information from GUI
        # ##############################
        exposure_tms = self.exposure_SpinBox.value()
        trigger_dmd = self.trigger_dmd_checkBox.isChecked()

        do_time_lapse = self.time_groupBox.isChecked()
        if do_time_lapse:
            ntimes = self.timepoints_spinBox.value()
            interval_ms = self.interval_spinBox.value()
        else:
            ntimes = 1
            interval_ms = 0.

        # ##############################
        # grab cameras
        # ##############################
        # todo: make camera selectable
        cam_index = int(self.camera_comboBox.currentText())
        mmc = self._mmcores[cam_index]

        # ##############################
        # turn off live mode if on
        # ##############################
        for core in self._mmcores:
            core.stopSequenceAcquisition()

        # ##################################
        # get camera and set up
        # ##################################
        cam = mmc.getCameraDevice()
        mmc.setExposure(cam, exposure_tms)
        nx = mmc.getImageWidth()
        ny = mmc.getImageHeight()

        # ##################################
        # setup zarr
        # todo: ideally want same structure as used in sim_odt_widget.py
        # ##################################
        if save_path is not None:
            self.img_data = zarr.open(save_path, mode="w")
            self.img_data.attrs["save_directory"] = str(save_path)
        else:
            self.img_data = zarr.open(mode="w")

        # other metadata
        now = datetime.datetime.now()
        self.img_data.attrs["notes"] = self.notes_textEdit.toPlainText()
        self.img_data.attrs["date_time"] = '%04d_%02d_%02d_%02d;%02d;%02d' % (
        now.year, now.month, now.day, now.hour, now.minute, now.second)
        self.img_data.attrs["pattern_directory"] = str(pattern_dir)

        self.img_data.create_dataset("dmd_patterns",
                                shape=(npatterns, ny_patterns, nx_patterns),
                                chunks=(1, ny_patterns, nx_patterns),
                                dtype=bool,
                                compressor=packbits.PackBits())
        self.img_data.dmd_patterns.attrs["dimensions"] = ["pattern", "y", "x"]
        self.img_data.dmd_patterns[:] = patterns.astype(bool)

        self.img_data.create_dataset("times",
                                shape=(ntimes, npatterns),
                                chunks=(1, 1),
                                dtype=float)

        # dataset
        self.img_data.create_dataset("img",
                                shape=(ntimes, npatterns, ny, nx),
                                chunks=(1, 1, ny, nx),
                                dtype='uint16')
        self.img_data.img.attrs["dimensions"] = ["time", "pattern", "y", "x"]
        self.img_data.img.attrs["exposure_time_ms"] = exposure_tms

        def run():
            # ##################################
            # acquisition
            # ##################################
            # warmup
            self.daq.set_digital_lines_by_name(np.array([1, 1, 1], dtype=np.uint8),
                                               ["dmd_enable",
                                                "odt_laser",
                                                "odt_shutter"])
            time.sleep(5)

            if trigger_dmd:
                self.daq.set_digital_lines_by_name(np.array([1, 1, 1, 1], dtype=np.uint8),
                                                   ["dmd_enable", "dmd_advance",
                                                    "dmd2_enable", "dmd2_advance"])

                self.dmd.upload_pattern_sequence(patterns.astype(np.uint8),
                                                 clear_pattern_after_trigger=False,
                                                 triggered=True,
                                                 )
                self.dmd.start_stop_sequence("start")

                self.daq.set_digital_lines_by_name(np.array([0, 0], dtype=np.uint8),
                                                   ["dmd_advance", "dmd2_advance"])

            # load DMD patterns using software
            self.dmd.debug = False
            tstart = time.perf_counter()
            for nt in range(ntimes):
                for ip in range(npatterns):
                    tnow = time.perf_counter()
                    n_so_far = ip + nt * npatterns
                    n_left = npatterns * ntimes - n_so_far

                    if n_so_far > 0:
                        print(f"time {nt + 1:d}/{ntimes:d}, "
                              f"pattern {ip + 1:d}/{npatterns:d}, "
                              f"{tnow - tstart:.2f}s / {(tnow - tstart) / n_so_far * n_left:.2f}s",
                              end="\r")

                    # program the DMD
                    if trigger_dmd:
                        self.daq.set_digital_lines_by_name(np.array([1, 1], dtype=np.uint8),
                                                           ["dmd_advance", "dmd2_advance"])
                    else:
                        self.dmd.upload_pattern_sequence(patterns[ip].astype(np.uint8),
                                                         triggered=False)

                    # take pictures
                    mmc.snapImage()
                    self.img_data.img[nt, ip] = mmc.getImage()
                    self.img_data.times[nt, ip] = tnow - tstart

                    if trigger_dmd:
                        self.daq.set_digital_lines_by_name(np.array([0, 0], dtype=np.uint8),
                                                           ["dmd_advance", "dmd2_advance"])

            self.dmd.debug = True
            print("\nacquisition finished")

            # ##################################
            # set DAQ back to off state (for digital lines only)
            # ##################################
            self.daq.set_preset("off")

        self.run_thread = threading.Thread(target=run)
        self.run_thread.start()

    def _show_dataset(self):
        if self.img_data is not None and not np.any(np.array(self.img_data.img.shape) == 0):
            try:
                layer_name = Path(self.img_data.store.path).parent.name
            except:
                layer_name = "DMD acquisition preview"
            try:
                preview_layer = self.viewer.layers[layer_name]
                preview_layer.data = self.img_data.img
            except KeyError:
                preview_layer = self.viewer.add_image(self.img_data.img, name=layer_name)
            self.viewer.dims.axis_labels = ["times", "patterns", "y", "x"]

        return


if __name__ == "__main__":
    from qtpy.QtWidgets import QApplication

    app = QApplication([])
    window = DmdWidget()
    window.show()
    app.exec_()
