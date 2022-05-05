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
import mcsim.expt_ctrl.daq

# dmd
import mcsim.expt_ctrl.dlp6500
import time
import datetime
import zarr
import numpy as np
import tifffile
from PIL import Image

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

    # patterns
    patterns_lineEdit: QtW.QLineEdit
    browse_patterns_button: QtW.QPushButton

    # triggering
    triggering_checkBox: QtW.QCheckBox

    # exposure time
    exposure_SpinBox: QtW.QDoubleSpinBox

    # time lapse
    time_groupBox: QtW.QGroupBox
    timepoints_spinBox: QtW.QSpinBox
    interval_spinBox: QtW.QSpinBox
    time_comboBox: QtW.QComboBox

    # run sequence
    show_dataset_checkBox: QtW.QCheckBox
    run_Button: QtW.QPushButton
    pause_Button: QtW.QPushButton
    cancel_Button: QtW.QPushButton

    def setup_ui(self):
        uic.loadUi(self.UI_FILE, self)  # load QtDesigner .ui file
        self.pause_Button.hide()
        self.cancel_Button.hide()
        # button icon
        self.run_Button.setIcon(QIcon(str(ICONS / "play-button_1.svg")))
        self.run_Button.setIconSize(QSize(20, 0))


class DmdWidget(QtW.QWidget, _MultiDUI):

    # metadata associated with a given experiment
    SEQUENCE_META: dict[MDASequence, SequenceMeta] = {}

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

        self.pause_Button.released.connect(self._mmc.toggle_pause)
        self.cancel_Button.released.connect(self._mmc.cancel)

        # default value for exposure times
        self.exposure_SpinBox.setValue(10.)

        # connect buttons
        self.browse_save_Button.clicked.connect(self.set_multi_d_acq_dir)
        self.browse_patterns_button.clicked.connect(self.load_dmd_patterns)
        self.run_Button.clicked.connect(self._on_run_clicked)

        # events
        mmcore.events.sequenceStarted.connect(self._on_mda_started)
        mmcore.events.sequenceFinished.connect(self._on_mda_finished)
        mmcore.events.sequencePauseToggled.connect(self._on_mda_paused)

    def _set_enabled(self, enabled: bool):
        self.save_groupBox.setEnabled(enabled)
        self.time_groupBox.setEnabled(enabled)

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

        # ##############################
        # grab save information
        # ##############################

        if self.save_groupBox.isChecked() and not (
                self.fname_lineEdit.text() and Path(self.dir_lineEdit.text()).is_dir()):
            raise ValueError("Select a filename and a valid directory.")

        if self.save_groupBox.isChecked():
            subdir = self.fname_lineEdit.text()
            save_path = Path(self.dir_lineEdit.text()) / subdir / "sim_odt.zarr"
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

        npatterns = len(patterns)
        print("loaded %d patterns" % npatterns)

        # ##############################
        # grab sequence information from GUI
        # ##############################

        # exposure time
        exposure_tms = self.exposure_SpinBox.value()

        # time lapse
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
        mmc1 = self._mmcores[0]
        mmc2 = self._mmcores[1]

        if len(self._mmc.getLoadedDevices()) < 2:
            raise ValueError("Load a cfg file first.")

        # ##############################
        # turn off live mode if on
        # ##############################
        mmc1.stopSequenceAcquisition()
        mmc2.stopSequenceAcquisition()

        # ##############################
        # grab DAQ line info
        # ##############################
        # line info
        daq_do_map = self.daq.digital_line_names
        daq_ao_map = self.daq.analog_line_names
        daq_presets = self.daq.presets

        # ##################################
        # get odt camera and set up
        # ##################################
        cam = mmc2.getCameraDevice()

        # set camera properties
        mmc2.setProperty(cam, "Exposure", exposure_tms)
        # turn off despeckle correction
        # mmc2.setProperty(cam, 'PP  1   ENABLED', 'No')
        # mmc2.setProperty(cam, 'PP  2   ENABLED', 'No')
        # mmc2.setProperty(cam, 'PP  3   ENABLED', 'No')
        # mmc2.setProperty(cam, 'PP  4   ENABLED', 'No')
        # mmc2.setProperty(cam, 'PP  5   ENABLED', 'No')

        # ##################################
        # setup zarr
        # ##################################
        nx = mmc2.getImageWidth()
        ny = mmc2.getImageHeight()

        if save_path is not None:
            img_data = zarr.open(save_path, mode="w")
            img_data.attrs["save_directory"] = str(save_path)
        else:
            img_data = zarr.open(mode="w")

        # other metadata
        now = datetime.datetime.now()
        img_data.attrs["date_time"] = '%04d_%02d_%02d_%02d;%02d;%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)

        # dataset
        img_data.create_dataset("img", shape=(ntimes, npatterns, ny, nx), chunks=(1, 1, ny, nx), dtype='uint16', compressor="none")
        img_data.img.attrs["exposure_time_ms"] = exposure_tms


        # ##################################
        # acquisition
        # ##################################

        # warmup
        do_warmup = np.zeros(16, dtype=np.uint8)
        do_warmup[daq_do_map["odt_laser"]] = 1
        do_warmup[daq_do_map["odt_shutter"]] = 1
        do_warmup[daq_do_map["dmd_enable"]] = 1
        self.daq.set_digital_once(do_warmup)
        time.sleep(5)

        # load DMD patterns
        npics = npatterns * ntimes
        self.dmd.debug = False
        tstart = time.perf_counter()
        for nt in range(ntimes):
            for ip in range(npatterns):
                tnow = time.perf_counter()
                n_so_far = ip + nt * npatterns
                n_left = npatterns * ntimes - n_so_far

                if n_so_far > 0:
                    print("time %d/%d, pattern %d/%d, %0.2fs / %0.2fs" %
                          (nt + 1, ntimes, ip + 1, npatterns, tnow - tstart, (tnow - tstart) / n_so_far * n_left))

                # program the DMD
                img_inds, bit_inds = self.dmd.upload_pattern_sequence(patterns[ip].astype(np.uint8), 105, 0, triggered=False,
                                                                      num_repeats=0, compression_mode='erle')

                # take pictures
                mmc2.snapImage()
                img_data.img[nt, ip] = mmc2.getImage()
        self.dmd.debug = True
        print("acquisition finished")

        # ##################################
        # set DAQ back to off state (for digital lines only)
        # ##################################
        # off_do, off_ao = daq_map.preset_to_array(daq_presets["off"], daq_map.daq_do_map, daq_map.daq_ao_map,
        #                                          n_digital_channels=self.daq.n_digital_lines,
        #                                          n_analog_channels=self.daq.n_analog_lines
        #                                          )
        self.daq.set_preset("off")

        print("reset DAQ")


        # ##################################
        # optionally create viewer layer...
        # ##################################
        if self.show_dataset_checkBox.isChecked():

            # show odt
            if not np.any(np.array(img_data.img.shape) == 0):
                if subdir == "" or subdir is None:
                    layer_name = "DMD acquisition preview"
                else:
                    layer_name = subdir + " DMD acquisition preview"

                try:
                    preview_layer = self.viewer.layers[layer_name]
                    preview_layer.data = img_data.img
                except KeyError:
                    preview_layer = self.viewer.add_image(img_data.img, name=layer_name)

        return


if __name__ == "__main__":
    from qtpy.QtWidgets import QApplication

    app = QApplication([])
    window = DmdWidget()
    window.show()
    app.exec_()
