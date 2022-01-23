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
import zarr

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

    def __init__(self, mmcores: list[RemoteMMCore], daq: mcsim.expt_ctrl.daq.daq, dmd: mcsim.expt_ctrl.dlp6500.dlp6500, parent=None):

        mmcore = mmcores[0]
        self._mmcores = mmcores
        self._mmc = self._mmcores[0]

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

        if self.save_groupBox.isChecked() and not (
                self.fname_lineEdit.text() and Path(self.dir_lineEdit.text()).is_dir()):
            raise ValueError("Select a filename and a valid directory.")

        save_dir = Path(self.dir_lineEdit.text()) / self.fname_lineEdit.text()
        img_fname = save_dir / "sim_odt.zarr"

        mmc1 = self._mmcores[0]
        mmc2 = self._mmcores[1]

        if len(self._mmc.getLoadedDevices()) < 2:
            raise ValueError("Load a cfg file first.")

        # grab information
        channels = [self.channel_tableWidget.cellWidget(c, 0).currentText() for c in range(self.channel_tableWidget.rowCount())]
        exposure_tms_sim = self.sim_exposure_SpinBox.value()
        exposure_tms_odt = self.odt_exposure_SpinBox.value()

        # time lapse
        do_time_lapse = self.time_groupBox.isChecked()
        if do_time_lapse:
            ntimes = self.timepoints_spinBox.value()
            interval_ms = self.interval_spinBox.value()
        else:
            ntimes = 1
            interval_ms = 0.

        # zstack
        do_zstack = self.stack_groupBox.isChecked()
        if do_zstack:
            raise NotImplementedError()
        else:
            nz = 1

        # line info
        daq_do_map = mcsim.expt_ctrl.expt_map.daq_do_map
        daq_ao_map = mcsim.expt_ctrl.expt_map.daq_ao_map
        daq_presets = mcsim.expt_ctrl.expt_map.presets

        # ##################################
        # program DMD
        # ##################################

        # make sure DMD advance/enable trigger lines are low before we program the DMD
        digital_start = np.zeros(self.daq.n_digital_lines, dtype=np.uint8)
        digital_start[daq_do_map["dmd_enable"]] = 0
        digital_start[daq_do_map["dmd_advance"]] = 0
        self.daq.set_digital_once(digital_start)

        # program the DMD
        # todo: read from channel table
        modes = ["default"] * len(channels)
        pic_inds, bit_inds = mcsim.expt_ctrl.set_dmd_sim.program_dmd_seq(self.dmd, modes, channels, nrepeats=1,
                                                                         ndarkframes=0, blank=False,
                                                                         mode_pattern_indices=None,
                                                                         triggered=True, verbose=True)
        dmd_data = np.vstack((pic_inds, bit_inds))

        # ##################################
        # warmup todo: how to deal with this when multiple channels?
        # ##################################
        do_warmup = np.zeros(16, dtype=np.uint8)
        do_warmup[daq_do_map["odt_laser"]] = 1
        self.daq.set_digital_once(do_warmup)

        time.sleep(5)

        # ##################################
        # program daq
        # ##################################
        # number of patterns for single channel
        n_odt_patterns = len(mcsim.expt_ctrl.set_dmd_sim.channel_map["odt"]["default"]["picture_indices"])
        n_sim_patterns_channel = len(mcsim.expt_ctrl.set_dmd_sim.channel_map["blue"]["sim"]["picture_indices"])
        n_odt_per_sim = 1
        dt = 105e-6
        n_trig_width = int(np.ceil(3e-3 / dt))

        # total number of pictures
        nsim_pics = n_sim_patterns_channel * len([ch for ch in channels if ch != "odt"]) * ntimes * nz
        nodt_pics = n_odt_patterns * n_odt_per_sim * ntimes * nz * len([ch for ch in channels if ch == "odt"])

        digital_program, analog_program, dt = build_odt_sim_sequence(daq_do_map, daq_ao_map, channels,
                                                                     exposure_tms_odt*1e-3, exposure_tms_sim*1e-3,
                                                                     n_odt_patterns, n_sim_patterns_channel,
                                                                     dt=dt, interval=interval_ms*1e-3,
                                                                     n_odt_per_sim=n_odt_per_sim,
                                                                     n_trig_width=n_trig_width)

        # check program and number of pictures match
        if np.sum(digital_program[:, daq_do_map["odt_cam"]]) // n_trig_width != nodt_pics // ntimes:
            raise ValueError("number of odt pics (%d) did not match DAQ program (%d)" %
                             (nodt_pics, np.sum(digital_program[:, daq_do_map["odt_cam"]]) // n_trig_width))

        if np.sum(digital_program[:, daq_do_map["sim_cam"]]) // n_trig_width != nsim_pics // ntimes:
            raise ValueError("number of sim pics (%d) did not match DAQ program (%d)" %
                             (nsim_pics, np.sum(digital_program[:, daq_do_map["sim_cam"]]) // n_trig_width))

        self.daq.set_sequence(digital_program, analog_program, 1/dt)


        # ##################################
        # get odt camera and set up
        # ##################################
        odt_cam = mmc2.getCameraDevice()

        # set camera properties
        mmc2.setProperty(odt_cam, "Exposure", exposure_tms_odt)
        # set external triggering
        mmc2.setProperty(odt_cam, "TriggerMode", "Edge Trigger")

        # set ROI
        sx = 801
        cx = 1024
        sy = 511
        # cy = 1120
        cy = 715
        mmc2.setROI(cx - sx // 2, cy - sy // 2, sx, sy)

        # ##################################
        # get SIM camera and set properties
        # ##################################
        sim_cam = mmc1.getCameraDevice()

        # set camera properties
        mmc1.setProperty(sim_cam, "ScanMode", "2")
        mmc1.setProperty(sim_cam, "Exposure", exposure_tms_odt)
        # set external triggering
        # self._mmc.setProperty(sim_cam, "TRIGGER ACTIVE", "EDGE")
        # self._mmc.setProperty(sim_cam, "TRIGGER DELAY", "0.000")
        ## mmc.setProperty(odt_cam, "TRIGGER GLOBAL EXPOSURE", "DELAYED")
        ## self._mmc.setProperty(odt_cam, "TRIGGER GLOBAL EXPOSURE", "GLOBAL RESET")
        mmc1.setProperty(sim_cam, "TRIGGER SOURCE", "EXTERNAL")
        mmc1.setProperty(sim_cam, "TriggerPolarity", "POSITIVE")
        # mmc1.setProperty(sim_cam, "TRIGGER SOURCE", "INTERNAL")

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

        # ##################################
        # setup zarr
        # ##################################
        nx_sim = mmc1.getImageWidth()
        ny_sim = mmc1.getImageHeight()

        nx_odt = mmc2.getImageWidth()
        ny_odt = mmc2.getImageHeight()

        img_data = zarr.open(img_fname, mode="w")
        img_data.create_dataset("sim", shape=(nsim_pics, ny_sim, nx_sim), chunks=(1, ny_sim, nx_sim), dtype='int16', compressor="none")
        img_data.create_dataset("odt", shape=(nodt_pics, ny_odt, nx_odt), chunks=(1, ny_odt, nx_odt), dtype='int16', compressor="none")

        img_data.create_dataset("dmd_firmware_program", shape=dmd_data.shape, dtype='int16', compressor='none')
        img_data.dmd_firmware_program[:] = dmd_data

        img_data.create_dataset("daq_digital_program", shape=digital_program.shape, dtype='int8', compressor="none")
        img_data.daq_digital_program[:] = digital_program

        img_data.create_dataset("daq_analog_program", shape=analog_program.shape, dtype='float32', compressor="none")
        img_data.daq_analog_program[:] = analog_program

        img_data.attrs["date_time"] = 0


        # ##################################
        # burst acquisition
        # ##################################

        # set circular buffer
        mmc2.setCircularBufferMemoryFootprint(3000)

        # start camera
        mmc1.startSequenceAcquisition(nsim_pics, 0, True)
        mmc2.startSequenceAcquisition(nodt_pics, 0, True)

        # start daq
        self.daq.start_sequence()

        # read images and save
        pgm_time_s = dt * digital_program.shape[0] * ntimes
        timeout = 10 + pgm_time_s
        tstart = time.perf_counter()

        ii_sim = 0
        ii_odt = 0

        daq_stopped = False
        def get_remaining_image_count(): return mmc1.getRemainingImageCount(), mmc2.getRemainingImageCount()
        for icount in range(nsim_pics + nodt_pics):

            tnow = time.perf_counter() - tstart
            if tnow > pgm_time_s and not daq_stopped:
                self.daq.stop_sequence()
                daq_stopped = True

            while (get_remaining_image_count() == (0, 0)):
                tnow = time.perf_counter() - tstart
                if tnow > pgm_time_s and not daq_stopped:
                    self.daq.stop_sequence()
                    daq_stopped = True

                if tnow > timeout:
                    print("timeout reached")
                    break

            n1, n2 = get_remaining_image_count()
            if n1 > 0:
                img_data.sim[ii_sim] = mmc1.popNextImage()
                ii_sim += 1

            if n2 > 0:
                img_data.odt[ii_odt] = mmc2.popNextImage()
                ii_odt += 1

        print("remaining images in buffer: ", end="")
        print(get_remaining_image_count())

        # ##################################
        # stop sequence acquisition
        # ##################################
        mmc1.stopSequenceAcquisition()
        mmc2.stopSequenceAcquisition()

        if not daq_stopped:
            self.daq.stop_sequence()

        # ##################################
        # set DAQ back to off state (for digital lines only)
        # ##################################
        off_do, off_ao = mcsim.expt_ctrl.expt_map.preset_to_array(daq_presets["off"],
                                                              mcsim.expt_ctrl.expt_map.daq_do_map,
                                                              mcsim.expt_ctrl.expt_map.daq_ao_map,
                                                              n_digital_channels=self.daq.n_digital_lines,
                                                              n_analog_channels=self.daq.n_analog_lines
                                                              )
        self.daq.set_digital_once(off_do)

        print("ii_sim=%d/%d, ii_odt=%d/%d" % (ii_sim, nsim_pics, ii_odt, nodt_pics))
        print("acquisition finished")

        # ##################################
        # reset cameras to internal triggering...
        # ##################################
        mmc1.setProperty(sim_cam, "TRIGGER SOURCE", "INTERNAL")

        mmc2.setProperty(odt_cam, "TriggerMode", "Internal Trigger")
        mmc2.clearROI()

        return


if __name__ == "__main__":
    from qtpy.QtWidgets import QApplication

    app = QApplication([])
    window = SimOdtWidget()
    window.show()
    app.exec_()
