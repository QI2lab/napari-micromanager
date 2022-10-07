from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import napari
import numpy as np
import zarr
from pymmcore_plus import CMMCorePlus, DeviceType #, RemoteMMCore
from qtpy import QtWidgets as QtW
from qtpy import uic
from qtpy.QtCore import QSize, QTimer
from qtpy.QtGui import QColor, QIcon
from superqt.utils import create_worker, ensure_main_thread
from useq import MDASequence

from ._illumination import IlluminationDialog
from ._saving import save_sequence
from ._util import blockSignals, event_indices, extend_array_for_index
from .explore_sample import ExploreSample
from .sim_odt_widget import SimOdtWidget
from .dmd_widget import DmdWidget
from .multid_widget import MultiDWidget, SequenceMeta
from .prop_browser import PropBrowser

if TYPE_CHECKING:
    from typing import Dict

    import napari.layers
    import napari.viewer
    import useq
    from pymmcore_plus.core.events import QCoreSignaler
    from pymmcore_plus.mda import PMDAEngine

# dmd and daq control
import json
import re
from mcsim.expt_ctrl import dlp6500, daq, phantom_cam
import mcsim.analysis.analysis_tools as mctools
from mcsim.analysis.sim_reconstruction import fit_modulation_frq
from localize_psf import fit
from skimage.restoration import unwrap_phase
from numpy import fft
from scipy.ndimage import maximum_filter, minimum_filter


ICONS = Path(__file__).parent / "icons"
CAM_ICON = QIcon(str(ICONS / "vcam.svg"))
CAM_STOP_ICON = QIcon(str(ICONS / "cam_stop.svg"))


class _MainUI:
    UI_FILE = str(Path(__file__).parent / "_ui" / "micromanager_gui.ui")

    # The UI_FILE above contains these objects:
    cfg_LineEdit: QtW.QLineEdit
    browse_cfg_Button: QtW.QPushButton
    load_cfg_Button: QtW.QPushButton
    cfg2_LineEdit: QtW.QLineEdit
    browse_cfg2_Button: QtW.QPushButton
    load_cfg2_Button: QtW.QPushButton
    dmd_cfg_lineEdit: QtW.QLineEdit
    dmd_load_cfg_Button: QtW.QPushButton
    browse_dmd_cfg_Button: QtW.QPushButton
    daq_cfg_lineEdit: QtW.QLineEdit
    daq_load_cfg_Button: QtW.QPushButton
    browse_daq_cfg_Button: QtW.QPushButton
    microscope_cfg_lineEdit: QtW.QLineEdit
    browse_microscope_cfg_Button: QtW.QPushButton
    microscope_load_cfg_Button: QtW.QPushButton
    objective_groupBox: QtW.QGroupBox
    objective_comboBox: QtW.QComboBox
    camera_groupBox: QtW.QGroupBox
    bin_comboBox: QtW.QComboBox
    bit_comboBox: QtW.QComboBox
    position_groupBox: QtW.QGroupBox
    x_lineEdit: QtW.QLineEdit
    y_lineEdit: QtW.QLineEdit
    z_lineEdit: QtW.QLineEdit
    stage_groupBox: QtW.QGroupBox
    XY_groupBox: QtW.QGroupBox
    Z_groupBox: QtW.QGroupBox
    left_Button: QtW.QPushButton
    right_Button: QtW.QPushButton
    y_up_Button: QtW.QPushButton
    y_down_Button: QtW.QPushButton
    up_Button: QtW.QPushButton
    down_Button: QtW.QPushButton
    xy_step_size_SpinBox: QtW.QSpinBox
    z_step_size_doubleSpinBox: QtW.QDoubleSpinBox
    tabWidget: QtW.QTabWidget
    snap_live_tab: QtW.QWidget
    multid_tab: QtW.QWidget
    snap_channel_groupBox: QtW.QGroupBox
    snap_channel_comboBox: QtW.QComboBox
    exp_spinBox: QtW.QDoubleSpinBox

    # image processing
    image_proc_mode_comboBox: QtW.QComboBox
    use_affine_xform_checkBox: QtW.QCheckBox
    fx_doubleSpinBox: QtW.QDoubleSpinBox
    fy_doubleSpinBox: QtW.QDoubleSpinBox
    threshold_SpinBox: QtW.QDoubleSpinBox
    guess_holo_frq_Button: QtW.QPushButton
    fit_holo_frq_Button: QtW.QPushButton
    fit_holo_curvature_Button: QtW.QPushButton
    set_affine_ref_Button: QtW.QPushButton
    track_affine_checkBox: QtW.QCheckBox

    #
    snap_Button: QtW.QPushButton
    live_Button: QtW.QPushButton
    max_min_val_label: QtW.QLabel
    max_scale_doubleSpinBox: QtW.QDoubleSpinBox
    min_scale_doubleSpinBox: QtW.QDoubleSpinBox
    autoscale_Button: QtW.QPushButton
    px_size_doubleSpinBox: QtW.QDoubleSpinBox
    properties_Button: QtW.QPushButton
    illumination_Button: QtW.QPushButton
    snap_on_click_xy_checkBox: QtW.QCheckBox
    snap_on_click_z_checkBox: QtW.QCheckBox
    set_camera_comboBox: QtW.QComboBox
    set_channel_Button: QtW.QPushButton
    channel_comboBox: QtW.QComboBox
    mode_comboBox: QtW.QComboBox
    daq_shutter_checkBox: QtW.QCheckBox

    # dmd frame
    pattern_time_SpinBox: QtW.QDoubleSpinBox
    pic_index_spinBox: QtW.QSpinBox
    bit_index_spinBox: QtW.QSpinBox
    set_dmd_pattern_index_pushButton: QtW.QPushButton
    dmd_snap_checkBox: QtW.QCheckBox

    # daq frame
    daq_channel_groupBox: QtW.QGroupBox
    daq_channel_tableWidget: QtW.QTableWidget
    add_ch_Button: QtW.QPushButton
    clear_ch_Button: QtW.QPushButton
    remove_ch_Button: QtW.QPushButton
    daq_update_immediately_checkBox: QtW.QCheckBox
    daq_update_pushButton: QtW.QPushButton
    daq_snap_checkBox: QtW.QCheckBox

    def setup_ui(self):
        uic.loadUi(self.UI_FILE, self)  # load QtDesigner .ui file

        # button icons
        for attr, icon in [
            ("left_Button", "left_arrow_1_green.svg"),
            ("right_Button", "right_arrow_1_green.svg"),
            ("y_up_Button", "up_arrow_1_green.svg"),
            ("y_down_Button", "down_arrow_1_green.svg"),
            ("up_Button", "up_arrow_1_green.svg"),
            ("down_Button", "down_arrow_1_green.svg"),
            ("snap_Button", "cam.svg"),
            ("live_Button", "vcam.svg"),
        ]:
            btn = getattr(self, attr)
            btn.setIcon(QIcon(str(ICONS / icon)))
            btn.setIconSize(QSize(30, 30))


class MainWindow(QtW.QWidget, _MainUI):
    def __init__(self, viewer: napari.viewer.Viewer, remote=False):
        super().__init__()
        self.setup_ui()

        self.viewer = viewer
        self.streaming_timer = None

        # create connection to mmcore server or process-local variant
        # create two cores, the first is the main core, the second only runs the second camera
        # self._mmcores = [RemoteMMCore(port=54333) if remote else CMMCorePlus(),
        #                  RemoteMMCore(port=54334) if remote else CMMCorePlus()]
        self._mmcores = [CMMCorePlus(), CMMCorePlus()]

        self._mmc = self._mmcores[0]
        self._mmc_cam = self._mmcores[1]

        # placeholders for later
        # since these are passed through to the other widgets, they can be updated but not reassigned
        self.phcam = phantom_cam.phantom_cam()
        self.dmd = dlp6500.dlp6500win(initialize=False)
        self.daq = daq.nidaq(initialize=False)
        self.cfg_data = {}
        self.cam_affine_xform_napari_cam2_to_cam1 = None

        # connect mmcore signals
        sig: QCoreSignaler = self._mmc.events

        # note: don't use lambdas with closures on `self`, since the connection
        # to core may outlive the lifetime of this particular widget.

        # mda events
        # sig.sequenceStarted.connect(self._on_mda_started)
        # sig.sequenceFinished.connect(self._on_mda_finished)
        # self._mmc.mda.events.frameReady.connect(self._on_mda_frame)
        # self._mmc.mda.events.sequenceStarted.connect(self._on_mda_started)
        # self._mmc.mda.events.sequenceFinished.connect(self._on_mda_finished)
        # self._mmc.events.mdaEngineRegistered.connect(self._update_mda_engine)

        sig.systemConfigurationLoaded.connect(self._refresh_options)
        sig.XYStagePositionChanged.connect(self._on_xy_stage_position_changed)
        sig.stagePositionChanged.connect(self._on_stage_position_changed)
        sig.exposureChanged.connect(self._on_exp_change)
        sig.channelGroupChanged.connect(self._refresh_channel_list)
        sig.configSet.connect(self._on_config_set)

        # connect buttons
        self.load_cfg_Button.clicked.connect(self.load_cfg)
        self.browse_cfg_Button.clicked.connect(self.browse_cfg)
        self.load_cfg2_Button.clicked.connect(self.load_cfg2)
        self.browse_cfg2_Button.clicked.connect(self.browse_cfg2)
        self.browse_dmd_cfg_Button.clicked.connect(self.browse_dmd_cfg)
        self.dmd_load_cfg_Button.clicked.connect(self.load_dmd_cfg)
        self.browse_daq_cfg_Button.clicked.connect(self.browse_daq_cfg)
        self.daq_load_cfg_Button.clicked.connect(self.load_daq_cfg)
        self.browse_microscope_cfg_Button.clicked.connect(self.browse_microscope_cfg)
        self.microscope_load_cfg_Button.clicked.connect(self.load_microscope_cfg)
        self.left_Button.clicked.connect(self.stage_x_left)
        self.right_Button.clicked.connect(self.stage_x_right)
        self.y_up_Button.clicked.connect(self.stage_y_up)
        self.y_down_Button.clicked.connect(self.stage_y_down)
        self.up_Button.clicked.connect(self.stage_z_up)
        self.down_Button.clicked.connect(self.stage_z_down)
        self.autoscale_Button.clicked.connect(self.autoscale_active_layer)
        self.snap_Button.clicked.connect(self.snap)
        self.live_Button.clicked.connect(self.toggle_live)
        self.illumination_Button.clicked.connect(self.illumination)
        self.properties_Button.clicked.connect(self._show_prop_browser)
        self.set_dmd_pattern_index_pushButton.clicked.connect(self._set_dmd_pattern_index)
        self.set_affine_ref_Button.clicked.connect(self._set_affine_ref)

        # connect buttons
        self.add_ch_Button.clicked.connect(self.add_channel)
        self.remove_ch_Button.clicked.connect(self.remove_channel)
        self.clear_ch_Button.clicked.connect(self.clear_channel)
        self.daq_update_pushButton.clicked.connect(self._on_daq_setting_change)
        self.daq_update_immediately_checkBox.clicked.connect(self._on_channel_changed)


        # update mode combo box when channel combo box is changed
        self.channel_comboBox.currentTextChanged.connect(self._refresh_mode_options)

        # set channel/mode combination
        self.set_channel_Button.clicked.connect(self.set_channel_and_mode)


        # connect comboBox
        self.objective_comboBox.currentIndexChanged.connect(self.change_objective)
        self.bit_comboBox.currentIndexChanged.connect(self.bit_changed)
        self.bin_comboBox.currentIndexChanged.connect(self.bin_changed)
        self.snap_channel_comboBox.currentTextChanged.connect(self._channel_changed)
        self.set_camera_comboBox.currentTextChanged.connect(self._camera_changed)

        # set up processing modes combo box
        proc_modes = ["normal", "fft", "hologram", "hologram unwrapped"]
        self.image_proc_mode_comboBox.addItems(proc_modes)
        self.snap_channel_comboBox.setCurrentText(proc_modes[0])
        self.guess_holo_frq_Button.clicked.connect(self.guess_holo_frq)
        self.fit_holo_frq_Button.clicked.connect(self.fit_holo_frq)
        self.fit_holo_curvature_Button.clicked.connect(self.fit_holo_curvature)
        self.threshold_SpinBox.setValue(50.)

        # DMD
        self.pattern_time_SpinBox.setValue(0.105)

        # connect spinboxes
        self.exp_spinBox.valueChanged.connect(self._update_exp)
        self.exp_spinBox.setKeyboardTracking(False)

        self.fx_doubleSpinBox.setValue(1600.)
        self.fy_doubleSpinBox.setValue(1400.)

        # refresh options in case a config is already loaded by another remote
        self._refresh_options()

        self.viewer.layers.events.connect(self.update_max_min)
        self.viewer.layers.selection.events.active.connect(self.update_max_min)
        self.viewer.dims.events.current_step.connect(self.update_max_min)

        # tab widgets
        self.sim_odt_acq = SimOdtWidget(self._mmcores, self.daq, self.dmd, self.viewer, self.phcam,
                                        configuration=self.cfg_data)
        self.tabWidget.addTab(self.sim_odt_acq, "SIM/ODT Acquisition")

        self.dmd_widget = DmdWidget(self._mmcores, self.daq, self.dmd, self.viewer)
        self.tabWidget.addTab(self.dmd_widget, "DMD")


    def illumination(self):
        if not hasattr(self, "_illumination"):
            self._illumination = IlluminationDialog(self._mmc, self)
        self._illumination.show()

    def _show_prop_browser(self):
        pb = PropBrowser(self._mmc, self)
        pb.exec()

    def _on_config_set(self, groupName: str, configName: str):
        if groupName == self._mmc.getOrGuessChannelGroup():
            with blockSignals(self.snap_channel_comboBox):
                self.snap_channel_comboBox.setCurrentText(configName)

    def _set_enabled(self, enabled):
        self.objective_groupBox.setEnabled(enabled)
        self.camera_groupBox.setEnabled(enabled)
        self.XY_groupBox.setEnabled(enabled)
        self.Z_groupBox.setEnabled(enabled)
        self.snap_live_tab.setEnabled(enabled)
        self.snap_live_tab.setEnabled(enabled)

    def _update_exp(self, exposure: float):
        self._mmc_cam.setExposure(exposure)
        if self.streaming_timer:
            self.streaming_timer.setInterval(int(exposure))
            self._mmc_cam.stopSequenceAcquisition()
            self._mmc_cam.startContinuousSequenceAcquisition(exposure)

    def _on_exp_change(self, camera: str, exposure: float):
        with blockSignals(self.exp_spinBox):
            self.exp_spinBox.setValue(exposure)
        if self.streaming_timer:
            self.streaming_timer.setInterval(int(exposure))

    def browse_cfg(self):
        self._mmc.unloadAllDevices()  # unload all devicies
        print(f"Loaded Devices: {self._mmc.getLoadedDevices()}")

        # clear spinbox/combobox without accidently setting properties
        boxes = [
            self.objective_comboBox,
            self.bin_comboBox,
            self.bit_comboBox,
            self.snap_channel_comboBox,
        ]
        with blockSignals(boxes):
            for box in boxes:
                box.clear()

        file_dir = QtW.QFileDialog.getOpenFileName(self, "", "⁩", "cfg(*.cfg)")
        self.cfg_LineEdit.setText(str(file_dir[0]))
        self.max_min_val_label.setText("None")
        self.load_cfg_Button.setEnabled(True)

    def load_cfg(self):
        #self.load_cfg_Button.setEnabled(False)
        print("loading", self.cfg_LineEdit.text())
        self._mmc.loadSystemConfiguration(self.cfg_LineEdit.text())

        # is this run already by loadSystemConfiguration()?
        if "System" in self._mmcores[0].getAvailableConfigGroups():
            if "Startup" in self._mmcores[0].getAvailableConfigs("System"):
                self._mmcores[0].setConfig("System", "Startup")

        self._set_affine_ref()

    def browse_cfg2(self):
        self._mmcores[1].unloadAllDevices()  # unload all devicies
        print(f"Loaded Devices: {self._mmcores[1].getLoadedDevices()}")

        # clear spinbox/combobox without accidently setting properties
        # boxes = [
        #     self.objective_comboBox,
        #     self.bin_comboBox,
        #     self.bit_comboBox,
        #     self.snap_channel_comboBox,
        # ]
        # with blockSignals(boxes):
        #     for box in boxes:
        #         box.clear()

        file_dir = QtW.QFileDialog.getOpenFileName(self, "", "⁩", "cfg(*.cfg)")
        self.cfg2_LineEdit.setText(str(file_dir[0]))
        self.max_min_val_label.setText("None")
        self.load_cfg2_Button.setEnabled(True)

    def load_cfg2(self):
        #self.load_cfg2_Button.setEnabled(False)
        print("loading", self.cfg2_LineEdit.text())
        self._mmcores[1].loadSystemConfiguration(self.cfg2_LineEdit.text())

        # is this run already by loadSystemConfiguration()?
        if "System" in self._mmcores[1].getAvailableConfigGroups():
            if "Startup" in self._mmcores[1].getAvailableConfigs("System"):
                self._mmcores[1].setConfig("System", "Startup")

    def browse_dmd_cfg(self):
        file_dir = QtW.QFileDialog.getOpenFileName(self, "", "⁩", "json(*.json)")
        self.dmd_cfg_lineEdit.setText(str(file_dir[0]))

    def load_dmd_cfg(self):
        fname_dmd_config = Path(self.dmd_cfg_lineEdit.text())

        fname_patterns = fname_dmd_config.parent / "dmd_firmware_patterns.zarr"
        if fname_patterns.exists():
            fware_patterns = np.array(zarr.open(fname_patterns, "r").dmd_patterns).astype(bool)
        else:
            fware_patterns = None

        self.dmd.initialize(debug=True, config_file=fname_dmd_config, firmware_patterns=fware_patterns)

    def browse_daq_cfg(self):
        file_dir = QtW.QFileDialog.getOpenFileName(self, "", "⁩", "json(*.json)")
        self.daq_cfg_lineEdit.setText(str(file_dir[0]))

    def browse_microscope_cfg(self):
        file_dir = QtW.QFileDialog.getOpenFileName(self, "", "⁩", "json(*.json)")
        self.microscope_cfg_lineEdit.setText(str(file_dir[0]))

    def load_microscope_cfg(self):
        fname_microscope_config = self.microscope_cfg_lineEdit.text()
        with open(fname_microscope_config, "r") as f:
            self.cfg_data.update(json.load(f))

        self.sim_odt_acq.set_cfg()

        # load affine transformation
        swap_xy = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        cam_affine_xform_cam1_to_cam2 = np.array(self.cfg_data["camera_affine_transforms"]["xform"])
        cam_affine_xform_napari_cam1_to_cam2 = swap_xy.dot(cam_affine_xform_cam1_to_cam2.dot(swap_xy))
        self.cam_affine_xform_napari_cam2_to_cam1 = np.linalg.inv(cam_affine_xform_napari_cam1_to_cam2)

    def load_daq_cfg(self):
        fname_daq_config = self.daq_cfg_lineEdit.text()
        self.daq.initialize(dev_name="Dev1", digital_lines="port0/line0:15",
                            analog_lines=["ao0", "ao1", "ao2", "ao3"],
                            config_file=fname_daq_config)

        # populate channel combo box
        self.channel_comboBox.clear()
        self.channel_comboBox.addItems(list(self.daq.presets.keys()))

    def _refresh_camera_options(self):
        cam_device = self._mmc_cam.getCameraDevice()
        if not cam_device:
            return
        cam_props = self._mmc_cam.getDevicePropertyNames(cam_device)
        if "Binning" in cam_props:
            bin_opts = self._mmc_cam.getAllowedPropertyValues(cam_device, "Binning")
            with blockSignals(self.bin_comboBox):
                self.bin_comboBox.clear()
                self.bin_comboBox.addItems(bin_opts)
                self.bin_comboBox.setCurrentText(
                    self._mmc_cam.getProperty(cam_device, "Binning")
                )

        if "PixelType" in cam_props:
            px_t = self._mmc_cam.getAllowedPropertyValues(cam_device, "PixelType")
            with blockSignals(self.bit_comboBox):
                self.bit_comboBox.clear()
                self.bit_comboBox.addItems(px_t)
                self.bit_comboBox.setCurrentText(
                    self._mmc_cam.getProperty(cam_device, "PixelType")
                )

    def _refresh_objective_options(self):
        if "Objective" in self._mmc.getLoadedDevices():
            with blockSignals(self.objective_comboBox):
                self.objective_comboBox.clear()
                self.objective_comboBox.addItems(self._mmc.getStateLabels("Objective"))
                self.objective_comboBox.setCurrentText(
                    self._mmc.getStateLabel("Objective")
                )

    def _refresh_channel_list(self, channel_group: str = None):
        if channel_group is None:
            channel_group = self._mmc.getOrGuessChannelGroup()
        if channel_group:
            channel_list = list(self._mmc.getAvailableConfigs(channel_group))
            with blockSignals(self.snap_channel_comboBox):
                self.snap_channel_comboBox.clear()
                self.snap_channel_comboBox.addItems(channel_list)
                self.snap_channel_comboBox.setCurrentText(
                    self._mmc.getCurrentConfig(channel_group)
                )

    def _refresh_camera_list(self):
        ncores = len(self._mmcores)

        core_inds = list(range(ncores))
        core_inds = [str(ind) for ind in core_inds]
        core_inds += ["phantom"]

        self.set_camera_comboBox.clear()
        self.set_camera_comboBox.addItems(core_inds)
        self.set_camera_comboBox.setCurrentText(core_inds[0])

    def _refresh_positions(self):
        if self._mmc.getXYStageDevice():
            x, y = self._mmc.getXPosition(), self._mmc.getYPosition()
            self._on_xy_stage_position_changed(self._mmc.getXYStageDevice(), x, y)
        if self._mmc.getFocusDevice():
            self.z_lineEdit.setText(f"{self._mmc.getZPosition():.1f}")

    def _refresh_mode_options(self):
        self.mode_comboBox.clear()

        if self.dmd.initialized:
            chan = self.channel_comboBox.currentText()
            modes = list(self.dmd.presets[chan].keys())
            self.mode_comboBox.addItems(modes)
            self.mode_comboBox.setCurrentText("default") # note: modes are required to have a mode named "default"

    def _refresh_options(self):
        self._refresh_camera_options()
        self._refresh_objective_options()
        self._refresh_channel_list()
        self._refresh_positions()
        self._refresh_camera_list()
        self._refresh_mode_options()

    def bit_changed(self):
        if self.bit_comboBox.count() > 0:
            bits = self.bit_comboBox.currentText()
            self._mmc_cam.setProperty(self._mmc_cam.getCameraDevice(), "PixelType", bits)

    def bin_changed(self):
        if self.bin_comboBox.count() > 0:
            bins = self.bin_comboBox.currentText()
            cd = self._mmc_cam.getCameraDevice()
            self._mmc_cam.setProperty(cd, "Binning", bins)

    def _channel_changed(self, newChannel: str):
        channel_group = self._mmc.getOrGuessChannelGroup()
        if channel_group:
            self._mmc.setConfig(channel_group, newChannel)

    def _camera_changed(self, newCamera: str):
        # self._mmc.setCameraDevice(newCamera)
        try:
            self._mmc_cam = self._mmcores[int(newCamera)]
        except ValueError:
            if newCamera == "phantom":
                self._mmc_cam = self.phcam

    def _set_affine_ref(self):
        self.affine_ref = [self._mmc.getXPosition(), self._mmc.getYPosition()]

    def _on_xy_stage_position_changed(self, name, x, y):
        self.x_lineEdit.setText(f"{x:.1f}")
        self.y_lineEdit.setText(f"{y:.1f}")

    def _on_stage_position_changed(self, name, value):
        if "z" in name.lower():  # hack
            self.z_lineEdit.setText(f"{value:.1f}")

    def stage_x_left(self):
        self._mmc.setRelativeXYPosition(-float(self.xy_step_size_SpinBox.value()), 0.0)

        # todo: for some reason with the MadCity stage doesn't update without this code even though the on_xy_stage_changed signal should be connected...
        x, y = self._mmc.getXPosition(), self._mmc.getYPosition()
        self._on_xy_stage_position_changed(self._mmc.getXYStageDevice(), x, y)

        if self.snap_on_click_xy_checkBox.isChecked():
            self.snap()

    def stage_x_right(self):
        self._mmc.setRelativeXYPosition(float(self.xy_step_size_SpinBox.value()), 0.0)

        x, y = self._mmc.getXPosition(), self._mmc.getYPosition()
        self._on_xy_stage_position_changed(self._mmc.getXYStageDevice(), x, y)

        if self.snap_on_click_xy_checkBox.isChecked():
            self.snap()

    def stage_y_up(self):
        self._mmc.setRelativeXYPosition(
            0.0,
            float(self.xy_step_size_SpinBox.value()),
        )

        x, y = self._mmc.getXPosition(), self._mmc.getYPosition()
        self._on_xy_stage_position_changed(self._mmc.getXYStageDevice(), x, y)

        if self.snap_on_click_xy_checkBox.isChecked():
            self.snap()

    def stage_y_down(self):
        self._mmc.setRelativeXYPosition(
            0.0,
            -float(self.xy_step_size_SpinBox.value()),
        )

        x, y = self._mmc.getXPosition(), self._mmc.getYPosition()
        self._on_xy_stage_position_changed(self._mmc.getXYStageDevice(), x, y)

        if self.snap_on_click_xy_checkBox.isChecked():
            self.snap()

    def stage_z_up(self):
        self._mmc.setRelativeXYZPosition(
            0.0, 0.0, float(self.z_step_size_doubleSpinBox.value())
        )
        if self.snap_on_click_z_checkBox.isChecked():
            self.snap()

    def stage_z_down(self):
        self._mmc.setRelativeXYZPosition(
            0.0, 0.0, -float(self.z_step_size_doubleSpinBox.value())
        )
        if self.snap_on_click_z_checkBox.isChecked():
            self.snap()

    def autoscale_active_layer(self):

        min_p = self.min_scale_doubleSpinBox.value()
        max_p = self.max_scale_doubleSpinBox.value()

        # only scale the visible layer, which is the first layer which is not hidden
        # todo: this doesn't always work ... how to figure out which layer is "on top"
        for l in reversed(self.viewer.layers):
            if l.visible:
                # get slider position and only scale for the visible slice
                # channel_dim = 4
                # current_step = list(self.viewer.dims.current_step[:-2])
                # current_step.pop(channel_dim)

                # don't let this code breaking crash the program...
                try:
                    # todo: think this only works if don't have scale = dxy set
                    current_slice = l.world_to_data(self.viewer.dims.current_step)[:-2]
                    current_slice = tuple([int(c) for c in current_slice])

                    vmin = np.percentile(l.data[tuple(current_slice)].ravel(), min_p)
                    vmax = np.percentile(l.data[tuple(current_slice)].ravel(), max_p)

                    # converting to float will force dask to compute
                    l.contrast_limits = (float(vmin), float(vmax))
                except Exception as e:
                    print(e)

                break


    def change_objective(self):
        if self.objective_comboBox.count() <= 0:
            return

        zdev = self._mmc.getFocusDevice()

        currentZ = self._mmc.getZPosition()
        self._mmc.setPosition(zdev, 0)
        self._mmc.waitForDevice(zdev)
        self._mmc.setProperty(
            "Objective", "Label", self.objective_comboBox.currentText()
        )
        self._mmc.waitForDevice("Objective")
        self._mmc.setPosition(zdev, currentZ)
        self._mmc.waitForDevice(zdev)

        # define and set pixel size Config
        self._mmc.deletePixelSizeConfig(self._mmc.getCurrentPixelSizeConfig())
        curr_obj_name = self._mmc.getProperty("Objective", "Label")
        self._mmc.definePixelSizeConfig(curr_obj_name)
        self._mmc.setPixelSizeConfig(curr_obj_name)

        # get magnification info from the objective name
        # and set image pixel sixe (x,y) for the current pixel size Config
        match = re.search(r"(\d{1,3})[xX]", curr_obj_name)
        if match:
            mag = int(match.groups()[0])
            self.image_pixel_size = self.px_size_doubleSpinBox.value() / mag
            self._mmc.setPixelSizeUm(
                self._mmc.getCurrentPixelSizeConfig(), self.image_pixel_size
            )

    def update_viewer(self, data=None):
        # TODO: - fix the fact that when you change the objective
        #         the image translation is wrong
        #       - are max and min_val_lineEdit updating in live mode?
        if data is None:
            try:
                data = self._mmc_cam.getLastImage()
            except (RuntimeError, IndexError):
                # circular buffer empty
                return

        ny, nx = data.shape

        # get image processing mode
        mode = self.image_proc_mode_comboBox.currentText()

        # different layers for different cameras and modes
        cam_name = self._mmc_cam.getCameraDevice()

        use_affine = self.use_affine_xform_checkBox.isChecked() and mode == "normal" and cam_name != self._mmcores[0].getCameraDevice()

        if mode == "normal":
            pass
        elif mode == "fft":
            data_ft = np.abs(fft.fftshift(fft.fft2(fft.ifftshift(data))))
        elif mode == "hologram" or mode == "hologram unwrapped":
            data_ft = fft.fftshift(fft.fft2(fft.ifftshift(data)))

            # todo: grab these values from configuration file
            # todo: could display in some nicer way...but...
            # dxy = 6.5 / 60
            dxy = self.cfg_data["camera_settings_phantom"]["dxy"]
            # fmax = 1 / 0.785
            fmax = self.cfg_data["camera_settings_phantom"]["na_detection"] / 0.785

            # get frequency coordinates
            fx = fft.fftshift(fft.fftfreq(nx, dxy))
            fy = fft.fftshift(fft.fftfreq(ny, dxy))
            fxfx, fyfy = np.meshgrid(fx, fy)
            ff = np.sqrt(fxfx**2 + fyfy**2)
            dfx = fx[1] - fx[0]
            dfy = fy[1] - fy[0]

            # convert from pixels in image to real units
            holo_frq = np.array([dfx * (self.fx_doubleSpinBox.value() - nx // 2),
                                 dfy * (self.fy_doubleSpinBox.value() - ny // 2)])

            # instead multiply by expected phase ramp
            # ft_xlated = mctools.translate_ft(data_ft, -holo_frq[0], -holo_frq[1], drs=(dxy, dxy), use_gpu=False)
            ft_xlated = mctools.translate_ft(data_ft, -holo_frq[0], -holo_frq[1], drs=(dxy, dxy), use_gpu=False)
            ft_xlated[ff > fmax] = 0
            im_holo = fft.fftshift(fft.ifft2(fft.ifftshift(ft_xlated)))

            # mask
            threshold = self.threshold_SpinBox.value()
            mask = np.abs(im_holo) > threshold

            # todo: could also display amplitude data...
            holo_amp = np.abs(im_holo)
            if mode == "hologram":
                holo_angle = np.angle(im_holo)
            else:
                holo_angle = unwrap_phase(np.angle(im_holo))

            # add center value back
            holo_angle -= np.mean(holo_angle[mask])

        else:
            raise ValueError(f"mode must be 'normal', 'fft', 'hologram', or 'hologram unwrapped' but was '{mode:s}'")

        # display image
        layer_name = f"{cam_name:s}"

        if self.track_affine_checkBox.isChecked():
            xnow = self._mmc.getXPosition()
            ynow = self._mmc.getYPosition()
            dxy_um = self.cfg_data["camera_settings_1"]["dxy"]
            translation = [-(self.affine_ref[1] - ynow) / dxy_um, -(self.affine_ref[0] - xnow) / dxy_um]
            layer_name += f"x={xnow:.0f}, y={ynow:.0f}"
        else:
            translation = [0, 0]

        try:
            preview_layer = self.viewer.layers[layer_name]
            preview_layer.data = data
        except KeyError:
            if use_affine:
                preview_layer = self.viewer.add_image(data, name=layer_name, affine=self.cam_affine_xform_napari_cam2_to_cam1)
            else:
                preview_layer = self.viewer.add_image(data, name=layer_name, translate=translation)

                if self.track_affine_checkBox.isChecked():
                    self.viewer.camera.center = (0, translation[0] + data.shape[0]//2, translation[1] + data.shape[0]//2)
                    self.autoscale_active_layer()

        # make our most recent snapped image the only visible layer
        self.viewer.layers[layer_name].visible = True

        layer_names = [self.viewer.layers[ii].name for ii in range(len(self.viewer.layers))]
        for ln in layer_names:
            # only make layers which are translucent invisible. This way we can see two layers in different colors
            # if change to "additive"
            # todo: do this in a smart way integrated with showing affine positions
            # if ln != layer_name and self.viewer.layers[ln].blending == 'translucent':
            #     self.viewer.layers[ln].visible = False
            pass

        self.update_max_min()

        # show FFT next to real space image
        # todo: update for affine transformation possibility
        if mode != "normal":
            layer_name_ft = f"{cam_name:s} fft"
            point_layer_name = f"{cam_name:s} fft holo reference"

            try:
                preview_layer = self.viewer.layers[layer_name_ft]
                preview_layer.data = np.abs(data_ft)
            except KeyError:
                preview_layer = self.viewer.add_image(np.abs(data_ft), name=layer_name_ft, translate=[0, nx],
                                                      contrast_limits=[0, np.percentile(np.abs(data_ft), 99.9)], gamma=0.1)

            pts = np.array([[self.fy_doubleSpinBox.value(), self.fx_doubleSpinBox.value() + nx]])
            try:
                point_layer = self.viewer.layers[point_layer_name]
                point_layer.data = pts
            except KeyError:
                point_layer = self.viewer.add_points(pts, name=point_layer_name,
                                                     face_color=[0, 0, 0, 0], edge_color="red", size=10)



        if mode == "hologram" or mode == "hologram unwrapped":
            layer_name_holo_phase = f"{cam_name:s} {mode:s} phase"

            try:
                preview_layer = self.viewer.layers[layer_name_holo_phase]
                preview_layer.data = holo_angle
            except KeyError:
                if mode == "hologram":
                    lims = [-np.pi, np.pi]
                else:
                    lims = [-2*np.pi, 2*np.pi]

                preview_layer = self.viewer.add_image(holo_angle, name=layer_name_holo_phase, translate=[ny, 0],
                                                      colormap="twilight_shifted", contrast_limits=lims)


            layer_name_holo_amp = f"{cam_name:s} {mode:s} amp"
            try:
                preview_layer = self.viewer.layers[layer_name_holo_amp]
                preview_layer.data = holo_amp
            except KeyError:
                preview_layer = self.viewer.add_image(holo_amp, name=layer_name_holo_amp, translate=[ny, nx])

    def guess_holo_frq(self):
        """
        guess offset-holography frequency from data
        """
        # find correct layer
        is_fft_layer = [l.name[-3:] == "fft" for l in self.viewer.layers]

        if np.sum(is_fft_layer) == 1:
            ind = int(np.where(is_fft_layer)[0])
            img_ft = self.viewer.layers[ind].data

            ny, nx = img_ft.shape

            dxy = self.cfg_data["camera_settings_phantom"]["dxy"]
            fmax_int = 2 * self.cfg_data["camera_settings_phantom"]["na_detection"] / 0.785

            fxs = fft.fftshift(fft.fftfreq(nx, dxy))
            dfx = fxs[1] - fxs[0]
            fys = fft.fftshift(fft.fftfreq(ny, dxy))
            dfy = fys[1] - fys[0]

            fxfx, fyfy = np.meshgrid(fxs, fys)
            ff_perp = np.sqrt(fxfx ** 2 + fyfy ** 2)

            # exclude points along lines
            guess_mask = np.logical_and.reduce((np.abs(fxfx) > dfx, # not along x=0
                                               np.abs(fyfy) > dfy, # not along y=0
                                               fyfy <= 0,
                                               ff_perp > fmax_int))

            guess_ind_1d = np.argmax(np.abs(img_ft) * guess_mask)
            guess_ind = np.unravel_index(guess_ind_1d, img_ft.shape)

            frq_guess = np.array([fxfx[guess_ind], fyfy[guess_ind]])

            self.fx_doubleSpinBox.setValue(guess_ind[1])
            self.fy_doubleSpinBox.setValue(guess_ind[0])
    def fit_holo_frq(self):
        """
        fit offset-holography frequency from data. Use current frequency value as guess value
        """
        # find correct layer
        is_fft_layer = [l.name[-3:] == "fft" for l in self.viewer.layers]

        if np.sum(is_fft_layer) == 1:
            ind = int(np.where(is_fft_layer)[0])
            img_ft = self.viewer.layers[ind].data

            ny, nx = img_ft.shape

            dxy = self.cfg_data["camera_settings_phantom"]["dxy"]
            fmax_int = 2 * self.cfg_data["camera_settings_phantom"]["na_detection"] / 0.785

            fxs = fft.fftshift(fft.fftfreq(nx, dxy))
            dfx = fxs[1] - fxs[0]
            fys = fft.fftshift(fft.fftfreq(ny, dxy))
            dfy = fys[1] - fys[0]

            fxfx, fyfy = np.meshgrid(fxs, fys)

            frq_guess = np.array([dfx * (self.fx_doubleSpinBox.value() - nx // 2),
                                  dfy * (self.fy_doubleSpinBox.value() - ny // 2)])

            frq_fit, mask, _ = fit_modulation_frq(img_ft, img_ft, dxy, frq_guess=frq_guess, roi_pix_size=50)

            indx_fit = frq_fit[0] / dfx + nx // 2
            indy_fit = frq_fit[1] / dfy + ny // 2

            self.fx_doubleSpinBox.setValue(indx_fit)
            self.fy_doubleSpinBox.setValue(indy_fit)

    def fit_holo_curvature(self):
        """
        fit hologram phase curvature
        """

        threshold = self.threshold_SpinBox.value()

        # find correct layer
        is_holo_amp = [l.name[-22:] == "hologram unwrapped amp" for l in self.viewer.layers]
        is_holo_phase = [l.name[-24:] == "hologram unwrapped phase" for l in self.viewer.layers]

        if np.sum(is_holo_phase) == 1 and np.sum(is_holo_amp) == 1:
            ind = int(np.where(is_holo_phase)[0])
            phase_unwrapped = self.viewer.layers[ind].data

            ind_amp = int(np.where(is_holo_amp)[0])
            holo_amp = self.viewer.layers[ind_amp].data

            ny, nx = phase_unwrapped.shape

            # get frequency data
            dxy = self.cfg_data["camera_settings_phantom"]["dxy"]
            fmax_int = 2 * self.cfg_data["camera_settings_phantom"]["na_detection"] / 0.785
            fxs = fft.fftshift(fft.fftfreq(nx, dxy))
            dfx = fxs[1] - fxs[0]
            fys = fft.fftshift(fft.fftfreq(ny, dxy))
            dfy = fys[1] - fys[0]

            k = 2 * np.pi / 0.785

            # fit mask
            # fit defocus
            to_fit_pix = holo_amp > threshold
            # dilate/erode to close holes
            footprint = np.ones((10, 10), dtype=bool)
            to_fit_pix = maximum_filter(to_fit_pix, footprint=footprint)
            to_fit_pix = minimum_filter(to_fit_pix, footprint=footprint)

            # exclude region around edges
            edge_exclude_size = 20
            to_fit_pix[:edge_exclude_size] = False
            to_fit_pix[-edge_exclude_size:] = False
            to_fit_pix[:, :edge_exclude_size] = False
            to_fit_pix[:, -edge_exclude_size:] = False

            # fit defocus phase
            def defocus_phase_fn(p, x, y):
                """
                @param p: [k/Rx, k/Ry, cx, cy, offset, theta]
                @param x:
                @param y:
                @return phase:
                """
                xrot = (x - p[2]) * np.cos(p[5]) + (y - p[3]) * np.sin(p[5])
                yrot = -(x - p[2]) * np.sin(p[5]) + (y - p[3]) * np.cos(p[5])
                phase = 0.5 * p[0] * xrot ** 2 + 0.5 * p[1] * yrot ** 2 + p[4]
                return phase

            def fit_fn(p): return defocus_phase_fn(p, xx[to_fit_pix], yy[to_fit_pix])

            xx, yy = np.meshgrid(range(nx), range(ny))
            xx = (xx - nx//2) * dxy
            yy = (yy - ny//2) * dxy

            init_params = [0, 0,
                           np.sum(xx * to_fit_pix) / np.sum(to_fit_pix),
                           np.sum(yy * to_fit_pix) / np.sum(to_fit_pix),
                           np.mean(phase_unwrapped[to_fit_pix]),
                           0]

            lbs = [-np.inf, -np.inf, np.min(xx[to_fit_pix]), np.min(yy[to_fit_pix]), -np.inf, -np.inf]
            ubs = [np.inf, np.inf, np.max(xx[to_fit_pix]), np.max(yy[to_fit_pix]), np.inf, np.inf]

            fit_data = phase_unwrapped[to_fit_pix]
            # fit_data = np.concatenate((phase_unwrapped[to_fit_pix], amp[to_fit_pix]))

            results = fit.fit_model(fit_data,
                                    fit_fn,
                                    init_params=init_params,
                                    bounds=(lbs, ubs))
            fp = results["fit_params"]
            # radius of curvature in um
            rx = k / fp[0]
            ry = k / fp[1]

            phase_fit_plot = defocus_phase_fn(results["fit_params"], xx, yy)
            phase_fit_plot[np.logical_not(to_fit_pix)] = np.nan

            layer_name_fit = f"phase fit"
            try:
                preview_layer = self.viewer.layers[layer_name_fit]
                preview_layer.data = phase_fit_plot
            except KeyError:
                lims = [-2*np.pi, 2*np.pi]

                preview_layer = self.viewer.add_image(phase_fit_plot, name=layer_name_fit, translate=[2*ny, 0],
                                                      colormap="twilight_shifted", contrast_limits=lims)


            pts_layer_name = f"phase fit center"
            shapes_layer_name = f"shape fit center"


            # todo: text layer not working
            text = {'string': 'Rx={rx:.1f}mm (red)\nRy={ry:.1f}mm (blue)', #, Ry={rx:.1f}mm',
                    'size': 20,
                    'color': 'red',
                    'translation': np.array([-10, -nx//2]),
                    }

            pts = np.array([fp[3] / dxy + ny // 2 + 2*ny, fp[2] / dxy + nx // 2])
            features = {"rx": np.array([rx / 1e3]), "ry": np.array([ry / 1e3])}

            theta = fp[5]
            length = nx//10
            shapes = [np.array([[pts[0], pts[1]], [pts[0] + length * np.cos(theta), pts[1] + length * np.sin(theta)]]),
                      np.array([[pts[0], pts[1]], [pts[0] - length * np.sin(theta), pts[1] + length * np.cos(theta)]])
                      ]

            try:
                point_layer = self.viewer.layers[pts_layer_name]
                point_layer.data = pts
                point_layer.features = features
            except KeyError:
                point_layer = self.viewer.add_points(pts, name=pts_layer_name,
                                                     features=features,
                                                     text=text,
                                                     symbol="disc", opacity=1, face_color=[0, 0, 0, 0], edge_color="red", size=10,
                                                     )

            try:
                shape_layer = self.viewer.layers[shapes_layer_name]
                shape_layer.data = shapes
            except KeyError:
                shape_layer = self.viewer.add_shapes(shapes, name=shapes_layer_name,
                                                     shape_type="line", edge_width=10,
                                                     edge_color=["red", "blue"])

    def update_max_min(self, event=None):

        if self.tabWidget.currentIndex() != 0:
            return

        min_max_txt = ""

        for layer in self.viewer.layers.selection:

            if isinstance(layer, napari.layers.Image) and layer.visible:

                col = layer.colormap.name

                if col not in QColor.colorNames():
                    col = "gray"

                # min and max of current slice
                min_max_show = tuple(layer._calc_data_range(mode="slice"))
                min_max_txt += f'<font color="{col}">{min_max_show}</font>'

        self.max_min_val_label.setText(min_max_txt)

    def snap(self):
        self.stop_live()

        if self.daq_shutter_checkBox.isChecked():
            self.daq.set_digital_lines_by_name(np.array([1], dtype=np.uint8), ["sim_shutter"])

        self._mmc_cam.snapImage()
        img = self._mmc_cam.getImage()

        if self.daq_shutter_checkBox.isChecked():
            self.daq.set_digital_lines_by_name(np.array([0], dtype=np.uint8), ["sim_shutter"])

        self.update_viewer(img)

    def start_live(self):

        self._mmc_cam.startContinuousSequenceAcquisition(self.exp_spinBox.value())
        self.streaming_timer = QTimer()
        self.streaming_timer.timeout.connect(self.update_viewer)

        if self.daq_shutter_checkBox.isChecked():
            self.daq.set_digital_lines_by_name(np.array([1], dtype=np.uint8), ["sim_shutter"])

        self.streaming_timer.start(int(self.exp_spinBox.value()))
        self.live_Button.setText("Stop")

    def stop_live(self):
        self._mmc_cam.stopSequenceAcquisition()

        if self.daq_shutter_checkBox.isChecked():
            self.daq.set_digital_lines_by_name(np.array([0], dtype=np.uint8), ["sim_shutter"])


        if self.streaming_timer is not None:
            self.streaming_timer.stop()
            self.streaming_timer = None
        self.live_Button.setText("Live")
        self.live_Button.setIcon(CAM_ICON)

    def toggle_live(self, event=None):
        if self.streaming_timer is None:

            ch_group = self._mmc.getOrGuessChannelGroup()
            if ch_group != []:
                self._mmc.setConfig(ch_group, self.snap_channel_comboBox.currentText())

            self.start_live()
            self.live_Button.setIcon(CAM_STOP_ICON)
        else:
            self.stop_live()
            self.live_Button.setIcon(CAM_ICON)

    def set_channel_and_mode(self):
        # get info from boxes
        channel = self.channel_comboBox.currentText()
        mode = self.mode_comboBox.currentText()

        # get daq values for channel
        self.daq.set_preset(channel)

        # if using shutter, make sure it is closed
        if self.daq_shutter_checkBox.isChecked():
            self.daq.set_digital_lines_by_name(np.array([0], dtype=np.uint8), ["sim_shutter"])

        # time to display DMD pattern before moving to the next
        frame_time_us = int(np.round(self.pattern_time_SpinBox.value() * 1000))

        # set dmd
        self.dmd.program_dmd_seq(mode, channel,
                                 nrepeats=1,
                                 noff_before=0,
                                 noff_after=0,
                                 exp_time_us=frame_time_us,
                                 triggered=False,
                                 clear_pattern_after_trigger=False,
                                 verbose=True)

    def _set_dmd_pattern_index(self):
        pic_ind = self.pic_index_spinBox.value()
        bit_ind = self.bit_index_spinBox.value()
        self.dmd.start_stop_sequence('stop')

        self.dmd.set_pattern_sequence([pic_ind], [bit_ind], 105, 0,
                                      triggered=False,
                                      clear_pattern_after_trigger=False,
                                      bit_depth=1,
                                      num_repeats=0,
                                      mode='pre-stored')

        if self.dmd_snap_checkBox.isChecked():
            self.snap()

    # add, remove, clear DAQ channel table
    def add_channel(self):
        digital_channels = list(self.daq.digital_line_names.keys())
        analog_channels = list(self.daq.analog_line_names.keys())

        # add channel
        idx = self.daq_channel_tableWidget.rowCount()
        self.daq_channel_tableWidget.insertRow(idx)

        # create a combo_box for channels in the table
        self.daq_channel_comboBox = QtW.QComboBox(self)
        self.daq_value_spinBox = QtW.QDoubleSpinBox(self)

        pks = digital_channels + analog_channels
        self.daq_channel_comboBox.addItems(pks)

        self.daq_channel_tableWidget.setCellWidget(idx, 0, self.daq_channel_comboBox)
        self.daq_channel_tableWidget.setCellWidget(idx, 1, self.daq_value_spinBox)

        self.channel_comboBox.currentTextChanged.connect(self._on_channel_changed)

        # call function to make sure updated
        self._on_channel_changed()

    def _on_channel_changed(self):
        digital_channels = list(self.daq.digital_line_names.keys())
        analog_channels = list(self.daq.analog_line_names.keys())

        for ii in range(self.daq_channel_tableWidget.rowCount()):
            ch = self.daq_channel_tableWidget.cellWidget(ii, 0).currentText()

            if ch in digital_channels:
                self.daq_channel_tableWidget.cellWidget(ii, 1).setDecimals(0)
                self.daq_channel_tableWidget.cellWidget(ii, 1).setSingleStep(1)
                self.daq_channel_tableWidget.cellWidget(ii, 1).setMinimum(0)
                self.daq_channel_tableWidget.cellWidget(ii, 1).setMaximum(1)

                index = self.daq.digital_line_names[ch]
                last_val_known = self.daq.last_known_digital_val[index]
                self.daq_channel_tableWidget.cellWidget(ii, 1).setValue(last_val_known)

            elif ch in analog_channels:
                self.daq_channel_tableWidget.cellWidget(ii, 1).setDecimals(3)
                self.daq_channel_tableWidget.cellWidget(ii, 1).setSingleStep(0.01)
                self.daq_channel_tableWidget.cellWidget(ii, 1).setMinimum(-10.)
                self.daq_channel_tableWidget.cellWidget(ii, 1).setMaximum(10.)

                index = self.daq.analog_line_names[ch]
                last_val_known = self.daq.last_known_analog_val[index]
                self.daq_channel_tableWidget.cellWidget(ii, 1).setValue(last_val_known)
            else:
                raise ValueError(f"channel '{ch:s}' was not present in analog or digital channels")

            # if update immediately, connect
            if self.daq_update_immediately_checkBox.isChecked():
                self.daq_channel_tableWidget.cellWidget(ii, 1).valueChanged.connect(self._on_daq_setting_change)
                self.daq_channel_tableWidget.cellWidget(ii, 1).setKeyboardTracking(False)
            else:
                # disconnect will fail if not connected to anything
                try:
                    self.daq_channel_tableWidget.cellWidget(ii, 1).valueChanged.disconnect()
                except TypeError:
                    pass

    def remove_channel(self):
        # remove selected position
        rows = {r.row() for r in self.daq_channel_tableWidget.selectedIndexes()}
        for idx in sorted(rows, reverse=True):
            self.daq_channel_tableWidget.removeRow(idx)

    def clear_channel(self):
        # clear all positions
        self.daq_channel_tableWidget.clearContents()
        self.daq_channel_tableWidget.setRowCount(0)


    def _on_daq_setting_change(self):
        # grab analog/digital lines from channels
        digital_channels = list(self.daq.digital_line_names.keys())
        analog_channels = list(self.daq.analog_line_names.keys())

        dig_ch_now = {}
        an_ch_now = {}
        for ii in range(self.daq_channel_tableWidget.rowCount()):
            ch_name = self.daq_channel_tableWidget.cellWidget(ii, 0).currentText()
            val = self.daq_channel_tableWidget.cellWidget(ii, 1).value()

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

        if self.daq_snap_checkBox.isChecked():
            self.snap()