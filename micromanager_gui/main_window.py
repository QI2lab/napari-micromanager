from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import napari
import numpy as np
from pymmcore_plus import CMMCorePlus, RemoteMMCore, DeviceType
from qtpy import QtWidgets as QtW
from qtpy import uic
from qtpy.QtCore import QSize, QTimer
from qtpy.QtGui import QColor, QIcon

from ._illumination import IlluminationDialog
from ._saving import save_sequence
from ._util import blockSignals, event_indices, extend_array_for_index
from .explore_sample import ExploreSample
from .sim_odt_widget import SimOdtWidget
from .multid_widget import MultiDWidget, SequenceMeta
from .prop_browser import PropBrowser

if TYPE_CHECKING:
    import napari.layers
    import napari.viewer
    import useq

# dmd and daq control
import mcsim.expt_ctrl.dlp6500 as dmd_ctrl
import mcsim.expt_ctrl.set_dmd_sim as dmd_map
import mcsim.expt_ctrl.daq
import mcsim.expt_ctrl.expt_map as daq_map
from mcsim.expt_ctrl.setup_optotune_mre2 import initialize_mre2
import mcsim.analysis.analysis_tools as mctools
from skimage.restoration import unwrap_phase
from numpy import fft


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
    image_proc_mode_comboBox: QtW.QComboBox
    fx_doubleSpinBox: QtW.QDoubleSpinBox
    fy_doubleSpinBox: QtW.QDoubleSpinBox
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

    pic_index_spinBox: QtW.QSpinBox
    bit_index_spinBox: QtW.QSpinBox
    set_dmd_pattern_index_pushButton: QtW.QPushButton

    def setup_ui(self):
        uic.loadUi(self.UI_FILE, self)  # load QtDesigner .ui file

        # set some defaults
        # self.cfg_LineEdit.setText("demo")
        self.cfg_LineEdit.setText(r"C:/Users/q2ilab/Documents/mcsim_private/mcSIM/mcsim/expt_ctrl/sim_odt_nidaq_ham_c1.cfg")
        self.cfg2_LineEdit.setText(r"C:/Users/q2ilab/Documents/mcsim_private/mcSIM/mcsim/expt_ctrl/sim_odt_nidaq_ham_c2.cfg")

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
    def __init__(self, viewer: napari.viewer.Viewer, remote=True):
        super().__init__()
        self.setup_ui()

        self.viewer = viewer
        self.streaming_timer = None

        # create connection to mmcore server or process-local variant
        # create two cores, the first is the main core, the second only runs the second camera
        self._mmcores = [RemoteMMCore(port=54333) if remote else CMMCorePlus(),
                         RemoteMMCore(port=54334) if remote else CMMCorePlus()]

        self._mmc = self._mmcores[0]
        self._mmc_cam = self._mmcores[1]

        # connect to DMD
        self.dmd = dmd_ctrl.dlp6500win(debug=True)
        # connect to daq
        self.daq = mcsim.expt_ctrl.daq.nidaq()

        # tab widgets
        self.sim_odt_acq = SimOdtWidget(self._mmcores, self.daq, self.dmd, self.viewer)
        self.mda = MultiDWidget(self._mmc)
        self.explorer = ExploreSample(self.viewer, self._mmc)
        self.tabWidget.addTab(self.sim_odt_acq, "SIM/ODT Acquisition")
        self.tabWidget.addTab(self.mda, "Multi-D Acquisition")
        self.tabWidget.addTab(self.explorer, "Sample Explorer")

        # connect mmcore signals
        sig = self._mmc.events

        # note: don't use lambdas with closures on `self`, since the connection
        # to core may outlive the lifetime of this particular widget.
        sig.sequenceStarted.connect(self._on_mda_started)
        sig.sequenceFinished.connect(self._on_mda_finished)
        sig.systemConfigurationLoaded.connect(self._refresh_options)
        sig.XYStagePositionChanged.connect(self._on_xy_stage_position_changed)
        sig.stagePositionChanged.connect(self._on_stage_position_changed)
        sig.exposureChanged.connect(self._on_exp_change)
        sig.frameReady.connect(self._on_mda_frame)
        sig.channelGroupChanged.connect(self._refresh_channel_list)
        sig.configSet.connect(self._on_config_set)

        # connect buttons
        self.load_cfg_Button.clicked.connect(self.load_cfg)
        self.browse_cfg_Button.clicked.connect(self.browse_cfg)
        self.load_cfg2_Button.clicked.connect(self.load_cfg2)
        self.browse_cfg2_Button.clicked.connect(self.browse_cfg2)
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

        # populate channel combo box
        pks = list(mcsim.expt_ctrl.expt_map.presets.keys())
        self.channel_comboBox.addItems(pks)
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

        # connect spinboxes
        self.exp_spinBox.valueChanged.connect(self._update_exp)
        self.exp_spinBox.setKeyboardTracking(False)

        # self.fx_doubleSpinBox.setValue(2.212)
        # self.fy_doubleSpinBox.setValue(1.445)
        self.fx_doubleSpinBox.setValue(1600.)
        self.fy_doubleSpinBox.setValue(1400.)

        # refresh options in case a config is already loaded by another remote
        self._refresh_options()

        self.viewer.layers.events.connect(self.update_max_min)
        self.viewer.layers.selection.events.active.connect(self.update_max_min)
        self.viewer.dims.events.current_step.connect(self.update_max_min)

        # todo: find a better place to put initialization code ... maybe should have a mechanism for running a startup script...
        initialize_mre2()

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

    def _on_mda_started(self, sequence: useq.MDASequence):
        """ "create temp folder and block gui when mda starts."""
        self._set_enabled(False)

    def _on_mda_frame(self, image: np.ndarray, event: useq.MDAEvent):
        meta = self.mda.SEQUENCE_META.get(event.sequence) or SequenceMeta()

        if meta.mode != "mda":
            return

        # pick layer name
        file_name = meta.file_name if meta.should_save else "Exp"
        channelstr = (
            f"[{event.channel.config}_idx{event.index['c']}]_"
            if meta.split_channels
            else ""
        )
        layer_name = f"{file_name}_{channelstr}{event.sequence.uid}"

        try:  # see if we already have a layer with this sequence
            layer = self.viewer.layers[layer_name]

            # get indices of new image
            im_idx = tuple(
                event.index[k]
                for k in event_indices(event)
                if not (meta.split_channels and k == "c")
            )

            # make sure array shape contains im_idx, or pad with zeros
            new_array = extend_array_for_index(layer.data, im_idx)
            # add the incoming index at the appropriate index
            new_array[im_idx] = image
            # set layer data
            layer.data = new_array
            for a, v in enumerate(im_idx):
                self.viewer.dims.set_point(a, v)

        except KeyError:  # add the new layer to the viewer
            seq = event.sequence
            _image = image[(np.newaxis,) * len(seq.shape)]
            layer = self.viewer.add_image(_image, name=layer_name, blending="additive")

            # dimensions labels
            labels = [i for i in seq.axis_order if i in event.index] + ["y", "x"]
            self.viewer.dims.axis_labels = labels

            # add metadata to layer
            layer.metadata["useq_sequence"] = seq
            layer.metadata["uid"] = seq.uid
            # storing event.index in addition to channel.config because it's
            # possible to have two of the same channel in one sequence.
            layer.metadata["ch_id"] = f'{event.channel.config}_idx{event.index["c"]}'

    def _on_mda_finished(self, sequence: useq.MDASequence):
        """Save layer and add increment to save name."""
        meta = self.mda.SEQUENCE_META.pop(sequence, SequenceMeta())
        save_sequence(sequence, self.viewer.layers, meta)
        # reactivate gui when mda finishes.
        self._set_enabled(True)

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
        self.load_cfg_Button.setEnabled(False)
        print("loading", self.cfg_LineEdit.text())
        self._mmc.loadSystemConfiguration(self.cfg_LineEdit.text())

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
        self.load_cfg2_Button.setEnabled(False)
        print("loading", self.cfg2_LineEdit.text())
        self._mmcores[1].loadSystemConfiguration(self.cfg2_LineEdit.text())

        try:
            # turn off prime BSI express crazy speckle correction
            cam = self._mmcores[1].getCameraDevice()
            self._mmcores[1].setProperty(cam, 'PP  1   ENABLED', 'No')
            self._mmcores[1].setProperty(cam, 'PP  2   ENABLED', 'No')
            self._mmcores[1].setProperty(cam, 'PP  3   ENABLED', 'No')
            self._mmcores[1].setProperty(cam, 'PP  4   ENABLED', 'No')
        except:
            print("error disabling photometrics camera despeckle correction")

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

    # def _refresh_camera_list(self):
    #     devs = self._mmc.getLoadedDevices()
    #     if devs:
    #         dtypes = [self._mmc.getDeviceType(d) for d in devs]
    #
    #         camera_list = [d for d, dt in zip(devs, dtypes) if dt == DeviceType.CameraDevice]
    #
    #         self.snap_camera_comboBox.clear()
    #         self.snap_camera_comboBox.addItems(camera_list)
    #         self.snap_camera_comboBox.setCurrentText(self._mmc.getCameraDevice())

    def _refresh_camera_list(self):
        ncores = len(self._mmcores)

        core_inds = list(range(ncores))
        core_inds = [str(ind) for ind in core_inds]

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
        chan = self.channel_comboBox.currentText()
        modes = list(dmd_map.channel_map[chan].keys())

        self.mode_comboBox.clear()
        self.mode_comboBox.addItems(modes)
        self.mode_comboBox.setCurrentText(modes[0])

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
            pass


    def _on_xy_stage_position_changed(self, name, x, y):
        self.x_lineEdit.setText(f"{x:.1f}")
        self.y_lineEdit.setText(f"{y:.1f}")

    def _on_stage_position_changed(self, name, value):
        if "z" in name.lower():  # hack
            self.z_lineEdit.setText(f"{value:.1f}")

    def stage_x_left(self):
        self._mmc.setRelativeXYPosition(-float(self.xy_step_size_SpinBox.value()), 0.0)
        if self.snap_on_click_xy_checkBox.isChecked():
            self.snap()

    def stage_x_right(self):
        self._mmc.setRelativeXYPosition(float(self.xy_step_size_SpinBox.value()), 0.0)
        if self.snap_on_click_xy_checkBox.isChecked():
            self.snap()

    def stage_y_up(self):
        self._mmc.setRelativeXYPosition(
            0.0,
            float(self.xy_step_size_SpinBox.value()),
        )
        if self.snap_on_click_xy_checkBox.isChecked():
            self.snap()

    def stage_y_down(self):
        self._mmc.setRelativeXYPosition(
            0.0,
            -float(self.xy_step_size_SpinBox.value()),
        )
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

        # only scale the visible layer, which is the last layer which is visible
        for l in reversed(self.viewer.layers):
            if l.visible:
                vmin = np.percentile(l.data, min_p)
                vmax = np.percentile(l.data, max_p)

                l.contrast_limits = (vmin, vmax)
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

        # get image processing mode
        mode = self.image_proc_mode_comboBox.currentText()

        # different layers for different cameras and modes
        layer_name = "%s %s" % (self._mmc_cam.getCameraDevice(), mode)

        if mode == "normal":
            pass
        elif mode == "fft":
            data = np.abs(fft.fftshift(fft.fft2(fft.ifftshift(data))))
            # dxy = 6.5 / 50
            # ny, nx = data.shape
            # scale = (1 / (ny * dxy), 1 / (ny * dxy))
        elif mode == "hologram" or mode == "hologram unwrapped":
            ft = fft.fftshift(fft.fft2(fft.ifftshift(data)))
            ny, nx = ft.shape

            # todo: could display in some nicer way...but...
            dxy = 6.5 / 50
            fx = fft.fftshift(fft.fftfreq(nx, dxy))
            fy = fft.fftshift(fft.fftfreq(ny, dxy))
            fxfx, fyfy = np.meshgrid(fx, fy)
            ff = np.sqrt(fxfx**2 + fyfy**2)
            dfx = fx[1] - fx[0]
            dfy = fy[1] - fy[0]
            fmax = 0.55 / 0.785

            # convert from pixels in image to real units
            holo_frq = np.array([dfx * (self.fx_doubleSpinBox.value() - nx // 2),
                                 dfy * (self.fy_doubleSpinBox.value() - ny // 2)])

            ft_xlated = mctools.translate_ft(ft, -holo_frq, drs=(dxy, dxy), use_gpu=False)
            ft_xlated[ff > fmax] = 0

            im = fft.fftshift(fft.ifft2(fft.ifftshift(ft_xlated)))

            # todo: could also display amplitude data...
            if mode == "hologram":
                data = np.angle(im)
            else:
                data = unwrap_phase(np.angle(im))
                # add center back ...
                data -= np.mean(data[ny//2 - 2: ny//2 + 2, nx//2 - 2: nx//2 + 2])

        else:
            raise ValueError()

        try:
            preview_layer = self.viewer.layers[layer_name]
            preview_layer.data = data
        except KeyError:
            preview_layer = self.viewer.add_image(data, name=layer_name)

        self.update_max_min()

        if self.streaming_timer is None:
            self.viewer.reset_view()

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
        self._mmc_cam.snapImage()
        self.update_viewer(self._mmc_cam.getImage())

    def start_live(self):
        self._mmc_cam.startContinuousSequenceAcquisition(self.exp_spinBox.value())
        self.streaming_timer = QTimer()
        self.streaming_timer.timeout.connect(self.update_viewer)
        self.streaming_timer.start(int(self.exp_spinBox.value()))
        self.live_Button.setText("Stop")

    def stop_live(self):
        self._mmc_cam.stopSequenceAcquisition()
        if self.streaming_timer is not None:
            self.streaming_timer.stop()
            self.streaming_timer = None
        self.live_Button.setText("Live")
        self.live_Button.setIcon(CAM_ICON)

    def toggle_live(self, event=None):
        if self.streaming_timer is None:

            ch_group = self._mmc.getOrGuessChannelGroup()
            if ch_group is not None:
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
        preset = daq_map.presets[channel]
        digital_array, analog_array = daq_map.preset_to_array(preset, mcsim.expt_ctrl.expt_map.daq_do_map,
                                                              mcsim.expt_ctrl.expt_map.daq_ao_map,
                                                              n_digital_channels=self.daq.n_digital_lines,
                                                              n_analog_channels=self.daq.n_analog_lines)

        # set daq
        self.daq.set_analog_once(analog_array)
        self.daq.set_digital_once(digital_array)

        # set dmd
        dmd_map.program_dmd_seq(self.dmd, mode, channel, 1, 0, False, None, False, True)

    def _set_dmd_pattern_index(self):
        pic_ind = self.pic_index_spinBox.value()
        bit_ind = self.bit_index_spinBox.value()
        self.dmd.start_stop_sequence('stop')

        self.dmd.set_pattern_sequence([pic_ind], [bit_ind], 105, 0, triggered=False,
                                 clear_pattern_after_trigger=False, bit_depth=1, num_repeats=0, mode='pre-stored')