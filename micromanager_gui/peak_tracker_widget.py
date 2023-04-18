from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import napari.layers.image.image
from qtpy import QtWidgets as QtW
from qtpy import uic
from qtpy.QtCore import QSize, Qt, QTimer
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
import dask.array as da
import threading
# custom code
from localize_psf import affine, fit, rois
# daq and dmd
import mcsim.expt_ctrl.daq
from mcsim.expt_ctrl.program_sim_odt import get_sim_odt_sequence
from mcsim.expt_ctrl import dlp6500
import mcsim.expt_ctrl.phantom_cam as phc


ICONS = Path(__file__).parent / "icons"
OBJECTIVE_DEVICE = "Objective"
# Once the PR #43 is merged, we pass the objective device to this variable


class _PeakTrackerDUI:
    UI_FILE = str(Path(__file__).parent / "_ui" / "peak_tracker.ui")

    # The UI_FILE above contains these objects:

    # saving
    save_groupBox: QtW.QGroupBox
    fname_lineEdit: QtW.QLineEdit
    dir_lineEdit: QtW.QLineEdit
    browse_save_Button: QtW.QPushButton

    # selection
    fit_roi_tableWidget: QtW.QTableWidget
    add_pushButton: QtW.QPushButton
    remove_pushButton: QtW.QPushButton
    clear_pushButton: QtW.QPushButton
    model_comboBox: QtW.QComboBox
    layer_comboBox: QtW.QComboBox
    update_pushButton: QtW.QPushButton
    roi_checkBox: QtW.QCheckBox
    compare_checkBox: QtW.QCheckBox

    # running
    show_Button: QtW.QPushButton
    run_Button: QtW.QPushButton
    pause_Button: QtW.QPushButton

    def setup_ui(self):
        uic.loadUi(self.UI_FILE, self)  # load QtDesigner .ui file


class PeakTrackerWidget(QtW.QWidget, _PeakTrackerDUI):

    def __init__(self,
                 mmcores: list[RemoteMMCore],
                 daq: mcsim.expt_ctrl.daq.daq, dmd: dlp6500,
                 viewer,
                 phcam=None,
                 parent=None,
                 configuration=None):

        self._mmcores = mmcores
        self._mmc = self._mmcores[0]
        self.daq = daq
        self.dmd = dmd
        self.phcam = phcam
        self.configuration = configuration

        self.viewer = viewer
        super().__init__(parent)
        self.setup_ui()

        # connect buttons
        self.browse_save_Button.clicked.connect(self._set_save_dir)
        self.add_pushButton.clicked.connect(self._add_roi)
        self.remove_pushButton.clicked.connect(self._remove_roi)
        self.clear_pushButton.clicked.connect(self._clear_roi)
        self.run_Button.clicked.connect(self._on_run_clicked)
        self.pause_Button.clicked.connect(self._on_stop_clicked)
        self.show_Button.clicked.connect(self._show_dataset)

        # todo: want combobox to update when it is clicked on, but couldn't figure out
        self.update_pushButton.clicked.connect(self._refresh)
        # self.layer_comboBox.mousePressEvent.connect(self._refresh_layers)

        # fit models
        self.fit_models = {"gaussian": fit.gauss2d(),
                           }

        # self.fit_params_lock = threading.Lock()
        self.fit_params = None
        # self.run_fit_thread = threading.Event()
        self.streaming_timer = None

    def _refresh(self):
        self._refresh_layers()
        self._refresh_models()

    def _refresh_layers(self):
        layers = [l.name for l in self.viewer.layers if isinstance(l, napari.layers.image.image.Image)]
        self.layer_comboBox.clear()
        self.layer_comboBox.addItems(layers)

    def _refresh_models(self):
        models = self.fit_models.keys()
        self.model_comboBox.clear()
        self.model_comboBox.addItems(models)

    def _fit_peaks(self):

        # model info
        try:
            model = self.fit_models[self.model_comboBox.currentText()]
        except KeyError as e:
            print(e)
            return

        ind_x = [ii for ii, name in enumerate(model.parameter_names) if name == "cx"][0]
        ind_y = [ii for ii, name in enumerate(model.parameter_names) if name == "cy"][0]

        # grab data
        layer_name = self.layer_comboBox.currentText()
        layer_list = [l for l in self.viewer.layers if l.name == layer_name]

        if layer_list == []:
            return

        layer = layer_list[0]
        img = layer.data
        if img.ndim != 2:
            raise ValueError(f"img.ndim={img.ndim:d}, but only 2D images are supported by _fit_peak()")

        xx, yy = np.meshgrid(range(img.shape[0]), range(img.shape[1]))

        # get rois
        nroi = self.fit_roi_tableWidget.rowCount()
        roi_list = []
        fit_params_roi = []
        for ii in range(nroi):
            # grab variables
            # cx = self.cx_doubleSpinBox.value()
            # cy = self.cy_doubleSpinBox.value()
            # nroi = self.roi_size_SpinBox.value()
            cy = self.fit_roi_tableWidget.cellWidget(ii, 0).value()
            cx = self.fit_roi_tableWidget.cellWidget(ii, 1).value()
            size_roi = self.fit_roi_tableWidget.cellWidget(ii, 2).value()

            # construct ROI
            roi = rois.get_centered_roi([cy, cx], [size_roi, size_roi], min_vals=[0, 0])

            # cut rois
            img_roi = np.abs(rois.cut_roi(roi, img))
            xx_roi = rois.cut_roi(roi, xx)
            yy_roi = rois.cut_roi(roi, yy)

            # do fitting
            # if we have a previous fit, use that value as default
            lbs = [None] * len(model.parameter_names)
            ubs = [None] * len(model.parameter_names)
            lbs[ind_x] = xx_roi.min()
            ubs[ind_x] = xx_roi.max()
            lbs[ind_y] = yy_roi.min()
            ubs[ind_y] = yy_roi.max()

            if self.fit_params[ii] == []:
                init_params = None
            else:
                last_fps_in_lbs = [lb is None or ip >= lb for ip, lb in zip(self.fit_params[ii][-1], lbs)]
                last_fps_in_ubs = [b is None or ip >= b for ip, b in zip(self.fit_params[ii][-1], ubs)]

                if np.all(last_fps_in_lbs) and np.all(last_fps_in_ubs):
                    init_params = self.fit_params[ii][-1]
                else:
                    init_params = None

            fit_results = model.fit(img_roi,
                                    (yy_roi, xx_roi),
                                    init_params=init_params,
                                    fixed_params=None,
                                    bounds=(lbs, ubs)
                                    )
            fit_params = fit_results["fit_params"]

            roi_list.append(roi)
            fit_params_roi.append(fit_params)
            self.fit_params[ii].append(fit_params)


        # plot fit point
        plot_layer_name = layer_name + " fit"
        plot_layer = [l for l in self.viewer.layers if l.name == plot_layer_name]
        if plot_layer == []:
            plot_layer = None
        else:
            plot_layer = plot_layer[0]


        plot_data = np.array([[fp[-1][ind_y], fp[-1][ind_x]] for fp in self.fit_params])

        features = dict(zip(model.parameter_names,
                            [np.array([fp[-1][ii] for fp in self.fit_params]) for ii in range(len(model.parameter_names))]
                            )
                        )

        if self.compare_checkBox.checkState():
            center = np.abs(img[img.shape[0] // 2, img.shape[1] // 2])
            features.update({"A/center": np.array([fp[-1][0] / center for fp in self.fit_params])})

        if plot_layer is not None:
            plot_layer.data = plot_data
            plot_layer.features = features
        else:
            self.viewer.add_points(plot_data,
                                   face_color=[0, 0, 0, 0],
                                   edge_color=[1, 0, 0, 1],
                                   features=features,
                                   text={'string': "; ".join([f'{k:s}={{{k:s}:.3g}}' for k in features.keys()]),
                                         'size': 10,
                                         'color': 'red',
                                         'translation': np.array([-7, 0]),
                                        },
                                   size=10,
                                   name=plot_layer_name,
                                   translate=layer.translate,
                                   affine=layer.affine)

        #
        if self.roi_checkBox.checkState():
            roi_layer_name = layer_name + " roi"
            roi_layer = [l for l in self.viewer.layers if l.name == roi_layer_name]

            if roi_layer == []:
                roi_layer = None
            else:
                roi_layer = roi_layer[0]

            roi_data = np.array([
                                 [[roi[0] - 1, roi[2] - 1],
                                  [roi[0] - 1, roi[3]],
                                  [roi[1], roi[3]],
                                  [roi[1], roi[2] - 1]
                                  ] for roi in roi_list
                                 ]
                                )
            if roi_layer is not None:
                roi_layer.data = roi_data
            else:
                self.viewer.add_shapes(roi_data,
                                       shape_type="polygon",
                                       edge_width=1,
                                       face_color=[0, 0, 0, 0],
                                       edge_color=[1, 0, 0, 1],
                                       name=roi_layer_name,
                                       translate=layer.translate,
                                       affine=layer.affine
                                       )

    def _add_roi(self):
        idx = self.fit_roi_tableWidget.rowCount()
        self.fit_roi_tableWidget.insertRow(idx)

        # create a combo_box for channels in the table
        cy_spinBox = QtW.QSpinBox(self)
        cy_spinBox.setMinimum(0)
        cy_spinBox.setMaximum(1e5)

        cx_spinBox = QtW.QSpinBox(self)
        cx_spinBox.setMinimum(0)
        cx_spinBox.setMaximum(1e5)

        size_spinBox = QtW.QSpinBox(self)
        size_spinBox.setMinimum(0)
        size_spinBox.setMaximum(1e5)
        size_spinBox.setValue(10)

        # create combo_boxes in table
        self.fit_roi_tableWidget.setCellWidget(idx, 0, cy_spinBox)
        self.fit_roi_tableWidget.setCellWidget(idx, 1, cx_spinBox)
        self.fit_roi_tableWidget.setCellWidget(idx, 2, size_spinBox)

    def _remove_roi(self):
        # remove selected position
        rows = {r.row() for r in self.fit_roi_tableWidget.selectedIndexes()}
        for idx in sorted(rows, reverse=True):
            self.channel_tableWidget.removeRow(idx)

    def _clear_roi(self):
        self.fit_roi_tableWidget.clearContents()
        self.fit_roi_tableWidget.setRowCount(0)

    def _clear(self):
        self._on_stop_clicked()
        self.fit_params = None

    def _on_run_clicked(self):
        nrois = self.fit_roi_tableWidget.rowCount()
        self.fit_params = [[] for _ in range(nrois)]

        self.streaming_timer = QTimer()
        self.streaming_timer.timeout.connect(self._fit_peaks)
        self.streaming_timer.start(100)

    def _on_stop_clicked(self):
        try:
            self.streaming_timer.stop()
            self.streaming_timer = None
        except AttributeError:
            pass

    def _show_dataset(self):
      pass

    def _set_save_dir(self):
        pass


if __name__ == "__main__":
    from qtpy.QtWidgets import QApplication

    app = QApplication([])
    window = PeakTrackerWidget()
    window.show()
    app.exec_()
