from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.lines
import napari.layers.image.image
from qtpy import QtWidgets as QtW
from qtpy import uic
from qtpy.QtCore import QSize, Qt, QTimer, QPoint
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
import mcsim.analysis.analysis_tools as tools
from mcsim.expt_ctrl import dlp6500
import matplotlib
import matplotlib.pyplot as plt


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
    mouse_select_pushButton: QtW.QPushButton
    fit_roi_tableWidget: QtW.QTableWidget
    add_pushButton: QtW.QPushButton
    remove_pushButton: QtW.QPushButton
    clear_pushButton: QtW.QPushButton
    model_comboBox: QtW.QComboBox
    layer_comboBox: QtW.QComboBox
    update_pushButton: QtW.QPushButton
    roi_checkBox: QtW.QCheckBox
    compare_checkBox: QtW.QCheckBox
    plot_trace_checkBox: QtW.QCheckBox
    show_on_napari_checkBox: QtW.QCheckBox

    memory_time_doubleSpinBox: QtW.QDoubleSpinBox
    update_rate_doubleSpinBox: QtW.QDoubleSpinBox

    # display
    # tracking_tableWidget: QtW.QTableWidget

    # running
    show_Button: QtW.QPushButton
    run_Button: QtW.QPushButton
    pause_Button: QtW.QPushButton

    def setup_ui(self):
        uic.loadUi(self.UI_FILE, self)  # load QtDesigner .ui file


class PeakTrackerWidget(QtW.QWidget, _PeakTrackerDUI):

    def __init__(self,
                 mmcores: list[RemoteMMCore],
                 daq: mcsim.expt_ctrl.daq.daq,
                 dmd: dlp6500,
                 viewer,
                 phcam=None,
                 parent=None,
                 configuration=None):

        matplotlib.use("Qt5Agg")

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
        self.mouse_select_pushButton.clicked.connect(self._select_rois_with_mouse)
        self.add_pushButton.clicked.connect(self._add_roi)
        self.remove_pushButton.clicked.connect(self._remove_roi)
        self.clear_pushButton.clicked.connect(self._clear_roi)
        self.run_Button.clicked.connect(self._on_run_clicked)
        self.pause_Button.clicked.connect(self._on_stop_clicked)
        self.show_Button.clicked.connect(self._show_dataset)

        # todo: want combobox to update when it is clicked on, but couldn't figure out
        self.update_pushButton.clicked.connect(self._refresh)
        self.compare_checkBox.setChecked(True)
        self.roi_checkBox.setChecked(True)
        self.plot_trace_checkBox.setChecked(True)
        self.show_on_napari_checkBox.setChecked(False)
        self.memory_time_doubleSpinBox.setValue(180.)
        self.update_rate_doubleSpinBox.setValue(100.)
        # self.layer_comboBox.mousePressEvent.connect(self._refresh_layers)

        # fit models
        self.fit_models = {"gaussian": fit.gauss2d(),
                           }
        self.fit_params = None
        self.phases = None
        self.fit_times = None
        self.streaming_timer = None

        # matplotlib figure
        self._model = None
        self._figure = None

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


        ind_x = [ii for ii, name in enumerate(self._model.parameter_names) if name == "cx"][0]
        ind_y = [ii for ii, name in enumerate(self._model.parameter_names) if name == "cy"][0]

        # grab data
        layer_name = self.layer_comboBox.currentText()
        layer_list = [l for l in self.viewer.layers if l.name == layer_name]

        if layer_list == []:
            return

        layer = layer_list[0]
        img = layer.data
        if img.ndim != 2:
            raise ValueError(f"img.ndim={img.ndim:d}, but only 2D images are supported by _fit_peak()")

        # check for phase layer
        phase_layer_name = layer_name + " phase"
        phase_layer_list = [l for l in self.viewer.layers if l.name == phase_layer_name]

        use_phase = True
        if phase_layer_list == []:
            use_phase = False

        # coordinates
        xx, yy = np.meshgrid(range(img.shape[0]), range(img.shape[1]))

        # ######################
        # fit ROIs
        # ######################
        self.fit_times = np.concatenate((self.fit_times, np.array([time.time()]))) # time since epoch in seconds

        nroi = self.fit_roi_tableWidget.rowCount()
        roi_list = []
        for ii in range(nroi):
            if ii > len(self.fit_params):
                continue

            # grab variables
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
            lbs = [None] * len(self._model.parameter_names)
            ubs = [None] * len(self._model.parameter_names)
            lbs[ind_x] = xx_roi.min()
            ubs[ind_x] = xx_roi.max()
            lbs[ind_y] = yy_roi.min()
            ubs[ind_y] = yy_roi.max()

            if len(self.fit_params[ii]) == 0:
                init_params = None
            else:
                last_fps_in_lbs = [lb is None or ip >= lb for ip, lb in zip(self.fit_params[ii][-1], lbs)]
                last_fps_in_ubs = [b is None or ip >= b for ip, b in zip(self.fit_params[ii][-1], ubs)]

                if np.all(last_fps_in_lbs) and np.all(last_fps_in_ubs):
                    init_params = self.fit_params[ii][-1]
                else:
                    init_params = None

            fit_results = self._model.fit(img_roi,
                                    (yy_roi, xx_roi),
                                    init_params=init_params,
                                    fixed_params=None,
                                    bounds=(lbs, ubs)
                                    )
            fit_params = fit_results["fit_params"]

            roi_list.append(roi)
            self.fit_params[ii] = np.concatenate((self.fit_params[ii], fit_params[None, :]), axis=0)

            # phases
            if use_phase:
                phases_roi = rois.cut_roi(roi, phase_layer_list[0].data)
                try:
                    phase = tools.get_peak_value(phases_roi, xx_roi[0, :], yy_roi[:, 0], np.array([fit_params[ind_x], fit_params[ind_y]]))
                except:
                    phase = np.nan
            else:
                phase = np.nan

            self.phases[ii] = np.concatenate((self.phases[ii], np.array([phase])[None, :]), axis=0)


        # get center value for comparison (really for FFT)
        if self.compare_checkBox.checkState():
            center = np.abs(img[img.shape[0] // 2, img.shape[1] // 2])
            ratios = np.array([fp[-1][0] / center for fp in self.fit_params])
        else:
            ratios = np.array([np.nan for fp in self.fit_params])

        # ######################
        # put data in table
        # ######################
        # todo: runs slow if do this
        # for ii in range(nroi):
        #     for jj in range(model.nparams):
        #         self.tracking_tableWidget.cellWidget(ii, jj).setText(f"{self.fit_params[ii][-1][jj]:.4g}")
        #
        #     self.tracking_tableWidget.cellWidget(ii, model.nparams).setText(f"{ratios[ii]:.4g}")

        # data to be plotted on the screen below
        features = {"A/center": ratios,
                    "roi_index": np.arange(nroi)}
        features.update(dict(zip(self._model.parameter_names,
                            [np.array([fp[-1, ii] for fp in self.fit_params]) for ii in range(len(self._model.parameter_names))]
                            )
                        )
                        )

        if self.show_on_napari_checkBox.isChecked():
            # ######################
            # titles
            # ######################
            plot_layer_name = layer_name + " titles"
            plot_layer = [l for l in self.viewer.layers if l.name == plot_layer_name]
            if plot_layer == []:
                plot_layer = None
            else:
                plot_layer = plot_layer[0]

            plot_data = np.zeros((1, 2))

            ttl_str = [f"{s:15}" for s in list(features.keys())]
            features_titles = {"titles": "; ".join(ttl_str)}

            if plot_layer is not None:
                plot_layer.data = plot_data
                plot_layer.features = features_titles
            else:
                self.viewer.add_points(plot_data,
                                       face_color=[0, 0, 0, 0],
                                       edge_color=[1, 0, 0, 0],
                                       features=features_titles,
                                       text={'string': "{titles:s}",
                                             'size': 15,
                                             'color': 'red',
                                             'translation': np.array([0, 0]),
                                             'anchor': 'upper_left'
                                             },
                                       size=10,
                                       name=plot_layer_name,
                                       translate=layer.translate,
                                       affine=layer.affine)

            # ######################
            # put fit parameters in points at top of screen
            # ######################

            plot_layer_name = layer_name + " print fits"
            plot_layer = [l for l in self.viewer.layers if l.name == plot_layer_name]
            if plot_layer == []:
                plot_layer = None
            else:
                plot_layer = plot_layer[0]

            plot_data = np.zeros((nroi, 2))
            plot_data[:, 0] = (np.arange(nroi) + 1) * 50

            if plot_layer is not None:
                plot_layer.data = plot_data
                plot_layer.features = features
            else:
                self.viewer.add_points(plot_data,
                                       face_color=[0, 0, 0, 0],
                                       edge_color=[1, 0, 0, 0],
                                       features=features,
                                       text={'string': "{roi_index:d}:  " + "; ".join([f'{{{k:s}:+15.2e}}' for k in features.keys()]),
                                             'size': 15,
                                             'color': 'red',
                                             'translation': np.array([0, 0]),
                                             'anchor': 'upper_left'
                                             },
                                       size=10,
                                       name=plot_layer_name,
                                       translate=layer.translate,
                                       affine=layer.affine)

        # ######################
        # plot fit results in napari
        # ######################
        plot_layer_name = layer_name + " fit"
        plot_layer = [l for l in self.viewer.layers if l.name == plot_layer_name]
        if plot_layer == []:
            plot_layer = None
        else:
            plot_layer = plot_layer[0]


        plot_data = np.array([[fp[-1, ind_y], fp[-1, ind_x]] for fp in self.fit_params])

        if plot_layer is not None:
            plot_layer.data = plot_data
            plot_layer.features = features
        else:
            self.viewer.add_points(plot_data,
                                   face_color=[0, 0, 0, 0],
                                   edge_color=[1, 0, 0, 1],
                                   features=features,
                                   text={#'string': "; ".join([f'{k:s}={{{k:s}:.3g}}' for k in features.keys()]),
                                         'string': "{roi_index:d}",
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

        # update plot
        if self.plot_trace_checkBox.isChecked():
            self._update_figure()

    def _select_rois_with_mouse(self):
        # much better to use viewer events rather than delve into QT
        # self.viewer.window.qt_viewer.canvas.events.mouse_press.connect(self._add_roi_by_point)
        self.viewer.mouse_drag_callbacks.append(self._add_roi_by_point)

        try:
            self.mouse_select_pushButton.clicked.disconnect()
        except Exception as e:
            print(e)

        self.mouse_select_pushButton.clicked.connect(self._stop_select_rois_with_mouse)
        self.mouse_select_pushButton.setText("Stop selecting...")

    def _stop_select_rois_with_mouse(self):

        self.viewer.mouse_drag_callbacks.pop()

        try:
            self.mouse_select_pushButton.clicked.disconnect()
        except Exception as e:
            print(e)

        self.mouse_select_pushButton.clicked.connect(self._select_rois_with_mouse)
        self.mouse_select_pushButton.setText("Select with mouse")

    def _add_roi_by_point(self, viewer, event):
        # center in canvas coordinates
        canvas_coord = event.position

        layer_name = self.layer_comboBox.currentText()
        layer_list = [l for l in self.viewer.layers if l.name == layer_name]

        if layer_list == []:
            print("no layers named `{layer_name:s}` were found")
            return

        layer = layer_list[0]
        cy, cx = layer.world_to_data(canvas_coord)

        self._add_roi()
        idx = self.fit_roi_tableWidget.rowCount() - 1
        self.fit_roi_tableWidget.cellWidget(idx, 0).setValue(int(np.round(cy)))
        self.fit_roi_tableWidget.cellWidget(idx, 1).setValue(int(np.round(cx)))

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
            self.fit_roi_tableWidget.removeRow(idx)

    def _clear_roi(self):
        self.fit_roi_tableWidget.clearContents()
        self.fit_roi_tableWidget.setRowCount(0)

    def _clear(self):
        self._on_stop_clicked()
        self.fit_params = None
        self.phases = None
        self.fit_times = None

    def _on_run_clicked(self):

        try:
            self._model = self.fit_models[self.model_comboBox.currentText()]
        except KeyError as e:
            print(e)
            return

        nrois = self.fit_roi_tableWidget.rowCount()
        self.fit_params = [np.zeros((0, self._model.nparams)) for _ in range(nrois)]
        self.phases = [np.zeros((0, 1)) for _ in range(nrois)]
        self.fit_times = np.zeros((0))

        self._prepare_figure()
        # self._prepare_display_table(nrois)

        self.streaming_timer = QTimer()
        self.streaming_timer.timeout.connect(self._fit_peaks)

        update_rate_ms = self.update_rate_doubleSpinBox.value()
        self.streaming_timer.start(update_rate_ms)

    def _on_stop_clicked(self):
        try:
            self.streaming_timer.stop()
            self.streaming_timer = None
        except AttributeError:
            pass

    # def _prepare_display_table(self, nrois):
    #     # clear
    #     self.tracking_tableWidget.clearContents()
    #     self.tracking_tableWidget.setRowCount(0)
    #
    #     # get current model
    #     nparams = self._model.nparams
    #
    #     for jj in range(nparams + 1):
    #         self.tracking_tableWidget.insertColumn(jj)
    #         # todo: set name
    #         # self.tracking_tableWidget.
    #
    #     # set new
    #     for idx in range(nrois):
    #         self.tracking_tableWidget.insertRow(idx)
    #
    #         for jj in range(nparams + 1):
    #             self.tracking_tableWidget.setCellWidget(idx, jj, QtW.QLabel(self))


    def _prepare_figure(self):

        try:
            plt.close(self._figure)
            # fignum = self._figure.number
            # make_new_figure = not plt.fignum_exists(fignum)
        except:
            # make_new_figure = True
            pass

        # if make_new_figure:

        self._figure = plt.figure(figsize=(30, 10))

        nrows = self._model.nparams + 1


        grid = self._figure.add_gridspec(nrows=nrows, ncols=1, hspace=0.3)
        for ii in range(nrows):
            ax = self._figure.add_subplot(grid[ii, 0])

            if ii < self._model.nparams:
                ax.set_ylabel(self._model.parameter_names[ii])
            else:
                ax.set_ylabel("phases")

            ax.axhline(0, color="k")

            if ii == (nrows - 1):
                ax.set_xlabel("Time (s)")
            else:
                ax.set_xticks([])
                ax.set_xticklabels([])

        # self._figure.canvas.draw()
        plt.show(block=False)

    def _update_figure(self):
        # todo plot all ROIs

        nrois = len(self.fit_params)
        colors = plt.cm.get_cmap('brg')(np.linspace(0, 1, nrois))

        # t_range_s = 10
        t_range_s = self.memory_time_doubleSpinBox.value()

        relative_times = self.fit_times - self.fit_times[0]

        max_time = relative_times.max()
        min_time = np.max([0, max_time - t_range_s])
        to_use = relative_times >= min_time
        t = relative_times[to_use]

        nrows = self._model.nparams + 1
        for ii in range(nrows):
            ax = self._figure.axes[ii]

            if ii < self._model.nparams:
                name = self._model.parameter_names[ii]
            else:
                name = "phase"

            ylows = []
            yhighs = []
            means = []
            stds = []
            for jj in range(nrois):

                if ii < self._model.nparams:
                    data = self.fit_params[jj][to_use, ii] - self.fit_params[jj][0, ii]
                    means.append(np.nanmean(self.fit_params[jj][to_use, ii]))
                    stds.append(np.nanstd(self.fit_params[jj][to_use, ii]))
                else:
                    data = np.unwrap(self.phases[jj][to_use]) - self.phases[jj][0]
                    means.append(np.nanmean(self.phases[jj][to_use]))
                    stds.append(np.nanstd(self.phases[jj][to_use]))

                # try to get line by label
                line_list = [ch for ch in ax.get_children() if isinstance(ch, matplotlib.lines.Line2D)
                             and ch.get_label() == f"{jj:d}"]

                if line_list == []:
                    ax.plot(t, data, c=colors[jj], label=f"{jj:d}")
                    ylows.append(np.min(data))
                else:
                    line_list[0].set_data(t, data)

                    # update limits
                    ax.set_xlim([min_time, max_time])

                ylows.append(np.nanmin(data))
                yhighs.append(np.nanmax(data))

            # set limits
            ylows = np.array(ylows)
            yhighs = np.array(yhighs)

            ylows[np.isinf(ylows)] = np.nan
            yhighs[np.isinf(yhighs)] = np.nan

            if not np.all(np.isnan(ylows)):
                ylow = np.nanmin(ylows)
            else:
                ylow = 0

            if not np.all(np.isnan(yhighs)):
                yhigh = np.nanmax(yhighs)
            else:
                yhigh = 0

            if yhigh == ylow:
                yhigh += 1e-3


            ax.set_ylim([ylow, yhigh])

            # set title
            ttl_str = f"{name:s};" + \
                      "; ".join([f" avg={means[jj]:.3f};"
                      f" std={stds[jj]:.3f}" for jj in range(nrois)])

            ax.set_title(ttl_str)

        self._figure.canvas.draw()
        self._figure.canvas.flush_events()


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
