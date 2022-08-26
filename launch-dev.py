"""
Launch application

useful place to call other startup commands that don't want hardcoded in plugin
"""
import matplotlib
matplotlib.use("TkAgg")
from pathlib import Path
import napari

# initialize mre2 mirror
from mcsim.expt_ctrl.setup_optotune_mre2 import initialize_mre2
initialize_mre2()

# launch viewer
v = napari.Viewer()
dw, main_window = v.window.add_plugin_dock_widget("micromanager")

# set default configurations
root_dir = Path(r"C:/Users/q2ilab/Documents/mcsim_private/mcSIM/mcsim/expt_ctrl")
main_window.cfg_LineEdit.setText(str(root_dir / "sim_odt_nidaq_c1.cfg"))
main_window.cfg2_LineEdit.setText(str(root_dir / "sim_odt_nidaq_c2.cfg"))
main_window.dmd_cfg_lineEdit.setText(str(root_dir / "dmd_config.json"))
main_window.daq_cfg_lineEdit.setText(str(root_dir / "daq_config.json"))
main_window.microscope_cfg_lineEdit.setText(str(root_dir / "config.json"))

# load default configurations
main_window.load_cfg()
# main_window.load_cfg2()
main_window.load_dmd_cfg()
main_window.load_daq_cfg()
main_window.load_microscope_cfg()

# grab devices
mmc1, mmc2 = main_window._mmcores
dmd = main_window.dmd
daq = main_window.daq
phcam = main_window.phcam


napari.run()
