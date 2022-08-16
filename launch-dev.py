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
# todo: find a better place to put initialization code ... maybe should have a mechanism for running a startup script...
initialize_mre2()

# launch viewer
v = napari.Viewer()
dw, main_window = v.window.add_plugin_dock_widget("micromanager")

# set default configurations
main_window.cfg_LineEdit.setText(r"C:/Users/q2ilab/Documents/mcsim_private/mcSIM/mcsim/expt_ctrl/sim_odt_nidaq_c1.cfg")
main_window.cfg2_LineEdit.setText(r"C:/Users/q2ilab/Documents/mcsim_private/mcSIM/mcsim/expt_ctrl/sim_odt_nidaq_c2.cfg")
main_window.dmd_cfg_lineEdit.setText(r"C:\Users\q2ilab\Documents\mcsim_private\mcSIM\mcsim\expt_ctrl\dmd_config.json")
main_window.daq_cfg_lineEdit.setText(r"C:\Users\q2ilab\Documents\mcsim_private\mcSIM\mcsim\expt_ctrl\daq_config.json")
main_window.microscope_cfg_lineEdit.setText(r"C:\Users\q2ilab\Documents\mcsim_private\mcSIM\mcsim\expt_ctrl\config.json")

# load default configurations
main_window.load_cfg()
main_window.load_cfg2()
main_window.load_dmd_cfg()
main_window.load_daq_cfg()
main_window.load_microscope_cfg()

# grab devices
core = main_window._mmc
dmd = main_window.dmd
daq = main_window.daq


napari.run()
