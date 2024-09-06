"""
Launch application

useful place to call other startup commands that don't want hardcoded in plugin
"""
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
main_window.load_cfg("MM config", str(root_dir / "sim_odt_nidaq_c1.cfg"))
# main_window.load_cfg("Cam 2", str(root_dir / "sim_odt_nidaq_c2.cfg"))
# main_window.load_cfg("Cam 2", str(root_dir / "blackfly_flir_cam.cfg"))
main_window.load_cfg("DMD", str(root_dir / "dmd_config.zarr"))
main_window.load_cfg("DMD 2", str(root_dir / "dmd2_config.zarr"))
main_window.load_cfg("microscope", str(root_dir / "config.json"))

# grab devices
mmc1, mmc2 = main_window._mmcores
dmd = main_window.dmd
dmd2 = main_window.dmd2
daq = main_window.daq
phcam = main_window.phcam

pk_track = main_window.peak_tracker_widget
sim_odt_widget = main_window.sim_odt_acq

napari.run()
