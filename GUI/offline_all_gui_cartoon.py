#%%
import sys
import os
import pickle
import pathlib
from multiprocessing import Queue
from tkinter import Tk, Button
from tkinter import S
# import keras_resnet
# import keras_resnet.models
# import torch
import numpy as np


#%%
FILE_PATH = pathlib.Path(__file__).absolute()
CODES_LOC = FILE_PATH.parent.parent
GUI_LOC = FILE_PATH.parent
AUG2022_LOC = FILE_PATH.parent.parent.parent
MODELS_LOC = os.path.join(AUG2022_LOC, "MLmodels", "Training1_Jan2023", "models")
DYNAMICTEST_LOC = os.path.join(AUG2022_LOC, "MLmodels", "Dynamic1_Jan2023")
VIDEO_LOC = os.path.join(AUG2022_LOC, "ProcessedVids")
GRAPHICS_LOC = os.path.join(AUG2022_LOC, "Codes", "GUI", "assets")

sys.path.insert(0, str(CODES_LOC))
sys.path.insert(0, str(AUG2022_LOC))
from Codes.Dynamic.helpers.procothersources_helper import ProcOtherSourcesOffline
from Codes.Dynamic.helpers.procpreds_helper import ProcPredsOffline
from Codes.Dynamic.helpers.procdata_helper import proc_data
from Codes.GUI.gui_helpers.gui_windows_helper import iFlynetWindows
from Codes.GUI.gui_helpers.stream_helper import StreamOffline

#%%
INCLUDE_STATE_PRED = True
INCLUDE_STALL_PRED = True
INCLUDE_LIFTDRAG_PRED = True

STALL_MODEL_NAME = "model_stall_1DCNN_ph10"
STATE_MODEL_NAME = "model_fno1d_valSplit"
LIFTDRAG_MODEL_NAME = "model_fno1d_valSplit_LiftDrag"
DYNAMICTEST_NAME = "dynamic1_run15"
ENCODER_NAME = None #encoder_ResNet18_dynVal for state model model_ResNet18_dynVal_ph10

# DAQ properties
DAQ_SAMPLERATE = 10000
WINDOW_SIZE = 333 #(10000/30=333) Inference in every 1/30 seconds
DATA_PER_SECOND = int(DAQ_SAMPLERATE/WINDOW_SIZE) #30 predictions per second
MEAN_DICT = {'PZT 1': 1.1964636626483654e-05, 'PZT 2': 1.516750592542323e-05, 'PZT 3': 1.2442148594709227e-05, 'PZT 4': 2.4788622977256767e-05, 'PZT 5': -3.7253732809026377e-07, 'PZT 6': 1.0824012458478593e-05, 'PZT 7': 1.1674180171877221e-05, 'SG 1': -6.594330432620959e-05, 'SG 2': -0.00010346779513860188, 'SG 5': -7.36565906801972e-05}
STD_DICT = {'PZT 1': 8.041994739151683e-05, 'PZT 2': 7.549385118018473e-05, 'PZT 3': 8.428371907009408e-05, 'PZT 4': 0.0001256689655046649, 'PZT 5': 8.544440378484743e-05, 'PZT 6': 9.837018189303782e-05, 'PZT 7': 0.00013566630247286284, 'SG 1': 4.3046127066776684e-05, 'SG 2': 7.459676011368966e-05, 'SG 5': 5.162082415922712e-05}

ALL_SENSORS = ['PZT 1', 'PZT 2', 'PZT 3', 'PZT 4', 'PZT 5', 'PZT 6', 'PZT 7',
               'SG 1', 'SG 2', 'SG 4', 'SG 5', 'SG 6', 'SG LE', 'SG TE']
ACTIVE_SENSORS = ['PZT 1', 'PZT 2', 'PZT 3', 'PZT 4', 'PZT 5', 'PZT 6', 'PZT 7',
                  'SG 1', 'SG 2', 'SG 5']

# GUI properties
VISIBLE_DURATION = 30 #seconds
PLOT_REFRESH_RATE = 0.1 #seconds. This should be equal to or slower than 30hz (equal to or more than 0.033)

# %%
# Add start button
def start_offline_button(gui_app, queue):
  startoffline_button = Button(gui_app.parent, text='Click to start the stream...', command=lambda : queue.put(False))
  startoffline_button.grid(row=0, column=0, rowspan=1, columnspan=8, sticky=S)

# %%
test_data = np.load(os.path.join(DYNAMICTEST_LOC, "data", DYNAMICTEST_NAME+".npy"), allow_pickle=True)
with open(os.path.join(DYNAMICTEST_LOC, "labels", DYNAMICTEST_NAME+"_truth_nopeaks.pkl"), 'rb') as f:
  test_truth_state = pickle.load(f)
# Here test_data.shape = (#of sensors, #of sensor readings in the experiment) (14, 5366200 for Aug. 2022 Dynamic1 Run15)
# test_data: 10,000sps * ~9 minutes per run 

# %%
# Process the sensor data
test_data, test_truth_state = proc_data(test_data, test_truth_state, ALL_SENSORS, ACTIVE_SENSORS, WINDOW_SIZE, MEAN_DICT, STD_DICT)
#test_data is reshaped to (-1, WINDOW_SIZE, #of active sensors) (16114, 333, 10) for Aug. 2022 Dynamic1 Run15

# %%
# Bring in the ground truth data
offline_other_sources = ProcOtherSourcesOffline(test_truth_state)
truth_data = offline_other_sources.get_truth_state()

# %%
# Bring in the i-FlyNet predictions
dynamictest_preds_loc = os.path.join(DYNAMICTEST_LOC, "predictions")
offline_preds = ProcPredsOffline(test_data, truth_data, dynamictest_preds_loc, DYNAMICTEST_NAME)

# State predictions:
if INCLUDE_STATE_PRED:
  airspeed_preds, aoa_preds = offline_preds.get_preds_state(MODELS_LOC, STATE_MODEL_NAME, ENCODER_NAME)

# Stall predictions
if INCLUDE_STALL_PRED:
  stall_preds = offline_preds.get_preds_stall(MODELS_LOC, STALL_MODEL_NAME)

# Lift/drag predictions
if INCLUDE_LIFTDRAG_PRED:
  lift_preds, drag_preds = offline_preds.get_preds_liftdrag(MODELS_LOC, LIFTDRAG_MODEL_NAME)

# %%
# Bring in the UAV data:
uav_data = offline_other_sources.get_uav_state()

# %%
###
# Initialize the GUI
###
root = Tk()
root.title ("i-FlyNet UAV")
GUIapp = iFlynetWindows(root)

# %%
###
# Initialize the data
###
streamhold_queue = Queue()
stream = StreamOffline(streamhold_queue, GUIapp)

if INCLUDE_STATE_PRED:
  stream.prep_data_state(truth_data['airspeed'], truth_data['aoa'], airspeed_preds, aoa_preds, uav_data['airspeed'], uav_data['aoa'])
if INCLUDE_STALL_PRED:
  stream.prep_data_stall(stall_preds)
if INCLUDE_LIFTDRAG_PRED:
  stream.prep_data_liftdrag(truth_data['lift'], truth_data['drag'], lift_preds, drag_preds)

# %%
###
# Place the GUI elements
###
GUIapp.draw_midrow("Flight Characteristics", os.path.join(GUI_LOC, "assets", "legend4.png"))

stream.initialize_video("Experiment Camera", VIDEO_LOC, DYNAMICTEST_NAME)
stream.initialize_plots_wcomparison(INCLUDE_STATE_PRED, INCLUDE_LIFTDRAG_PRED, PLOT_REFRESH_RATE, VISIBLE_DURATION, DATA_PER_SECOND)
stream.initialize_cartoon(GRAPHICS_LOC)

# %%
###
#Finalize the GUI
###
GUIapp.place_on_grid()

# %%
# Start streaming
start_offline_button(GUIapp, streamhold_queue)
root.mainloop()