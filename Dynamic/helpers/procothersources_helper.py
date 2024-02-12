import os
import sys
import pickle
import pathlib
import numpy as np

class ProcOtherSourcesOffline():
  def __init__ (self, other_source_data):
    self.other_source_data = other_source_data

  def get_truth_state (self):
    airspeed_truth = self.other_source_data["WT_Airspeed"].to_numpy()
    aoa_truth = self.other_source_data["WT_AoA"].to_numpy()
    lift_truth = self.other_source_data["Lift (lbf)"].to_numpy()
    drag_truth = self.other_source_data["Drag (lbf)"].to_numpy()
    truth_data = {'airspeed':airspeed_truth, 'aoa':aoa_truth, 'lift':lift_truth, 'drag':drag_truth}
    return truth_data
  
  def get_uav_state (self):
    airspeed_uav = self.other_source_data["UAV_TAS"].to_numpy()
    aoa_uav = self.other_source_data["UAV_Pitch"].to_numpy()
    uav_data = {'airspeed':airspeed_uav, 'aoa':aoa_uav}
    return uav_data