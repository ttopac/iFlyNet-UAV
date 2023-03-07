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
    return (airspeed_truth, aoa_truth)
  
  def get_uav_state (self):
    airspeed_uav = self.other_source_data["UAV_TAS"].to_numpy()
    aoa_uav = self.other_source_data["UAV_Pitch"].to_numpy()
    return (airspeed_uav, aoa_uav)