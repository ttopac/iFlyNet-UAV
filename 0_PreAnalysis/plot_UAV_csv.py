#%%
import pathlib, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import math

test_date = "08_02_2022"
test_folder = "Day2_Training1"
test_name = "WTRUN2_day2_training1_UAV"

#%%
def read_csv (test_csv, header_cnt):
  test_df = pd.read_csv(test_csv, header=header_cnt)
  return test_df

def plot_data (test_df, sensors_to_plot, plot_time=False):
  fig, ax = plt.subplots()
  
  start_ix = 0
  end_ix = len(test_df.iloc[:,0])
  
  try:
    x_axis = test_df["STATIC_PRESSURE_TIME"]
  except:
    x_axis = np.arange(end_ix)
  for sensor in sensors_to_plot:
    if "ALPHA" in sensor and "Deg" not in sensor:
      ax.plot(x_axis[start_ix:end_ix], test_df[sensor][start_ix:end_ix]*180/math.pi, label=sensor)
    else:
      ax.plot(x_axis[start_ix:end_ix], test_df[sensor][start_ix:end_ix], label=sensor)
  
  ax.set_xlabel("Datapoint")
  
  ax.legend(loc='upper right')
  plt.show()

if __name__ == "__main__":
  cur_dir = pathlib.Path(__file__).parent.resolve()
  up_dir = cur_dir.parent.resolve().parent.resolve()
  data_dir = os.path.join(up_dir, test_date+"_Tests", "testdata", test_folder)
  test_csv = os.path.join(data_dir, test_name+".csv")

  sensors_to_plot = ["Pitch (deg)"]
  df = read_csv(test_csv, header_cnt=0)

  plot_data (df, sensors_to_plot, plot_time=False)
