#%%
import pathlib, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

test_date = "08_02_2022"
test_folder = "Day2_Training1"
test_name = "DateTimed_WTRUN2_day2_training1_EDS"
sensors_to_plot = ["Inclination (deg)"]

#%%
def read_csv (header_cnt):
  test_df = pd.read_csv(test_csv, header=header_cnt, index_col=False)
  test_df.insert(10, "Airspeed (m/s)", test_df["MPH"]*0.447)
  return test_df

def plot_data (test_df):
  _, ax = plt.subplots()
  start_ix = 0
  # start_ix = 1000
  end_ix = len(test_df["Flag"])
  # end_ix = 3000

  test_df.insert(1, "Date/Time (formatted)", test_df["Parsed Date & Time"])
  test_df["Date/Time (formatted)"] = df["Date/Time (formatted)"].apply(lambda x: datetime.strptime(x, "%H hours %M minutes %S seconds %f milliseconds"))
  x_axis = test_df["Date/Time (formatted)"]

  for sensor in sensors_to_plot:
    ax.plot(x_axis[start_ix:end_ix], test_df[sensor][start_ix:end_ix], label=sensor)
  
  # ax.set_xlim([0,1])
  # ax.set_ylim([0,1])
  ax.legend(loc='upper right')
  plt.show()

if __name__ == "__main__":
  cur_dir = pathlib.Path(__file__).parent.resolve()
  up_dir = cur_dir.parent.resolve().parent.resolve()
  data_dir = os.path.join(up_dir, test_date+"_Tests", "testdata", test_folder)
  test_csv = os.path.join(data_dir, test_name+".csv")
#%%
  df = read_csv(header_cnt=0)
  plot_data (df)