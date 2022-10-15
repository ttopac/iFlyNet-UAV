#%%
import pathlib, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import ndimage

test_date = "08_01_2022"
test_folder = "Day1_Training1"
test_name = "compensated_normalized_WTRUN2_training_sweep1_2022-08-01_17-24-43-50_rtd-str"

#%%
def read_csv (header_cnt):
  if "normalized" not in test_csv:
    _ = input("The .csv file doesn't seem to be normalized. Are you sure you'd like to continue?")
  test_df = pd.read_csv(test_csv, header=header_cnt)
  return test_df

def plot_data (test_df, plot_time=False, plot_compensated=False):
  _, ax = plt.subplots()
  
  # start_ix = 0
  start_ix = 1674128 - 2000
  end_ix = 1674128
  # end_ix = len(test_df.iloc[:,0])
  # end_ix = 1681720 + 2000
  
  if plot_time:
    test_df.insert(1, "Date/Time (formatted)", test_df["Date/Time"])
    test_df["Date/Time (formatted)"] = test_df["Date/Time (formatted)"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d_%H-%M-%S-%f"))
    x_axis = test_df["Date/Time (formatted)"]
  else:
    try:
      x_axis = test_df["Datapoint"]
    except:
      x_axis = np.arange(end_ix)

  for sensor in sensors_to_plot:
    if sensor == "RTD 6 (V)":
      rtd6_dat = test_df[sensor + " (normalized)"]
      test_df[sensor + " (normalized)"] = ndimage.uniform_filter1d(rtd6_dat, size = 100)
      ax.plot(x_axis[start_ix:end_ix], test_df[sensor + " (normalized)"][start_ix:end_ix], label=sensor, zorder=0) #for normalized SGs/RTDs
    else:
      # ax.plot(x_axis[start_ix:end_ix], test_df[sensor + " (normalized)"][start_ix:end_ix], label=sensor) #for normalized SGs/RTDs
      if plot_compensated and "SG" in sensor:
        ax.plot(x_axis[start_ix:end_ix], test_df[sensor + " (normalized) (compensated)"][start_ix:end_ix], label=sensor + " (compensated)") #for normalized SGs/RTDs
  
  ax.set_xlabel("Datapoint")
  ax.set_ylabel("Voltage (V)")
  ax.legend(loc='upper right')

  # ax.set_xlim([0,1])
  # ax.set_ylim([0,1])

#%%
if __name__ == "__main__":
  cur_dir = pathlib.Path(__file__).parent.resolve()
  up_dir = cur_dir.parent.resolve().parent.resolve()
  data_dir = os.path.join(up_dir, test_date+"_Tests", "testdata", test_folder)
  test_csv = os.path.join(data_dir, test_name+".csv")

  # sensors_to_plot = ["SG 1 (V)", "SG 2 (V)", "SG 4 (V)", "SG 5 (V)", "SG 6 (V)", "SG TE (V)", "SG LE (V)", "RTD 1 (V)", "RTD 2 (V)", "RTD 4 (V)", "RTD 5 (V)", "RTD 6 (V)"]
  sensors_to_plot = ["SG 2 (V)"]
  df = read_csv(header_cnt=0) #header_cnt = 0 for compensated&normalized, header_cnt=1 for normalized, header_cnt=0 for non-normalized
  plot_data (df, plot_time=False, plot_compensated=True)

  plt.show()
# %%
