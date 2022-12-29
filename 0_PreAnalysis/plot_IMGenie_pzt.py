#%%
import copy
import pathlib, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import ndimage
from os import listdir
from scipy import signal

test_date = "08_02_2022"
test_folder = "Day2_Training1"

#%%
def get_test_files():
  test_names = [f for f in listdir(data_dir) if "pzt" in f]
  return sorted(test_names)

def concat_dfs():
  if len(dfs) > 1:
    return pd.concat(dfs, axis=0)
  else:
    return dfs[0]

def read_csv (test_csv, header_cnt):
  test_df = pd.read_csv(test_csv, header=header_cnt)
  return test_df

def plot_data (test_df, normalize_pzts=False):
  _, ax = plt.subplots()
  

  start_ix = 16200000 - 50000
  end_ix = 16200000 + 50000

  # end_ix = len(test_df.iloc[:,0])
  # end_ix = 132345200 + 60*10000 + 150000 #10000 * 60 * mins (PZT data is sampled at 10,000 sps)
  # end_ix = 60000
  

  try:
    x_axis = test_df["Datapoint"]
  except:
    x_axis = np.arange(end_ix)
  
  for sensor in sensors_to_plot:
    if normalize_pzts:
      raw_data = test_df[sensor]
      norm_data = copy.copy(raw_data)
      norm_data = raw_data-np.mean(raw_data[0:10000])
      
      # filtered = signal.medfilt(norm_data, 5)
      # sos = signal.butter(1, 3, 'highpass', fs=10000, output='sos')
      sos = signal.butter(5, 2000, 'lowpass', fs=10000, output='sos')
      filtered = signal.sosfilt(sos, norm_data)
      
      ax.plot(x_axis[start_ix:end_ix], norm_data[start_ix:end_ix], label=sensor, linewidth=0.1, color='g')
      ax.plot(x_axis[start_ix:end_ix], filtered[start_ix:end_ix], label=sensor+" (filtered)", linewidth=0.1, color='r')
    else:
      filtered = signal.medfilt(test_df[sensor], 3)
      # ax.plot(x_axis[start_ix:end_ix], filtered[start_ix:end_ix], label=sensor, linewidth=0.1, color='g')
  
  ax.set_xlabel("Datapoint")
  ax.set_ylabel("Voltage (V)")
  ax.legend(loc='upper right')

  # ax.set_xlim([0,1])
  ax.set_ylim([-0.0008,0.0008])

#%%
if __name__ == "__main__":
  cur_dir = pathlib.Path(__file__).parent.resolve()
  up_dir = cur_dir.parent.resolve().parent.resolve()
  data_dir = os.path.join(up_dir, test_date+"_Tests", "testdata", test_folder)
  dfs = list()
  
  test_names = get_test_files()
  for test_name in test_names:
    test_csv = os.path.join(data_dir, test_name)
    df = read_csv(test_csv, header_cnt=0) #header_cnt=0 for PZTs
    dfs.append(df)

  df_pile = concat_dfs()

  #%%
  # sensors_to_plot = ["PZT 1", "PZT 2", "PZT 3", "PZT 4", "PZT 5", "PZT 6", "PZT 7"]
  sensors_to_plot = ["PZT 1"]
  plot_data (df_pile, normalize_pzts=True)

  plt.show()
