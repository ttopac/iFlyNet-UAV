import os
import pathlib
import copy
import pandas as pd
import numpy as np

# test_date = "08_01_2022"
# test_folder = "preliminary"
# test_name = "heating_pretest_rtd-str"
# test_date = "08_01_2022"
# test_folder = "Day1_Training1"
# test_name = "WTRUN2_training_sweep1_2022-08-01_17-24-43-50_rtd-str"
test_date = "08_02_2022"
test_folder = "Day2_Training1"
test_name = "WTRUN2_day2_training1_2022-08-02_12-38-30-01_rtd-str"
# test_date = "08_02_2022"
# test_folder = "Day2_Dynamic1"
# test_name = "WTRUN2_day2_dynamic1_2022-08-02_14-32-54-11_rtd-str"

def process_csv (data_dir, test_name, test_csv, header=None, remove_extremes=False):
  test_data = pd.read_csv(test_csv, header=header)
  
  if remove_extremes:
    for col in test_data.columns:
      badcell = test_data.loc[test_data[col] == 1.65]
      if badcell.shape[0] > 0:
        print ("There are bad cell(s)!")
        for ix in range(badcell.shape[0]):
          col_index = badcell.iloc[[ix]].index
          test_data[col].iloc[[col_index[0]]] = test_data[col].iloc[[col_index[0]-1]]

  num_list = list(range(1, len(test_data)+1))
  IMGenie_list = ["STR 1", "STR 2", "STR 3", "STR 4", "STR 5", "STR 6", "STR 7", "STR 8", "RTD 1", "RTD 2", "RTD 3", "STR 9"] #Aug 2022 tests
  sensor_list = ["SG 1 (V)", "SG 2 (V)", "SG 4 (V)", "SG 5 (V)", "SG 6 (V)", "SG TE (V)", "SG LE (V)", "RTD 1 (V)", "RTD 2 (V)", "RTD 4 (V)", "RTD 5 (V)", "RTD 6 (V)"] #Aug 2022 tests
  # IMGenie_list = ["STR 1", "STR 2", "STR 3", "STR 4", "STR 5", "STR 6", "STR 7", "STR 8", "STR 9"] #Aug 2022 tests (B)
  # sensor_list = ["SG 1 (V)", "SG 2 (V)", "SG 4 (V)", "SG 5 (V)", "SG 6 (V)", "SG TE (V)", "SG LE (V)", "RTD 1 (V)", "RTD 6 (V)"] #Aug 2022 tests (B)

  test_data.rename(columns={test_data.columns[0] : 'Date/Time'}, inplace=True)
  test_data.insert(1, "Temp", "")
  test_data.insert(2, "Datapoint", num_list)
  test_data.insert(3, "IMGenie Channel", "")

  #Add a new empty row just below the header
  new_row = pd.DataFrame(index=[-0.5])
  test_data = test_data.append(new_row, ignore_index=False)
  test_data = test_data.sort_index().reset_index(drop=True)
  
  #Rename the first few columns of this new row
  test_data["Date/Time"][0] = "Date/Time"
  test_data["Temp"][0] = "Temp"
  test_data["Datapoint"][0] = "Datapoint"
  test_data["IMGenie Channel"][0] = "Sensor ID"
  
  for cnt, col_num in enumerate(range(4,len(IMGenie_list)*2+3,2)):
    #Rename the column title
    # test_data.columns.values[col_num] = IMGenie_list[cnt]
    test_data.rename(columns={test_data.columns[col_num] : IMGenie_list[cnt]}, inplace=True)

    #Rename the second row with correct label
    test_data[IMGenie_list[cnt]][0] = sensor_list[cnt]
    
    #Calculate the normalized values
    raw_data = test_data[IMGenie_list[cnt]]
    norm_data = copy.copy(raw_data)
    norm_data[0] = raw_data[0]+" (normalized)"
    # if sensor_list[cnt] == "RTD 6 (V)":
    #   norm_data[1:] = raw_data[1:]-np.mean(raw_data[1:200])
    # else:
      # norm_data[1:] = raw_data[1:]-raw_data[1]
    norm_data[1:] = raw_data[1:]-np.mean(raw_data[1:1300])

    #Add the normalized values next to the raw values
    test_data.insert(col_num+1, IMGenie_list[cnt]+" (normalized)", norm_data)

  #Save as a new csv file
  export_file = os.path.join(data_dir, "normalized_"+test_name+"_Jan2023.csv")
  test_data.to_csv(export_file, sep=',', index=False)

if __name__ == "__main__":
  cur_dir = pathlib.Path(__file__).parent.resolve()
  up_dir = cur_dir.parent.resolve().parent.resolve()
  data_dir = os.path.join(up_dir, test_date+"_Tests", "testdata", test_folder)
  test_csv = os.path.join(data_dir, test_name+".csv")

  process_csv(data_dir, test_name, test_csv, header='infer', remove_extremes=False)