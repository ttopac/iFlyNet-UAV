import pathlib, os
import pandas as pd
import datetime

test_date = "06_16_2022"
test_name = "day3_initial_test_2022-06-16_12-46-07-32_rtd-str"

def read_csv (test_csv, header_cnt):
  test_df = pd.read_csv(test_csv, header=header_cnt)
  return test_df

def distance_btw_dps(test_df):
  test_df.insert(1, "Date/Time (formatted)", test_df["Date/Time"])
  test_df["Date/Time (formatted)"] = df["Date/Time (formatted)"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d_%H-%M-%S-%f"))
  
  threshold = datetime.timedelta(microseconds=20000)
  old_time = test_df["Date/Time (formatted)"][0]
  data_cnt = 0
  exceed_cnt = 0
  for index, row in test_df.iterrows():
    if old_time != row['Date/Time (formatted)']:
      new_time = row['Date/Time (formatted)']
      delta = new_time - old_time
      if delta > threshold:
        # print ("We exceed the threshold")
        # print ("Delta is {}".format(delta))
        exceed_cnt += 1
      old_time = new_time
      data_cnt += 1
    else:
      pass
  print ("{} samples exceded the threshold of {}.".format(exceed_cnt, threshold))
  

if __name__ == "__main__":
  cur_dir = pathlib.Path(__file__).parent.resolve()
  data_dir = os.path.join(cur_dir, test_date+"_Tests", "testdata")
  test_csv = os.path.join(data_dir, test_name+".csv")

  df = read_csv(test_csv, header_cnt=0)

  distance_btw_dps (df)