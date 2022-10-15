#%%
import pathlib, os
from math import atan2, asin, pi
import numpy as np

#%%
test_date = "08_02_2022"
test_folder = "Day2_Training1_B"
test_name = "WTRUN2_day2_training1_B_UAV"

def read_dat (test_csv, delimiter):
  dat = np.genfromtxt(test_csv, delimiter=delimiter)
  return dat

def convert_quats(dat, delimiter):
  numdp = dat.shape[0]
  rpy = np.zeros((numdp-2, 3))
  for i in range(numdp-2):
    quat = dat[i+2, 36:40]

    roll = atan2(2*(quat[2]*quat[3] + quat[0]*quat[1]), quat[0]*quat[0] - quat[1]*quat[1] - quat[2]*quat[2] + quat[3]*quat[3])
    pitch = asin(2*(quat[0]*quat[2] - quat[1]*quat[3]))
    yaw = atan2(2*(quat[1]*quat[2] + quat[0]*quat[3]), quat[0]*quat[0] + quat[1]*quat[1] - quat[2]*quat[2] - quat[3]*quat[3])
    
    rpy[i,:] = [roll, pitch, yaw]

  rpy_deg = rpy * 180/pi
  np.savetxt("rpy_deg.csv", rpy_deg, header="Roll (deg),Pitch (deg),Yaw (deg)", delimiter=delimiter)

if __name__ == "__main__":
  cur_dir = pathlib.Path(__file__).parent.resolve()
  up_dir = cur_dir.parent.resolve()
  data_dir = os.path.join(up_dir, test_date+"_Tests", "testdata", test_folder)
  test_csv = os.path.join(data_dir, test_name+".csv")

  dat = read_dat(test_csv, ',')
  convert_quats (dat, ',')
