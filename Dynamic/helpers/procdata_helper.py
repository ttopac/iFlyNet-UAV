import numpy as np

def proc_data(test_data, test_truth, all_sensors, active_sensors, window_size, mean_dict, std_dict):
  # Mask out the sensors not used
  mask = np.in1d(np.asarray(all_sensors), np.asarray(active_sensors))
  examples = test_data[mask,:]

  #Convert the data to ML-ready form
  sensor_count = examples.shape[0]
  line_count = examples.shape[1]
  splitparameter = line_count//window_size
  examples_t = examples.T[0:window_size*splitparameter,:]
  examples = np.reshape(examples_t, (-1,window_size,sensor_count))   # examples.shape = (16114, 333, 10)
  truth = test_truth[0:examples.shape[0]]   # truth.shape = (16114, 13)

  #Standardize the data based on the mean and standard deviation of training set
  if len(active_sensors) != examples.shape[2]:
    raise Exception("Number of sensors declared active in the problem doesn't match with the data size") 
  
  for cnt, sensor in enumerate(active_sensors):
    mean = mean_dict[sensor]
    stddev = std_dict[sensor]
    examples[:,:,cnt] = (examples[:,:,cnt]-mean)/stddev
  
  return examples, truth