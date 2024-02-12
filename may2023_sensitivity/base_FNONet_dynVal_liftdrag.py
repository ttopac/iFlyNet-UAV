# %%
import pathlib
import sys
import os
import pickle
import gc
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy import signal
import importlib

FILE_PATH = pathlib.Path(__file__).absolute()
FBF_PATH = FILE_PATH.parent.parent
sys.path.insert(0, str(FBF_PATH))
print (FILE_PATH)

# %%
training_dir = "/home/tanay/onedrive_su/KerasML/Training1_Jan2023"
dyn_dir = "/home/tanay/onedrive_su/KerasML/Dynamic1_Jan2023"
data_dir = os.path.join(training_dir, "data")

data_filename = "sttr_aug2022_data_multidf.npy"
liftdraglabels_filename = "sttr_aug2022_liftdraglabels.npy"

allExamples = np.load(os.path.join(data_dir, data_filename), allow_pickle=True)
allLiftdrag = np.load(os.path.join(data_dir, liftdraglabels_filename), allow_pickle=True)

# Here data.shape = (#of states, #of sensors, #data per each state) (238, 14, 599200 for Aug. 2022 test)
# states: 7m/s_0deg, 7m/s_1deg, ... 20m/s_15deg, 20m/s_16deg.
# sensors: ['PZT 1', 'PZT 2', 'PZT 3', 'PZT 4', 'PZT 5', 'PZT 6', 'PZT 7', 'SG 1', 'SG 2', 'SG 4', 'SG 5', 'SG 6', 'SG LE', 'SG TE']
# data: 10,000sps * ~60 seconds 

# %%
# Get rid of the nan's in liftdrag data
nan_mask = np.isnan(allLiftdrag)
masked = np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), allLiftdrag[~nan_mask])
allLiftdrag[nan_mask] = masked

# %%
# from tensorflow.python.client import device_lib
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# print("Available GPU: ", tf.test.gpu_device_name())
# print(device_lib.list_local_devices())

# %%
all_airspeeds = [7, 8.3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
all_aoas = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
all_sensors = ['PZT 1', 'PZT 2', 'PZT 3', 'PZT 4', 'PZT 5', 'PZT 6', 'PZT 7', 
                    'SG 1', 'SG 2', 'SG 4', 'SG 5', 'SG 6', 'SG LE', 'SG TE']

# available_airspeeds = [7, 8.3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# available_aoas = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# available_sensors = ['PZT 1', 'PZT 2', 'PZT 3', 'PZT 4', 'PZT 5', 'PZT 6', 'PZT 7', 
#                     'SG 1', 'SG 2', 'SG 4', 'SG 5', 'SG 6', 'SG LE', 'SG TE']

available_airspeeds = [7, 8.3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
available_aoas = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
available_sensors = ['PZT 1', 'PZT 2', 'PZT 3', 'PZT 4', 'PZT 5', 'PZT 6', 'PZT 7',
                     'SG 1', 'SG 2', 'SG 5']

# %%
# First get the number of states to be included in the training dataset
test_lines = allExamples.shape[2]
output_lines = allLiftdrag.shape[2]

sensor_count = len(available_sensors)
state_count = 0
for airspeed in all_airspeeds:
  if airspeed in available_airspeeds:
    for aoa in all_aoas:
      if aoa in available_aoas:
        state_count += 1
train_data_np = np.zeros((state_count, sensor_count, test_lines))
train_liftdrag_np = np.zeros((state_count, 2, output_lines))

# Second create a sensor mask to select the sensors to use among all sensors.
sensor_mask = list()
sensor_ix = -1
for sensor in all_sensors:
  sensor_ix += 1
  if sensor in available_sensors:
    sensor_mask.append(sensor_ix)

# Finally, construct the train numpy arrays.
np_ix = -1
airspeed_ix = -1
aoa_ix = -1
for airspeed in all_airspeeds:
  airspeed_ix += 1
  if airspeed in available_airspeeds:
    for aoa in all_aoas:
      aoa_ix += 1
      if aoa in available_aoas:
        np_ix += 1
        state_ix = airspeed_ix*len(available_aoas) + aoa_ix
        train_data_np[np_ix, :, :] = allExamples[state_ix, sensor_mask, :]
        train_liftdrag_np[np_ix,:,:] = allLiftdrag[state_ix,:,:]
    aoa_ix = -1

print (train_data_np.shape)
print (train_liftdrag_np.shape)

# %%
del allExamples

# %%
# Block to take the first 50-second of the data
train_data_np = train_data_np[:,:,0:500000]

print (train_data_np.shape)

# %%
# Plot PZT1 ('7m/s_0deg')
# legendkey = list()
# for i in [0]: #0 is PZT1
# #   dat = np.transpose(train_data_np,[1,0,2]).reshape(train_data_np.shape[1],-1)
#   dat = train_data_np
#   plt.plot(dat[0,i,0:333], linewidth=0.3)
#   legendkey.append(i)
# plt.legend(legendkey,loc='upper left', prop={'size':14})
# plt.xlabel("Time (sec)")

# %%
# Plot SG5 ('7m/s_0deg')
# legendkey = list()
# for i in [0]: #10 is SG5
# #   dat = np.transpose(train_data_np,[1,0,2]).reshape(train_data_np.shape[1],-1)
#   dat = train_data_np
#   plt.plot(dat[0,i,0:100000], linewidth=0.3)
#   legendkey.append(i)
# plt.legend(legendkey,loc='upper left', prop={'size':14})
# plt.xlabel("Time (sec)")

# %%
###
# Prepare the data for Machine Learning
###

# Find out parameters
dataCount = train_data_np.shape[0] 
sensorCount = train_data_np.shape[1] #9
lineCount = train_data_np.shape[2]
windowSize = 333 #(10000/30=333) Inference in every 1/30 seconds.
splitparameter = lineCount//windowSize

# Perform transformations
examplesT = np.transpose(train_data_np,[0,2,1])[:,0:windowSize*splitparameter,:]
examples = np.reshape(examplesT, (dataCount*splitparameter,windowSize,sensorCount))

# Adjust the liftdrag data
train_liftdrag_np = train_liftdrag_np[:,:,0:splitparameter] #First, take the first 50 seconds
liftdragT = np.transpose(train_liftdrag_np,[1,0,2])
liftdrag = np.reshape(liftdragT, (2,-1)).T

print (examples.shape) #(number of states * splitparameter = number of examples, number of datapoints in each example = size, # of channels in each example)
print (liftdrag.shape) #(number of examples)


# %%
# Standardize examples within each sensor
meanDict = dict()
stdDict = dict()
for i in range(len(available_sensors)):
  mean = np.mean(examples[:,:,i])
  stddev = np.std(examples[:,:,i])
  meanDict[available_sensors[i]] = mean
  stdDict[available_sensors[i]] = stddev
  examples[:,:,i] = (examples[:,:,i]-mean)/stddev
print (meanDict)
print (stdDict)

# %%
# Plot PZT1 ('7m/s_0deg') (standardized)
# legendkey = list()
# for i in [0]: #0 is PZT1
# #   dat = np.transpose(train_data_np,[1,0,2]).reshape(train_data_np.shape[1],-1)
#   dat = examples
#   plt.plot(dat[1,:,i], linewidth=0.3)
#   legendkey.append(i)
# plt.legend(legendkey,loc='upper left', prop={'size':14})
# plt.xlabel("Time (sec)")

# %%
# Plot SG5 ('7m/s_0deg') (standardized)
# legendkey = list()
# for i in [10]: #10 is SG5
# #   dat = np.transpose(train_data_np,[1,0,2]).reshape(train_data_np.shape[1],-1)
#   dat = examples
#   plt.plot(dat[1,:,i], linewidth=0.3)
#   legendkey.append(i)
# plt.legend(legendkey,loc='upper left', prop={'size':14})
# plt.xlabel("Time (sec)")


# %%
# Shuffle the data to remove any bias.
# This operation removes of the inter- & intra-state sequentiality.
def unison_shuffled_copies(b):
  p = np.random.permutation(len(b))
  return b[p], p

examples, p = unison_shuffled_copies(examples)
liftdrag = liftdrag[p]
trainingLiftdrag = liftdrag

# %%
# Prepare dynamic data for validation
dyn_data_dir = os.path.join(dyn_dir, "data")
dyn_labels_dir = os.path.join(dyn_dir, "labels")

# Pre-process dynamic data
dyn_data = np.load(os.path.join(dyn_data_dir, "dynamic1_run15.npy"))
dyn_data_masked = dyn_data[sensor_mask, :]

lineCount = dyn_data_masked.shape[1]
splitparameter = lineCount//windowSize
dyn_data_masked = dyn_data_masked[:,0:splitparameter*windowSize]
dyn_data_split = np.reshape(dyn_data_masked, (sensorCount, splitparameter, windowSize))
dyn_data_transposed = np.transpose(dyn_data_split,[1,2,0]) #shape:(num_examples, 333, num_sensors)

# %%
# Standardize the dynamic data
for cnt,sensor in enumerate(meanDict.keys()):
  mean = meanDict[sensor]
  stddev = stdDict[sensor]
  dyn_data_transposed[:,:,cnt] = (dyn_data_transposed[:,:,cnt]-mean)/stddev

# %%
# Pre-process dynamic lift/drag labels (work with FLOAT data)
with open(os.path.join(dyn_labels_dir,'dynamic1_run15_truth.pkl'), 'rb') as f:
    dyn_labels = pickle.load(f)
dyn_labels = dyn_labels[0:splitparameter]    
float_lift_labels = dyn_labels["Lift (lbf)"].apply(lambda x: float(x)).to_numpy(dtype=object)
float_drag_labels = dyn_labels["Drag (lbf)"].apply(lambda x: float(x)).to_numpy(dtype=object)
dynLiftDrag = np.stack((float_lift_labels, float_drag_labels), axis=1)

#Filter the dynamic data so we only take the samples with seen labels
minlift, maxlift = min(liftdrag[:,0]), max(liftdrag[:,0])
mindrag, maxdrag = min(liftdrag[:,1]), max(liftdrag[:,1])

lift_lowerbound = float_lift_labels.astype('float')>=minlift
lift_upperbound = float_lift_labels.astype('float')<=maxlift
drag_lowerbound = float_drag_labels.astype('float')>=mindrag
drag_upperbound = float_drag_labels.astype('float')<=maxdrag
dyn_mask = np.where([lift_lowerbound, lift_upperbound, drag_lowerbound, drag_upperbound], True, False)
dyn_mask = dyn_mask.all(axis = 0)
contain_ixs = np.nonzero(dyn_mask)[0]

contained_dyn_liftdrag = dynLiftDrag[contain_ixs]
contained_dyn_data_transposed = dyn_data_transposed[contain_ixs[0:len(dyn_data_transposed)], :, :]

# %%
# Finalize the training data
trainingX = examples
trainingY = trainingLiftdrag

# %%
# Finalize the validation data
# OPTION 1: 
# Val data from dynamic test 15 (only datapoints that exist in training states)
# valX = contained_dyn_data_transposed
# valY = np.stack((contained_dyn_vels, contained_dyn_aoas), axis=1)

# OPTION 2: 
# Val data is 15% of training data
split_ix = int(0.9 * trainingX.shape[0])
valX = trainingX[split_ix:]
valY = trainingY[split_ix:]
trainingX = trainingX[0:split_ix]
trainingY = trainingY[0:split_ix]


# %%
# Train the model
from networks import fnonet_1d
# importlib.reload(fnonet_1d)

def save_checkpoint_torch(epoch_cnt, model_state, optimizer_state, save_path):
  torch.save({'epoch': epoch_cnt, 'model_state': model_state,
              'optimizer_state_dict': optimizer_state}, save_path)

val_data = torch.from_numpy(valX).float().cuda()
val_labels = torch.from_numpy(valY).float().cuda()

modes = 128
width = 20
trainN = trainingX.shape[0]
signal_len = trainingX.shape[1]
in_chans = trainingX.shape[2]
out_chans = 2
valN = valX.shape[0]

learning_rate = 0.001
num_epochs = 75
batch_size = 256
best_val_loss = float('inf')

net = fnonet_1d.FNO1d(modes, width, signal_len, in_chans, out_chans).cuda()

print("Number of parameters: ", fnonet_1d.count_params(net))
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)

for epoch in range(0, num_epochs):  # loop over the dataset
  for step in range(0,trainN//batch_size):

    input_batch = torch.from_numpy(trainingX[step*batch_size:(step+1)*batch_size]).float().cuda()
    label_batch = torch.from_numpy(trainingY[step*batch_size:(step+1)*batch_size]).float().cuda()
    # print('shape of input', input_batch.shape)
    # print('shape of label', label_batch.shape)
    
    # forward + backward + optimize
    optimizer.zero_grad()
    output = fnonet_1d.directstep(net.cuda(),input_batch)
    # print('shape of FNO1D output',output.shape)
    loss = fnonet_1d.mse_loss(output, label_batch)
    loss.backward()
    optimizer.step()

    if step % 100 == 0:    # print train progress every 100 mini-batches
      print('[%d, %5d] train loss: %.3f' % (epoch + 1, step + 1, loss))

  with torch.no_grad():
    val_output = fnonet_1d.directstep(net, val_data)
    val_loss = fnonet_1d.mse_loss(val_output, val_labels)
  
  print('[%d, %5d] val loss: %.3f' % (epoch + 1, step + 1, val_loss)) # print val progress at end of each epoch.

  save_checkpoint_torch(epoch, net.state_dict(), optimizer.state_dict(), 'ckpt.tar')
  if val_loss < best_val_loss:
    save_checkpoint_torch(epoch, net.state_dict(), optimizer.state_dict(), 'best_ckpt.tar')
    best_val_loss = val_loss
    print ("Val. loss improved to: ", best_val_loss)

print('Finished Training')

# %%
# Evaluate model performance on test set
unfiltered_dyn_data = torch.from_numpy(dyn_data_transposed).float().cuda()
unfiltered_dyn_labels = dynLiftDrag.astype(float)
unfiltered_dyn_labels = torch.from_numpy(unfiltered_dyn_labels).float().cuda()
with torch.no_grad():
  unfiltered_test_output = fnonet_1d.directstep(net, unfiltered_dyn_data)
  unfiltered_test_dyn_loss = fnonet_1d.mse_loss(unfiltered_test_output, unfiltered_dyn_labels)
print('Finally MSE loss on unfiltered Dynamic 15 data: ', unfiltered_test_dyn_loss)

filtered_dyn_data = torch.from_numpy(contained_dyn_data_transposed).float().cuda()
filtered_dyn_labels = contained_dyn_liftdrag.astype(float)
filtered_dyn_labels = torch.from_numpy(filtered_dyn_labels).float().cuda()
with torch.no_grad():
  filtered_test_output = fnonet_1d.directstep(net, filtered_dyn_data)
  filtered_test_dyn_loss = fnonet_1d.mse_loss(filtered_test_output, filtered_dyn_labels)
print('Finally MSE loss on filtered Dynamic 15 data: ', filtered_test_dyn_loss)

# Evaluate BEST model performance on test set
best_ckpt = torch.load('best_ckpt.tar')
net.load_state_dict(best_ckpt['model_state'])
net.eval ()

with torch.no_grad():
  unfiltered_test_output = fnonet_1d.directstep(net, unfiltered_dyn_data)
  unfiltered_test_dyn_loss = fnonet_1d.mse_loss(unfiltered_test_output, unfiltered_dyn_labels)
print('Finally MSE loss on unfiltered Dynamic 15 data (best model): ', unfiltered_test_dyn_loss)

with torch.no_grad():
  filtered_test_output = fnonet_1d.directstep(net, filtered_dyn_data)
  filtered_test_dyn_loss = fnonet_1d.mse_loss(filtered_test_output, filtered_dyn_labels)
print('Finally MSE loss on filtered Dynamic 15 data (best model): ', filtered_test_dyn_loss)
# %%
