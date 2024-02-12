import os
import sys
import pickle
import pathlib
import numpy as np
import pandas as pd

FILE_PATH = pathlib.Path(__file__).absolute().parent
AUG2022_LOC = FILE_PATH.parent.parent.parent
CODES_LOC = FILE_PATH.parent.parent

sys.path.append(str(AUG2022_LOC))

class ProcPredsOffline():
  def __init__ (self, test_data, truth_data, predictions_dir, test_name):
    self.test_data = test_data
    self.truth_data = truth_data
    self.predictions_dir = predictions_dir
    self.test_name = test_name

  def get_preds_state(self, models_loc, model_name, encoder_name):
    predictions_dir_state = os.path.join(self.predictions_dir, "state")
    self.predictions_dir_state = predictions_dir_state
    
    if not os.path.isdir(predictions_dir_state):
      os.mkdir(predictions_dir_state)
      preds = self._make_preds_state(models_loc, model_name, encoder_name)
    elif len(os.listdir(predictions_dir_state)) < 1:
      preds = self._make_preds_state(models_loc, model_name, encoder_name)
    elif self.test_name+"_"+model_name+"_preds.npy" not in os.listdir(predictions_dir_state):
      preds = self._make_preds_state(models_loc, model_name, encoder_name)
    else:
      preds = np.load(os.path.join(predictions_dir_state,self.test_name+"_"+model_name+"_preds.npy"))
    preds_df = pd.DataFrame(preds)

    if encoder_name is not None:
      preds_ser = preds_df[0].str.decode("utf-8")
      vel = preds_ser.str.split("m", expand=True)[0].to_numpy(dtype=float)
      aoa = preds_ser.str.split("_", expand=True)[1]
      aoa = aoa.str.split("d", expand=True)[0].to_numpy(dtype=float)
    else:
      vel = preds_df[0].to_numpy(dtype=float)
      aoa = preds_df[1].to_numpy(dtype=float)
    
    return vel, aoa
  
  def get_preds_stall(self, models_loc, model_name):
    predictions_dir_stall = os.path.join(self.predictions_dir, "stall")
    self.predictions_dir_stall = predictions_dir_stall
    
    if not os.path.isdir(predictions_dir_stall):
      os.mkdir(predictions_dir_stall)
      preds = self._make_preds_stall(models_loc, model_name)
    elif len(os.listdir(predictions_dir_stall)) < 1:
      preds = self._make_preds_stall(models_loc, model_name)
    elif self.test_name+"_"+model_name+"_preds.npy" not in os.listdir(predictions_dir_stall):
      preds = self._make_preds_stall(models_loc, model_name)
    else:
      preds = np.load(os.path.join(predictions_dir_stall,self.test_name+"_"+model_name+"_preds.npy"))

    return preds
  
  def get_preds_liftdrag(self, models_loc, model_name):
    predictions_dir_liftdrag = os.path.join(self.predictions_dir, "liftdrag")
    self.predictions_dir_liftdrag = predictions_dir_liftdrag
    
    if not os.path.isdir(predictions_dir_liftdrag):
      os.mkdir(predictions_dir_liftdrag)
      preds = self._make_preds_liftdrag(models_loc, model_name)
    elif len(os.listdir(predictions_dir_liftdrag)) < 1:
      preds = self._make_preds_liftdrag(models_loc, model_name)
    elif self.test_name+"_"+model_name+"_preds_nopeaks.npy" not in os.listdir(predictions_dir_liftdrag):
      preds = self._make_preds_liftdrag(models_loc, model_name)
    else:
      preds = np.load(os.path.join(predictions_dir_liftdrag,self.test_name+"_"+model_name+"_preds.npy"))
    
    return preds[:,0], preds[:,1]
  



  def _make_preds_state(self, models_loc, model_name, encoder_name):
    from sklearn import preprocessing
    
    if encoder_name is not None:
      with open (os.path.join(models_loc, encoder_name+".p"), "rb") as file:
        self.encoder = pickle.load(file)

      if "ResNet" in model_name:
        preds = self._make_preds_resnet(models_loc, model_name)
      if "ResNeXt" in model_name:
        preds = self._make_preds_resnet(models_loc, model_name)

      argmax_preds = np.argmax(preds, axis=1)
      out_preds = self.encoder.inverse_transform(argmax_preds)
    else:
      if "ResNet" in model_name:
        out_preds = self._make_preds_resnet_regression("state", models_loc, model_name)
        out_preds = out_preds.astype('float64')
      if "fno" in model_name:
        out_preds = self._make_preds_fno("state", models_loc, model_name)

    np.save(os.path.join(self.predictions_dir_state,self.test_name+"_"+model_name+"_preds.npy"), out_preds)
    np.savetxt(os.path.join(self.predictions_dir_state,self.test_name+"_"+model_name+"_preds.csv"), out_preds, fmt="%s", delimiter=",")

    return out_preds
  

  def _make_preds_stall(self, models_loc, model_name):
    from keras.models import load_model
    model = load_model(os.path.join(models_loc, model_name+".tf"))

    preds = model.predict(self.test_data) #shape:(num_examples, num_classes)(16144, 2 for Aug. 2022 Dynamic1 Run15)

    argmax_preds = np.argmax(preds, axis=1)

    np.save(os.path.join(self.predictions_dir_stall,self.test_name+"_"+model_name+"_preds.npy"), argmax_preds)
    np.savetxt(os.path.join(self.predictions_dir_stall,self.test_name+"_"+model_name+"_preds.csv"), argmax_preds, fmt="%s", delimiter=",")

    return argmax_preds
  

  def _make_preds_liftdrag(self, models_loc, model_name):
    if "ResNet" in model_name:
      preds = self._make_preds_resnet_regression("liftdrag", models_loc, model_name)
      preds = preds.astype('float64')
    elif "fno" in model_name:
      preds = self._make_preds_fno("liftdrag", models_loc, model_name)
    else:
      raise NotImplementedError

    np.save(os.path.join(self.predictions_dir_liftdrag,self.test_name+"_"+model_name+"_preds_nopeaks.npy"), preds)
    np.savetxt(os.path.join(self.predictions_dir_liftdrag,self.test_name+"_"+model_name+"_preds_nopeaks.csv"), preds, fmt="%s", delimiter=",")

    return preds
  



  def _make_preds_resnet(self, models_loc, model_name):
    import tensorflow as tf
    import keras
    import keras.layers
    import keras_resnet
    import keras_resnet.models
    from keras.models import load_model
    
    shape = (self.test_data.shape[2], self.test_data.shape[1])
    class_cnt = len(self.encoder.classes_)
    x = keras.layers.Input(shape)

    leakyrelu = tf.keras.layers.LeakyReLU(alpha=0.02)
    resnet_model = keras_resnet.models.ResNet1D18(x, classes=class_cnt, freeze_bn=True)
    resnet_bn_layer = keras_resnet.layers.BatchNormalization(freeze=True)

    model = load_model(os.path.join(models_loc, model_name+".tf"), custom_objects={'LeakyReLU': leakyrelu, 'ResNet1D18':resnet_model, 'BatchNormalization':resnet_bn_layer})

    unfiltered_out = model.predict(self.test_data)

    self.compute_metrics_tf("state", model, self.test_data, unfiltered_out)

    return unfiltered_out #shape:(num_examples, num_classes)(16144, 238 for Aug. 2022 Dynamic1 Run15)


  def _make_preds_resnext(self, models_loc, model_name):
    from MLmodels.ModelClasses.mod_resnet import ModResNet
    from keras.models import load_model

    model = load_model(os.path.join(models_loc, model_name+".tf"))

    return model.predict(self.test_data) #shape:(num_examples, num_classes)(16144, 238 for Aug. 2022 Dynamic1 Run15)


  def _make_preds_resnet_regression(self, out_type, models_loc, model_name):
    import tensorflow as tf
    import keras
    import keras.layers
    import keras_resnet
    import keras_resnet.models
    from keras.models import load_model
    
    shape = (self.test_data.shape[2], self.test_data.shape[1])
    preds_cnt = 2 #lift and drag OR vel and aoa
    x = keras.layers.Input(shape)

    resnet_model = keras_resnet.models.ResNet1D18Regression(inputs=x, preds=preds_cnt, freeze_bn=True)
    resnet_bn_layer = keras_resnet.layers.BatchNormalization(freeze=True)

    model = load_model(os.path.join(models_loc, model_name+".tf"), custom_objects={'ResNet1D18':resnet_model, 'BatchNormalization':resnet_bn_layer}, compile=False)
    model.compile()
    unfiltered_out = model.predict(self.test_data) #shape:(num_examples, 2)(16144, 2 for Aug. 2022 Dynamic1 Run15)

    self.compute_metrics_tf(out_type, model, self.test_data, unfiltered_out)

    return unfiltered_out
  

  def _make_preds_fno(self, out_type, models_loc, model_name):
    import torch
    from MLmodels.networks import fnonet_1d
    from MLmodels.networks import afnonet_1d
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    
    modes = 128
    width = 20
    signal_len = self.test_data.shape[1]
    in_chans = self.test_data.shape[2]
    out_chans = 2

    if "afno" in model_name:
      net = afnonet_1d.AFNONet().to(device)
      unfiltered_test_data = np.transpose(self.test_data, (0,2,1))
    elif "fno" in model_name:
      net = fnonet_1d.FNO1d(modes, width, signal_len, in_chans, out_chans).to(device)
      unfiltered_test_data = self.test_data
    else:
      raise Exception ("No other models are implemented yet")
    
    unfiltered_test_data = torch.from_numpy(unfiltered_test_data).float()
    net.zero_grad()
    
    checkpoint = torch.load(os.path.join(models_loc, model_name+".tar"), map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['model_state'])
    net.eval()

    unfiltered_out = fnonet_1d.directstep(net, unfiltered_test_data)
    unfiltered_out = unfiltered_out.detach().numpy()

    self.compute_metrics_torch(out_type, net, unfiltered_test_data, unfiltered_out)

    return unfiltered_out #shape:(num_examples, 2)(16144, 2 for Aug. 2022 Dynamic1 Run15)
  

  def compute_metrics_torch(self, out_type, model, test_data, predictions):
    import torch
    from MLmodels.networks import fnonet_1d

    truth_data, filtered_test_data, filtered_truth_data = self._get_processed_data(out_type, test_data)

    filtered_out = fnonet_1d.directstep(model, filtered_test_data)
    filtered_out = filtered_out.detach().numpy()

    unfiltered_loss = fnonet_1d.mse_loss(predictions, truth_data)
    filtered_loss = fnonet_1d.mse_loss(filtered_out, filtered_truth_data)
    print (f"Test mse on {out_type} prediction (unfiltered) = ", unfiltered_loss)
    print (f"Test mse on {out_type} prediction (filtered) = ", filtered_loss)

  def compute_metrics_tf(self, out_type, model, test_data, predictions, encoder_name=None):
    from MLmodels.networks import fnonet_1d
    if encoder_name is not None:
      raise Exception("Not implemented")
    
    truth_data, filtered_test_data, filtered_truth_data = self._get_processed_data(out_type, test_data)

    filtered_out = model.predict(filtered_test_data)

    unfiltered_loss = fnonet_1d.mse_loss(predictions, truth_data)
    filtered_loss = fnonet_1d.mse_loss(filtered_out, filtered_truth_data)
    print (f"Test mse on {out_type} prediction (unfiltered) = ", unfiltered_loss)
    print (f"Test mse on {out_type} prediction (filtered) = ", filtered_loss)


  def _get_processed_data(self, out_type, test_data):
    if out_type == 'state':
      airspeed_truth = self.truth_data['airspeed']
      aoa_truth = self.truth_data['aoa']
      truth_data = np.stack((airspeed_truth, aoa_truth)).T

      #Create a mask to filter unseen data in training set
      airspeed_lowerbound = truth_data[:,0]>=7
      airspeed_upperbound = truth_data[:,0]<=20
      aoa_lowerbound = truth_data[:,1]>=0
      aoa_upperbound = truth_data[:,1]<=16
      dyn_mask = np.where([airspeed_lowerbound, airspeed_upperbound, aoa_lowerbound, aoa_upperbound], True, False)
      dyn_mask = dyn_mask.all(axis = 0)
      
      contain_ixs = np.nonzero(dyn_mask)[0]
      filtered_truth_data = truth_data[contain_ixs]
      filtered_test_data = test_data[contain_ixs]
    elif out_type == 'liftdrag':
      lift_truth = self.truth_data['lift']
      drag_truth = self.truth_data['drag']
      truth_data = np.stack((lift_truth, drag_truth)).T

      #Create a mask to filter unseen data in training set
      lift_lowerbound = truth_data[:,0]>=-0.04 #Taken from training data
      lift_upperbound = truth_data[:,0]<=12.06 #Taken from training data
      drag_lowerbound = truth_data[:,1]>=0.09 #Taken from training data
      drag_upperbound = truth_data[:,1]<=3.76 #Taken from training data
      dyn_mask = np.where([lift_lowerbound, lift_upperbound, drag_lowerbound, drag_upperbound], True, False)
      dyn_mask = dyn_mask.all(axis = 0)
      
      contain_ixs = np.nonzero(dyn_mask)[0]
      filtered_truth_data = truth_data[contain_ixs]
      filtered_test_data = test_data[contain_ixs]

    return truth_data, filtered_test_data, filtered_truth_data