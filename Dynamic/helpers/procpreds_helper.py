import os
import sys
import pickle
import pathlib
import numpy as np
import pandas as pd

FILE_PATH = pathlib.Path(__file__).absolute().parent
AUG2022_LOC = FILE_PATH.parent.parent.parent
KERASML_LOC = os.path.join(FILE_PATH.parent.parent.parent, "KerasML")
CODES_LOC = FILE_PATH.parent.parent

sys.path.append(str(AUG2022_LOC))
from KerasML.ModelClasses.mod_resnet import ModResNet

class ProcPredsOffline():
  def __init__ (self, test_data, predictions_dir, test_name):
    self.test_data = test_data
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
    preds_ser = preds_df[0].str.decode("utf-8")
    
    vel = preds_ser.str.split("m", expand=True)[0].to_numpy(dtype=float)
    aoa = preds_ser.str.split("_", expand=True)[1]
    aoa = aoa.str.split("d", expand=True)[0].to_numpy(dtype=float)
    
    return vel, aoa

  def _make_preds_state(self, models_loc, model_name, encoder_name):
    from sklearn import preprocessing
    with open (os.path.join(models_loc, encoder_name+".p"), "rb") as file:
      self.encoder = pickle.load(file)

    if "ResNet" in model_name:
      preds = self._make_preds_state_resnet(models_loc, model_name)
    if "ResNeXt" in model_name:
      preds = self._make_preds_state_resnext(models_loc, model_name)

    argmax_preds = np.argmax(preds, axis=1)
    decoded_preds = self.encoder.inverse_transform(argmax_preds)

    np.save(os.path.join(self.predictions_dir_state,self.test_name+"_"+model_name+"_preds.npy"), decoded_preds)
    np.savetxt(os.path.join(self.predictions_dir_state,self.test_name+"_"+model_name+"_preds.csv"), decoded_preds, fmt="%s", delimiter=",")

    return decoded_preds
    

  def _make_preds_state_resnet(self, models_loc, model_name):
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

    return model.predict(self.test_data) #shape:(num_examples, num_classes)(16144, 238 for Aug. 2022 Dynamic1 Run15)

  
  def _make_preds_state_resnext(self, models_loc, model_name):
    from KerasML.ModelClasses.mod_resnet import ModResNet
    from keras.models import load_model

    model = load_model(os.path.join(models_loc, model_name+".tf"))

    return model.predict(self.test_data) #shape:(num_examples, num_classes)(16144, 238 for Aug. 2022 Dynamic1 Run15)