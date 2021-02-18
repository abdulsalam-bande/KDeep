# Training the SqueezeNet
# Mahmudulla Hassan
# The University of Texas at El Paso
# Last modified: 02/22/2019

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from keras.models import Model
from keras.layers import Input
from keras import optimizers
from keras.utils import multi_gpu_model
#from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import multiprocessing as mp
import numpy as np
import h5py, json, pickle, os, glob
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error
from generators import DataGeneratorFromDir
from model import SqueezeModel2 as SqueezeModel
from time import time
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

def train(nb_batch, nb_epochs, l_rate, multi_gpu, train_data_dir, test_data, output_dir, weight_file=None):
  # Load the test data
  h5f = h5py.File(test_data, "r")
  test_x, test_y = h5f['x'][:], h5f['y'][:]
  h5f.close()
  
  # Weight file to be generated
  if not os.path.isdir(output_dir): os.makedirs(output_dir)
  output_weight = os.path.join(output_dir, "weights.h5")
  
  # Build the model
  model_input = Input(shape=(test_x.shape[1:]))
  squeeze_model = Model(inputs=model_input, outputs=SqueezeModel(model_input))
  model = multi_gpu_model(squeeze_model, gpus=multi_gpu) if multi_gpu > 1 else squeeze_model

  # Compile the model
  model.compile(optimizer=optimizers.adam(lr=l_rate, beta_1=0.99, beta_2=0.999),
                loss='mean_squared_error')
	
  if weight_file is not None:
    print("Loaded weight: {}".format(weight_file))
    model.load_weights(weight_file)
	
  # Parameters
  params = {'data_dir': train_data_dir,
            'dim': test_x.shape[1:4],
            'batch_size': nb_batch,
            'n_channels': test_x.shape[-1],
            'shuffle': True}

  # Load the sample labels and the partition
  with open(os.path.join(train_data_dir, "labels.json"), "r") as f:
    labels = json.load(f)
  with open(os.path.join(train_data_dir, "partition.json"), "r") as f:
    partition = json.load(f)

  data_gen = DataGeneratorFromDir(partition['train'], labels, **params)

  # Train
  history = model.fit_generator(generator = data_gen,
                                # validation_data = (valid_x, valid_y),
                                epochs = nb_epochs,
                                max_queue_size = 20,
                                verbose = True,
                                use_multiprocessing = False,
                                workers = mp.cpu_count())

  # Save the model weights and the training history
  squeeze_model.save_weights(output_weight)
  # Get the prediction from the model
  pred = model.predict(test_x).ravel()
  # Find the mean prediction for each sample
  pred = np.mean(pred.reshape((-1, 24)), axis=1)
  # Get the actual values for each sample
  test_y = np.mean(test_y.reshape((-1, 24)), axis=1)

  pearson_r = pearsonr(test_y, pred)[0]
  rmse = np.sqrt(mean_squared_error(y_true=test_y, y_pred=pred))
  output_text = "TRAINING DATA: {} \nTEST_DATA: {} \n\n".format(os.path.abspath(train_data_dir), os.path.abspath(test_data))
  output_text += "TRAINING PARAMETERS: \n"
  output_text += "\tLEARNING RATE: {} \n".format(l_rate)
  output_text += "\tBATCH SIZE: {} \n".format(nb_batch)
  output_text += "\tEPOCHS: {} \n\n".format(nb_epochs)
  output_text += "TEST PEARSON R: {:.3f}, TEST RMSE: {:.3f} \n".format(pearson_r, rmse)

  # Write the results
  with open(os.path.join(output_dir, 'log.txt'), 'w') as f:
    f.write(output_text)

  print(output_text, "Log is written to " + os.path.abspath(os.path.join(output_dir, 'log.txt')))
  
  return output_weight


if __name__=="__main__":
  train_data_dir = "dataset/dataset2"
  test_data = "dataset/dataset2/test_data.h5"
  
  output_dir = "weights"
  l_rate = 1e-4
  nb_batch = 128
  nb_epochs = 100
  multi_gpu = 0 # Value more than 1 enables multi GPU training. Number of GPUs = multi_gpu
  input_weight = None

  for i in range(1, nb_epochs):
    output_dir = "weights_" + str(i)
    input_weight = train(nb_batch, 1, l_rate, multi_gpu, train_data_dir, test_data, output_dir, input_weight)
