# Training the SqueezeNet
# Mahmudulla Hassan
# The University of Texas at El Paso
# Last modified: 03/13/2019

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from keras.models import Model
from keras.layers import Input
from keras import optimizers
from keras.utils import multi_gpu_model
#from keras.utils.training_utils import multi_gpu_model # Doesn't work in windows for god knows why!!
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import multiprocessing as mp
import numpy as np
import h5py, json, pickle, os, glob
#import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error
from generators import DataGeneratorFromDir
from model import SqueezeModel
from time import time

## required for efficient GPU use
import tensorflow as tf
from keras.backend import tensorflow_backend
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

def train(nb_batch, nb_epochs, l_rate, multi_gpu, train_data_dir, test_data, output_dir, plot_figure=False, early_stop=False):
  # Check if output dir exists, otherwise create one
  if os.path.isdir(output_dir):
      print("OUTPUT DIRECTORY {} ALREADY EXISTS! EXITING THE PROGRAM!!!".format(output_dir))
      return
  os.makedirs(output_dir)
  # Load the test data
  h5f = h5py.File(test_data, "r")
  test_x, test_y = h5f['x'][:], h5f['y'][:]
  h5f.close()

  # Build the model
  model_input = Input(shape=(test_x.shape[1:]))
  squeeze_model = Model(inputs=model_input, outputs=SqueezeModel(model_input))
  print("MODEL SUMMARY: ")
  print(squeeze_model.summary())
  model = multi_gpu_model(squeeze_model, gpus=multi_gpu) if multi_gpu > 1 else squeeze_model

  # Compile the model
  model.compile(optimizer=optimizers.adam(lr=l_rate, beta_1=0.99, beta_2=0.999),
                loss='mean_squared_error')

  weights_filepath = os.path.join(output_dir, 'weights.{epoch:02d}-{val_loss:.2f}.ckpt')
  callbacks_list = [ModelCheckpoint(weights_filepath,
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='auto', period=1),
                    TensorBoard(log_dir=output_dir+"/logs/{}".format(time()))]
  if early_stop:
      callbacks_list.append(EarlyStopping(monitor='val_loss', verbose=1, patience=10))

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

  # Downsampling
  #test_x = test_x[:, ::2, ::2, ::2]
  valid_x, valid_y = test_x[::24], test_y[::24]  # Take only the original samples

  # Train
  history = model.fit_generator(generator = data_gen,
                                validation_data = (valid_x, valid_y),
                                epochs = nb_epochs,
                                max_queue_size = 20,
                                verbose = True,
                                use_multiprocessing = False,
                                workers = mp.cpu_count(),
                                callbacks = callbacks_list)

  if plot_figure:
	  # Plot training history
	  plt.figure(figsize=(10, 5))
	  plt.plot(history.history['loss'])
	  plt.plot(history.history['val_loss'])
	  plt.xlabel("Epochs")
	  plt.ylabel("Loss (MSE)")
	  plt.legend(['Train Loss', 'Validation Loss'])
	  plt.savefig(os.path.join(output_dir, 'training_history.png'), format='png', dpi=1000)

  # Save the model weights and the training history
  squeeze_model.save_weights(os.path.join(output_dir, "weights.h5"))
  squeeze_model.save(os.path.join(output_dir, "model.h5"))

  with open(os.path.join(output_dir, "history.pickle"), "wb") as f:
    pickle.dump(history.history, f)

  # Evaluate the model
  pred = model.predict(test_x).ravel()
  pred = np.mean(pred.reshape((-1, 24)), axis=1)
  test_y = np.mean(test_y.reshape((-1, 24)), axis=1)
  pearson_r = pearsonr(test_y, pred)[0]
  rmse = np.sqrt(mean_squared_error(y_true=test_y, y_pred=pred))
  output_text = "TRAINING DATA: {} \nTEST_DATA: {} \n\n".format(os.path.abspath(train_data_dir), os.path.abspath(test_data))
  output_text += "TRAINING PARAMETERS: \n"
  output_text += "\tLEARNING RATE: {} \n".format(l_rate)
  output_text += "\tBATCH SIZE: {} \n".format(nb_batch)
  output_text += "\tEPOCHS: {} \n\n".format(nb_epochs)
  output_text += "TEST PEARSON R: {:.2f}, TEST RMSE: {:.2f}".format(pearson_r, rmse)

  # Evaluate the model using the last checkpoint
  checkpoints = glob.glob(os.path.join(output_dir, "*.ckpt"))
  best_checkpoint = max(checkpoints, key=os.path.getctime) # Last one is the best
  squeeze_model = Model(inputs=model_input, outputs=SqueezeModel(model_input))
  print("Loading the checkpoint {}".format(best_checkpoint))
  squeeze_model.load_weights(best_checkpoint)

  # Evaluate the model
  pred = model.predict(test_x).ravel()
  pred = np.mean(pred.reshape((-1, 24)), axis=1)
  pearson_r = pearsonr(test_y, pred)[0]
  rmse = np.sqrt(mean_squared_error(y_true=test_y, y_pred=pred))
  output_text += "\n[CHECKPOINT RESULT] TEST PEARSON R: {:.3f}, TEST RMSE: {:.3f}".format(pearson_r, rmse)

  # Write the results
  with open(os.path.join(output_dir, 'log.txt'), 'w') as f:
    f.write(output_text)

  print(output_text, "\nLog is written to " + os.path.abspath(os.path.join(output_dir, 'log.txt')))


if __name__=="__main__":
  train_data_dir = "../dataset/dataset2"
  test_data = "../dataset/dataset2/test_data.h5"

  #train_data_dir = "dataset/npy_data32"
  #test_data = "dataset/test_data_augmented.h5"
  output_dir = "test_run"
  l_rate = 1e-4
  nb_batch = 128
  nb_epochs = 50
  multi_gpu = 1 # Value more than 1 enables multi GPU training. Number of GPUs = multi_gpu
  train(nb_batch, nb_epochs, l_rate, multi_gpu, train_data_dir, test_data, output_dir)


  #batch_sizes = [32, 64, 128, 256, 512]
  #learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
  #counter = 0
  #for i, bs in enumerate(batch_sizes):
  #  for j, lr in enumerate(learning_rates):
  #    counter += 1
  #    output_dir = "weights_" + str(counter)
  #    if os.path.isdir(output_dir): continue
  #    print("TRAINING WITH BATCH SIZE: {} AND LEARNING RATE: {}".format(bs, lr))
  #    train(bs, nb_epochs, lr, multi_gpu, train_data_dir, test_data, output_dir)

  # if os.path.isdir(output_dir):
    # raise Exception('Output directory \"{}\" already exists, choose a different directory'.format(output_dir))
  # else:
    # try:
      # os.makedirs(output_dir)
    # except Exception as e:
      # print(str(e))
