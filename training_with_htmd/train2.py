# Training the SqueezeNet
# Mahmudulla Hassan
# The University of Texas at El Paso
# Last modified: 10/08/2018

from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense, Input, Add, merge, concatenate
from keras.layers.convolutional import Conv3D
from keras.layers.pooling import MaxPooling3D, GlobalAveragePooling3D, AveragePooling3D
from keras import metrics
from keras import optimizers
from keras.utils import plot_model
from keras import backend as K
from keras.utils.training_utils import multi_gpu_model
from keras.utils.data_utils import Sequence
from keras.callbacks import ModelCheckpoint
from keras.initializers import he_uniform
from keras.initializers import glorot_uniform
import os
import numpy as np
import sys
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import json
import multiprocessing as mp
import pickle

sys.path.append("models/")
sys.path.append("scripts/")
from generators import DataGenerator, AugmentedDataGenerator, DataGeneratorFromDir
from models import Squeeze_model

data_dir = "../dataset"

def train(nb_batch=32, nb_epochs=100, l_rate=1e-4, multi_gpu=0):

  # Load the data
  h5f = h5py.File(os.path.join(data_dir, "data.h5"), 'r')
  # train_x, train_y = h5f['train_x'][:], h5f['train_y'][:]
  # valid_x, valid_y = h5f['valid_x'][:], h5f['valid_y'][:]
  test_x, test_y = h5f['test_x'][:], h5f['test_y'][:]
  h5f.close()

  print("Test data shape: ", test_x.shape)

  # Training parameters
  if multi_gpu : nb_batch = multi_gpu*nb_batch # Assigning same batch size to all the gpus

  # Build the model
  model_input = Input(shape=(24, 24, 24, 16))
  model = Model(inputs=model_input, outputs=Squeeze_model(model_input))

  if multi_gpu: model = multi_gpu_model(model, gpus=multi_gpu)

  # Compile the model
  model.compile(optimizer=optimizers.adam(lr=l_rate, beta_1=0.99, beta_2=0.999),
                loss='mean_squared_error')

  # checkpoint
  outputFolder = "weights"
  if not os.path.isdir(outputFolder): os.makedirs(outputFolder)
  #weigts_filepath = os.path.join(outputFolder, "weights.h5")
  #callbacks_list = [ModelCheckpoint(weigts_filepath,
  #                                  monitor='val_loss',
  #                                  verbose=1,
  #                                  save_best_only=True,
  #                                  save_weights_only=True,
  #                                  mode='auto', period=1)]

  # Train without generators
  # history = model.fit(x=train_x, y=train_y,
  #                     batch_size=nb_batch,
  #                     epochs=nb_epochs,
  #                     callbacks=callbacks_list,
  #                     validation_data=(valid_x, valid_y),
  #                     verbose=True)


  # Generators
  print("TRINING ON AUGMENTED DATA")
  # Parameters
  params = {'dim': (24, 24, 24),
            'batch_size': nb_batch,
            'n_channels': 16,
            'shuffle': True}

  # Load the sample labels and the partition
  with open("labels.json", "r") as f:
    labels = json.load(f)
  with open("partition.json", "r") as f:
    partition = json.load(f)

  data_gen = DataGeneratorFromDir(partition['train'], labels, **params)

  # Train
  history = model.fit_generator(generator=data_gen, epochs=nb_epochs, max_queue_size=10, verbose=True,
                              use_multiprocessing=True, workers=mp.cpu_count())

  # Plot training history
  #plt.figure()
  #plt.plot(history['loss'])
  #plt.plot(history['val_loss'])
  #plt.xlabel("Epochs")
  #plt.ylabel("Loss (MSE)")
  #plt.legend(['Train Loss', 'Validation Loss'])
  #plt.savefig('training_history.png', format='png', dpi=1000)
  #plt.show()

  # Save the model weights and the training history
  model.save_weights(os.path.join(outputFolder, "weights.h5"))
  with open(os.path.join(outputFolder, "history.pickle"), "wb") as f:
    pickle.dump(history.history)


  # Evaluate the model's performance
  #train_r2 = r2_score(y_true=train_y, y_pred=model.predict(train_x))
  #print("Train r2: ", train_r2)

  test_r2 = r2_score(y_true=test_y, y_pred=model.predict(test_x))
  print("Test r2: ", test_r2)



if __name__=="__main__":
  # Without augmentation
  #train(nb_epochs=100)
  # With augmented data (24 rotation)
  train(nb_epochs=1, nb_batch=10, multi_gpu=0)
