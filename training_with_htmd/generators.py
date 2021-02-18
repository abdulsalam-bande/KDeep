# Script for data augmentation #
# Mahmudulla Hassan
# The University of Texas at El Paso
# Last modified: 09/10/2018


import numpy as np
import keras
from itertools import permutations
from scipy.ndimage import interpolation
import randrot

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x, y, batch_size=16, dim=(24, 24, 24), n_channels=16, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.y = y
        self.x = x
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        rand_ids = np.random.choice(self.x.shape[0], self.batch_size)

        return self.x[rand_ids], self.y[rand_ids]


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


class DataGeneratorFromDir(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
        shuffle=True):
    #def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
    #             n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        #self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('../dataset/npy_data/' + ID + '.npy')
            #X[i,] = np.load('../../CNN_experiments/npy_data/' + ID + '.npy')
            # Store class
            y[i] = self.labels[ID]

        return X, y

class AugmentedDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x, y, batch_size=16, dim=(24, 24, 24), n_channels=16, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.y = y
        self.x = x
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        rand_ids = np.random.choice(self.x.shape[0], self.batch_size)

        return self._aug_data_generator(self.x[rand_ids], self.y[rand_ids])


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    
    def _rotate_sample(self, sample):
        output = np.zeros((24,)+sample.shape) #24 possible rotation
        counter = 0
        axes = [0, 1, 2]
        rotation_plane = permutations(axes, 2)
        rotated_sample = sample
        
        for plane in rotation_plane:
            #for angle in [0, 90, 180, 270]:
            output[counter] = rotated_sample
            counter = counter + 1
            for _ in range(3):
                rotated_sample = np.rot90(rotated_sample, axes=plane) #interpolation.rotate(input=sample, angle=angle, axes=plane, reshape=False)
                output[counter] = rotated_sample
                counter = counter + 1

        return output
        
    def _aug_data_generator(self, sample_x, sample_y):
        aug_count = 24 # 24 possible rotation
        
        aug_data_x = np.zeros((sample_x.shape[0]*aug_count, ) + sample_x.shape[1:])
        aug_data_y = np.zeros((sample_y.shape[0]*aug_count))
        
        for i in range(sample_x.shape[0]):
            
            aug_x = self._rotate_sample(sample_x[i])
            aug_y = np.repeat(sample_y[i], aug_x.shape[0])
            
            aug_data_x[i*aug_count:i*aug_count+aug_count] = aug_x
            aug_data_y[i*aug_count:i*aug_count+aug_count] = aug_y
            
        return aug_data_x, aug_data_y



class AugmentedDataGeneratorRandom(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x, y, batch_size=16, dim=(24, 24, 24), n_channels=16, rotation_count = 16, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.y = y
        self.x = x
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.rotation_count = rotation_count
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        rand_ids = np.random.choice(self.x.shape[0], self.batch_size)

        return self._random_aug_data_generator(self.x[rand_ids], self.y[rand_ids])


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    
    def _random_rotation(self, nz_positions):
	    # Get the indices where attributes are non-zero
	    #nz = np.nonzero(x)
	    #nz_positions = [(nz[0][i], nz[1][i], nz[2][i]) for i in range(len(nz[0]))]
	    # Remove duplicate coordinates as np.nonzero returns same indices multiple times for having more than one nonzero values
	    #nz_positions = list(set(nz_positions))
	    
	    # Since the given coordinates are matrix indices, they are all positive "coordinates". 
	    # To rotate them, let's make it "centered"
	    indices = np.array(nz_positions) - 12
	        
	    # Get a random rotation matrix. It's a matrix, not an ndarray!
	    R = randrot.generate_3d()
	    
	    # Rotate!
	    rotated_indices = np.transpose(np.dot(R, np.transpose(indices)))

	    # Round to int
	    rotated_indices = np.rint(rotated_indices) - 1
	    
	    # Make it all positive 
	    rotated_indices = np.array(rotated_indices + 14, dtype=np.int) # Move it 2 boxes more on the positive side
	    
	    # If any of the indices is out of bounds, then redo it.
	    if np.any(rotated_indices > 27) or np.any(rotated_indices < 0):
	        return self._random_rotation(nz_positions)
	    
	    # Convert the matrix into array and return
	    return rotated_indices
        

    def _random_rotate_samples(self, samples):
	    """
	    Rotate a sample randomly
	    
	    :param x
	    :param rotation_count
	    
	    """
	    output = np.zeros((len(samples)*(1 + self.rotation_count), 28, 28, 28, 16))
	    
	    for i, x in enumerate(samples):
	        # Get the indices where nonzero elements are
	        nz_indices = np.nonzero(x)
	        
	        # Add the original sample after padding
	        output[i*self.rotation_count+i] = np.pad(x, [(2, 2), (2, 2), (2, 2), (0, 0)], mode = 'constant')
	        
	        # Get the indices where attributes are non-zero
	        nz = np.nonzero(x)
	        nz_positions = [(nz[0][i], nz[1][i], nz[2][i]) for i in range(len(nz[0]))]
	        
	        # Remove duplicate coordinates as np.nonzero returns same indices multiple times for having more than one nonzero values
	        nz_positions = np.array(list(set(nz_positions)), dtype=np.int)
	        
	        # Add the rotated samples
	        for j in range(1, self.rotation_count+1, 1):
	            rotated_indices = self._random_rotation(nz_positions)
	            output[i*self.rotation_count+i + j, 
	                   rotated_indices[:, 0], 
	                   rotated_indices[:, 1], 
	                   rotated_indices[:, 2]] = x[nz_positions[:, 0],
	                                              nz_positions[:, 1],
	                                              nz_positions[:, 2]]
	    
	    return output


    def _random_aug_data_generator(self, sample_x, sample_y):
	    #rotation_count = 16
	    
	    aug_data_x = self._random_rotate_samples(sample_x)
	    aug_data_y = np.zeros(aug_data_x.shape[0])

	    for i in range(sample_x.shape[0]):
	        start = i*(self.rotation_count + 1)
	        end = i*(self.rotation_count + 1) + (self.rotation_count + 1)
	        aug_data_y[start:end] = np.repeat(sample_y[i], self.rotation_count + 1)
	    
	    # print(aug_data_x.shape, aug_data_y.shape)
	    return aug_data_x, aug_data_y
