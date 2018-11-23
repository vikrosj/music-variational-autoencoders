import numpy as np
import keras
import h5py

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(16,131), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
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
        """Generates data containing batch_size samples. Gets encoder and
        decoder input from separate files, with same ID"""

        # Initialization
        encoder_inputs = np.empty((self.batch_size, *self.dim))
        decoder_inputs = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store enc input
            encoder_inputs[i,] = np.load('encoder_inputs/' + ID + '.npy')

            decoder_inputs[i,] = np.load('decoder_inputs/' + ID + '.npy')

        return ([encoder_inputs, decoder_inputs], encoder_inputs)

class MDNDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    
    def __init__(self, list_IDs, 
                 dataset_path="h5_files/mdn_dataset_b0.2.h5", 
                 batch_size=32, 
                 dim=(16,64), 
                 shuffle=True):
        
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.dataset_path = dataset_path
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
        """Generates data containing batch_size samples. Gets encoder and
        decoder input from separate files, with same ID"""
        
        with h5py.File(self.dataset_path, "r") as h5pyFile:

            # Initialization
            mdn_inputs = np.empty((self.batch_size, *self.dim))
            target = np.empty((self.batch_size, *self.dim))

            # Generate data
            for i, ID in enumerate(list_IDs_temp):

                # Store enc input
                mdn_inputs[i,] = np.array(h5pyFile.get('z_x_' + ID )).reshape(*self.dim)

                target[i,] = np.array(h5pyFile.get('z_y_' + ID )).reshape(*self.dim)

        return mdn_inputs, target
