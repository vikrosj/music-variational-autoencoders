import numpy as np
import os
import shutil
import tqdm
import h5py

np.set_printoptions(threshold=10e6)

avg_song_len = 113

class VAEdataPrep:

    def __init__(self, timesteps, features):
        self.timesteps = timesteps
        self.features = features


    def slice_sequences(self, dataset):
        """Slice a dataset into sequences of lenght num_steps."""

        slices = []

        if len(np.array(dataset).shape) == 1:

            maxlen = len(dataset)
            steps = int(maxlen / self.timesteps)

            if steps > int(3*avg_song_len):
                print("Song too long, skipping. No. of bars:", steps)
                    
            elif steps <= 1:
                print("Song too short, skipping. No. of bars:", steps)

            else:
                idx = 0
                for i in range(steps):
                    # saving all bars from song
                    slices.append(dataset[idx:idx+self.timesteps])
                    idx += self.timesteps

        else:
            for seq in dataset:
                maxlen = len(seq)
                steps = int(maxlen / self.timesteps)

                if steps > int(3*avg_song_len):
                    print("Song too long, skipping. No. of bars:", steps)
                    
                elif steps <= 1:
                    print("Song too short, skipping. No. of bars:", steps)

                else:
                    idx = 0
                    for i in range(steps):
                        # saving all bars from song
                        slices.append(seq[idx:idx+self.timesteps])
                        idx += num_steps

        return slices

    def slice_songs(self, dataset):
        """Slice a dataset into sequences of lenght num_steps."""

        songs = []

        for seq in dataset:

            slices = []

            maxlen = len(seq)
            steps = int(maxlen / self.timesteps)
            
            if steps > int(3*avg_song_len):
                print("Song too long, skipping. No. of bars:", steps)
                    
            elif steps <= 1:
                print("Song too short, skipping. No. of bars:", steps)
            
            else:
                idx = 0
                for i in range(steps):
                    # saving all bars from song
                    subsequence = seq[idx:idx+self.timesteps]

                    # start of first sequence must be
                    # "start of sequence" token, n_unique
                    if idx == 0:
                        subsequence[0] = self.features-1

                    slices.append(subsequence)
                    idx += self.timesteps

                songs.append(slices)
        return songs

    def one_hot_encode(self, sequence):
        """
        Helper function to one hot encode a sequence.
        
        :param sequence: list of integers with length 'timesteps' (sliced in
        'slice_sequences'. 
        
        returns: 2D numpy array with shape (timesteps, features) 
        """
        
        encoding = np.zeros((len(sequence),self.features))
        encoding[np.arange(len(sequence)), sequence] = 1
        return encoding


    def one_hot_decode(self, encoded_seq):
        """
        Helper function to one hot decode a sequence.
        
        :param encoded_seq: 2D numpy array with shape (timesteps, features),
        meaning one one hot encoded bar.
        
        returns: list of length timesteps*features
        """
        return [np.argmax(vector) for vector in encoded_seq]


    def one_hot_decode_song(self, encoded_seq):
        """
        Helper function to one hot decode a song. 
        
        :param encoded_seq: 3D numpy array with shape (bars, timesteps, features),
        meaning one one hot encoded song.
        
        returns: 1D numpy array with shape (bars*timesteps, )
        """
        full_song = []

        for seq in encoded_seq:

            full_song.append(self.one_hot_decode(seq))

        return np.ravel(np.array(full_song))

    # convert encoded sequence to supervised learning
    def to_supervised(self, sequence):
        """
        Creating set for self-supervised learning. Inputs one sequence and creates
        a time shifted sequence (t-1). Adds a start of sequence token to the time
        shifted sequence.
        
        :param sequence: sequence to create self-supervised set of
        
        returns: sequence and time shifted sequence (t-1)
        """

        # initialise once
        prepend_decoder = np.zeros(self.features)
        prepend_decoder[self.features-1] = 1

        seq_len = len(sequence)
        X1 = sequence[0::]
        X2 = np.insert(X1[0:-1], [0], prepend_decoder, axis=0)

        return X1, X2

    def get_songs_dataset(self, dataset):
        """
        Creating dataset of songs. Slices songs into 3D numpy
        arrays with shape (songs, bars, timesteps), then 
        one hot encodes each song and return this 4D numpy array
        (songs, bars, timesteps, features).
        
        :param dataset: 2D numpy array of integers (songs, song_lengths)
        
        returns: 4D array of one hot encoded songs (songs, bars, timesteps, features) 
        """

        songs = self.slice_songs(dataset)
        encoded_songs = []

        for song in songs:
            encoded_bars = []
            for bar in song:
                encoded = self.one_hot_encode(bar)
                encoded_bars.append(encoded)
            encoded_songs.append(encoded_bars)

        return np.array(encoded_songs)


    def get_bars_dataset(self, dataset):
        
        """
        Slices a dataset into sequences of predefined (timesteps, features).
        One hot encodes all bars and creates encoder/decoder input pairs.
        
        :param dataset: 2D numpy array of integers (songs, song_lengths)
        
        returns: two 3D arrays of one hot encoded encoder and decoder inputs (bars, timesteps, features)
        """

        # Slice the sequences:
        # split into two sequences, one for encoder input
        # and one for decoder input

        encoder_inputs = []
        decoder_inputs = []
        target = []

        slices = self.slice_sequences(dataset)

        #for i in range(num_samples):
        for bar in slices:
            # one hot encode
            encoded = self.one_hot_encode(bar)

            # convert to X,y pairs
            X1, X2 = self.to_supervised(encoded)

            encoder_inputs.append(X1)
            decoder_inputs.append(X2)

        return np.array(encoder_inputs), np.array(decoder_inputs)


    def create_bars_dir(self,
                        encoder_directory,
                        decoder_directory,
                        dataset_file,
                        generator_IDs_file,
                        print_progress=True):
        """
        Creates two directories defined by user. If the directories exists they will be deleted and replaced.
        Loads dataset and one hot encodes all bars in the dataset. Creates one encoder and one decoder input for each bar. 
        Decoder input is the encoder input shifted backwards one time step. The first item in decoder input is 
        replaced with 'start of sequence token' 130.
        
        
        Saves each encoder/decoder input pair as a numpy-array in user defined directories.
        
        :param encoder_directory: user defined name for directory of encoder bars
        :param decoder_directory: user defined name for directory of decoder bars
        :param dataset_file: name of dataset to process
        :param generator_IDs_file: 
        :param print_progress: both boolean option and integer. 
        
        If True: prints progress for each song. 
        If False: prints nothing.
        If integer: prints progress for each k times the integer (k being a natural number)
        """
        

        # removes previous directory
        if os.path.isdir(encoder_directory):
            shutil.rmtree(encoder_directory)

        if os.path.isdir(decoder_directory):
            shutil.rmtree(decoder_directory)

        os.mkdir(encoder_directory)
        os.mkdir(decoder_directory)

        pointer = 0

        dataset = np.load(dataset_file)

        for i in range(dataset.shape[0]):

            encoder_inputs, decoder_inputs = self.get_bars_dataset(dataset[i])

            if encoder_inputs.shape[0] == 0: continue

            if print_progress:
                if pointer % print_progress == 0:
                    print("Encoder inputs shape", encoder_inputs.shape)
                    print("Decoder inputs shape",decoder_inputs.shape)
                    print("One hot decoded, 1st, encoder inputs",
                          self.one_hot_decode(encoder_inputs[0]))
                    print("One hot decoded, 1st, decoder inputs",
                          self.one_hot_decode(decoder_inputs[0]))
                    print("Total amount of sequences:", pointer, "\n\n")

            # saving encoder and decoder inputs in separate folders with same name
            for j in range(encoder_inputs.shape[0]):
                encoder_filename = encoder_directory + "/id-" + str(pointer) + ".npy"
                decoder_filename = decoder_directory + "/id-" + str(pointer) + ".npy"
                np.save(encoder_filename, encoder_inputs[j])
                np.save(decoder_filename, decoder_inputs[j])
                pointer += 1

        # generator setup
        size_of_dataset = len([f for f in os.listdir(encoder_directory + "/")])

        training_set_size = int(2 * size_of_dataset / 3)
        validation_set_size = int(size_of_dataset / 6)
        test_set_size = int(size_of_dataset / 6)

        # initialise dictionaries
        inputs = {}

        # create ID lists
        training_IDs = ['id-'+ str(i) for i in range(training_set_size)]
        validation_IDs = ['id-'+str(i+training_set_size) for i in range(validation_set_size)]
        test_IDs = ['id-'+str(i+training_set_size+validation_set_size) for i in range(test_set_size)]

        # fill in dictionaries
        inputs["train"] = training_IDs
        inputs["validation"] = validation_IDs
        inputs["test"] = test_IDs

        np.save(generator_IDs_file, inputs)


    def create_songs_dir(self,
                        encoder_directory,
                        inference_ID_file,
                        dataset_file,
                        print_progress=True):   
        """
        Creates a directory defined by user. If the directory exists it will be deleted and replaced.
        Loads dataset and one hot encodes all songs,
        saves each song a numpy-array in user defined directory.
        
        :param encoder_directory: user defined name for directory of songs
        :param inference_ID_file: user defined directory, important during inference, just place in datasets/id_lists/
        :param dataset_file: name of dataset to process
        :param print_progress: both boolean option and integer. 
        If True: prints progress for each song. 
        If False: prints nothing.
        If integer: prints progress for each k times the integer (k being a natural number)
        """

        # removes previous directory
        if os.path.isdir(encoder_directory):
            shutil.rmtree(encoder_directory)

        os.mkdir(encoder_directory)

        pointer = 0

        dataset = np.load(dataset_file)

        encoder_inputs = self.get_songs_dataset(dataset)

        # Saving just encoder inputs
        for j in range(encoder_inputs.shape[0]):

            if print_progress:
                    if pointer % print_progress == 0:
                        print("One hot decoded, 1st bar, song {}".format(j),
                              self.one_hot_decode(encoder_inputs[j][0]))

                        print("One hot decoded, 2nd bar, song {}".format(j),
                              self.one_hot_decode(encoder_inputs[j][1]))

                        print("Total amount of songs:", pointer, "\n\n")

            encoder_filename = encoder_directory + "/id-" + str(pointer) + ".npy"
            np.save(encoder_filename, encoder_inputs[j])

            pointer += 1
            
        inputs = {}
        # generator setup
        size_of_dataset = len([f for f in os.listdir(encoder_directory + "/")])

        IDs = ['id-'+ str(i) for i in range(size_of_dataset)]
        inputs["test"] = IDs

        np.save(inference_ID_file, inputs)


class MDNdataPrep:
    
    def __init__(self, timesteps, features, z_dim, z_dataset_file):
        self.timesteps = timesteps
        self.features = features
        self.z_dim = z_dim
        self.z_dataset_file = z_dataset_file
        
    
    def create_z_array(self, vae_enc, song):
        """
        Generates array of z's given source sequence.
        Helper function for function 'create_mm_data'.
        
        :param vae_enc: encoder from variational autoencoder (keras object)
        :param song: 3 dimensional array (bars, timesteps, features)
        :param z_dim: latent space dimesionality (int)
        
        returns: numpy array of latent vectors for one song
        """
        
        z_list = []

        for bar in song:

            bar = bar.reshape(1, self.timesteps, self.features)

            # encode
            encoder_output = vae_enc.predict(bar)

            z = encoder_output[2]
            z = z.reshape(1, 1, self.z_dim)
            z_list.append(z)

        return np.array(z_list)

    def create_mm_data(self, vae_enc, encoder_dir_songs):
        """
        Infers latent vectors for all files in song directory.
        
        :param vae_enc: encoder from variational autoencoder (keras object)
        :param z_dim: :param z_dim: latent space dimesionality (int)
        :param encoder_dir_songs: directory of one hot encoded song files
        :param z_dataset_file: name of z-dataset file to be created       
        """
        

        name_list = []

        for root, dirs, files in os.walk(encoder_dir_songs):
            for file in files:
                name_list.append(root + os.sep + file)

        h5f = h5py.File(self.z_dataset_file, 'w')

        i = 0

        for name in tqdm.tqdm_notebook(name_list):
            song_from_file = np.load(name)
            song_len = song_from_file.shape[0]

            # reshaping to work as input to lstm
            song = np.array(song_from_file).reshape(song_len, self.timesteps, self.features)

            # predicting list of z's
            z_array = self.create_z_array(vae_enc, song)

            # appending list of z's to dataset
            h5f.create_dataset("z_list" + str(i), data=z_array)

            i+=1

        h5f.close()
        
    def slice_z_data_for_mdn(self, sliced_z_dataset_file, generator_IDs_file, mdn_seq_len):

        zs_file = h5py.File(self.z_dataset_file, 'r')

        mdn_file = h5py.File(sliced_z_dataset_file, 'w')

        counter = 0

        for i in tqdm.tqdm_notebook(range(len(zs_file.keys()))):

            z_list = zs_file.get('z_list' + str(i))

            z_list_i = np.array(z_list)

            len_z_list = z_list_i.shape[0]

            num_steps = int(len_z_list / mdn_seq_len)

            # don't keep short songs
            if num_steps == 0: continue

            else:

                #avoiding errors if the number of steps leaves no
                #room for an extra +1 for the target
                if len_z_list % mdn_seq_len == 0:
                    num_steps = num_steps - 1

                idx = 0

                for j in range(num_steps):

                    data = z_list_i[idx : idx + mdn_seq_len]
                    target = z_list_i[idx + 1 : idx + mdn_seq_len + 1]

                    mdn_file.create_dataset("z_x_id-" + str(counter), data=data)
                    mdn_file.create_dataset("z_y_id-" + str(counter), data=target)

                    counter += 1

                    idx += mdn_seq_len

        zs_file.close()
        mdn_file.close()
        
        
        hf_mdn = h5py.File(sliced_z_dataset_file, 'r')

        size_of_dataset = len(list(hf_mdn.keys()))//2

        hf_mdn.close()

        training_set_size = 2 * size_of_dataset // 3
        validation_set_size = size_of_dataset // 6
        test_set_size = size_of_dataset // 6

        inputs = {}
        
        training_IDs = ['id-'+str(i) for i in range(training_set_size)]
        validation_IDs = ['id-'+str(i+training_set_size) for i in range(validation_set_size)]
        test_IDs = ['id-'+str(i+training_set_size+validation_set_size) for i in range(test_set_size)]
        
        inputs["train"] = training_IDs
        inputs["validation"] = validation_IDs
        inputs["test"] = test_IDs
        
        np.save(generator_IDs_file, inputs)