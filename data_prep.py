import numpy as np
np.set_printoptions(threshold=10e6)

n_unique = 131
prepend_decoder = np.zeros(n_unique)
prepend_decoder[n_unique-1] = 1

def slice_sequences(dataset, num_steps):
    """Slice a dataset into sequences of lenght num_steps."""

    slices = []

    for seq in dataset:
        maxlen = len(seq)
        steps = int(maxlen / num_steps)

        idx = 0
        for i in range(steps):
            # saving all bars from song
            slices.append(seq[idx:idx+num_steps])
            idx += num_steps

    return slices

def slice_songs(dataset, num_steps):
    """Slice a dataset into sequences of lenght num_steps."""

    songs = []

    for seq in dataset:

        slices = []

        maxlen = len(seq)
        steps = int(maxlen / num_steps)
        idx = 0
        for i in range(steps):
            # saving all bars from song
            subsequence = seq[idx:idx+num_steps]

            # start of first sequence must be
            # "start of sequence" token, n_unique
            if idx == 0:
                subsequence[0] = n_unique-1

            slices.append(subsequence)
            idx += num_steps

        songs.append(slices)
    return songs

def generate_sequence(seq):
    rp = np.random.randint(0,len(seq))
    new_sequence = seq[rp]
    return new_sequence

def one_hot_encode(sequence, n_unique=n_unique):
    encoding = np.zeros((len(sequence),n_unique))
    encoding[np.arange(len(sequence)), sequence] = 1
    return encoding

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return [np.argmax(vector) for vector in encoded_seq]

# convert encoded sequence to supervised learning
def to_supervised(sequence):
    """
    Creating two sequence lists for encoder and decoder,
    and one output sequence
    """

    seq_len = len(sequence)
    X1 = sequence[0::]
    X2 = np.insert(X1[0:-1], [0], prepend_decoder, axis=0)

    return X1, X2

def get_songs_dataset(dataset, timesteps):
    """Creating dataset of songs to further get dataset of z's"""

    encoder_inputs = []
    decoder_inputs = []
    target = []

    songs = slice_songs(dataset, timesteps)
    seq = np.array(songs)

    encoded_songs = []

    for song in songs:
        encoded_bars = []
        for bar in song:
            encoded = one_hot_encode(bar)
            encoded_bars.append(encoded)
        encoded_songs.append(encoded_bars)

    return encoded_songs


def get_bars_dataset(dataset, timesteps):

    # Slice the sequences:
    # split into two sequences, one for encoder input
    # and one for decoder input

    encoder_inputs = []
    decoder_inputs = []
    target = []

    slices = slice_sequences(dataset, timesteps)

    #for i in range(num_samples):
    for bar in slices:
        # one hot encode
        encoded = one_hot_encode(bar)

        # convert to X,y pairs
        X1, X2 = to_supervised(encoded)

        encoder_inputs.append(X1)
        decoder_inputs.append(X2)

    return np.array(encoder_inputs), np.array(decoder_inputs)
