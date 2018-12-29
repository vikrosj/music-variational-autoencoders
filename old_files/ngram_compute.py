import h5py
from numba import jit
from math import log


def int_list_to_str(input_list):
    """
    Converting int list to str for comparing in h5 file
    """
    
    int_string = " ".join(str(i) for i in input_list)
    return int_string

@jit
def compute_prob(prob_n, prob_n_1):
    """
    MLE with Laplace smoothing. 
    
    No. of unique tokens is 130.
    """
    return log((prob_n + 1)/(prob_n_1 + 130))


def compute_ngram_prob(sequence):

    with h5py.File("h5_files/ngram_songs_post(orig_range).h5", "r") as some:
        prob = 0

        for i in range(len(sequence)-4):
            
            ngram5_str = int_list_to_str(sequence[i:i+5])
            ngram4_str = int_list_to_str(sequence[i:i+4])

            try:
                prob5 = list(some.get(ngram5_str))[0][0]

            except Exception as e:
                #print("Error:", e)
                prob5 = 0
            try:
                prob4 = list(some.get(ngram4_str))[0][0]

            except Exception as e:
                #print("Error:", e)
                prob4 = 0

            prob += compute_prob(prob5, prob4)
            
        return prob


