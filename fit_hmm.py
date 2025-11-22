from numpy.typing import NDArray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sequence_path = "./seq_to_coding_file.csv"


# assume 2 states, coding non coding
def init_cpts(window_size: int):
    n_possible_emissions = 4 ** window_size
    
    # assume less likely to start coding
    pi = np.array([0.9, 0.1])
    
    # as regions are contigous make keep state higher
    alpha = np.array([[0.9, 0.1], [0.1, 0.9]])

    beta  = np.random.rand(n_possible_emissions, 2)
    beta = beta / beta.sum(axis=1).reshape(-1, 1)

    return pi, alpha, beta

if __name__ == "__main__":
    window_size = 1
    seq_df = pd.read_csv(sequence_path)
    pi, alpha, beta = init_cpts(window_size)
    
    print(pi)
    print(alpha)
    print(beta)

    
    

