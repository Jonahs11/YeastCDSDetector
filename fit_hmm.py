from numpy.typing import NDArray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sequence_path = "./seq_to_coding_file.csv"


def _helper(seq_lis, window_size):
    nucs = ["A", "T", "C", "G"]
    sizes = []
    for seq in seq_lis:
        sizes.append(len(seq))
    
    assert np.all([size == sizes[0] for size in sizes])
    size = sizes[0]
    if size == window_size:
        return seq_lis
    
    all_seqs = []
    for nuc in nucs:
        new_seqs = [f"{seq}{nuc}" for seq in seq_lis]
        all_seqs.extend(_helper(new_seqs, window_size))
    
    return all_seqs
        
    

def create_all_possible_combos(window_size):
    possible_seqs = _helper([""], window_size)
    return possible_seqs
            

# assume 2 states, coding non coding
def init_cpts(window_size: int):
    n_possible_emissions = 4 ** window_size
    
    # assume less likely to start coding
    pi = np.array([0.9, 0.1])
    
    # as regions are contigous make keep state higher
    transitions = np.array([[0.9, 0.1], [0.1, 0.9]])

    emissions  = np.random.rand(2, n_possible_emissions)
    emissions = emissions / emissions.sum(axis=1).reshape(-1, 1)
    
    all_possible_seqs = create_all_possible_combos(window_size)
    seq_to_inx = {}
    for inx, seq in enumerate(all_possible_seqs):
        seq_to_inx[seq] = inx
    
    return pi, transitions, emissions, seq_to_inx


# needed expectations
# P(state at time 1 | sequence)
# P(state at time t, state at time t+1 | sequence)
# P(state at time t, emission t | sequence)
def run_forward_algo(pi, transitions_mat, emissions_mat, seq, seq_to_inx):
    n_states = transitions_mat.shape[0]
    T = len(seq)
    #init alpha (n, T)
    alpha = np.zeros((n_states, T))
    
    print(seq_to_inx)
    # initialization
    first_seq = seq[0]
    first_inx = seq_to_inx[first_seq]
    emissions_first = emissions_mat[:, first_inx]
    alpha[:, 0] = pi * emissions_first
    
    for t_inx in range(1, T):
        for state_inx in range(n_states):
            prev_alpha = alpha[:, t_inx-1]
            transitions = transitions_mat[:, state_inx]
            current_emission_inx = seq_to_inx[seq[t_inx]]
            emission = emissions_mat[state_inx, current_emission_inx]
            alpha[state_inx, t_inx] += np.sum(prev_alpha * transitions * emission)
    return alpha

def run_backward_algo(pi, transitions_mat, emissions_mat, seq, seq_to_inx):
    n_states = transitions_mat.shape[0]
    T = len(seq)
    
    beta = np.zeros((n_states, T))
    # set last col to ones
    beta[:, -1] = 1
    
    for inx in range(T-2, -1, -1):
        next_beta = beta[:, inx + 1]
        future_state_id = seq_to_inx[seq[inx+1]]
        emission_prob = emissions_mat[:, future_state_id]

        for state_inx in range(n_states):
            transition_probs = transitions_mat[state_inx,: ]
            beta[state_inx, inx] += np.sum(next_beta * transition_probs * emission_prob)
        
    return beta




if __name__ == "__main__":
    window_size = 1
    seq_df = pd.read_csv(sequence_path)
    seq_df = seq_df.head(100)
    seq = np.array(seq_df["seq"])
    pi, transitions_mat, emissions_mat, seq_to_inx = init_cpts(window_size)
    alpha = run_forward_algo(pi, transitions_mat, emissions_mat, seq, seq_to_inx)
    beta = run_backward_algo(pi, transitions_mat, emissions_mat, seq, seq_to_inx)

    print(alpha.shape)
    print(beta.shape)
    
    

