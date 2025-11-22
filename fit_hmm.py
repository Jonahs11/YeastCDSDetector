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


def run_forward_algo(pi, transitions_mat, emissions_mat, seq, seq_to_inx):
    n_states = transitions_mat.shape[0]
    T = len(seq)
    #init alpha (n, T)
    alpha = np.zeros((n_states, T)).astype(np.float64)
    
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
    
    beta = np.zeros((n_states, T)).astype(np.float64)
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


def compute_prob_states_given_obs(alpha, beta, t_inx):
    num = alpha[:,t_inx] * beta[:,t_inx]
    denom =  np.sum(alpha[:, t_inx] * beta[:, t_inx])
    state_i_probs = num / denom
    # state_i_probs = alpha[:,t_inx] * beta[:,t_inx] / np.sum(alpha[:, t_inx] * beta[:, t_inx])
    return state_i_probs

def compute_prob_state_and_next_given_obs(alpha, beta, transitions_mat, emissions_mat, seq, seq_to_inx, t_inx):
    n_states = transitions_mat.shape[0]
    res_mat = np.zeros((n_states, n_states)).astype(np.float64)
    
    next_obs_id = seq_to_inx[seq[t_inx + 1]]
    
    denom = np.sum(alpha[:, t_inx] * beta[:, t_inx])
    for next_state_inx in range(n_states):
        num = (alpha[:, t_inx] * transitions_mat[:, next_state_inx] * emissions_mat[next_state_inx, next_obs_id] * beta[next_state_inx, t_inx+1]) / denom
        res_mat[:, next_state_inx] = num
        
    return res_mat

# needed expectations
def em(alpha, beta, transitions_mat, emissions_mat, seq, seq_to_inx):
    
    T = len(seq)
    n_states = transitions_mat.shape[0]
    new_pi = compute_prob_states_given_obs(alpha, beta, 0)

    new_transition_mat = np.zeros(transitions_mat.shape).astype(np.float64)
    denom = np.zeros((n_states, )).astype(np.float64)
    for t_inx in range(T-1):
        new_transition_mat += compute_prob_state_and_next_given_obs(alpha, beta, transitions_mat, emissions_mat, seq, seq_to_inx, t_inx)
        denom += compute_prob_states_given_obs(alpha, beta, t_inx)

    new_transition_mat /= denom
    
    new_emission_mat = np.zeros(emissions_mat.shape).astype(np.float64)
    denom_emission = np.zeros((n_states, )).astype(np.float64)
    for t_inx in range(T):
        emission_id = seq_to_inx[seq[t_inx]]
        prob_state_given_obs = compute_prob_states_given_obs(alpha, beta, t_inx)
        new_emission_mat[:, emission_id] += prob_state_given_obs
        denom_emission += compute_prob_states_given_obs(alpha, beta, t_inx)
        
    new_emission_mat /= denom_emission.reshape(-1, 1)
    return new_pi, new_transition_mat, new_emission_mat
        

def fit_hmm_loop(pi, transition_mat, emissions_mat, seq, seq_to_inx, iterations=100):
    
    
    log_likes = []
    for iter in range(iterations):
        alpha = run_forward_algo(pi, transition_mat, emissions_mat, seq, seq_to_inx)
        beta = run_backward_algo(pi, transition_mat, emissions_mat, seq, seq_to_inx)
        print(alpha)
        print(beta)
        epsilon = 0.00000001
        ll = np.sum(np.log(alpha[:,-1] + epsilon))
        log_likes.append(ll)
        pi, transition_mat, emissions_mat = em(alpha, beta, transition_mat, emissions_mat, seq, seq_to_inx)

    return log_likes



if __name__ == "__main__":
    window_size = 1
    iterations=1
    seq_df = pd.read_csv(sequence_path)
    seq_df = seq_df.head(1000)
    seq = np.array(seq_df["seq"])
    pi, transitions_mat, emissions_mat, seq_to_inx = init_cpts(window_size)
    print(transitions_mat)
    print(emissions_mat)
    log_likes = fit_hmm_loop(pi, transitions_mat, emissions_mat, seq, seq_to_inx, iterations=iterations)
    # print(log_likes)

    
    

