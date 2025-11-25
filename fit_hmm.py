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
    log_alpha = np.zeros((n_states, T)).astype(np.float64)
    
    # initialization
    first_seq = seq[0]
    first_inx = seq_to_inx[first_seq]
    emissions_first = emissions_mat[:, first_inx]
    log_alpha[:, 0] = np.log(pi) + np.log(emissions_first)
    
    for t_inx in range(1, T):
        for state_inx in range(n_states):
            prev_log_alpha = log_alpha[:, t_inx-1]
            transitions = transitions_mat[:, state_inx]
            current_emission_inx = seq_to_inx[seq[t_inx]]
            emission = emissions_mat[state_inx, current_emission_inx]
            # alpha[state_inx, t_inx] += np.sum(prev_alpha * transitions * emission)
            # log_alpha[state_inx, t_inx] += np.log(np.sum(np.exp(prev_log_alpha + np.log(transitions) + np.log(emission))))

            # add numerical stability
            log_terms = prev_log_alpha + np.log(transitions)
            shift = np.max(log_terms)
            log_sum_exp = shift + np.log(np.sum(np.exp(log_terms - shift)))
            log_alpha[state_inx, t_inx] = log_sum_exp + np.log(emission)

    return log_alpha

def run_backward_algo(pi, transitions_mat, emissions_mat, seq, seq_to_inx):
    n_states = transitions_mat.shape[0]
    T = len(seq)
    
    log_beta = np.zeros((n_states, T)).astype(np.float64)
    # set last col to ones
    log_beta[:, -1] = 0
    
    for inx in range(T-2, -1, -1):
        next_log_beta = log_beta[:, inx + 1]
        future_state_id = seq_to_inx[seq[inx+1]]
        emission_prob = emissions_mat[:, future_state_id]

        for state_inx in range(n_states):
            transition_probs = transitions_mat[state_inx,: ]
            # log_beta[state_inx, inx] += np.log(np.sum(np.exp(next_log_beta + np.log(transition_probs) + np.log(emission_prob))))
            # add numerical stability
            log_terms = next_log_beta + np.log(transition_probs) + np.log(emission_prob)
            shift = np.max(log_terms)
            log_beta[state_inx, inx] = shift + np.log(np.sum(np.exp(log_terms - shift)))
        
    return log_beta

# P(S_t = i | emisions)
def compute_prob_states_given_obs(log_alpha, log_beta, t_inx):

    # num = alpha[:,t_inx] * beta[:,t_inx]
    # denom =  np.sum(alpha[:, t_inx] * beta[:, t_inx])

    log_num = log_alpha[:,t_inx] + log_beta[:,t_inx]
    shift = np.max(log_num)
    log_denom = shift + np.log(np.sum(np.exp(log_num - shift)))

    # num = np.exp(log_alpha[:,t_inx] + log_beta[:,t_inx])
    # denom =  np.sum(np.exp(log_alpha[:, t_inx] + log_beta[:, t_inx]))
    state_i_probs = np.exp(log_num - log_denom)
    # state_i_probs = alpha[:,t_inx] * beta[:,t_inx] / np.sum(alpha[:, t_inx] * beta[:, t_inx])
    return state_i_probs

# P(S_t = i, St+1 = j | emisions)
def compute_prob_state_and_next_given_obs(log_alpha, log_beta, transitions_mat, emissions_mat, seq, seq_to_inx, t_inx):
    n_states = transitions_mat.shape[0]
    res_mat = np.zeros((n_states, n_states)).astype(np.float64)
    
    next_obs_id = seq_to_inx[seq[t_inx + 1]]
    
    # log_denom = np.sum(np.exp(log_alpha[:, t_inx] + log_beta[:, t_inx]))

    # Compute full log numerator for all i,j at once
    log_num_mat = (log_alpha[:, t_inx][:, None] + 
                np.log(transitions_mat) + 
                np.log(emissions_mat[:, next_obs_id])[None, :] + 
                log_beta[:, t_inx+1][None, :])

    # Stable logsumexp over all i,j
    shift = np.max(log_num_mat)
    log_denom = shift + np.log(np.sum(np.exp(log_num_mat - shift)))

    # Result
    res_mat = np.exp(log_num_mat - log_denom)
        
    return res_mat

# needed expectations
def em(alpha, beta, transitions_mat, emissions_mat, seq, seq_to_inx):
    
    T = len(seq)
    n_states = transitions_mat.shape[0]
    new_pi = compute_prob_states_given_obs(alpha, beta, 0)

    # Update transition matrix
    new_transition_mat = np.zeros(transitions_mat.shape, dtype=np.float64)
    for t_inx in range(T-1):
        new_transition_mat += compute_prob_state_and_next_given_obs(
            alpha, beta, transitions_mat, emissions_mat, seq, seq_to_inx, t_inx
        )
    # Normalize each row to sum to 1
    new_transition_mat /= new_transition_mat.sum(axis=1)[:, None]
    
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
        # print("Iter: ",str(iter))
        log_alphas = run_forward_algo(pi, transition_mat, emissions_mat, seq, seq_to_inx)
        log_betas = run_backward_algo(pi, transition_mat, emissions_mat, seq, seq_to_inx)
        # print("Alphas: ")
        # print(log_alphas)
        # print("Betas: ")
            
        epsilon = 0.00000001
        ll = np.sum(log_alphas[:,-1] + epsilon)
        log_likes.append(ll)
        pi, transition_mat, emissions_mat = em(log_alphas, log_betas, transition_mat, emissions_mat, seq, seq_to_inx)

    return log_likes, pi, transition_mat, emissions_mat



if __name__ == "__main__":
    window_size = 1
    iterations=20
    seq_df = pd.read_csv(sequence_path)
    seq_df = seq_df.head(1000)
    seq = np.array(seq_df["seq"])
    pi, transitions_mat, emissions_mat, seq_to_inx = init_cpts(window_size)
    print("Initial Pi")
    print(pi)
    print("Initial Transitions")
    print(transitions_mat)
    print("Initial Emissions")
    print(emissions_mat)
    log_likes, pi, transitions_mat, emissions_mat = fit_hmm_loop(pi, transitions_mat, emissions_mat, seq, seq_to_inx, iterations=iterations)
    print("Log Likelihoods")
    print(np.round(log_likes,4))

    print("Final Pi")
    print(np.round(pi,3))
    print("Final Transitions")
    print(np.round(transitions_mat,3))
    print("Final Emissions")
    print(np.round(emissions_mat,3))


# try to run on simulated data:
def runSimulatedData():
    # HMM parameters
    states = ['S1', 'S2']  # hidden states
    observations = ['A', 'T', 'C', 'G']  # observed symbols
    true_transition = np.array([[0.99, 0.01],
                [0.01, 0.99]])
    true_emission = np.array([[0.05, 0.6, 0.3, 0.05],   
                [0.3, 0.2, 0.2, 0.3]])
    true_pi = np.array([0.6, 0.4])
    T_seq = 10000

    # Generate HMM sequence
    hidden_states = [np.random.choice(len(states), p=true_pi)]
    observed_seq = [np.random.choice(len(observations), p=true_emission[hidden_states[0]])]

    for t in range(1, T_seq):
        hidden_states.append(np.random.choice(len(states), p=true_transition[hidden_states[-1]]))
        observed_seq.append(np.random.choice(len(observations), p=true_emission[hidden_states[-1]]))

    hidden_states = np.array(hidden_states)
    observed_seq = np.array(observed_seq)

    # Map numerical observations to letters
    observed_seq_letters = [observations[i] for i in observed_seq]

    nstates = 2
    noutputs = 4
    init_pi = np.random.rand(2)
    init_pi /= init_pi.sum()

    init_transition = np.random.rand(nstates, nstates)
    init_transition /= init_transition.sum(axis=1, keepdims=True)
    # print(init_transition)

    init_emission = np.random.rand(nstates, noutputs)
    init_emission /= init_emission.sum(axis=1, keepdims=True) 

    log_likes, pi, transitions_mat, emissions_mat = fit_hmm_loop(init_pi, init_transition, init_emission, np.array(observed_seq_letters), seq_to_inx, iterations=30)
    print("True Pi")
    print(true_pi)
    print("Final Pi")
    print(np.round(pi,3))
    print("True Transitions")
    print(true_transition)
    print("Final Transitions")
    print(np.round(transitions_mat,3))
    print("True Emissions")
    print(true_emission)
    print("Final Emissions")
    print(np.round(emissions_mat,3))
    print(np.round(log_likes,3))
runSimulatedData()