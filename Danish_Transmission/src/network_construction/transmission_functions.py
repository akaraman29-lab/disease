#### File containing helper functions for calculating probabilities of observing genetic and 
#### testing data between two individuals, A and B, if A and B form a transmission pair

import numpy as np
import seaborn as sns
import pandas as pd
import polars as pl
from tqdm import tqdm
from matplotlib import pyplot as plt
from datetime import datetime
from joblib import Parallel, delayed
import scipy as sp
cmap = sns.color_palette('Set2')


# Generation Time Parameters
gmean = 4.87
gsd = 1.98
a = gmean**2 / gsd**2
b = gsd**2 / gmean 

# Mean and Variance of testing delay distribution
ln_mean = np.log(4.0)
ln_var = (np.log(4.7) - ln_mean)*2

# Time from infection to testing positive
def time_to_pcr_pos(t_diff, m, sig2):
    return sp.stats.lognorm.cdf(t_diff + 0.5, m, sig2) - sp.stats.lognorm.cdf(t_diff - 0.5, m, sig2) 


# Generation time + discrete version
def gamma_probability(t_diff, a, b, shift = 0):
    return sp.stats.gamma.pdf(t_diff-shift, a, b)

def gamma_probability_discrete(t_diff, a = a, b = b, shift = 0):
    return sp.stats.gamma.cdf(t_diff-shift + 0.5, a, b) - sp.stats.gamma.cdf(t_diff-shift - 0.5, a, b) 

# Probability for observing a number of substitutions
def substitution_probability(nsubs, t_diff, sub_rate, inhomogeneous = False):
    if inhomogeneous == False:
        return ((sub_rate*t_diff)**nsubs)*np.exp(-sub_rate * t_diff) / sp.special.gamma(nsubs +1)
    else:
        prob = 0
        # Choose a sensible cutoff
        for t in range(10):
            if t<=t_diff:
                prob += ((sub_rate*t_diff)**nsubs)*np.exp(-sub_rate * t_diff) / sp.special.gamma(nsubs +1) * time_to_pcr_pos(t, ln_mean, ln_var)
            else:
                t_before_test = t - t_diff
                prob += ((sub_rate*(t_before_test+t))**nsubs)*np.exp(-sub_rate * (t_before_test+t)) / sp.special.gamma(nsubs +1) * time_to_pcr_pos(t, ln_mean, ln_var)
        return prob


def calculate_probabilities(t_diff, nsubs, sub_rate = 1/11, shift = 0, 
                            serial_interval = gamma_probability_discrete,
                            poisson_model = substitution_probability):
    return poisson_model(nsubs, t_diff, sub_rate = sub_rate) * serial_interval(t_diff, shift = shift)




# Probability of observing a number of substitutions between two individuals in a potential transmission pair.
def scenario1_probability(t_diff, nsubs, sub_rate = 1/11, shift = 0, 
                          serial_interval = gamma_probability_discrete, 
                          poisson_model = substitution_probability, 
                          testing_delay = time_to_pcr_pos):
    prob = 0
    # Choose a sensible cutoff -30 days
    for t in range(30):
        if t<=t_diff:
            prob += poisson_model(nsubs, t_diff, sub_rate = sub_rate) * serial_interval(t_diff, shift = shift) * time_to_pcr_pos(t, ln_mean, ln_var)
        else:
            t_before_test = t - t_diff
            prob += poisson_model(nsubs, t_before_test + t, sub_rate = sub_rate) * serial_interval(t_diff, shift = shift) * time_to_pcr_pos(t, ln_mean, ln_var)
    return prob


# Scenario 1 probability with inflation at day zero, with zero hamming distance. 
def scenario1_probability_zero_inflate(t_diff, nsubs, sub_rate = 1/11, shift = 0,
                                       serial_interval = gamma_probability_discrete, 
                                       poisson_model = substitution_probability,
                                       testing_delay = time_to_pcr_pos, zero_inflate = 1e-5):
    prob = 0
    # Choose a sensible cutoff -30 days
    if t_diff == 0 and nsubs == 0:
        return zero_inflate
    else:
        for t in range(30):
            if t<=t_diff:
                prob += (poisson_model(nsubs, t_diff, sub_rate = sub_rate) * 
                         serial_interval(t_diff, shift = shift) * 
                         time_to_pcr_pos(t, ln_mean, ln_var))
            else:
                t_before_test = t - t_diff
                prob += (poisson_model(nsubs, t_before_test + t, sub_rate = sub_rate) * 
                         serial_interval(t_diff, shift = shift) * 
                         time_to_pcr_pos(t, ln_mean, ln_var))
        return prob * (1-zero_inflate)

# Probability of observing a number of substitutions between two individuals in a potential transmission pair.
def scenario1_probability_tests(t_diff, nsubs, t_diff_test, sub_rate = 1/11, shift = 0,
                                serial_interval = gamma_probability_discrete, 
                                poisson_model = substitution_probability,
                                testing_delay = time_to_pcr_pos):
    prob = 0
    # Choose a sensible cutoff -30 days
    for t in range(30):
        if t<=t_diff:
            prob += poisson_model(nsubs, t_diff, sub_rate = sub_rate) * serial_interval(t_diff_test, shift = shift) * time_to_pcr_pos(t, ln_mean, ln_var)
        else:
            t_before_test = t - t_diff
            prob += poisson_model(nsubs, t_before_test + t, sub_rate = sub_rate) * serial_interval(t_diff_test, shift = shift) * time_to_pcr_pos(t, ln_mean, ln_var)
    return prob

# Scenario 1 with added probability of zero hamming distance and zero days between testing 
def scenario1_probability_tests_inflate_zero(t_diff, nsubs, t_diff_test, sub_rate = 1/11, shift = 0, 
                                             serial_interval = gamma_probability_discrete,
                                             poisson_model = substitution_probability, testing_delay = time_to_pcr_pos, 
                                            zero_inflate = 1e-5):
    
    prob = 0
    if t_diff == 0 and nsubs == 0:
        return zero_inflate
    else:
    # Choose a sensible cutoff -30 days
        for t in range(30):
            if t<=t_diff:
                prob += poisson_model(nsubs, t_diff, sub_rate = sub_rate) * serial_interval(t_diff_test, shift = shift) * time_to_pcr_pos(t, ln_mean, ln_var)
            else:
                t_before_test = t - t_diff
                prob += poisson_model(nsubs, t_before_test + t, sub_rate = sub_rate) * serial_interval(t_diff_test, shift = shift) * time_to_pcr_pos(t, ln_mean, ln_var)
        return prob * (1-zero_inflate)
        
    


# Adjacency matrix without weights -- we do not typically use this as we use the weights in the network. 
# Now use weighted_adj_mat, which appears directly in the .py files for generating the transmission networks. 

def adjacency_matrix(hamming_mat, daydiff_mat, idxs, nseqs):
    infection_adj_mat = np.zeros((nseqs, nseqs))
    for i in tqdm(range(nseqs)):
        for j in range(nseqs):
            infection_adj_mat[i, j] = ((int(hamming_mat[i, j]), daydiff_mat[i, j]) in idxs) * (daydiff_mat[i, j] >= 0)  

        infection_adj_mat[i, i] = 0
    return infection_adj_mat
