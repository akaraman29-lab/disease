import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from Bio import SeqIO
import regex
import hammingdist
import importlib
import transmission_functions as tf
importlib.reload(tf)
import time
from glob import glob
from Bio.SeqRecord import SeqRecord
from Bio import Seq
from Bio import Phylo
import scipy as sp
import os.path 
from pathlib import Path
import polars as pl


# Flag for calcualting weighted matrix or not. 
weights = 1
# Sensitivity analysis or not (this means shifting the generation interval by shift_sensitivity = -2)
sensitivity = 0
low_sr = 1

sensitivity_output_str_list = ['', '_sensitivity']
sensitivity_output_str = sensitivity_output_str_list[sensitivity]
shift_list = [0, -2]
shift_sensitivity = shift_list[sensitivity]
# weights = int(input('Include weights in matrix (input True or False): '))

low_sr_output_str_list = ['', '_low_sr']
low_sr_output_str = low_sr_output_str_list[low_sr]

reweight_zero = False
reweight_zero_str = ['', '_reweight_zero'][int(reweight_zero)]

read_sparse = True

print('Weighted graph = ' + str(weights == 1))
print('Sensitivity Analysis: ' + str(bool(sensitivity)))
print('Reweighting zero: ' + str(reweight_zero) )

sequences_folder = ''

# Convert sequences to strings from Seq objects
def convert_to_string(Seq):
    string = str(Seq).upper()
    return regex.sub(r'[^ACTG]', '-', string)



# Max number of days - 21 is a decent cutoff to make transmission >21 days unlikely
ndays = 20 + 1
# Max number of substitutions - 9 is a decent cutoff to make transmission >9 substitutions unlikely. In practice, limit is 2
nsubs = 9
# Substitution rate - subject to sensitivity! On average 1 mutation per 11 days.
if low_sr:
    sr = 0.06
else:
    sr = 1/11
    
days = np.arange(ndays)
subs = np.arange(nsubs)
# prob_mat is a matrix whose (i, j)th entry is the probability that a transmission pair (A, B) has i days between testing and a Hamming distance of j.
prob_mat = np.zeros((ndays, nsubs+1))

for sub in range(nsubs+1):
    for d, day in enumerate(days):

        prob = tf.scenario1_probability(day, sub, sr, shift = shift_sensitivity)
        prob_mat[d, sub] = prob
prob_mat = prob_mat.T
if reweight_zero:
    prob_mat += 5e-6
    prob_mat /= np.sum(prob_mat)

# Function for calculating (unweighted) adjacency matrix - in practice we do not use this any more
def adjacency_matrix(hamming_mat, daydiff_mat, idxs, nseqs):
    infection_adj_mat = np.zeros((nseqs, nseqs))
    for i in range(nseqs):
        for j in range(nseqs):
            infection_adj_mat[i, j] = ((int(hamming_mat[i, j]), daydiff_mat[i, j]) in idxs) * (daydiff_mat[i, j] >= 0) 

    np.fill_diagonal(infection_adj_mat, 0)
    return infection_adj_mat

# Function for calculating weighted adjacency matrix - this forms the network
def weighted_adj_mat(hamming_mat, daydiff_mat, idxs, nseqs, prob_mat):
    weighted_mat = np.zeros((nseqs, nseqs))
    for i in tqdm(range(nseqs)):
        for j in range(nseqs):
            hdist = int(hamming_mat[i, j])
            ddiff = int(daydiff_mat[i, j])
            if (hdist, ddiff) in valid_idxs:
                weighted_mat[i, j] = prob_mat[hdist, ddiff] * (ddiff >= 0) 
    np.fill_diagonal(weighted_mat, 0)
    return weighted_mat
    
# Alternative (more efficient) function for calculating weighted adjacency matrix
def weighted_adj_mat_2(hamming_mat, daydiff_mat, idxs, prob_mat = prob_mat):
    new_dist = hamming_mat
    new_dates = daydiff_mat.astype(int)
    
    valid_distances = [v for (v, d) in idxs]
    valid_daydiffs = [d for (v, d) in idxs]

    invalid_distances = ~np.isin(new_dist, valid_distances)
    invalid_daydiffs =  ~np.isin(new_dates, valid_daydiffs)

    invalid_idxs = invalid_daydiffs + invalid_distances
    weighted_mat = prob_mat[new_dist*(1-invalid_idxs), new_dates*(1-invalid_idxs)]
    np.fill_diagonal(weighted_mat, 0)
    weighted_mat[invalid_idxs] = 0
    return weighted_mat


# Get indices of probability space corresponding to plausible transmission via a Poisson process, based on prob_mat
def get_idxs(ndays = 20 + 1,
            nsubs = 9,
            sr = sr, 
            shift = 0):

    days = np.arange(ndays)
    subs = np.arange(nsubs)
    prob_mat = np.zeros((ndays, nsubs+1))
    for sub in range(nsubs+1):
        for d, day in enumerate(days):

            prob = tf.scenario1_probability(day, sub, sr)
            prob_mat[d, sub] = prob
        # prob_mats += [prob_mat]
    prob_mat = prob_mat.T
    time = np.arange(21)
    cprob = (np.cumsum(tf.gamma_probability_discrete(time)))
    p_cutoff = 0.95
    t_cutoff = time[np.argwhere(cprob > p_cutoff)[0][0]]
    prob_mat_cutoff = np.zeros_like(prob_mat)
    all_subs = np.arange(10)
    inhomogeneous = False
    tdiff1 = 6

    for sub in range(nsubs+1):

        for d, day in enumerate(days):
            cprob_sub = (np.sum(tf.substitution_probability(all_subs[:sub+1], d, sr, inhomogeneous=inhomogeneous)))
            cprob_day = (np.sum(tf.gamma_probability_discrete(time[:d+1], shift = shift)))
            

            prob_mat_cutoff[sub, d] = (int(cprob_sub <= p_cutoff) * int(d <=t_cutoff))
    valid_idxs = [(0, 0)] + list(zip(*np.where(prob_mat_cutoff == 1)))
    return valid_idxs


valid_idxs = get_idxs(shift = shift_sensitivity)

print('Reading Sequences and Metadata')
trees_folder = "../Large_Networks/"
# Read in metadata
metadata = pd.read_csv('../../sequence_metadata.csv')
seq_path_list = ['Eta_B.1.525_like++(Other)',
                 "Alpha_B.1.1++(Other_B.1.1)", 
                 "Omicron_BA.1_like++(Probable_Omicron_BA.1_like_and_Unassigned)", 
                 'wildtype']

# Read in sequences
file_path = '../Large_networks/*'


sequences_path = sequences_folder + './sequences.fasta'
start = time.time()
data = SeqIO.parse(sequences_path, "fasta")
records = [(record.id, record.seq) for record in data]
ids, sequences = list(zip(*records))
stop = time.time()
print('Sequences read in ' + str(stop-start) + ' seconds')
sequences_df = pd.DataFrame({'strain' : ids, 'SeqRecords' : sequences})
total_df = metadata.merge(sequences_df, on = 'strain', how = 'inner')




# Calculate an adjacency matrix (and, hence, a network)
for sl in tqdm(seq_path_list):
    
    
    print('Calculating adjacency matrix for strain: ' + sl)
    
    if sl != 'wildtype':
        
        tree = Phylo.read(trees_folder + "/" + sl + "/" + sl + "_MAPLE.tree", format = 'newick')
        tree_ids = [n.name for n in tree.get_terminals()]

        tree_df = pd.DataFrame(np.array(tree_ids).T, columns = ['strain'])
        tree_df = tree_df.merge(total_df, on = 'strain', how = 'inner').filter(['strain', 'SeqRecords', 'SampleDateTime'])
        tree_seq_ids = tree_df.strain.values
        tree_seqs = tree_df.SeqRecords.values
        tree_seqs_string = [convert_to_string(seq) for seq in tree_seqs]
        tree_seqs_seq = [Seq.Seq(seq) for seq in tree_seqs_string]
        nseqs = len(tree_ids)
        tree_df.filter(['strain']).to_csv('../Large_Networks/' + sl + '/' + sl + '_ids_full.csv')
        new_records = [SeqRecord(tree_seqs_seq[i], id = tree_ids[i], name = tree_ids[i], description = tree_ids[i]) for i in range(nseqs)]
        SeqIO.write(new_records, "../Large_Networks/" + sl + "/" + sl + "_sequences_maple.fasta", "fasta")
        
        start = time.time()
    
        hamming_path = Path("../Large_Networks/" + sl + "/"  + 'hamming.csv')
        if not hamming_path.is_file():

            hammingdist.from_fasta("../Large_Networks/" + sl + "/" + sl + "_sequences_maple.fasta", 
                                   remove_duplicates = False).dump("../Large_Networks/" + sl + "/"  + 'hamming.csv')
        print('Reading hamming csv')
        if read_sparse:
            hamming_tree = sp.sparse.load_npz( "../Large_Networks/" + sl + "/"  + 'hamming_sparse.npz').toarray()
        else:
            hamming_tree = pd.read_csv("../Large_Networks/" + sl + "/"  + 'hamming.csv', header = None).values
        stop = time.time()
        print('Hamming Distance matrix calcualted in ' + str(stop-start) + ' seconds')

        sampledatetimes = tree_df['SampleDateTime'].values

        start = time.time()
        datetimes_first_sample = [datetime.strptime(dstr.split(' ')[0], '%Y-%m-%d') for dstr in tree_df['SampleDateTime'].values]
        
        datediffs = np.zeros((nseqs, nseqs)).astype(int)
        for i, dt in tqdm(enumerate(datetimes_first_sample)):
            for j in range(i):
                datediffs[i, j] = (datetimes_first_sample[i] - datetimes_first_sample[j]).days
                 # datediffs should be anti-symmetric
                datediffs[j, i] = - datediffs[i, j]
        end = time.time()

    
    
    else:
        
        
        tree_df = pd.DataFrame(np.array(tree_ids).T, columns = ['strain'])
        tree_df = tree_df.merge(total_df, on = 'strain', how = 'inner').filter(['strain', 'SeqRecords', 'SampleDateTime'])
        tree_seq_ids = tree_df.strain.values
        tree_seqs = tree_df.SeqRecords.values
        tree_seqs_string = [convert_to_string(seq) for seq in tree_seqs]
        tree_seqs_seq = [Seq.Seq(seq) for seq in tree_seqs_string]
        nseqs = len(tree_ids)
        tree_df.filter(['strain']).to_csv('../Large_Networks/' + sl + '/' + sl + '_ids_full.csv')
        new_records = [SeqRecord(tree_seqs_seq[i], id = tree_ids[i], name = tree_ids[i], description = tree_ids[i]) for i in range(nseqs)]
        SeqIO.write(new_records, "../Large_Networks/" + sl + "/" + sl + "_sequences_maple.fasta", "fasta")
        
        new_sequences_path = '../Large_Networks/wildtype/sequences.fasta'
        start = time.time()
        data = SeqIO.parse(new_sequences_path, "fasta")
        records = [(record.id, record.seq) for record in data]
        ids, sequences = list(zip(*records))
        print('Finished reading wildtype sequences!')
        tree_seqs_string = [convert_to_string(seq) for seq in sequences]
        tree_ids = list(ids)
    
        metadata_wildtype = pd.read_csv('../Large_Networks/wildtype/metadata_wildtype.csv')
        sequences_df_wildtype = pd.DataFrame({'strain' : ids, 'SeqRecords' : sequences})
        
        total_df_wildtype = sequences_df_wildtype.merge(metadata_wildtype, on = 'strain', how = 'inner')
        total_df_wildtype = total_df_wildtype.loc[~pd.isna(total_df_wildtype['SAMPLEDATETIME'])]
        tree_seqs_string = [convert_to_string(seq) for seq in total_df_wildtype.SeqRecords.values]
        tree_seqs_seq = [Seq.Seq(seq) for seq in tree_seqs_string]
        tree_ids = total_df_wildtype.filter(['strain']).values.flatten()
        nseqs = len(tree_ids)
        total_df_wildtype.filter(['strain']).to_csv('../Large_Networks/' + sl + '/' + sl + '_ids_full.csv')
        new_records = [SeqRecord(tree_seqs_seq[i], id = tree_ids[i], name = tree_ids[i], description = tree_ids[i]) for i in range(nseqs)]
        SeqIO.write(new_records, "../Large_Networks/" + sl + "/" + sl + "_sequences_maple.fasta", "fasta")


        start = time.time()
    
        hamming_path = Path("../Large_Networks/" + sl + "/"  + 'hamming.csv')
        if not hamming_path.is_file():

            hammingdist.from_fasta( "../Large_Networks/" + sl + 
                                   "/" + sl +
                                   "_sequences_maple.fasta",
                                   remove_duplicates = False).dump("../Large_Networks/" + sl + "/"  + 'hamming.csv')
        print('Reading hamming csv')
        hamming_tree = pd.read_csv("../Large_Networks/" + sl + "/"  + 'hamming.csv', header = None).values
        stop = time.time()
        print('Hamming Distance matrix calcualted in ' + str(stop-start) + ' seconds')

        sampledatetimes = total_df_wildtype['SAMPLEDATETIME'].values

        start = time.time()
        datetimes_first_sample = [datetime.strptime(dstr.split(' ')[0], 
                                                    '%Y-%m-%d') if dstr != np.nan else np.nan
                                  for dstr in sampledatetimes ]


        datediffs = np.zeros((nseqs, nseqs)).astype(int)
        for i, dt in tqdm(enumerate(datetimes_first_sample)):
            for j in range(i):
                datediffs[i, j] = (datetimes_first_sample[i] - datetimes_first_sample[j]).days
                 # datediffs should be anti-symmetric
                datediffs[j, i] = - datediffs[i, j]
        end = time.time()
       

    
    Calculate Hamming distance matrix
    start = time.time()
    
    hamming_path = Path("../Large_Networks/" + sl + "/"  + 'hamming.csv')
    if not hamming_path.is_file():

        hammingdist.from_fasta("../Large_Networks/" + sl + "/" + sl + "_sequences.fasta", remove_duplicates = False).dump("../Large_Networks/" + sl + "/"  + 'hamming.csv')
    print('Reading hamming csv')
    hamming_tree = pd.read_csv("../Large_Networks/" + sl + "/"  + 'hamming.csv', header = None).values
    stop = time.time()
    print('Hamming Distance matrix calcualted in ' + str(stop-start) + ' seconds')

    sampledatetimes = tree_df['SampleDateTime'].values
    
    start = time.time()
    datetimes_first_sample = [datetime.strptime(dstr.split(' ')[0], '%Y-%m-%d') for dstr in tree_df['SampleDateTime'].values]
    
    
    datediffs = np.zeros((nseqs, nseqs)).astype(int)
    for i, dt in tqdm(enumerate(datetimes_first_sample)):
        for j in range(i):
            datediffs[i, j] = (datetimes_first_sample[i] - datetimes_first_sample[j]).days
             # datediffs should be anti-symmetric
            datediffs[j, i] = - datediffs[i, j]
    end = time.time()

    print('Date difference matrix calcualted in ' + str(end-start) + ' seconds')
   
    sparse_dates = sp.sparse.csr_matrix(np.tril(datediffs))
    print('Calculating adjacency matrix')
    if weights == 0:
        
        tree_adj_mat = adjacency_matrix(hamming_tree, datediffs, valid_idxs, nseqs=nseqs)
        sparse = sp.sparse.csr_matrix(tree_adj_mat)
        sp.sparse.save_npz("../Large_Networks/" + sl + "/" + sl + "_adjacency_full_sparse" + sensitivity_output_str + ".npz", sparse)
        sp.sparse.save_npz("../Large_Networks/" + sl + "/" + sl + "_dates_full_sparse" + sensitivity_output_str + ".npz", sparse_dates)
    else:
        
        start = time.time()
        tree_adj_mat = weighted_adj_mat_2(hamming_tree, datediffs, valid_idxs, prob_mat)
        end = time.time()
        print('Weighted adjacency matrix calcualted in ' + str(end-start) + ' seconds')
        
        
        sparse = sp.sparse.csr_matrix(tree_adj_mat)
        sp.sparse.save_npz("../Large_Networks/" + sl + "/" + sl + "weighted_adjacency_full_sparse" 
                           + sensitivity_output_str + reweight_zero_str + low_sr_output_str  +".npz", sparse)
        sp.sparse.save_npz("../Large_Networks/" + sl + "/" + sl + "_dates_full_sparse" 
                          + sensitivity_output_str + reweight_zero_str + ".npz", sparse_dates)
    