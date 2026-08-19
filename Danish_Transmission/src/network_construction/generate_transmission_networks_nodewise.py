#### Some trees contain too many sequences to be stored in memory, and so the direct approach doesn't work. Here we use an approach
#### that is more memory efficient, but that takes longer to run, which we use to run trees for the Delta variant. We create a dictionary looping over
#### all sequences and find the plausible infectors for each one, finally saving the dictionary instead of a large adjacency matrix. 
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
from Bio import SeqIO
import regex
import hammingdist
import importlib
import transmission_functions as tf
importlib.reload(tf)
import scipy as sp
import polars as pl
import pandas as pd
import time
import pickle as pkl



weights = 1
sensitivity = False
reweight_zero = False
reweight_zero_prob = 5e-6
reweight_zero_str = ['', '_reweight_zero'][int(reweight_zero)]
low_sr = 1
low_sr_output_str_list = ['', '_low_sr']
low_sr_output_str = low_sr_output_str_list[low_sr]

shift = [0, -2][sensitivity]
# Number of cores for parallel computing
n_cores = 10

settings = True
refactor_settings = False

print('Weighted graph = ' + str(weights == 1))
print('Sensitivity = ' + str(sensitivity == 1))
print('Settings accounted for in generation time: ' + str(refactor_settings == True))
print('Number of cores used = ' + str(n_cores))



ndays = 20 + 1
# Max number of substitutions - 9 is a decent cutoff to make transmission >9 substitutions unlikely
nsubs = 9
# Substitution rate - subject to sensitivity!
if low_sr:
    sr = 0.06
else:
    sr = 1/11
days = np.arange(ndays)
subs = np.arange(nsubs)

# Convert sequences to strings
def convert_to_string(Seq):
    string = str(Seq).upper()
    return regex.sub(r'[^ACTG]', '-', string)


lineage_idx = 0
sensitivity_string = ['', '_sensitivity'][sensitivity]
lineage_names = ['Delta', 'wildtype', 'Alpha', 'Eta', 'Omicron']
lineage_filepaths = ['', 
                     'wildtype', 
                     'Alpha_B.1.1++(Other_B.1.1)', 
                     'Eta_B.1.525_like++(Other)', 
                     'Omicron_BA.1_like++(Probable_Omicron_BA.1_like_and_Unassigned)']


print('Running analysis for:  ' + lineage_names[lineage_idx])
# Input and output file paths
lineage_strain_paths = ['../sliding_window_delta/delta_network_ids_all.csv', 
                       '../Large_Networks/wildtype/wildtype_ids_full.csv', 
                       '../Large_Networks/Alpha_B.1.1++(Other_B.1.1)/' + lineage_filepaths[2] + '_ids_full.csv', 
                       '../Large_Networks/Eta_B.1.525_like++(Other)/' + lineage_filepaths[3] + '_ids_full.csv', 
                       '../Large_Networks/Omicron_BA.1_like++(Probable_Omicron_BA.1_like_and_Unassigned)/' + lineage_filepaths[4] + '_ids_full.csv']
lineage_strain_path = lineage_strain_paths[lineage_idx]

output_start_list = ['../sliding_window_delta/infectors_dict' + sensitivity_string +  reweight_zero_str + low_sr_output_str + 
                     '/infectors_dict_', 
                    '../Large_Networks/wildtype/infectors_dict' + sensitivity_string +  reweight_zero_str + '/infectors_dict_', 
                    '../Large_Networks/Alpha_B.1.1++(Other_B.1.1)/infectors_dict' + sensitivity_string  +  reweight_zero_str +
                     '/infectors_dict_', 
                    '../Large_Networks/Eta_B.1.525_like++(Other)/infectors_dict' + sensitivity_string + '/infectors_dict_', 
                    '../Large_Networks/Omicron_BA.1_like++(Probable_Omicron_BA.1_like_and_Unassigned)/infectors_dict' +
                     sensitivity_string + '/infectors_dict_']


output_start = output_start_list[lineage_idx]
print(output_start)
all_attributes_path_list = ['../sliding_window_delta/sliding_window_data/all_attributes_work_school.csv', 
                           '../Large_Networks/wildtype/all_attributes_work_school.csv', 
                           '../Large_Networks/Alpha_B.1.1++(Other_B.1.1)/all_attributes_work_school.csv', 
                           '../Large_Networks/Eta_B.1.525_like++(Other)/all_attributes_work_school.csv', 
                           '../Large_Networks/Omicron_BA.1_like++(Probable_Omicron_BA.1_like_and_Unassigned)/all_attributes_work_school.csv']

all_attributes_work_school = pl.read_csv(all_attributes_path_list[lineage_idx])

school_id_name = 'INSTNR'

## Read in family edgelist

df_family_edgelist = pl.read_csv('../sliding_window_delta'+ '/df_family_edgelist.csv', has_header = False)
df_family_edgelist = df_family_edgelist.rename({'column_1' : 'PERSON_ID_1', 'column_2' : 'PERSON_ID_2'})
family_array = df_family_edgelist.select(pl.col('PERSON_ID_1', 'PERSON_ID_2')).drop_nulls().to_numpy()
family_pairs_set = set([(e, v) for e, v in family_array] + [(v, e) for e, v in family_array])

lineage_strain_ids = pl.read_csv(lineage_strain_path,
                                 has_header = False,
                                 new_columns = ['index', 'strain'])
# Read in metadata
if lineage_idx == 1:
    metadata = pd.read_csv('../Large_Networks/wildtype/metadata_wildtype.csv')
    metadata = metadata.loc[~pd.isna(metadata['SAMPLEDATETIME'])]
    metadata = metadata.loc[~pd.isna(metadata['DATESAMPLING'])]
    metadata = pl.from_pandas(metadata).rename({'SAMPLEDATETIME' : 'SampleDateTime', 
                                                'DATESAMPLING':'DateSamplingLinelist'})
    
else:
    metadata = pl.from_pandas(pd.read_csv('../../newids_combined_metadata_ID_seqname_filename_pango.csv'))

lineage_metadata = lineage_strain_ids.join(metadata, on = 'strain', how = 'left')
lineage_metadata = lineage_metadata.with_columns(pl.col('DateSamplingLinelist', 
                                     'SampleDateTime').str.split(' ')
                              .list.first().str.to_date('%Y-%m-%d'))


start = time.time()
# Read in sequences
if lineage_idx == 1:
    sequences_path = '../Large_Networks/wildtype/sequences.fasta'
else:
    sequences_path = '../sequences_folder/newids_masked_consensus_aligned_2021.fasta'
data = SeqIO.parse(sequences_path, "fasta")
records = [(record.id, record.seq) for record in data]
ids, sequences = list(zip(*records))
stop = time.time()
print('Data read in ' + str(stop-start) + ' seconds')

sequences_df = pd.DataFrame({'strain' : ids, 'SeqRecords' : sequences})

assert(len(ids) == len(sequences))
print('Converting sequences to strings')
sequences_dict = dict([(idx, convert_to_string(seq)) for idx, seq in tqdm(records)])

valid_idxs = [(0, 0)]
for i in range(12):
    sub_prob = np.cumsum(tf.substitution_probability(np.arange(5), i, sr, inhomogeneous=False))
    valid_idxs += [(j, i) for j in np.argwhere(sub_prob<= 0.95).flatten() ]
    
# Get probabilities for observing data from a transmission pair (A, B)
def get_probability(distance_datediff_zip, inhomogeneous = True, share_settings = None):
    distance_datediff_pair, datediff_test = distance_datediff_zip
    # Adjusted generation time for pairs that share a setting - we no longer use this in our analysis! 
    if share_settings == True:

        return tf.scenario1_probability_tests(distance_datediff_pair[1], distance_datediff_pair[0], datediff_test,
                                              sr, shift = -3)
    elif share_settings == False:
        return tf.scenario1_probability_tests(distance_datediff_pair[1], distance_datediff_pair[0], datediff_test,
                                              sr, shift = 0)
    # If we don't use the share_settings approach - this corresponds to our main analysis! 
    else:
        if (distance_datediff_pair[1] == 0) and (distance_datediff_pair[1] == 0) and (reweight_zero == True):
            return reweight_zero_prob
        elif (reweight_zero == True):
            return tf.scenario1_probability_tests(distance_datediff_pair[1], distance_datediff_pair[0], datediff_test,
                                                  sr, shift = shift) *(1-reweight_zero_prob)
        else:
            return tf.scenario1_probability_tests(distance_datediff_pair[1], distance_datediff_pair[0], datediff_test,
                                                  sr, shift = shift)
            

    

ids_lineage = lineage_metadata.select('strain').to_numpy().flatten()
# Partition the sequences into a list of lists - for parallelisation
if lineage_idx == 1:
    
    ids_lineage_list = [part.select('strain').to_numpy().flatten() for part in lineage_metadata.with_row_count().with_columns(pl.col('row_nr') // 1000).partition_by('row_nr')]
    
else:
    
    ids_lineage_list = [part.select('strain').to_numpy().flatten() for part in lineage_metadata.with_columns(pl.col('X.1') // 10000).partition_by('X.1')]
# Generate dictionary of plausible infectors for each individual
def generate_infectors_dict(ids_lineage, output_index, output_start,
                            lineage_metadata = lineage_metadata, 
                            all_attributes_work_school = all_attributes_work_school.select(school_id_name, 
                                                                                            'strain', 
                                                                                            'UNIQUE_ADDRESS_ID', 
                                                                                            'ARB_NR')):
    # Output is a dictionary whose keys are the sequence IDs, and whose values are a dictionary containing:
                    # - A list of the sequence IDs of plausible infectors
                    # - A list of the associated probabilities (to be used as edge weights)
                    # - The Hamming distance and date difference between plausible infector and infectee 
    
    infectors_dict = {}


    start = time.time()
    print('Starting thread ' + str(output_index) + '!')
    for seq_id in ids_lineage:
        first_test_date = lineage_metadata.filter(pl.col('strain') == seq_id).select('DateSamplingLinelist').item()
        
        if first_test_date == None:
            continue
            
        first_sample_test_date = lineage_metadata.filter(pl.col('strain') == seq_id).select('SampleDateTime').item()
        first_sample_person_id = lineage_metadata.filter(pl.col('strain') == seq_id).select('PERSON_ID').item()
        first_sample_address = all_attributes_work_school.filter(pl.col('strain') == seq_id).select('UNIQUE_ADDRESS_ID').item()
        first_sample_school = all_attributes_work_school.filter(pl.col('strain') == seq_id).select(school_id_name).item()
        first_sample_workplace = all_attributes_work_school.filter(pl.col('strain') == seq_id).select('ARB_NR').item()
        
        all_attributes_work_school_join = all_attributes_work_school
        lineage_metadata_subset = lineage_metadata.filter((pl.col('DateSamplingLinelist')
                                                  .is_between(first_test_date - datetime.timedelta(days = 14), 
                                                              first_test_date + datetime.timedelta(days = 0))) & 
                                                 (pl.col('strain')!= seq_id)).join(all_attributes_work_school_join, 
                                                                                   on = 'strain', 
                                                                                   how = 'left')
        ids_subset = lineage_metadata_subset.select(pl.col('strain')).to_numpy().flatten()
        dates_subset = lineage_metadata_subset.select(pl.col('DateSamplingLinelist')).to_numpy().flatten()
        sample_dates_subset = lineage_metadata_subset.select(pl.col('SampleDateTime')).to_numpy().flatten()
        addresses_subset = lineage_metadata_subset.select(pl.col('UNIQUE_ADDRESS_ID')).to_numpy().flatten()
        schools_subset = lineage_metadata_subset.select(pl.col(school_id_name)).to_numpy().flatten()
        workplaces_subset = lineage_metadata_subset.select(pl.col('ARB_NR')).to_numpy().flatten()
        person_ids_subset = lineage_metadata_subset.select(pl.col('PERSON_ID')).to_numpy().flatten()

        hamming_dists = np.array(([hammingdist.distance(sequences_dict[seq_id],
                                                        sequences_dict[idx]) for idx in ids_subset if idx != seq_id]))
        
        share_address_subset = np.array([adx == first_sample_address 
                                         if type(first_sample_address) == str 
                                         else False
                                         for adx in addresses_subset])
        share_school_subset = np.array([sdx == first_sample_school 
                                         if (first_sample_school != None 
                                         and ~np.isnan(first_sample_school)) 
                                         else False
                                         for sdx in schools_subset])
        share_workplace_subset = np.array([wdx == first_sample_workplace 
                                         if (first_sample_workplace != None 
                                         and ~np.isnan(first_sample_workplace)) 
                                         else False
                                         for wdx in workplaces_subset])
        
        share_family_subset = np.array([len(set([(first_sample_person_id, pidx),
                                         (pidx, first_sample_person_id)]).intersection(family_pairs_set)) > 0
                               
                                         for pidx in person_ids_subset])
        
        share_settings_subset = (share_address_subset + share_school_subset + 
                                 share_workplace_subset + share_family_subset).astype(bool)

        plausible_idxs = np.argwhere(hamming_dists <= 2).flatten()
        plausible_ids = ids_subset[plausible_idxs]
        plausible_hammings = hamming_dists[plausible_idxs]
        plausible_dates = dates_subset[plausible_idxs]
        plausible_sample_dates = sample_dates_subset[plausible_idxs]

        plausible_sample_date_diffs = np.array([(first_sample_test_date-d.item()).days for d in plausible_sample_dates])
        plausible_date_diffs = np.array([(first_test_date-d.item()).days for d in plausible_dates])

        valid_pair_idxs_bool = np.array([(sdd, dd) in valid_idxs for 
                                         sdd, dd in zip(plausible_hammings, plausible_sample_date_diffs )])
        if len(valid_pair_idxs_bool)!=0:
            valid_ids = plausible_ids[valid_pair_idxs_bool]
            valid_distances_datediffs = np.array([(i, j) for i, j in zip(plausible_hammings[valid_pair_idxs_bool], 
                                            plausible_sample_date_diffs[valid_pair_idxs_bool])])
            valid_date_diffs = plausible_date_diffs[valid_pair_idxs_bool]
            
            if not refactor_settings:
                probabilities = [get_probability(p) for p in zip(valid_distances_datediffs, 
                                                             valid_date_diffs)]
            else:
                probabilities = [get_probability(p, share_settings = share_settings_subset[i]) 
                                 for i, p in enumerate(zip(valid_distances_datediffs, valid_date_diffs))]
                
            infectors_dict[seq_id] = dict(zip(valid_ids,
                                             zip(probabilities, 
                                                            zip(valid_distances_datediffs, 
                                                                               valid_date_diffs))))
        else:
            infectors_dict[seq_id] = None


            
    stop = time.time()
    print('Processed ' + str(len(ids_lineage)) + ' sequences in ' + str(stop-start) + ' seconds')


    print('Finished identifying infectors! Saving to pickle:')

    
    # Save dictionary as pickle file
    start = time.time()
    with open(output_start + str(output_index) + '.pickle', 'wb') as handle:
        pkl.dump(infectors_dict, handle, protocol = pkl.HIGHEST_PROTOCOL)    
    stop = time.time()
    print('Data saved in ' + str(stop-start) + ' seconds')
          

print('Number of infector dictionaries to be generated is: ' + str(len(ids_lineage_list)))
# Run dictionaries for sequences in parallel
Parallel(n_jobs = n_cores)(delayed(generate_infectors_dict)(ids_lineage_list[j], j, output_start) for j in range(len(ids_lineage_list)))

    
    