#### Sample trees from a dictionary of plausible transmission pairs - primarily used for the Delta variant.
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
from Bio import SeqIO
import regex
import hammingdist
import importlib
import scipy as sp
import polars as pl
import pandas as pd
import time
import pickle as pkl
from glob import glob



weights = 1
sensitivity = False
reweight_zero = False
reweight_zero_str = ['', '_reweight_zero'][int(reweight_zero)]
# Max number of days - 21 is a decent cutoff to make transmission >21 days unlikely
ndays = 11 + 1
# Max number of substitutions - 9 is a decent cutoff to make transmission >9 substitutions unlikely
nsubs = 9
# Substitution rate - subject to sensitivity!
sr = 1/11
shift = [0, -2][sensitivity]
sensitivity_string = ['', '_sensitivity'][sensitivity]
# Number of cores for parallel computing
n_cores = 10

settings = True

print('Weighted graph = ' + str(weights == 1))
print('Sensitivity = ' + str(sensitivity == 1))
print('Settings: ' + str(settings == True))
print('Number of cores used = ' + str(n_cores))

### Decisions about schools:
school_id_name = 'INSTNR' # Choose UDD_ID, INSTNR or INSTNR + UDD
school_types = pl.read_csv('../school_types.csv', has_header = False)
school_types = school_types.rename({'column_1' : 'school', 'column_2' : 'school_type'})
higher_ed = True # Include higher education or only schools and gymnasium
print('Include higher ed:  ' + str(higher_ed))

# Convert sequences to strings
def convert_to_string(Seq):
    string = str(Seq).upper()
    return regex.sub(r'[^ACTG]', '-', string)


lineage_idx = 0
low_sr = 1
low_sr_output_str_list = ['', '_low_sr']
low_sr_str = low_sr_output_str_list[low_sr]

lineage_names = ['Delta', 'wildtype', 'Alpha', 'Eta', 'Omicron']
lineage_filepaths = ['', 
                     'wildtype', 
                     'Alpha_B.1.1++(Other_B.1.1)', 
                     'Eta_B.1.525_like++(Other)', 
                     'Omicron_BA.1_like++(Probable_Omicron_BA.1_like_and_Unassigned)']


print('Running analysis for:  ' + lineage_names[lineage_idx])

lineage_strain_paths = ['../sliding_window_delta/delta_network_ids_all.csv', 
                       '../Large_Networks/wildtype/wildtype_ids_full.csv', 
                       '../Large_Networks/Alpha_B.1.1++(Other_B.1.1)/' + lineage_filepaths[2] + '_ids_full.csv', 
                       '../Large_Networks/Eta_B.1.525_like++(Other)/' + lineage_filepaths[3] + '_ids_full.csv', 
                       '../Large_Networks/Omicron_BA.1_like++(Probable_Omicron_BA.1_like_and_Unassigned)/' + lineage_filepaths[4] + '_ids_full.csv']
lineage_strain_path = lineage_strain_paths[lineage_idx]

input_start_list = ['../sliding_window_delta/infectors_dict' +
                    sensitivity_string + reweight_zero_str + low_sr_str, 
                    '../Large_Networks/wildtype/infectors_dict' + sensitivity_string, 
                    '../Large_Networks/Alpha_B.1.1++(Other_B.1.1)/infectors_dict' + 
                    sensitivity_string + reweight_zero_str, 
                    '../Large_Networks/Eta_B.1.525_like++(Other)/infectors_dict' + sensitivity_string, 
                    '../Large_Networks/Omicron_BA.1_like++(Probable_Omicron_BA.1_like_and_Unassigned)/infectors_dict' +
                     sensitivity_string]


lineage_dir_list = ['../sliding_window_delta/', 
                    '../Large_Networks/wildtype/', 
                    '../Large_Networks/Alpha_B.1.1++(Other_B.1.1)/', 
                    '../Large_Networks/Eta_B.1.525_like++(Other)/', 
                    '../Large_Networks/Omicron_BA.1_like++(Probable_Omicron_BA.1_like_and_Unassigned)/']


all_attributes_path_list = ['../sliding_window_delta/sliding_window_data/all_attributes_work_school.csv', 
                           '../Large_Networks/wildtype/all_attributes_work_school.csv', 
                           '../Large_Networks/Alpha_B.1.1++(Other_B.1.1)/all_attributes_work_school.csv', 
                           '../Large_Networks/Eta_B.1.525_like++(Other)/all_attributes_work_school.csv', 
                           '../Large_Networks/Omicron_BA.1_like++(Probable_Omicron_BA.1_like_and_Unassigned)/all_attributes_work_school.csv']

family_edgelist_paths = ['../sliding_window_delta/sliding_window_data/df_family_edgelist.csv', 
                           '../Large_Networks/wildtype/df_family_edgelist.csv', 
                           '../Large_Networks/Alpha_B.1.1++(Other_B.1.1)/df_family_edgelist.csv', 
                           '../Large_Networks/Eta_B.1.525_like++(Other)/df_family_edgelist.csv', 
                           '../Large_Networks/Omicron_BA.1_like++(Probable_Omicron_BA.1_like_and_Unassigned)/df_family_edgelist.csv']



all_attributes_work_school = pl.read_csv(all_attributes_path_list[lineage_idx])

# school_id is not longer what we want it to be, so rename when we read it in and then later label our chosen variable as school_id
all_attributes_work_school = all_attributes_work_school.rename({'school_id' : 'CLASS_ID'})
## Choose between UDD_ID (original choice = CLASS_ID), INSTNR or INSTNR + UDD
school_id_name = 'INSTNR'  
all_attributes_work_school = all_attributes_work_school.rename({school_id_name : 'school_id'})
all_attributes_work_school = all_attributes_work_school.join(school_types,
                                                             left_on = 'UDD_ID',
                                                             right_on = 'school',
                                                             how = 'left')



full_network_dict = {}
import pickle as pkl
for filepath in tqdm(glob(input_start_list[lineage_idx] + '/*')):
    try:
        with open(filepath, 'rb') as handle:
            b = pkl.load(handle)
    except:
        continue
    full_network_dict.update(b)


df_family_edgelist = pl.read_csv(family_edgelist_paths[lineage_idx])
family_array = df_family_edgelist.select(pl.col('PERSON_ID_1', 'PERSON_ID_2')).drop_nulls().to_numpy()
family_pairs_set = set([(e, v) for e, v in family_array] + [(v, e) for e, v in family_array])
person_ids_dict = dict(all_attributes_work_school.select(pl.col('strain', 'PERSON_ID')).to_numpy())
strain_ids_dict = {v : k for k, v in person_ids_dict.items()}

print( lineage_dir_list[lineage_idx] + 'random_trees'+ 
                       sensitivity_string + '_macro_school' + reweight_zero_str + low_sr_str )
i = 0
n_samples = 100
use_settings = True
random_infector_dict = {}
random_infector_dict_settings = {}
np.random.seed(100)
for key in tqdm(full_network_dict.keys()):

    infectee_attrs = all_attributes_work_school.select('strain', 
                                                       'PERSON_ID',   
                                                       'school_id', 
                                                       'ARB_NR', 
                                                       'UNIQUE_ADDRESS_ID',
                                                       'Age_at_testing', 
                                                       'Complete_vaccination_before_sampling',
                                                       'school_type',
                                                      ).filter(pl.col('strain') == key)
    infectee_school = infectee_attrs.select(pl.col('school_id')).item()
    infectee_school_type = infectee_attrs.select(pl.col('school_type')).item()
    if higher_ed == True:
        if infectee_school == None:
            infectee_school = -1
    else:
        if (infectee_school == None) or (infectee_school_type != 'Gymnasiale uddannelser' and
                                         infectee_school_type != 'Grundskole'):
            infectee_school = -1
    infectee_workplace = infectee_attrs.select(pl.col('ARB_NR')).item()
    if infectee_workplace == None:
        infectee_workplace = -1
    infectee_household = infectee_attrs.select(pl.col('UNIQUE_ADDRESS_ID')).item()
    if infectee_household == None:
        infectee_household = 'No household'
    infectee_PERSON_ID = infectee_attrs.select(pl.col('PERSON_ID')).item()
        
    potential_infectors = full_network_dict[key]
    
    if potential_infectors == None:
        random_infector_dict[key] = None
        continue
    elif len(potential_infectors) == 0:
        random_infector_dict[key] = None
        continue
    elif type(potential_infectors) == list and len(potential_infectors) == 1:
        random_infector_dict[key] = [potential_infectors[0] for n in range(n_samples)]
        random_infector_dict_settings[key] = [potential_infectors[0] for n in range(n_samples)]
        continue
    if type(potential_infectors) == list:
        print(potential_infectors)
    potential_infectors_keys = ([k for k in potential_infectors.keys() if potential_infectors[k][0] > 0])
    
    if len(potential_infectors_keys) == 0:
        random_infector_dict[key] = None
        continue
    potential_infectors_df = all_attributes_work_school.filter(pl.col('strain').is_in(potential_infectors_keys)) 
    n_potential_infectors = len(potential_infectors_df)
    
    if use_settings == True:
        potential_infectors_setting = potential_infectors_df.filter((pl.col('school_id') == infectee_school) |
                                            (pl.col('ARB_NR') == infectee_workplace) |
                                            (pl.col('UNIQUE_ADDRESS_ID') == infectee_household))
        potential_infectors_keys_settings = potential_infectors_setting.select(pl.col('strain')).to_numpy().flatten()
                    

        
        family_set = set([(infectee_PERSON_ID, person_ids_dict[pi]) 
                     for pi in potential_infectors_keys_settings] + [(person_ids_dict[pi], infectee_PERSON_ID) 
                                                                     for pi in potential_infectors_keys])

        if len(potential_infectors_setting) > 0:

            potential_infectors_df_settings = potential_infectors_setting
            

            potential_infectors_settings = dict([(key, potential_infectors[key])
                                                 for key in potential_infectors_keys_settings])
            potential_infectors_person_ids_settings = [person_ids_dict[pi]
                                                           for pi in potential_infectors_keys_settings]
        elif len(family_set.intersection(family_pairs_set))>0:
            potential_infectors_PERSON_IDs_settings = np.array([person_ids_dict[pi] 
                                                          for pi 
                                                          in potential_infectors_keys_settings 
                                                          if person_ids_dict[pi]
                                                          in family_pairs_set])
            
            potential_infectors_keys_settings = np.array([strain_ids_dict[pi] 
                                                  for pi 
                                                  in potential_infectors_PERSON_IDs_settings])
            
            
            
        if len(potential_infectors_setting)>1:
            weights_settings = [p for p, d in potential_infectors_settings.values()]
            if sum(weights_settings) <=0:
                chosen_infectors_settings =  None
            else:
                chosen_infectors_settings = np.random.choice(potential_infectors_keys_settings,
                                                             p = weights_settings / np.sum(weights_settings),
                                                             size = n_samples)
            random_infector_dict_settings[key] = chosen_infectors_settings
            
        elif len(potential_infectors_setting)==1:
            assert(len(potential_infectors_keys_settings) == 1), potential_infectors_keys_settings
            weights_settings = [p for p, d in potential_infectors_settings.values()]
            if weights_settings[0] <= 0:
                chosen_infectors_settings =  None
            else:
                chosen_infectors_settings = [potential_infectors_keys_settings[0] for n in range(n_samples)]
        
            
            
            random_infector_dict_settings[key] = chosen_infectors_settings
        
    
    
    if len(potential_infectors) > 1:
        potential_infectors = dict([(key, potential_infectors[key]) for key in potential_infectors_keys])
        weights = [p for p, d in potential_infectors.values()]
        if sum(weights) <=0:
            random_infector_dict[key] = None
            continue
            
        chosen_infectors = np.random.choice(potential_infectors_keys, p = weights / np.sum(weights), size = n_samples)
    else:
        assert(len(potential_infectors_keys) == 1), key
        weights = [p for p, d in potential_infectors.values()]
        if weights[0] <= 0:
            random_infector_dict[key] = None
            continue
        chosen_infectors = [potential_infectors_keys[0] for n in range(n_samples)]
    
    random_infector_dict[key] = chosen_infectors
    if use_settings == True and len(potential_infectors_setting)==0:
        random_infector_dict_settings[key] = chosen_infectors



metadata = pl.from_pandas(pd.read_csv('../../newids_combined_metadata_ID_seqname_filename_pango.csv'))
lineage_strain_ids = pl.read_csv(lineage_strain_paths[lineage_idx])
lineage_metadata = lineage_strain_ids.join(metadata, on = 'strain', how = 'left')
lineage_metadata = lineage_metadata.with_columns(pl.col('SampleDateTime').str.split(' ')
                              .list.first().str.to_date('%Y-%m-%d'))   


random_infector_df = pd.DataFrame.from_dict(random_infector_dict)
random_infector_df = pl.from_pandas(random_infector_df.transpose())

index = pl.Series('strain', list(random_infector_dict.keys()))
insert_cols = random_infector_df.columns
insert_cols.insert(0, index)
random_infector_df = random_infector_df.select(insert_cols)


random_infector_df_settings = pd.DataFrame.from_dict(random_infector_dict_settings)
random_infector_df_settings = pl.from_pandas(random_infector_df_settings.transpose())
index = pl.Series('strain', list(random_infector_dict_settings.keys()))
insert_cols_settings = random_infector_df_settings.columns
insert_cols_settings.insert(0, index)
random_infector_df_settings = random_infector_df_settings.select(insert_cols_settings)






def add_out_degree_attributes(nodelist_test):

    if 'out_degree_family' in nodelist_test.columns:
        nodelist_test = nodelist_test.drop('out_degree_family')
    if 'out_degree_school' in nodelist_test.columns:
        nodelist_test = nodelist_test.drop('out_degree_school')
    if 'out_degree_workplace' in nodelist_test.columns:
        nodelist_test = nodelist_test.drop('out_degree_workplace')
    if 'out_degree_address' in nodelist_test.columns:
        nodelist_test = nodelist_test.drop('out_degree_address')

    n_nodes = len(nodelist_test)
    person_ids_df = nodelist_test.select(pl.col('PERSON_ID', 'Infector_PERSON_ID'))

    family_pairs = df_family_edgelist.filter(pl.col('PERSON_ID_2')
                                             .is_in(person_ids_df.select('PERSON_ID')
                                                    .to_numpy().flatten())).select(pl.col('PERSON_ID_1', 
                                                                                          'PERSON_ID_2')).to_numpy()

    family_pairs_list_flip = [tuple(i) for i in (np.flip(family_pairs, axis = 1))]#.shape
    family_pairs_list = [tuple(i) for i in (family_pairs)]
    nodelist_pairs = [tuple(i.astype(int)) if not np.isnan(i).any() else np.nan for i in person_ids_df.to_numpy()]

    household_pairs = nodelist_test.filter(pl.col('Infector_share_address').cast(pl.Boolean)).select(pl.col('PERSON_ID',
                                                                                   'Infector_PERSON_ID')).to_numpy()
    household_pairs = [tuple(i) for i in household_pairs]
    household_pairs_flip = [tuple(i) for i in np.flip(household_pairs, axis = 1)]
    
    nodelist_infectee_breakdown = nodelist_test.group_by(pl.col('Infector_ID')).sum().select(pl.col('Infector_ID', 
                                                                            'Infector_share_school', 
                                                                            'Infector_share_address', 
                                                                            'Infector_share_workplace', 
                                                                             'Infector_family_relation'))

    nodelist_infectee_breakdown = nodelist_infectee_breakdown.rename({'Infector_ID' : 'join_id', 
                                                                    'Infector_share_school' : 'out_degree_school', 
                                                                    'Infector_share_address' : 'out_degree_address', 
                                                                    'Infector_share_workplace' : 'out_degree_workplace', 
                                                                     'Infector_family_relation' : 'out_degree_family'})
    if 'index' in nodelist_test.columns:
        nodelist_test = nodelist_test.drop('index')
    nodelist_test = nodelist_test.with_row_index().with_columns(pl.col('index').cast(pl.Float64))
    
    nodelist_test = nodelist_test.join(nodelist_infectee_breakdown, 
                      left_on = 'index',
                       right_on = 'join_id', 
                       how = 'left').with_columns(pl.col('out_degree_school','out_degree_address',
                                                         'out_degree_workplace', 'out_degree_family').fill_null(0))
   
    return nodelist_test.to_pandas()

nodelist_test_alpha = pl.read_csv('../Large_Networks/Alpha_B.1.1++(Other_B.1.1)/random_trees/nodelist_0.csv')  


def get_df_from_edgelist(random_edgelist, join_metadata):
    random_edgelist = random_edgelist.join(join_metadata,
                         on = 'strain', 
                         how = 'left').join(join_metadata, 
                                            left_on = col,
                                            right_on = 'strain', 
                                           how = 'left', 
                                           suffix = '_infector').rename({col : 'strain_infector'})
    
    nodelist_test = random_edgelist.with_columns(Infector_share_address =
                                 pl.col('UNIQUE_ADDRESS_ID') == 
                                 pl.col('UNIQUE_ADDRESS_ID_infector').fill_null(0),
                                                Infector_share_school =
                                 pl.col('school_id') == 
                                 pl.col('school_id_infector').fill_null(0), 
                                                Infector_share_workplace =
                                 pl.col('ARB_NR') == 
                                 pl.col('ARB_NR_infector').fill_null(0))#.filter(pl.col('share_school'))
    
    nodelist_PERSON_IDs = nodelist_test.select(pl.col('PERSON_ID', 'PERSON_ID_infector')).to_numpy()
    infector_share_family = [len(set((tuple(nodelist_PERSON_IDs[i, :]),)).intersection(family_pairs_set)) > 0
                             for i in (range(len(nodelist_PERSON_IDs)))]
    nodelist_test = nodelist_test.with_columns(pl.Series(infector_share_family)
                                               .alias('Infector_family_relation').cast(pl.Int64))
    
    nodelist_test = nodelist_test.join(nodelist_test.group_by(pl.col('strain_infector')).len(), 
                      left_on = 'strain', right_on = 'strain_infector', 
                      how = 'left').rename({'len' : 'out_degree'}).with_columns(
        pl.col('out_degree').fill_null(0))
    
    nodelist_test = nodelist_test.with_columns(SampleDate = pl.col('SampleDateTime')
                                               .str.split('T')
                                               .list.first())
    nodelist_test = nodelist_test.with_columns(pl.col('Infector_share_address', 
                                  'Infector_share_school', 
                                  'Infector_share_workplace')
                           .cast(pl.Int64))
    
    nodelist_test = nodelist_test.with_columns(pl.col('Infector_share_address', 
                                  'Infector_share_school', 
                                  'Infector_share_workplace')
                           .fill_null(0))

    nodelist_test = nodelist_test.with_columns(pl.col('SampleDateTime',
                                                  'SampleDateTime_infector')
                                           .str.split('T')
                                           .list.first()
                                           .str.to_date())
    
    rename_dict = {'PERSON_ID_infector' : 'Infector_PERSON_ID', 
              'UNIQUE_ADDRESS_ID' : 'address', 
              'school_id' : 'school', 
              'ARB_NR' : 'workplace', 
              'KOMMUNE_NAME' : 'kommune', 
              'Region' : 'region', 
              'KOM' : 'Kommunekode', 
              'Days_since_first_vaccination' : 'days_since_first_vacc', 
              'Days_since_complete_vaccination' : 'days_since_second_vacc', 
              'Days_since_third_vaccination' : 'days_since_third_vacc'}
    nodelist_test_full = nodelist_test.join(all_attributes_work_school,
                              on = 'strain',
                              how = 'left',
                              suffix = '_2').rename(rename_dict)

    nodelist_test_full = nodelist_test_full.with_columns(first_vacc_complete = ~(pl.col('days_since_first_vacc').is_null()), 
                            second_vacc_complete = ~(pl.col('days_since_second_vacc').is_null()), 
                             third_vacc_complete = ~(pl.col('days_since_third_vacc').is_null()))
    nodelist_test_full = nodelist_test_full.select(pl.col(nodelist_test_alpha.columns[2:19] + nodelist_test_alpha.columns[21:-5]))
    nodelist_test_full = nodelist_test_full.with_row_index()
    
    Infector_ID_series = nodelist_test_full.select('Infector_PERSON_ID').join(nodelist_test_full.select('PERSON_ID', 'index'), 
                                      left_on = 'Infector_PERSON_ID', 
                                      right_on = 'PERSON_ID', 
                                      how = 'left').to_numpy()
    Infector_PERSON_ID_series = nodelist_test_full.select(pl.col('Infector_PERSON_ID')).to_numpy().flatten()
    Infector_ID_dict = dict(Infector_ID_series)
    Infector_ID_series = pl.Series([Infector_ID_dict[i] if ~np.isnan(i) else np.nan for i in Infector_PERSON_ID_series ])
    nodelist_test_full = nodelist_test_full.with_columns(Infector_ID_series.alias('Infector_ID'))
    nodelist_test_full = pl.from_pandas(add_out_degree_attributes(nodelist_test_full))
    return nodelist_test_full


#########################

print('Converting random choices to nodelists:')

# save_settings = True
for i in tqdm(range((100))):
    n_unique_infectors = []
    join_metadata = all_attributes_work_school.select(pl.col('strain', 
                                                   'PERSON_ID', 
                                                   'school_id', 
                                                   'ARB_NR',
                                                   'UNIQUE_ADDRESS_ID', 
                                                  'SampleDateTime'))
    col = str(i)
    
    random_edgelist_settings = random_infector_df_settings.select(pl.col('strain', col))
    random_edgelist = random_infector_df.select(pl.col('strain', col))
    
    nodelist_test = get_df_from_edgelist(random_edgelist, join_metadata)
    nodelist_test_settings = get_df_from_edgelist(random_edgelist_settings, join_metadata)
    
    
    nodelist_test_settings.write_csv(lineage_dir_list[lineage_idx] + 'random_trees_prioritise_settings'+
                       sensitivity_string + '_macro_school' + reweight_zero_str + low_sr_str + '/nodelist_' +
                       str(i) +
                       '.csv')
    nodelist_test.write_csv(lineage_dir_list[lineage_idx] + 'random_trees'+ 
                       sensitivity_string + '_macro_school' + reweight_zero_str + low_sr_str + '/nodelist_' + 
                       str(i) +
                       '.csv')


