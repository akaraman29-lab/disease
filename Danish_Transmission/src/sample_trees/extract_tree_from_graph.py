#### Sample a number of plausible transmission trees from the network of all plausible transmission pairs. Do this on a variant-by-variant basis. 
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import importlib
from glob import glob
import scipy as sp
from time import time
import seaborn as sns
from matplotlib import pyplot as plt
import networkx as nx
import polars as pl
import pickle
import glob
from joblib import Parallel, delayed
import oracledb
import random

cmp = sns.color_palette('Set2')

# Flag for whether to use the edge weights to create the random graphs or not. Weighted means we are explicitly using the edge probabilities, which is the preferred option, however sometimes we want to ignore the edge weights when looking at the structure of the network itself. 
use_edge_weights = True
settings = True
sensitivity = False
parallelise = False
reweight_settings = False
setting_factor_1 = 2
setting_factor_2 = 1
reweight_zero = True
reweight_zero_str = ['', '_reweight_zero'][int(reweight_zero)]

n_cores = 10

print('Using edge weights: ' + str(use_edge_weights))
print('Prioritise all settings: ' + str(settings))
print('Running sensitivity analysis: ' + str(sensitivity))
# print('Parallelised tree sampling: ' + str(parallelise) + ' on n_cores = ' + str(n_cores))
print('Reweighting zero: ' + str(reweight_zero))

### Decisions about schools:
school_id_name = 'INSTNR' # Choose UDD_ID, INSTNR or INSTNR + UDD
school_types = pl.read_csv('../school_types.csv', has_header = False)
school_types = school_types.rename({'column_1' : 'school', 'column_2' : 'school_type'})
higher_ed = False # Include higher education or only schools and gymnasium - False means macro_school, True means macro_education



# Which variant to extract trees from
sl_idx = 1
file_path = '../Large_networks/*'
sl_list = ['Alpha_B.1.1++(Other_B.1.1)', 
           'Omicron_BA.1_like++(Probable_Omicron_BA.1_like_and_Unassigned)', 
           'Eta_B.1.525_like++(Other)', 
           'wildtype']
sl = sl_list[sl_idx]
print('Extracting trees for strain: ' + sl)
# Number of random trees to extract
n_trees = 100
print('Extracting n=' + str(n_trees) + ' trees')


id_path = "../Large_Networks/" + sl + "/" + sl + "_ids_full.csv" 
ids = pl.read_csv(id_path).select(pl.col('strain')).to_numpy().flatten()
ids_df = pl.read_csv(id_path).select(pl.col('strain'))


start = time()
# Read in adjacency matrix calculated in generate_transmission_networks_large.py
if sensitivity:
    infection_adj_mat = sp.sparse.load_npz("../Large_Networks/" + sl + "/" + sl +
                                       "weighted_adjacency_full_sparse_sensitivity.npz")
elif reweight_zero:
    infection_adj_mat = sp.sparse.load_npz("../Large_Networks/" + sl + "/" + sl +
                                       "weighted_adjacency_full_sparse_reweight_zero.npz")
else:
    infection_adj_mat = sp.sparse.load_npz("../Large_Networks/" + sl + "/" + sl +
                                       "weighted_adjacency_full_sparse.npz")
end = time()

print('Sparse infection matrix read in: ' + str(np.round(end-start, 4)) + ' seconds!')

#### Networkx requires these matrices to be transposed for infector-infectee pairs to be the right way around
#### Generate networkx DiGraph object
G_digraph = nx.DiGraph(infection_adj_mat.T, sequence_ids = ids)
####

end2 = time()
print('Digraph created in: ' + str(np.round(end2-end, 4)) + ' seconds!')



df_family_edgelist = pl.read_csv('../Large_Networks/' + sl + '/df_family_edgelist.csv')
family_array = df_family_edgelist.select(pl.col('PERSON_ID_1', 'PERSON_ID_2')).drop_nulls().to_numpy()
family_pairs_set = set([(e, v) for e, v in family_array] + [(v, e) for e, v in family_array])




def create_random_sub_tree(G,
                           weighted = True):
    G_random = G.copy()
    G_random_nodes = list(G_random.nodes())
    random.shuffle(G_random_nodes)
    for node in G_random_nodes:
        edges = G_random.in_edges(node, data = True)
        if len(edges) > 1:
            weights = np.array([edge[2]['weight'] for edge in edges])
            node_idxs = np.arange(len(weights)).astype(int)
            if weighted:
                edge_choice = np.random.choice(node_idxs, p = weights/np.sum(weights))
            else:
                edge_choice = np.random.choice(node_idxs)
            ebunch = []

            ebunch = [(edge[0], edge[1]) for i, edge in enumerate(edges) if i!=edge_choice]
            ebunch += [(edge[1], edge[0]) for edge in ebunch]
            G_random.remove_edges_from(ebunch)
        
    return G_random


def create_random_sub_tree_reweight_settings(G, 
                                             weighted = True, 
                                             setting_factor = 2):
    ## Settings should now be reweighted so that they are multiplied by setting_factor, i.e. preferring but not forcing settings
    
    G_random_nodes = list(G_random.nodes())
    random.shuffle(G_random_nodes)
    for node in G_random_nodes:
        edges = G_random.in_edges(node, data = True)
        if len(edges) > 1:
            # Create mask for which edges correspond to infector sharing a setting with infectee 
            settings_mask = np.array([edge[2]['share_address'] 
                                 or edge[2]['share_school'] 
                                 or edge[2]['family_relation'] 
                                 or edge[2]['share_workplace'] for edge in edges])
            weights = np.array([edge[2]['weight'] for edge in edges])
            
            # Multiply masked edge weights by setting_factor
            
            weights[settings_mask] *= setting_factor
            node_idxs = np.arange(len(weights)).astype(int)
            if weighted:
                edge_choice = np.random.choice(node_idxs, p = weights/np.sum(weights))
            else:
                edge_choice = np.random.choice(node_idxs)
            ebunch = []

            ebunch = [(edge[0], edge[1]) for i, edge in enumerate(edges) if i!=edge_choice]
            ebunch += [(edge[1], edge[0]) for edge in ebunch]
            G_random.remove_edges_from(ebunch)
        
    return G_random


def create_random_sub_tree_settings(G, weighted = True, sensitivity = sensitivity):
    G_copy = G.copy()
    G_copy_nodes = list(G_copy.nodes())
    random.shuffle(G_copy_nodes)
    for node in G_copy_nodes:
        edges = G_copy.in_edges(node, data = True)
        if len(edges) > 1:
            
            settings = np.array([edge[2]['share_address'] for edge in edges])
            workplaces = np.array([edge[2]['share_workplace'] for edge in edges])  
            schools= np.array([edge[2]['share_school'] for edge in edges])
            families = np.array([edge[2]['family_relation'] for edge in edges])
            weights = np.array([edge[2]['weight'] for edge in edges])
            node_idxs = np.arange(len(edges)).astype(int)
            
            # Households prioritised first
            if (settings).any():
                edge_choice = np.argwhere((settings)).flatten()
                if len(edge_choice)>1:
                    setting_weights = weights[edge_choice]
                    setting_nodes = node_idxs[edge_choice]
                    edge_choice = np.random.choice(setting_nodes, p = setting_weights / np.sum(setting_weights))
                    
            # Then schools and workplaces - assumed equal prioritisation as little to no crossover. 
            elif (workplaces + schools).any():
                edge_choice = np.argwhere((workplaces + schools)).flatten()
                if len(edge_choice)>1:
                    setting_weights = weights[edge_choice]
                    setting_nodes = node_idxs[edge_choice]
                    edge_choice = np.random.choice(setting_nodes, p = setting_weights / np.sum(setting_weights))
            
            # Families explicitly under-prioritised compared to household, school, workplace
            elif families.any():
                edge_choice = np.argwhere((families)).flatten()
                if len(edge_choice)>1:
                    setting_weights = weights[edge_choice]
                    setting_nodes = node_idxs[edge_choice]
                    edge_choice = np.random.choice(setting_nodes, p = setting_weights / np.sum(setting_weights))
                
            else:
                if weighted:
                    edge_choice = np.random.choice(node_idxs, p = weights/np.sum(weights))
                else:
                    edge_choice = np.random.choice(node_idxs)
            ebunch = []

            ebunch = [(edge[0], edge[1]) for i, edge in enumerate(edges) if i!=edge_choice]
            ebunch += [(edge[1], edge[0]) for edge in ebunch]
            

            G_copy.remove_edges_from(ebunch)
            
         
        
    return G_copy





# Collect together node attributes for a given random graph e.g. vaccination status, age, who their infector is, shared attributes with infector such as household, workplace or school. 


def add_out_degree_attributes(nodelist_test):
    
    # Take a sampled tree (given as a polars dataframe representing a nodelist with a list of infectors, thus creating an edgelist)
    # Add required attributes to each node, in particular giving it an out_degree variable which is the number of infectees from that node.
    # Add other attributes as required 
    
    if 'out_degree_family' in nodelist_test.columns:
        nodelist_test = nodelist_test.drop('out_degree_family')
    if 'out_degree_school' in nodelist_test.columns:
        nodelist_test = nodelist_test.drop('out_degree_school')
    if 'out_degree_workplace' in nodelist_test.columns:
        nodelist_test = nodelist_test.drop('out_degree_workplace')
    if 'out_degree_address' in nodelist_test.columns:
        nodelist_test = nodelist_test.drop('out_degree_address')

    n_nodes = nodelist_test.select(pl.len()).item()#len(nodelist_test)
    person_ids_df = nodelist_test.select(pl.col('PERSON_ID', 'Infector_PERSON_ID'))

    family_pairs = df_family_edgelist.filter(pl.col('PERSON_ID_2')
                                             .is_in(person_ids_df.select('PERSON_ID')
                                                    .to_numpy().flatten())).select(pl.col('PERSON_ID_1', 
                                                                                          'PERSON_ID_2')).to_numpy()

    family_pairs_list_flip = [tuple(i) for i in (np.flip(family_pairs, axis = 1))]#.shape
    family_pairs_list = [tuple(i) for i in (family_pairs)]
    nodelist_pairs = [tuple(i.astype(int)) if not np.isnan(i).any() else np.nan for i in person_ids_df.to_numpy()]

    household_pairs = nodelist_test.filter('Infector_share_address').select(pl.col('PERSON_ID',
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

def convert_to_datediff(e1, e2, G):

    # Convert SampleDates to a datediff
    
    date_e2 = G.nodes[e2]['SampleDate']
    date_e1 = G.nodes[e1]['SampleDate']
    
    # Some of the datediffs are None types - this only affects 57 wildtype sequences, so replace these with np.nan
    
    if date_e1 == None or date_e2 == None:
        return np.nan
    
    else:
        return int((datetime.strptime(date_e2.split('T')[0], '%Y-%m-%d')
                - datetime.strptime(G.nodes[e1]['SampleDate'].split('T')[0], '%Y-%m-%d')).days )

def test_graph_serial_intervals(G):
    # Test to avoid any graph negative serial intervals - this will cause an assertion error if you have problems! 
    
    for n, d in G.nodes(data = True):
        assert((d['serial_interval_infector'] >=0 and d['serial_interval_infector'] <= 11) 
               or np.isnan(d['serial_interval_infector'])), 'Found serial intervals outside of generation time: ' + str(d['serial_interval_infector']) 
        return None

def create_nodelist_from_graph(G_r, attributes_table, output_file):

    # Take a nodelist and create a dataframe with node attributes to be saved as a csv
    
    G = G_r.copy()
    ## attributes_table should contain all of the node attributes that you want. This function extracts them, puts them
    ## in a dictionary and passes them as node attributes to the graph G. 
    
    ##Then makes a dataframe of the attributes and saves as a csv. 

    attributes_table = ids_df.join(attributes_table, on = 'strain', how = 'left')
    out_degrees = dict([(key, val) for key, val in G.out_degree()])
    nx.set_node_attributes(G, out_degrees, name = 'out_degree')
    
    age_attr_dict = dict([(i, a) for (i, a) in enumerate(attributes_table.select(pl.col('Age_at_testing')).to_numpy().flatten())])
    nx.set_node_attributes(G, age_attr_dict, name = 'Age_at_testing')
    # SampleDateTime
    sample_date_dict = dict([(i, a) for (i, a) in enumerate(attributes_table.select(pl.col('SampleDateTime')).to_numpy().flatten())])
    nx.set_node_attributes(G, sample_date_dict, name = 'SampleDate')

    # Regioncode
    region_attr_dict = dict([(i, a) for (i, a) in enumerate(attributes_table.select(pl.col('Regionskode')).to_numpy().flatten())])
    nx.set_node_attributes(G, region_attr_dict, name = 'Regionskode')
    # Kommunecode
    kommune_attr_dict = dict([(i, a) for (i, a) in enumerate(attributes_table.select(pl.col('KOM')).to_numpy().flatten())])
    nx.set_node_attributes(G, kommune_attr_dict, name = 'Kommunekode')
    
    
    addresses = attributes_table.select('UNIQUE_ADDRESS_ID').to_numpy().flatten()
    strain_ids = attributes_table.select('strain').to_numpy().flatten()
    
    address_attr_dict = dict([(i, addresses[i]) for i in range(len(addresses))])
    ids_attr_dict = dict([(i, strain_ids[i]) for i in range(len(strain_ids))])

    nx.set_node_attributes(G, ids_attr_dict, name = 'strain')
    nx.set_node_attributes(G, address_attr_dict, name = 'address')

    
    # Vaccination attributes
    days_since_complete_vaccination = attributes_table.select(pl.col('Days_since_complete_vaccination')).to_numpy().flatten()
    complete_vaccination_before_sampling = attributes_table.select(pl.col('Complete_vaccination_before_sampling')).to_numpy().flatten()

    days_since_first_vaccination = attributes_table.select(pl.col('Days_since_first_vaccination')).to_numpy().flatten()
    first_vaccination_before_sampling = attributes_table.select(pl.col('First_vaccination_before_sampling')).to_numpy().flatten()

    days_since_third_vaccination = attributes_table.select(pl.col('Days_since_third_vaccination')).to_numpy().flatten()
    third_vaccination_before_sampling = attributes_table.select(pl.col('Third_vaccination_before_sampling')).to_numpy().flatten()


    vacc_days_attr_dict = dict([(i, days_since_complete_vaccination[i]) for i in range(len(days_since_complete_vaccination))])
    vacc_complete_attr_dict = dict([(i, complete_vaccination_before_sampling[i]) for i in range(len(complete_vaccination_before_sampling))])

    first_vacc_days_attr_dict = dict([(i, days_since_first_vaccination[i]) for i in range(len(days_since_first_vaccination))])
    first_vacc_attr_dict = dict([(i, first_vaccination_before_sampling[i]) for i in range(len(first_vaccination_before_sampling))])

    third_vacc_days_attr_dict = dict([(i, days_since_third_vaccination[i]) for i in range(len(days_since_third_vaccination))])
    third_vacc_attr_dict = dict([(i, third_vaccination_before_sampling[i]) for i in range(len(third_vaccination_before_sampling))])


    nx.set_node_attributes(G, vacc_days_attr_dict, name = 'days_since_second_vacc')
    nx.set_node_attributes(G, vacc_complete_attr_dict, name = 'second_vacc_complete')
    nx.set_node_attributes(G, first_vacc_days_attr_dict, name = 'days_since_first_vacc')
    nx.set_node_attributes(G, first_vacc_attr_dict, name = 'first_vacc_complete')
    nx.set_node_attributes(G, third_vacc_days_attr_dict, name = 'days_since_third_vacc')
    nx.set_node_attributes(G, third_vacc_attr_dict, name = 'third_vacc_complete')
    
    # Predecessor in the tree
    nodes = G.nodes(data = True)
    predecessor_list = [[(n, p) for p in G.predecessors(n)] if [(n, p) for p in G.predecessors(n)]!= [] else [(n, np.nan)] for n in G.nodes]
    predecessor_dict = dict([l[0] for l in predecessor_list])
    predecessor_id_list = [l[0][1]  for l in predecessor_list]
    predecessor_id_df = pd.DataFrame(predecessor_id_list, columns = ['row_index'])

    
    predecessor_person_id_list = [nodes[int(p)]['PERSON_ID'] if ~np.isnan(p) else np.nan for p in predecessor_dict.values()] 
    predecessor_person_id_dict = dict([(i, idx) for i, idx in enumerate(predecessor_person_id_list)])       
    nx.set_node_attributes(G, predecessor_dict, name = 'Infector_ID')
    nx.set_node_attributes(G, predecessor_person_id_dict, name = 'Infector_PERSON_ID')
     
    share_addresses = [data['share_address'] for (e1, e2, data) in G.edges(data = True) ]
    share_addresses_end = dict([(end, list(G.in_edges(end, data = True))[0][2]['share_address']) for start, end in G.edges])
    nx.set_node_attributes(G, share_addresses_end, name = 'Infector_share_address')
    
    
    share_workplace = [data['share_workplace'] for (e1, e2, data) in G.edges(data = True) ]
    share_workplace_end = dict([(end, list(G.in_edges(end, data = True))[0][2]['share_workplace']) for start, end in G.edges])
    nx.set_node_attributes(G, share_workplace_end, name = 'Infector_share_workplace')
    
    share_school = [data['share_school'] for (e1, e2, data) in G.edges(data = True) ]
    share_school_end = dict([(end, list(G.in_edges(end, data = True))[0][2]['share_school']) for start, end in G.edges])
    nx.set_node_attributes(G, share_school_end, name = 'Infector_share_school')
    
    share_family = [data['family_relation'] for (e1, e2, data) in G.edges(data = True) ]
    share_family_end = dict([(end, list(G.in_edges(end, data = True))[0][2]['family_relation']) for start, end in G.edges])
    nx.set_node_attributes(G, share_family_end, name = 'Infector_family_relation')
    
    nx.set_node_attributes(G, np.nan, name = 'serial_interval_infector')
    serial_intervals = dict([(e2, convert_to_datediff(e1, e2, G)) for (e1, e2) in G.edges])
    nx.set_node_attributes(G, serial_intervals, name = 'serial_interval_infector')
    test_graph_serial_intervals(G)
    
    # Create dataframe of the attributes and save to output_file
    
    node_attr_df = pd.DataFrame.from_dict(dict((G.nodes(data = True))), orient = 'index')
    node_attr_df = add_out_degree_attributes(pl.from_pandas(node_attr_df))
    serial_intervals_df = node_attr_df['serial_interval_infector'].values
    serial_intervals_df = serial_intervals_df[~np.isnan(serial_intervals_df)]
    assert((serial_intervals_df >=0).all()), 'Found negative serial intervals'
    assert((serial_intervals_df <=11).all()), 'Found serial intervals longer than allowed generation time'
    
    
    node_attr_df.to_csv(output_file)
    return None
    
    


def create_node_output(G, attributes_table,
                       output_file = None,
                       use_weights = True, 
                       prioritise_settings = settings, 
                      reweight_settings = reweight_settings,
                       setting_factor = None):

    # Single function to create sub tree and then save to csv - for parallelisation
    
    if prioritise_settings:
        G_random = create_random_sub_tree_settings(G, 
                                                   weighted = use_weights)
    elif reweight_settings:
        G_random = create_random_sub_tree_reweight_settings(G,
                                                   weighted = use_weights,
                                                   setting_factor = setting_factor)
    else:
        G_random = create_random_sub_tree(G, 
                                          weighted = use_weights)
    create_nodelist_from_graph(G_random, attributes_table, output_file)
    return None

#####
# Read node attributes for creating transmission network. Then set node and edge attributes
#####

all_attributes = pl.read_csv('../Large_Networks/' + sl + '/all_attributes.csv')


    
address_ids = ids_df.join(all_attributes, on = 'strain', 
                          how = 'left').select(pl.col('strain', 'UNIQUE_ADDRESS_ID')).to_numpy()

address_attr_dict = dict([(i, address_ids[i, 1]) for i in range(len(address_ids[:, 1]))])

PERSON_ids = ids_df.join(all_attributes, on = 'strain', 
                          how = 'left').select(pl.col('strain', 'PERSON_ID')).to_numpy()

PERSON_ID_attr_dict = dict([(i, PERSON_ids[i, 1]) for i in range(len(PERSON_ids[:, 1]))])
nx.set_node_attributes(G_digraph, PERSON_ID_attr_dict, name = 'PERSON_ID')


all_attributes_work_school = pl.read_csv('../Large_Networks/' + sl + '/all_attributes_work_school.csv')


all_attributes_work_school = all_attributes_work_school.join(school_types,
                                                             left_on = 'UDD_ID',
                                                             right_on = 'school',
                                                             how = 'left')


school_ids = ids_df.join(all_attributes_work_school, 
                         on = 'strain', 
                          how = 'left').select(pl.col('strain', 
                                                      school_id_name)).to_numpy()

school_attr_dict = dict([(i, school_ids[i, 1]) for i in range(len(school_ids[:, 1]))])

school_types = ids_df.join(all_attributes_work_school, 
                         on = 'strain', 
                          how = 'left').select(pl.col('strain', 
                                                      'school_type')).to_numpy()

school_types_attr_dict = dict([(i, school_types[i, 1]) for i in range(len(school_types[:, 1]))])

# Workplaces
work_ids = ids_df.join(all_attributes_work_school, on = 'strain', 
                          how = 'left').select(pl.col('strain', 'ARB_NR')).to_numpy()

work_attr_dict = dict([(i, work_ids[i, 1]) for i in range(len(work_ids[:, 1]))])


kommune_ids = ids_df.join(all_attributes_work_school, on = 'strain', 
                          how = 'left').select(pl.col('strain', 'KOMMUNE_NAME')).to_numpy()
kommune_attr_dict = dict([(i, kommune_ids[i, 1]) for i in range(len(kommune_ids[:, 1]))])

region_ids = ids_df.join(all_attributes_work_school, on = 'strain', 
                          how = 'left').select(pl.col('strain', 'Region')).to_numpy()
region_attr_dict = dict([(i, region_ids[i, 1]) for i in range(len(region_ids[:, 1]))])


sample_dates = ids_df.join(all_attributes_work_school, on = 'strain', 
                          how = 'left').select(pl.col('strain', 'SampleDateTime')).to_numpy()
dates_attr_dict = dict([(i, sample_dates[i, 1]) for i in range(len(sample_dates[:, 1]))])

nx.set_node_attributes(G_digraph, address_attr_dict, name = 'address')
nx.set_node_attributes(G_digraph, school_attr_dict, name = 'school')
nx.set_node_attributes(G_digraph, school_types_attr_dict, name = 'school_type')
nx.set_node_attributes(G_digraph, work_attr_dict, name = 'workplace')
nx.set_node_attributes(G_digraph, kommune_attr_dict, name = 'kommune')
nx.set_node_attributes(G_digraph, region_attr_dict, name = 'region')
nx.set_node_attributes(G_digraph, dates_attr_dict, name = 'SampleDate')

if sl == 'wildtype':
    # some of the wildtype nodes have null dates (n=57) - remove these from the graph
    null_dates_nodes = [n for n, d in G_digraph.nodes(data = True) if d['SampleDate'] == None]
    G_digraph.remove_nodes_from(null_dates_nodes)

nodes_data = G_digraph.nodes(data = True)
for (start, end) in tqdm(G_digraph.edges):
    start_node = nodes_data[start]
    end_node = nodes_data[end]
    if start_node['address'] == end_node['address']:
        nx.set_edge_attributes(G_digraph, {(start, end) : True}, name = 'share_address')
    else:
        nx.set_edge_attributes(G_digraph, {(start, end) : False}, name = 'share_address')
        
    if higher_ed == True:

        if start_node['school'] == end_node['school']:
            nx.set_edge_attributes(G_digraph, {(start, end) : True}, name = 'share_school')
        else:
            nx.set_edge_attributes(G_digraph, {(start, end) : False}, name = 'share_school')
    else:
        if (start_node['school'] == end_node['school'] and 
        (start_node['school_type'] == 'Grundskole' or 
         start_node['school_type'] == 'Gymnasiale uddannelser')):
            
            nx.set_edge_attributes(G_digraph, {(start, end) : True}, name = 'share_school')
        else:
            nx.set_edge_attributes(G_digraph, {(start, end) : False}, name = 'share_school')

    if start_node['workplace'] == end_node['workplace']:
        nx.set_edge_attributes(G_digraph, {(start, end) : True}, name = 'share_workplace')
    else:
        nx.set_edge_attributes(G_digraph, {(start, end) : False}, name = 'share_workplace')
        
    if (start_node['kommune'] == end_node['kommune']) and (end_node['kommune'] != None):
        nx.set_edge_attributes(G_digraph, {(start, end) : True}, name = 'share_kommune')
    else:
        nx.set_edge_attributes(G_digraph, {(start, end) : False}, name = 'share_kommune')
        
    if (start_node['region'] == end_node['region']) and (end_node['region'] != None):
        nx.set_edge_attributes(G_digraph, {(start, end) : True}, name = 'share_region')
    else:
        nx.set_edge_attributes(G_digraph, {(start, end) : False}, name = 'share_region')
        
    if len(set([(start_node['PERSON_ID'],end_node['PERSON_ID']),
            (end_node['PERSON_ID'], start_node['PERSON_ID'])])
           .intersection(family_pairs_set)) > 0:
        
        
        
        nx.set_edge_attributes(G_digraph, {(start, end) : True}, name = 'family_relation')
    else:
        nx.set_edge_attributes(G_digraph, {(start, end) : False}, name = 'family_relation')
        




print('Creating random graphs')

####
# Running starts here!
####

# Different setup for parallelisation

if not parallelise:
    for i in tqdm(range(n_trees)):
        np.random.seed(i)


        if sensitivity:
            output_file_start = '../Large_Networks/' + sl + '/random_trees_sensitivity_macro_school/nodelist_' 
        elif reweight_zero:
            output_file_start = '../Large_Networks/' + sl + '/random_trees_macro_school_reweight_zero/nodelist_' 

        else:
            output_file_start = '../Large_Networks/' + sl + '/random_trees_macro_school/nodelist_' 

        create_node_output(G_digraph, all_attributes, output_file = output_file_start + str(i) + '.csv',
                           use_weights = use_edge_weights, prioritise_settings = False)


        if sensitivity:
            output_file_start = '../Large_Networks/' + sl + '/random_trees_prioritise_settings_sensitivity_macro_school/nodelist_'
        elif reweight_zero:
            output_file_start = '../Large_Networks/' + sl + '/random_trees_prioritise_settings_macro_school_reweight_zero/nodelist_'
        else:  
            output_file_start = '../Large_Networks/' + sl + '/random_trees_prioritise_settings_macro_school/nodelist_' 
        create_node_output(G_digraph, all_attributes, output_file = output_file_start + str(i) + '.csv',
                           use_weights = use_edge_weights, prioritise_settings = True)

else:
    if sensitivity:
            output_file_start = '../Large_Networks/' + sl + '/random_trees_sensitivity_macro_school/nodelist_' 
    else:
        output_file_start = '../Large_Networks/' + sl + '/random_trees_macro_school/nodelist_' 
    start = time()

    Parallel(n_jobs = n_cores)(delayed(create_node_output)(G_digraph, all_attributes, output_file = output_file_start + str(i) + '.csv',
                       use_weights = use_edge_weights, prioritise_settings = False) for i in range(n_trees))
    end = time()

    print('Trees without settings sampled in: ' + str(np.round(end-start, 4)) + ' seconds!')

    if sensitivity:
            output_file_start = '../Large_Networks/' + sl + '/random_trees_prioritise_settings_sensitivity_macro_school/nodelist_' 
    else:  
        output_file_start = '../Large_Networks/' + sl + '/random_trees_prioritise_settings_macro_school/nodelist_'
    start = time()       
    Parallel(n_jobs = n_cores)(delayed(create_node_output)(G_digraph, all_attributes, output_file = output_file_start + str(i) + '.csv',
                       use_weights = use_edge_weights, prioritise_settings = True) for i in range(n_trees))

    end = time()

    print('Trees with settings sampled in: ' + str(np.round(end-start, 4)) + ' seconds!')



    
    
    
    
    
    
    
    
    
    