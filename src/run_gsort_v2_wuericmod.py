import sys

import sklearn.cluster
import numpy as np
import sklearn.decomposition
import scipy.io as sio

from src.artifact_estimator_class import *
from src.old_labview_data_reader import *
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import scipy.stats
from functools import partial
from functools import partialmethod
from itertools import chain, combinations
# from small_sample_variance import *
from scipy.ndimage.interpolation import shift
import multiprocessing as mp
import src.eilib as eil
import warnings

from skimage.util.shape import view_as_windows as viewW



import src.elecresploader as el
import scipy.signal
import src.signal_alignment as sam


from itertools import product
from itertools import starmap

import fastconv.corr1d as corr1d # compiled dependency wueric, compiled already
from typing import List, Tuple


def run_movie(n, p, pi, ks, preloaded_data):
    
    electrode_list, data_on_cells, start_time_limit, end_time_limit, estim_analysis_path, noise = preloaded_data

    shift_window=[start_time_limit, end_time_limit]
    
    
    probs = []
    cosine_probs = []
    notes = []
    num_trials_list = []
    clustering_list = []
    cluster_label_list = []
    root_signals = []
    root_list = []
    all_signals = []
    p_to_cell= {}

    
    cell_ids, cell_eis, cell_error, cell_spk_times = data_on_cells
    significant_electrodes = np.arange(len(electrode_list))
    

    for k in range(ks):
    
        try:
            signal = get_oldlabview_pp_data(estim_analysis_path , p, k)[:,:,:]
        except:
            break

        num_trials = len(signal)
        raw_signal = signal[:, electrode_list, shift_window[0]:shift_window[1]].astype(float) 
        finished, G, (_, _), (event_labels_with_virtual, _), (_, _), edge_to_matched_signals, _, mask, note = spike_sorter_EA(n, significant_electrodes, electrode_list, raw_signal, 1, 1000, noise, data_on_cells, artifact_cluster_estimate=None)



        total_p, cell_in_clusters = get_probabilities(G, event_labels_with_virtual)


        cosine_similarity = 0.7
        graph_signal = edge_to_matched_signals
        edge_to_cell = {k:v for k, v in graph_signal.keys()}
        signal_index = np.argwhere([k[1] ==n for k in  graph_signal.keys()]).flatten()
        g_signal_error = 0
        clustering = event_labels_with_virtual
        if len(signal_index) != 0:

            edges = np.array([list(graph_signal.keys())[k][0] for k in signal_index])  
            low_power = [np.sum(mask*graph_signal[list(graph_signal.keys())[k]][0]*graph_signal[list(graph_signal.keys())[k]][1])/np.sqrt(np.sum(graph_signal[list(graph_signal.keys())[k]][1]**2))/np.sqrt(np.sum(mask*graph_signal[list(graph_signal.keys())[k]][0]**2))<=cosine_similarity for k in signal_index]
            g_signal_error += sum([sum(clustering == e[1])/len(clustering) for e in edges[low_power]])
            g_signal_error += sum([sum(clustering == n)/len(clustering) for e in edges[low_power] for n in nx.descendants(G, e[1])])
            cosine_probs += [total_p[n]-g_signal_error]
        else:
            cosine_probs += [total_p[n]]
   
        probs += [total_p[n]]

    return ( pi, [i for i in range(len(probs))], cosine_probs, probs)

def fine_cluster(signals,significant_electrodes, electrode_list, noise, note, unsatured_min= -1000, unsaturated_max = 40, damping = 0.5, sat_band = 5, resolution = 10):
    _, _, num_samples = signals.shape
    # Generate mask to deal with saturation
    signals_max = np.max(signals, axis = 0)
    signals_min = np.min(signals, axis = 0)
    mask = ((signals_max < unsaturated_max)*(signals_min > unsatured_min)).astype(np.float64)
    mask_loc = np.argwhere(1-mask)
    for m in mask_loc:
        mask[m[0],max(0,m[1]-sat_band):min(m[1]+sat_band, num_samples)] = 0
    
    
    # First unsupervised clustering of event cliques with affinity propagation
    # In case affinity propagation does not converge on the first few iterations, allow for 10 tries
    
    
    try: 
        seed_electrode = min(np.argwhere(np.sum(1-mask, axis = 1) == 0)).item()
        non_saturated_channels = list(np.argwhere(np.sum(1-mask, axis = 1) == 0).flatten())
        
        s_channels = np.array([s for s in significant_electrodes if s in non_saturated_channels])
    except:
        note += "all relevant electrodes saturated"
        s_channels = significant_electrodes
    unresolved = True
    new_index = 0
    index = 0
    
    
    segmented_lists = [[i for i in range(len(signals))]]
    
    while unresolved:
        new_segmented_lists = []
        for l in segmented_lists:
            
            selected_signals = signals[l]
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                for r_state in range(10):

                    ap = sklearn.cluster.AffinityPropagation(damping=damping, random_state=r_state)
                    try:
                        event_labels = ap.fit_predict(selected_signals[:, s_channels, :].reshape((len(selected_signals), -1)) )              
                        break
                    except:
                        pass

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if r_state == 9:
                    ap = sklearn.cluster.AffinityPropagation(damping=damping, random_state=r_state)
                    event_labels = ap.fit_predict(selected_signals[:, s_channels, :].reshape((len(selected_signals), -1)) ) 
            
            old_event_cliques = [set([i for i, x in enumerate(event_labels == l) if x]) for l in set(event_labels)]
            tries = 0

            # 2. Cluster merging
            while True:
                event_labels = first_merge_event_cliques_by_noise(electrode_list,selected_signals, event_labels,  mask, significant_electrodes,   noise)
                event_cliques = [set([i for i, x in enumerate(event_labels == l) if x]) for l in set(event_labels)]
                event_cliques = [set([i for i, x in enumerate(event_labels == l) if x]) for l in set(event_labels)]

                if event_cliques == old_event_cliques:
                    break
                old_event_cliques = event_cliques
                tries += 1
                if tries == 10:
                    break
            
            for label in set(event_labels):
                new_segmented_lists += [list(np.array(l)[event_labels==label])]
                

        index += 1
        if index == resolution:
            unresolved = False
        
        if (segmented_lists == new_segmented_lists):
            unresolved = False
        segmented_lists = new_segmented_lists
        
            
    final_event_labels = np.zeros(len(signals), dtype=int)

    for i, sets in enumerate(new_segmented_lists):
        final_event_labels[sets]=i
        
    old_event_cliques = [set([i for i, x in enumerate(event_labels == l) if x]) for l in set(final_event_labels)]
    tries = 0
    while True:
        final_event_labels = first_merge_event_cliques_by_noise(electrode_list,signals, final_event_labels,  mask, significant_electrodes,   noise)
        event_cliques = [set([i for i, x in enumerate(final_event_labels == l) if x]) for l in set(final_event_labels)]
        event_cliques = [set([i for i, x in enumerate(final_event_labels == l) if x]) for l in set(final_event_labels)]

        if event_cliques == old_event_cliques:
            break
        old_event_cliques = event_cliques
        tries += 1
        if tries == 10:
            break
        
    return mask, final_event_labels


def spike_sorter_EA(n, significant_electrodes, electrode_list, signals_tmp, iterations,max_iter, noise, data_on_cells, damping = 0.5, sat_band = 5, unsatured_min= -1000, unsaturated_max = 400, artifact_cluster_estimate = None, hierachical_cluster = False, no_time = False):
    """
    Main run script
    """
    
    total_p = Counter()
    total_p_gof = Counter()
    total_p_greedy = Counter()
    valid_graph = 1
    edge_to_cell_and_fit_info = {}
    

    signals = copy.copy(signals_tmp)
    num_trials, num_electrodes, num_samples = signals.shape
    cells_on_trials = {i:[] for i in range(num_trials)}
    note = ""
    if not hierachical_cluster:
        


        # Generate mask to deal with saturation
        signals_max = np.max(signals, axis = 0)
        signals_min = np.min(signals, axis = 0)
        mask = ((signals_max < unsaturated_max)*(signals_min > unsatured_min)).astype(np.float64)
        mask_loc = np.argwhere(1-mask)
        for m in mask_loc:
            mask[m[0],max(0,m[1]-sat_band):min(m[1]+sat_band, num_samples)] = 0


        # First unsupervised clustering of event cliques with affinity propagation
        # In case affinity propagation does not converge on the first few iterations, allow for 10 tries

        try: 
            seed_electrode = min(np.argwhere(np.sum(1-mask, axis = 1) == 0)).item()
            non_saturated_channels = list(np.argwhere(np.sum(1-mask, axis = 1) == 0).flatten())

            s_channels = np.array([s for s in significant_electrodes if s in non_saturated_channels])
        except:
            note += "all relevant electrodes saturated"
            s_channels = significant_electrodes
    #     print(signals_max, signals_min)

        # Clustering step
        # 1. Affinity propagation
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for r_state in range(10):

                ap = sklearn.cluster.AffinityPropagation(damping=damping, random_state=r_state)
                try:
                    event_labels = ap.fit_predict(signals[:, s_channels, 7:].reshape((num_trials, -1)) )              
                    break
                except:
                    pass

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if r_state == 9:
                ap = sklearn.cluster.AffinityPropagation(damping=damping, random_state=r_state)
                event_labels = ap.fit_predict(signals[:, s_channels, 7:].reshape((num_trials, -1)) ) 

        old_event_cliques = [set([i for i, x in enumerate(event_labels == l) if x]) for l in set(event_labels)]
        tries = 0

        # 2. Cluster merging
        while True:
            event_labels = first_merge_event_cliques_by_noise(electrode_list,signals, event_labels,  mask, significant_electrodes,   noise)
            event_cliques = [set([i for i, x in enumerate(event_labels == l) if x]) for l in set(event_labels)]
            event_cliques = [set([i for i, x in enumerate(event_labels == l) if x]) for l in set(event_labels)]

            if event_cliques == old_event_cliques:
                break
            old_event_cliques = event_cliques
            tries += 1
            if tries == 10:
                break
    else:
        mask, event_labels = fine_cluster(signals,significant_electrodes, electrode_list, noise, note)
    initial_event_set = set(event_labels)
    
    # Create nodes for explanation graph
    G = nx.DiGraph()
    G.add_nodes_from(list(set(event_labels)))
    state_to_cells ={e:[] for e in event_labels}

    
    initial_event_labels = copy.copy(event_labels)
    constrained = 'gof'
   
        
    # If all cliques have been merged, quit
    if len(set(event_labels))==1 or iterations == 0:
        note += " early completion, 1 cluster"
        return 1, G, (event_labels, signals), (event_labels, signals), (event_labels, signals), {}, 0, mask, note



    # Compute difference signals
    difference_signals, event_signal_ordering =compute_difference_signals(signals, event_labels)
    
   
    # Clean the difference signals
    difference_signals = clean_signals(difference_signals)

    # Find candidate edges in graph from comparing templates and difference signals
    edge_cell_to_fit_info, edge_cell_to_signal, edge_cell_to_error, edge_cell_to_loss, edge_cell_to_noise_loss = find_candidate_edges(n, electrode_list, data_on_cells, difference_signals, event_signal_ordering,  mask, event_labels, noise, constrained = constrained, artifact_estimate = artifact_cluster_estimate, no_time = no_time)
    
    sorted_edges = sorted(edge_cell_to_loss.items(), key=lambda item: item[1])
#     print("sorted_edges", sorted_edges)
    _, _, num_samples = signals.shape   
    
    finished = False
    check_points = {}
    check_points_loss = {}
    v_structure_edges_and_nodes = []
    nodes_to_virtual_nodes = {}
    initial_nodes = len(G.nodes)
    
    skip_edges = []
    tracker = 0
    downstream = {}
    
    # Add edges that satisfy contraints: no improper chains, cycles, and (in simple mode) v-structures
    while len(sorted_edges) != 0:

        
        e_c, loss = sorted_edges.pop(0)
        e = e_c[0]
        cell = e_c[1]
        
        edge_data = nx.get_edge_attributes(G,'cell')
        
        
        head = e[0]
        tail = e[1]
        skip = False
        for u, v, ddict in G.in_edges(head, data=True):
            if cell == ddict['cell']:
                skip = True
  
        
        predecessors_nodes = list(G.predecessors(tail))
        predecessors_edges = [(tail, s) for s in predecessors_nodes]
        
        predecessors_edges_cells = []
        while len(predecessors_edges) != 0:
            p_e = predecessors_edges.pop(0)
            predecessors_edges_cells += [edge_data[(p_e[1], p_e[0])]]
            predecessors_nodes = list(G.predecessors(p_e[1]))
            predecessors_edges += [(p_e[1], p) for p in predecessors_nodes]
       
        successors_nodes = list(G.successors(head))
        successors_edges = [(s, head) for s in successors_nodes]
        
            
        successors_edges_cells = []
        while len(successors_edges) != 0:
            s_e = successors_edges.pop(0)
          
            successors_edges_cells += [edge_data[(s_e[1], s_e[0])]]
            successors_nodes = list(G.successors(s_e[0]))
            successors_edges += [(s, s_e[0]) for s in successors_nodes]
        
            
        
        if cell in successors_edges_cells:
            skip = True
        elif cell in predecessors_edges_cells:
            skip = True
        elif len(set(successors_edges_cells).intersection(predecessors_edges_cells)) != 0:
            skip = True
            
        if G.in_degree(head) > 0:
            skip = True
            
        if not skip: 
            G.add_edge(tail, head, cell = cell)
            
        if len(list(nx.simple_cycles(G))) != 0:
            G.remove_edge(tail, head)
            skip = True
            
            
        for n in G.nodes:
            if len(nx.descendants(G, n)) + 1 == len(G.nodes):
                root = n
                finished = True
        
        
        if finished:
            break
   
            
        
    # Compute residual/root signal    
    event_labels_with_virtual = copy.copy(event_labels)
    edge_data = nx.get_edge_attributes(G,'cell')
    final_signals = copy.copy(signals)
    for n in list(reversed(list(nx.topological_sort(G)))):
        if len(list(G.in_edges(n))) == 0:
            continue
        u, v = list(G.in_edges(n))[0]
        cell = edge_data[(u,v)]
        sig = edge_cell_to_signal[((v, u), cell)]
        shifts = get_shifts_2(final_signals,event_labels, v, u)                       
        shifted_sig = shift_sig_2(sig,shifts)
        final_signals[event_labels == v] -= shifted_sig
        
        event_labels = np.array( [u if l == v else l for l in event_labels] )
    
    edge_to_matched_signals = {}
    
    # Store graph related info
    event_signal_ordering_tup = list(map(tuple, event_signal_ordering))
    for tail, head in G.edges:
        cell = edge_data[(tail,head)]
        d_sig = difference_signals[event_signal_ordering_tup.index((head, tail))]
        c_sig = edge_cell_to_signal[((head, tail), cell)]
        edge_to_matched_signals[((tail, head), cell)] = (d_sig, c_sig, edge_cell_to_loss[((head, tail), cell)], edge_cell_to_fit_info[((head, tail), cell)][2],edge_cell_to_fit_info[((head, tail), cell)][3])
    

    return finished, G, (initial_event_labels, signals_tmp), (event_labels_with_virtual, signals), (event_labels, final_signals), edge_to_matched_signals, tracker, mask, note


def noise_ball_clustering(signals_, mask, noise, electrode_list):
    signals = copy.copy(signals_)
    num_trials, num_electrodes, num_samples = signals.shape
    signals *= mask
    signals = (signals).reshape((num_trials, -1))
    event_labels = np.arange(num_trials)
    member_count = np.ones(num_trials)
    member_count = member_count[:, None]
    merging = True
    noise_threshold =  sum([noise[e]**2*sum(mask[i]) for i, e in enumerate(electrode_list)])
    
    while merging:
        power = np.sum(signals**2, axis = 1, keepdims = True)
        distance = power + power.T -2*signals @ signals.T
#         print("distance")
#         print(distance)
        threshold_array = (1/member_count + 1/member_count.T )* noise_threshold
#         print("threshold_array")
#         print(threshold_array)
        satisfaction = (distance <= threshold_array ) * (distance > 1 )
        close_enough = np.argwhere(satisfaction)
        if len(close_enough) == 0:
            break
        mc = close_enough[np.argmin(distance[satisfaction])]
        avg_signal = (signals[mc[0]] * member_count[mc[0]] + signals[mc[1]] * member_count[mc[1]])/(member_count[mc[0]] + member_count[mc[1]])
        new_label = min(mc[0], mc[1])
        tmp = copy.copy(event_labels)
        for m in mc:
            signals[event_labels == event_labels[m]] = avg_signal
            tmp[event_labels == event_labels[m]] = new_label
        event_labels = tmp
    return event_labels
        
            
        

def check_valid_graph(n, G):
    """
    Check whether the final explanation graph is valid or not.
    

    n: int
    G: networkx.Graph


    output: int
    """


    # First, check if the same cell repeats on any path in the final explanation graph
    chaini = chain.from_iterable
    roots = (v for v, d in G.in_degree() if d == 0)
    leaves = (v for v, d in G.out_degree() if d == 0)
    if len(G.nodes) == 1:
        return 1
    all_paths = partial(nx.all_simple_paths, G)
    ancestor_paths = list(chaini(starmap(all_paths, product(roots, leaves))))
    cell_on_edeges = nx.get_edge_attributes(G, "cell")
    
    cell_paths = []
    for p in ancestor_paths:
        cell_paths += [Counter([cell_on_edeges[(p[i], p[i+1])] for i in range(len(p)-1)])[n]]
    
    valid_graph_0 = int(max(cell_paths) <= 1)
    


    # Second, check if, when there are multiple distinct paths between two nodes, A) there are the same number of distict cells in each path, and B) the distint cell ids in all paths are found in each path
    valid_graph_1 = 1
    valid_graph_2 = 1
    
    roots = [v for v, d in G.in_degree() if d == 0]
    leaves = [v for v, d in G.out_degree() if d == 0]
    for r in roots:
        for l in leaves:
            simple_paths = list(nx.all_simple_paths(G, r, l))
            if len(simple_paths) > 1:
                overlapping_cell = []
                overlapping_cell_set = []
                for p in list(simple_paths):
                    overlapping_cell += [[cell_on_edeges[(p[i], p[i+1])] for i in range(len(p)-1)]]
                    overlapping_cell_set += [set([cell_on_edeges[(p[i], p[i+1])] for i in range(len(p)-1)])]
                path_lengths = [len(o) for o in overlapping_cell]
                valid_graph_1 *= int(max(path_lengths) == min(path_lengths))
                
                join_intersection = set.intersection(*overlapping_cell_set)
                valid_graph_2 *= int(all([join_intersection == ov for ov in overlapping_cell_set]))
                

    return valid_graph_0*valid_graph_1*valid_graph_2


def first_merge_event_cliques_by_noise(electrode_list, signals, event_labels_tmp,  mask, significant_electrodes, noise):
    """
    Iteratively merge event cliques according to average l2 distance.
    This implementation is a bit silly; will be improved later. 
    
    electrode_list: list<int>
    signals: np.array
    event_labels_tmp: list<int>
    state_to_cell: dict<key: int, value: list<tuple<int, int> > >
    mask: np.array
    significant_electrodes: np.array
    final_g: networkx.Graph
    noise: list<float>

    output: list<int>, dict<key: int, value: list<tuple<int, int> > >, networkx.Graph
    """
    event_labels = copy.copy(event_labels_tmp)
    
    # Compute noise distance threshold
    _, _, num_samples = signals.shape

    std = np.repeat([noise[e] for e in electrode_list], num_samples-5)
    
    std = math.sqrt(2) * std[None]
    
    k = num_samples*len(electrode_list)
    
    
    
    merging = True
    while merging:
        event_set = list(set(event_labels))
        num_events = len(event_set)
        threshold_array = (np.ones((num_events, num_events)) - np.eye(num_events)) * k 
        
        # Compute average noise distance between event cliques
        distance = np.zeros((num_events, num_events))
        for node_pair in list(itertools.combinations(np.arange(num_events), 2 )):
            X = signals[event_labels == event_set[node_pair[0]],:,5:] * mask[None,:,5:]
            Y = signals[event_labels == event_set[node_pair[1]],:,5:] * mask[None,:,5:]

            Z = all_pairwise(X,Y)
            
            num_diff_signals = Z.shape[0]
            Z = Z.reshape((num_diff_signals, -1))
            distance[node_pair[0], node_pair[1]] = np.sum((Z/std)**2)/num_diff_signals
            
            distance[node_pair[1], node_pair[0]] = np.sum((Z/std)**2)/num_diff_signals
            
        
            

        edges = np.argwhere((distance < threshold_array) *(distance != 0))
        if len(edges) == 0:
            break

        shortest_edge = edges[np.argmin([distance[e[0],e[1]] for e in edges])]
        cluster_members = [i for i,x in enumerate(event_labels) if x==event_set[shortest_edge[0]]]
        
        for i in [i for i,x in enumerate(event_labels) if x==event_set[shortest_edge[1]]]:
            cluster_members += [i]
        
        for i in cluster_members:
            event_labels[i] = min(shortest_edge[0], shortest_edge[1])
    return event_labels




def l2_max(event_pop_difference_signal, eis, ei_errors_, mask_, n):
    """
    Compute the optimal noise normalized MSE and negative log likelihood along with the corresponding alignment between the difference signal and cell templates.
    event_pop_difference_signal: np.array
    eis: np.array
    ei_errors_: np.array
    mask: np.array
    n: int

    output: np.array, np.array, np.array
    """

#     print("event_pop_difference_signal",event_pop_difference_signal.shape)
    difference_signals = event_pop_difference_signal[ 2:]
    event_pop = event_pop_difference_signal[:2]


    # Reverse the signal because convolution reverses it back
    difference_signal = np.expand_dims(difference_signals[::-1], 0)
    mask = np.expand_dims(mask_[::-1], 0)
    
    # Seperate cell variance from baseline noise variance
    # Scale noise variance to account for difference gaussian random variables
    # Scale cell variance to account for number of signals in the event clique
    ei_errors  = sum(event_pop) * ei_errors_

    # Decompose sliding || a - b ||**2 into multiple convolutions
    aTa = scipy.signal.convolve(1/ei_errors, (mask*difference_signal)**2, mode = 'valid')
    bTb = scipy.signal.convolve(eis**2/ei_errors, mask, mode = 'valid')
    aTb = scipy.signal.convolve(eis/ei_errors, mask*difference_signal, mode = 'valid')
    normalized_mse = aTa + bTb  - 2 * aTb 
    
#     ei_errors = ei_errors_
    
    aTa = scipy.signal.convolve(1/ei_errors, (mask*difference_signal)**2, mode = 'valid')

    var_bump = np.sum(np.log(ei_errors), axis = 1, keepdims=True)
    

    
    negative_log_likelihood =  aTa + bTb  - 2 * aTb  + var_bump
    return np.min(normalized_mse, axis = 1), np.argmin(normalized_mse, axis = 1), np.min(normalized_mse + var_bump, axis = 1), np.min(aTa, axis = 1)

def direct_similarity_exp(n, electrode_list, data_on_cells, difference_signals,  mask,event_pop, noise):
    """
    Apply l2_max function to each of the electrodes

    n: int
    electrode_list: list<int>
    data_on_cells: tuple<list<int>, np.array, np.array, np.array >
    difference_signals: np.array
    mask: np.array
    event_pop: np.array
    noise: list<float>

    output: np.array, dict<key: int, value: np.array>, np.array


    """
    cell_ids, cell_eis, cell_variance, _ = data_on_cells
    
    normalized_mse_total = np.zeros(( len(difference_signals), len(cell_ids)))
    neg_log_likelihood_total = np.zeros(( len(difference_signals), len(cell_ids)))
    noise_neg_log_likelihood_total = np.zeros(( len(difference_signals), len(cell_ids)))
    
    
    
    
    difference_signals = mask[None,:,:]/np.sum(event_pop, axis = 1)*difference_signals
    difference_signals = difference_signals[None, :,:,:]
    
    cell_eis = cell_eis[:,None,:,:]
    
    
    
        
        
    return normalized_mse_total, idxs_total, neg_log_likelihood_total, noise_neg_log_likelihood_total


def direct_similarity(n, electrode_list, data_on_cells, difference_signals,  mask,event_pop, noise):
    """
    Apply l2_max function to each of the electrodes

    n: int
    electrode_list: list<int>
    data_on_cells: tuple<list<int>, np.array, np.array, np.array >
    difference_signals: np.array
    mask: np.array
    event_pop: np.array
    noise: list<float>

    output: np.array, dict<key: int, value: np.array>, np.array


    """
    cell_ids, cell_eis, cell_variance, _ = data_on_cells
    
    normalized_mse_total = np.zeros(( len(difference_signals), len(cell_ids)))
    neg_log_likelihood_total = np.zeros(( len(difference_signals), len(cell_ids)))
    noise_neg_log_likelihood_total = np.zeros(( len(difference_signals), len(cell_ids)))
    idxs_total = {}
    
#     print("difference_signals shape",difference_signals.shape)

    for i,e in enumerate(electrode_list):
        args = (cell_eis[:, i,:], cell_variance[:, i,:], mask[i,:], noise[electrode_list[i]])
#         print("hstack shape", np.hstack((event_pop, difference_signals[:, i, :])).shape)
        A = np.apply_along_axis(l2_max, 1, np.hstack((event_pop, difference_signals[:, i, :])), *args)
        
        normalized_mse_total += A[:,0,:]
        neg_log_likelihood_total += A[:,2,:]
        noise_neg_log_likelihood_total += A[:, 3, :]
        idxs_total[e] = (A[:,1,:]).astype(int)
        
#     print("normalized_mse_total",normalized_mse_total)
#     print("neg_log_likelihood_total",neg_log_likelihood_total)
        
        
    return normalized_mse_total, idxs_total, neg_log_likelihood_total, noise_neg_log_likelihood_total

def direct_similarity_raw_wueric(n: int,
                             electrode_list: List[int],
                             data_on_cells: Tuple[List[int], np.ndarray, np.ndarray, np.ndarray],
                             difference_signals: np.ndarray,
                             mask: np.ndarray,
                             event_pop: np.ndarray,
                             noise: np.ndarray):
    """
    Alternative approach to direct_similarity, computes the correlations directly
        with the loops in C++ rather than Python.

    Single-threaded.

    :param n: int, ?? Is this unused??
    :param electrode_list: list of electrode indices
    :param data_on_cells: 4-tuple, elements are

        * cell_ids, List[int] of cell ids
        * cell_eis, np.ndarray of shape (n_cells, n_electrodes, n_timepoints_ei),
            contains reduced EIs of the cells
        * cell_variance, np.ndarray of shape (n_cells, n_electrodes, n_timepoints_ei),
        * ??

    :param difference_signals: np.ndarray, shape (2, n_electrodes, n_timepoints_diff_sig)
    :param mask: np.ndarray, shape (n_electrodes, n_timepoints_diff_sig)
    :param event_pop: np.ndarray
    :param noise: np.ndarray, shape (n_electrodes_total, ), noise for every channel in the array

    output: np.array, dict<key: int, value: np.array>, np.array


    """

    # cell_ids has length n_cells
    # cell_eis has shape (n_cells, n_electrodes, n_timepoints_ei)
    # cell_variance has shape (n_cells, n_electrodes, n_timepoints_ei)
    cell_ids, cell_eis, cell_variance, _ = data_on_cells
    
    n_cells, n_electrodes, n_timepoints_ei = cell_eis.shape

    cell_precision = np.ones_like(cell_variance)# shape (n_cells, n_electrodes, n_timepoints_ei)

    # shape (2, n_electrodes, n_timepoints_diff_sig)
    masked_difference_signal = mask[None, :, :] * difference_signals

    # shape (n_cells, n_electrodes, n_timepoints_ei)
    cell_eis_mul_prec = cell_eis * cell_precision

    # shape (n_cells, n_electrodes, n_timepoints_ei)
    cell_eis2_mul_prec = cell_eis * cell_eis_mul_prec


    # shape (n_cells, 2, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    aTa_parallel_corr = corr1d.batch_filter_batch_data_channel_correlate1D(
        cell_precision, (masked_difference_signal * masked_difference_signal)
    )

    # shape (n_cells, 1, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    bTb_parallel_corr = corr1d.batch_filter_batch_data_channel_correlate1D(
        cell_eis2_mul_prec, mask[None, :, :]
    )

    # shape (n_cells, 2, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    two_aTb_parallel_corr = corr1d.batch_filter_batch_data_channel_correlate1D(
        cell_eis_mul_prec, 2 * masked_difference_signal
    )

    event_pop_axis_sum = np.sum(event_pop, axis=1) # shape (2, )
    log_event_pop_sum = np.log(event_pop_axis_sum) # shape (2, )

    # shape (n_cells, )
    var_bump = np.sum(np.log(cell_variance), axis=(1, 2))[None, :] + \
                (n_electrodes * n_timepoints_ei * log_event_pop_sum[:, None])

    # shape (n_cells, 2, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    normalized_mse_all_no_event_pop = aTa_parallel_corr + bTb_parallel_corr - two_aTb_parallel_corr

    # shape (n_cells, 2, n_electrodes)
    min_mse_index = np.argmin(normalized_mse_all_no_event_pop, axis=3)

    # shape (n_cells, 2, n_electrodes, 1) -> (n_cells, 2, n_electrodes)
    min_mse_no_event_pop = np.take_along_axis(normalized_mse_all_no_event_pop,
                                              np.expand_dims(min_mse_index, 3), axis=3).squeeze(-1)

    # shape (n_cells, 2) -> (2, n_cells)
    normalized_mse_total = np.sum(min_mse_no_event_pop, axis=2).transpose(1, 0)

    # shape (2, n_cells) + (2, n_cells) -> (2, n_cells)
    neg_log_likelihood_total = normalized_mse_total #+ var_bump

    # shape (n_cells, 2, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    # -> (n_cells, 2, n_electrodes) -> (n_cells, 2) -> (2, n_cells)
    noise_neg_log_likelihood_total = np.sum(np.min(aTa_parallel_corr, axis=-1), axis=-1).transpose(1, 0) 

    idxs_total = {el_id : min_mse_index[:,:, idx].T for idx, el_id in enumerate(electrode_list)}

    return normalized_mse_total, idxs_total, neg_log_likelihood_total, noise_neg_log_likelihood_total



def direct_similarity_wueric(n: int,
                             electrode_list: List[int],
                             data_on_cells: Tuple[List[int], np.ndarray, np.ndarray, np.ndarray],
                             difference_signals: np.ndarray,
                             mask: np.ndarray,
                             event_pop: np.ndarray,
                             noise: np.ndarray):
    """
    Alternative approach to direct_similarity, computes the correlations directly
        with the loops in C++ rather than Python.

    Single-threaded.

    :param n: int, ?? Is this unused??
    :param electrode_list: list of electrode indices
    :param data_on_cells: 4-tuple, elements are

        * cell_ids, List[int] of cell ids
        * cell_eis, np.ndarray of shape (n_cells, n_electrodes, n_timepoints_ei),
            contains reduced EIs of the cells
        * cell_variance, np.ndarray of shape (n_cells, n_electrodes, n_timepoints_ei),
        * ??

    :param difference_signals: np.ndarray, shape (2, n_electrodes, n_timepoints_diff_sig)
    :param mask: np.ndarray, shape (n_electrodes, n_timepoints_diff_sig)
    :param event_pop: np.ndarray
    :param noise: np.ndarray, shape (n_electrodes_total, ), noise for every channel in the array

    output: np.array, dict<key: int, value: np.array>, np.array


    """

    # cell_ids has length n_cells
    # cell_eis has shape (n_cells, n_electrodes, n_timepoints_ei)
    # cell_variance has shape (n_cells, n_electrodes, n_timepoints_ei)
    cell_ids, cell_eis, cell_variance, _ = data_on_cells

    n_cells, n_electrodes, n_timepoints_ei = cell_eis.shape

    cell_precision = 1.0 / cell_variance # shape (n_cells, n_electrodes, n_timepoints_ei)

    # shape (2, n_electrodes, n_timepoints_diff_sig)
    masked_difference_signal = mask[None, :, :] * difference_signals

    # shape (n_cells, n_electrodes, n_timepoints_ei)
    cell_eis_mul_prec = cell_eis * cell_precision

    # shape (n_cells, n_electrodes, n_timepoints_ei)
    cell_eis2_mul_prec = cell_eis * cell_eis_mul_prec


    # shape (n_cells, 2, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    aTa_parallel_corr = corr1d.batch_filter_batch_data_channel_correlate1D(
        cell_precision, (masked_difference_signal * masked_difference_signal)
    )

    # shape (n_cells, 1, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    bTb_parallel_corr = corr1d.batch_filter_batch_data_channel_correlate1D(
        cell_eis2_mul_prec, mask[None, :, :]
    )

    # shape (n_cells, 2, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    two_aTb_parallel_corr = corr1d.batch_filter_batch_data_channel_correlate1D(
        cell_eis_mul_prec, 2 * masked_difference_signal
    )

    event_pop_axis_sum = np.sum(event_pop, axis=1) # shape (2, )
    log_event_pop_sum = np.log(event_pop_axis_sum) # shape (2, )

    # shape (n_cells, )
    var_bump = np.sum(np.log(cell_variance), axis=(1, 2))[None, :] + \
                (n_electrodes * n_timepoints_ei * log_event_pop_sum[:, None])

    # shape (n_cells, 2, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    normalized_mse_all_no_event_pop = aTa_parallel_corr + bTb_parallel_corr - two_aTb_parallel_corr

    # shape (n_cells, 2, n_electrodes)
    min_mse_index = np.argmin(normalized_mse_all_no_event_pop, axis=3)

    # shape (n_cells, 2, n_electrodes, 1) -> (n_cells, 2, n_electrodes)
    min_mse_no_event_pop = np.take_along_axis(normalized_mse_all_no_event_pop,
                                              np.expand_dims(min_mse_index, 3), axis=3).squeeze(-1)

    # shape (n_cells, 2) -> (2, n_cells)
    normalized_mse_total = np.sum(min_mse_no_event_pop, axis=2).transpose(1, 0) / event_pop_axis_sum[:, None]

    # shape (2, n_cells) + (2, n_cells) -> (2, n_cells)
    neg_log_likelihood_total = normalized_mse_total + var_bump

    # shape (n_cells, 2, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    # -> (n_cells, 2, n_electrodes) -> (n_cells, 2) -> (2, n_cells)
    noise_neg_log_likelihood_total = np.sum(np.min(aTa_parallel_corr, axis=-1), axis=-1).transpose(1, 0) / event_pop_axis_sum[:, None]

    idxs_total = {el_id : min_mse_index[:,:, idx].T for idx, el_id in enumerate(electrode_list)}

    return normalized_mse_total, idxs_total, neg_log_likelihood_total, noise_neg_log_likelihood_total



def infer_probability_from_graph(event_labels, edge_to_cell_and_fit_info, num_trials = 25):
    """
    Compute cell response probability from graph structure: 
    event_labels: list<int> 
    edge_to_cell_and_fit_info: dict<key: tuple<int, int>, value: list<tuple<int, float, int> > >
    num_trials: int, default = 25
    
    output: Counter 
    """
    counter = Counter()
    
    # Construct a graph from the edges assigned this round; update cell probability based on this iterations information
    G = nx.DiGraph([(k[1], k[0])  for k, v in edge_to_cell_and_fit_info.items()])
    event_labels_to_count = {l:sum(event_labels == l) for l in event_labels}
    
    for edge, info in edge_to_cell_and_fit_info.items():

        # Account for dependencies by accounting for descendants in graph
        count = sum([event_labels_to_count[i] for i in list(nx.descendants(G, edge[0]))]) + event_labels_to_count[edge[0]]

        # Only consider the addition of the probabilty if the template shift was in the proper range (determine in find_candidate_cells)
        counter[info[0][0]] += count/num_trials if not info[0][2] else 0
    
    return counter
            
        
    
def get_center_eis(n, electrode_list, ap, array_id = 1501,  num_samples = 25, snr_ratio = 1, power_threshold = 2, excluded_types = ['bad','dup'], excluded_cells = [], sample_len_left = 55,sample_len_right = 75 , with_noise = True):
    """
    Return templates and template parameters
    n: int
    electrode_list: list<int>
    ap: tuple<string, string> 
    array_id: int
    num_samples: int
    snr_ration: float
    
    output: list<int>, np.array, np.array, np.array
    """

    # Store relevant templates because on SNR on the stimulating electrode and SNR on the electrodes relevant to the cell-of-interest
    tl = TemplateLoader(ap[0], '', ap[1], array_id = array_id)
    
    if with_noise:
        tl.store_all_cells_except_with_noise(excluded_types)
    else:
        tl.store_all_cells_except(excluded_types)
    
    
    # Remove this condition in next set of run
    #tl.remove_templates_by_snr(electrode_list[0]+1, snr_ratio)
    tl.remove_templates_by_list(excluded_cells)
    tl.remove_templates_with_zero_variance(electrode_list)
#     tl.remove_templates_with_excess_variance(electrode_list)
    tl.remove_templates_by_elec_power(electrode_list, power_threshold, num_samples)
    
    if n not in tl.cellids:
        tl.store_cells_from_list([n])


    # Align the peak of each template along each electrode
    cell_eis = np.pad(np.array([tl.templates[i][electrode_list, :] for i in range(len(tl.cellids))]), ((0,0),(0,0),(sample_len_left, sample_len_right)), mode='edge')
    cell_eis_variance = np.pad(np.array([tl.templates_variance[i][electrode_list, :]**2 for i in range(len(tl.cellids))]), ((0,0),(0,0),(sample_len_left, sample_len_right)), mode='edge')

    peak_spike_times = np.argmin(cell_eis, axis = 2)
    peak_spike_times[peak_spike_times < sample_len_left] = sample_len_left
    
    cell_eis_tmp = np.zeros((cell_eis.shape[0], cell_eis.shape[1],sample_len_left + sample_len_right))
    cell_variance_tmp = np.zeros((cell_eis.shape[0], cell_eis.shape[1],sample_len_left + sample_len_right))
    

    for i in range(len(peak_spike_times)):
        for j in range(len(electrode_list)):
        
            cell_eis_tmp[i, j] = cell_eis[i,j, peak_spike_times[i][j]-sample_len_left:peak_spike_times[i][j]+sample_len_right]
            cell_variance_tmp[i, j] = cell_eis_variance[i,j, peak_spike_times[i][j]-sample_len_left:peak_spike_times[i][j]+sample_len_right]
#     for j in range(len(electrode_list)):
#         cell_variance_tmp[:,j][np.abs(cell_eis_tmp[:,j])<=1] = tl.noise[electrode_list][j]**2
#     for j in range(len(electrode_list)):
#         cell_eis_tmp[:,j][np.abs(cell_eis_tmp[:,j])<=1] = 0
        
        
        
    peak_spike_times = np.argmin(cell_eis_tmp, axis = 2)
    return tl.cellids, cell_eis_tmp, cell_variance_tmp, peak_spike_times

def clean_signals(signals, thr = 1):
    """
    Denoise signals with SVD decomposition + thresholding
    signals: np.array
    thr: int
    
    output: np.array
    """
   
    num_putative_signals = signals.shape[0]
    putative_signals_flat = signals.reshape((num_putative_signals, -1))
    
    # SVD of signals flattend across electrodes x time
    try:
        U, S, Vh = np.linalg.svd(putative_signals_flat)

        # Singular values are presorted; store indices with singular value above threshold
        sc = sum(S > thr) 

        # Reconstruct signal
        clean_putative_signals = U[:, :sc] @ np.diag(S[:sc]) @ Vh[:sc, :]
        return clean_putative_signals.reshape(signals.shape)
    except:
        return signals
    
    
    
def get_event_signals(electrode_list, signals, event_labels):
    """
    Return the median signal of each event clique
    electrode_list: list<int>
    signals: np.array
    event_labels: list<int>

    output: np.array
    """
    event_signals = np.zeros((len(set(event_labels)), len(electrode_list), signals.shape[2]))
    for i, l in enumerate(set(event_labels)):

        candidates = signals[event_labels == l, :, :]
        event_signals[i, :, :] = np.median(candidates, axis = 0)
    
    return event_signals

def remove_cell_invalid_paths(edge_to_cell_and_fit_info_tmp,  edge_to_loss):
    """
    Remove edges from graph that create necessarily incorrect structures in graph. 
    These sorts of error are not the type which can be dealt with (necessarily) by removing multipaths;
    however, they do constitute violation of basic fact surrounding the data. 

    edge_to_cell_and_fit_info: dict<key: tuple<int, int>, value: list<tuple<int, float, int> >
    edge_to_loss: dict<key: tuple<int, int>, value: float >

    output: dict<key: tuple<int, int>, value: list<tuple<int, float, int> >
    """
    edge_to_cell_and_fit_info = copy.copy(edge_to_cell_and_fit_info_tmp)

    # Create a mapping from cells to the edges they are assigned to    
    cell_to_edge = {}
    for k, v in edge_to_cell_and_fit_info.items():
        if v[0][0] not in cell_to_edge.keys():
            cell_to_edge[v[0][0]] = []
        cell_to_edge[v[0][0]].append(k)
    
    # In this first part, we remove invalid inverted v-structures.
    # An inverted v-structure is invalid if more than one of the edges is assigned to the same cell
    # This is because this graphical structure implies a cell is in superposition with itself
    valid_v_structures = []
    for c, es in cell_to_edge.items():
        end_nodes = np.array([el[0] for el in list(map(list, es))])
        for end_node in set(end_nodes):
            idxs = np.argwhere(end_nodes == end_node).flatten()
            
            # If there are repeated end nodes, then that mean there are distinct edge that end of the same node and are assigned to the same cell. 
            if len(idxs) > 1: 
                # Keep the edge with the highest likelihood/ lowest loss
                score = [edge_to_loss[es[i]] for i in idxs]
                valid_v_structures += [es[idxs[np.argmin(score)]]]
            else:
                valid_v_structures += [es[idxs.item()]]
    

    tmp_dict = {}
    for k, v in edge_to_cell_and_fit_info.items():
        if k in valid_v_structures:
            tmp_dict[k] = v  
    edge_to_cell_and_fit_info = tmp_dict
                

            
    # In this second part, we remove invalid linear structures.
    # A linear structure is invalid if more than one of the edges in a path is assigned to the same cell
    # This is because this graphical structure implies a cell is in superposition with itself
    bad_linear_structures = []
    edges = [(k[1], k[0])  for k, v in edge_to_cell_and_fit_info.items()]
    G = nx.DiGraph(edges)

    # Construct all paths between pairs of nodes
    roots = (v for v, d in G.in_degree() if d == 0)
    leaves = (v for v, d in G.out_degree() if d == 0)
    all_paths = partial(nx.all_simple_paths, G)
    chaini = chain.from_iterable
    full_paths = list(chaini(starmap(all_paths, product(roots, leaves))))

    
    edge_to_cell = {k:v[0][0] for k, v in edge_to_cell_and_fit_info.items()}
    # For each path,
    for p in full_paths:
        cells_in_path = []

        # Examine the edges in the path
        edges_in_path = []
        for j in range(len(p)-1):
            edges_in_path += [(p[j+1], p[j])]
        edges_in_path = set(edges_in_path)

        # Take the intersection of the path and the edges the cell is assigned to
        # If the cell is assigned to multiple edges in the same path, store those edges
        path_int = {c:edges_in_path & set(e) for c, e  in cell_to_edge.items() if len(edges_in_path & set(e)) > 1}
      
        # Retain the edge of the conflicting path with the maximum likelihood 
        for c, es in path_int.items():
            score = [edge_to_loss[e_tmp] for e_tmp in es]
            bad_linear_structures += [e for i, e in enumerate(es) if i != np.argmin(score)]
            
    tmp_dict = {}
    for k, v in edge_to_cell_and_fit_info.items():
        if k not in bad_linear_structures:
            tmp_dict[k] = v
    edge_to_cell_and_fit_info = tmp_dict
    

    # In this third part, we remove cycles.
    # To do so, we construct a graph, iteratively find cycles in the graph and remove one edge at a time with higest lost/ lowest likelihood
    edges = [(k[0], k[1])  for k, v in edge_to_cell_and_fit_info.items()]
    G = nx.DiGraph(edges)
    looking_for_cycles = True
    while looking_for_cycles:
        try:
            es  = list(map(tuple, nx.find_cycle(G, orientation="original")))
            es = [(e[0], e[1]) for e in es]
            i = np.argmax([edge_to_loss[e] for e in es])
            G.remove_edge(es[i][0], es[i][1])
    
        except:
            looking_for_cycles = False
            
    tmp_dict = {}
    for k, v in edge_to_cell_and_fit_info.items():
        if k in list(map(tuple, G.edges)):
            tmp_dict[k] = v
    edge_to_cell_and_fit_info = tmp_dict

    return edge_to_cell_and_fit_info 
                
def compute_difference_signals(signals, event_labels):
    """
    Compute all potential difference signals between event cliques and store the subtraction ordering
    signals: np.array
    event_labels: list<int>

    output: np.array, np.array
    """
    num_trials, num_electrodes, num_samples = signals.shape
    
    event_set = list(set(event_labels))
    num_event_signals = len(event_set)
    
    pairwise_signals = []
    signal_ordering = []
    

    for pair_of_event_cliques in list(itertools.combinations(np.arange(num_event_signals), 2 )):
        # Compute set of pairwise difference signals between event cliques
        X = signals[event_labels == event_set[pair_of_event_cliques[0]]]
        Y = signals[event_labels == event_set[pair_of_event_cliques[1]]]
        Z = np.einsum('ijk->jik',all_pairwise(X,Y)).reshape((num_electrodes, -1))

        # Align and take the average as the exemplar difference signal between the two event cliques
        list_of_tuples = [align_group_main(Z[i], sample_len = num_samples) for i in range(num_electrodes)]
        diff_signal = np.array([t__[0] for t__ in list_of_tuples])

       
        pairwise_signals += [diff_signal]
        pairwise_signals += [-diff_signal]
        signal_ordering += [(event_set[pair_of_event_cliques[0]], event_set[pair_of_event_cliques[1]])]
        signal_ordering += [(event_set[pair_of_event_cliques[1]], event_set[pair_of_event_cliques[0]])]
        
        
    pairwise_signals = np.array(pairwise_signals)
    signal_ordering = np.array(signal_ordering)
    
    return pairwise_signals, signal_ordering





def compute_dependencies(electrode_list, edge_to_cell_and_fit_info, edge_to_signal, event_labels_tmp, signals_tmp, node_state_to_cells, final_g, cells_on_trials, signal_len = 25, num_trials = 25, artifact_estimate = None):
    """
    Update signal set and graph to account for iterative signal merging via template subtraction
    electrode_list: list<int>
    edge_to_cell_and_fit_info: dict<key: tuple<int, int>, value: list<tuple<int, float, int> > >
    edge_to_signal: dict<key: tuple<int, int>, value: np.array >
    event_labels_tmp: list<int>
    signals_tmp: np.array
    node_state_to_cells: dict<key: int, value: list<tuple<int, int> > >
    final_g: nx.Graph

    output: np.array, list<int>, dict<key: int, value: list<tuple<int, int> > >, nx.Graph
    """
 
    signals = copy.copy(signals_tmp)
    event_labels = copy.copy(event_labels_tmp)

    # Construct edge set and corresponding graph
    edges = [(k[1], k[0])  for k, v in edge_to_cell_and_fit_info.items()]
    first_edge = edges[0]
    G = nx.DiGraph(edges)
    edge_to_cell = {k:v[0][0] for k, v in edge_to_cell_and_fit_info.items()}
    edge_to_oor = {k:v[0][2] for k, v in edge_to_cell_and_fit_info.items()}

    # Create generator of set of weakly connected nodes
    H_generator = nx.weakly_connected_components(G)

    ### This may not be necessary anymore; check after other changes have been made
    if len(edges) == 0:
        return signals, [], _, _
        
    
    # Add edges to explanation graph from this run
    final_g.add_edges_from([(e[0],e[1],{"cell":edge_to_cell[(e[1],e[0])]}) for e in edges])

    new_node = -1
    chaini = chain.from_iterable
    for node_set in H_generator:
        
        H = nx.DiGraph( G.out_edges(node_set) )

        # Compute and iterate over reverse topological sort of nodes
        rev_topo_sort = list(reversed(list(nx.topological_sort(H))))
        for n in rev_topo_sort:
            # Only check nodes with parents
            if list(H.in_edges(n)):

                parent_nodes = [edge[0] for edge in list(H.in_edges(n))]
                # If the node only has one parent
                if len(parent_nodes) == 1:
                    # Subtract shifted template from signal set
                    sig = edge_to_signal[(n, parent_nodes[0])]
                    shifts = get_shifts(signals,event_labels, n, parent_nodes[0])                   
                    shifted_sig = shift_sig(sig,shifts)                  
                    signals[event_labels == n, :,:] -= shifted_sig
                    
                    # Note cells on each trial
                    event_clique_trials = [i for i, e in enumerate(event_labels) if e == n]
                    for trial in event_clique_trials:
                        cells_on_trials[trial] += [edge_to_cell[(n, parent_nodes[0])]]
                    
                    
                    # Relabel event cliques to reflect merging
                    event_labels = [parent_nodes[0] if e == n else e for e in event_labels]

                    # Keep track of which nodes have which cells subtracted into them
                    node_state_to_cells[parent_nodes[0]] += node_state_to_cells[n]
                    node_state_to_cells[parent_nodes[0]] += [(edge_to_cell[(n, parent_nodes[0])],edge_to_oor[(n, parent_nodes[0])])]
                    
                    node_state_to_cells.pop(n, None)
       
                # If the node only has multiple parents
                else:
                    
                    trial_signals_tmp = np.zeros((num_trials, len(electrode_list), signal_len))
                    node_state_to_cells[new_node] = node_state_to_cells[n]
                    
                    
                    # Note cells on each trial
                    event_clique_trials = [i for i, e in enumerate(event_labels) if e == n]
                    
                    
                    for p_node in parent_nodes:
                        
                        sig = edge_to_signal[(n, p_node)]
                        shifts = get_shifts(signals,event_labels, n, p_node)                       
                        shifted_sig = shift_sig(sig,shifts)
                        trial_signals_tmp[event_labels == n, :,:] += shifted_sig
                        
                        node_state_to_cells[new_node] += [(edge_to_cell[(n, p_node)], edge_to_oor[(n, p_node)])] 
                        
                        
                        for trial in event_clique_trials:
                            cells_on_trials[trial] += [edge_to_cell[(n, p_node)]]
                        
                    node_state_to_cells.pop(n, None)
                    
                    # Subtract sum of shifted template from signal set (sum given multiple parents)
                    signals -= trial_signals_tmp
                    
                    # Relabel event cliques to reflect merging
                    event_labels = [new_node if e == n else e for e in event_labels]

                    # Add new node to explanation grpah
                    final_g.add_node(new_node)
                    new_node -= 1
                        

    # Update node labels to 0-index them
    node_state_to_cells = {k-new_node-1:v for k,v in node_state_to_cells.items()}
    g_relabel = {s:s-new_node-1 for s in list(final_g.nodes)}
    final_g =  nx.relabel_nodes(final_g, g_relabel)
    event_labels = np.array([e-new_node-1 for e in event_labels])

    if artifact_estimate != None:
        artifact_estimate += -new_node-1
    return signals, event_labels, node_state_to_cells, final_g, cells_on_trials, artifact_estimate

def get_shifts(signals,event_labels, m, n):
    """
    Compute difference signal shifts
    signals: np.array
    event_labels: list<int>
    m, n: int

    output: np.array
    """
    
    # Compute pairwise differences between (updating) event signal cliques
    _, num_electrodes, num_samples = signals.shape
    X = signals[event_labels == m]
    Y = signals[event_labels == n]
    Z = all_pairwise(X, Y)
    Z = np.einsum('ijk->jik',Z).reshape((num_electrodes, -1))

    # Store sample of minimum for each difference signal on each electrode
    list_of_signal_peaks = [np.argmin(Z[i].reshape((-1,num_samples)), axis = 1) for i in range(num_electrodes)]
    
    # Choose a representative shifts for the signal that is subtracted from
    shifts = np.median(np.array(list_of_signal_peaks).reshape((-1, len(X), len(Y))), axis = 2)
    
    return shifts
        
    
def shift_sig(sig, shifts):
    """
    Shift difference signal sig according to shifts
    
    sig: np.array
    shifts: np.array
    
    output: np.array
    """
    peak_times = np.argmin(sig, axis=1)
    offsets = shifts - peak_times[:, None]
    num_electrodes, num_relevant_trials = shifts.shape
    shifted_signals = np.einsum('ijk->jik', np.array([np.array([shift(sig[i],s,mode='nearest') for s in offset]) for i, offset in enumerate(offsets)]))
    return shifted_signals


def align_group_main(x, sample_len = 50):
    """
    Reshape signal and call align_group
    x: np.array
    """
    X = x.reshape((-1,sample_len))

    return align_group(X)


    
    
def align_group(X, window = 10, res = 2):
    """
    Align a set of set of signals to approximatetly minimize global l2 distance
    X: np.array
    window: int, default = 10

    output: tuple<np.array, tuple<int, int> >

    """
    
    # Special case: Only one signal in signal set
    if len(X) == 1:
        min_index = np.argmin(X,axis = 1)
        max_index = np.argmax(X,axis = 1)
        return (X[0], (min_index, max_index))
    else:
        
        # Upsample data for better signal alignment
        x = np.arange(0, len(X[0]))
        interpolant = scipy.interpolate.interp1d(x,X, kind='cubic')
        domain = np.linspace(0, len(x)-1, len(x)*res)
        high_res_X = interpolant(domain)
        
        min_index = np.argmin(X,axis = 1)
        max_index = np.argmax(X,axis = 1)
        
        # Find signals with median sample of either depolarization or hyperpolarization hump (if there is a cell firing) i.e. find an average signal
        c_i = np.argsort(min_index)[len(min_index)//2]
        
        # Store median signal to align all others to
        x = high_res_X[c_i:c_i + 1,::-1]
        
        one_window = np.zeros_like(x)
        one_window[:,::res] = 1
        one_window = one_window[:,::-1]
        
        # Compute l2 difference between reference and all other signals using convolution
        _, signal_len = x.shape
        conv_Xx = scipy.signal.convolve(high_res_X, x*one_window, mode = 'full')[:,(2*signal_len-1)//2-window*res:(2*signal_len-1)//2+window*res]
        conv_X21 = scipy.signal.convolve(high_res_X**2, one_window, mode = 'full')[:,(2*signal_len-1)//2-window*res:(2*signal_len-1)//2+window*res]
        conv_x21 = scipy.signal.convolve(x**2, one_window, mode = 'full')[:,(2*signal_len-1)//2-window*res:(2*signal_len-1)//2+window*res]
        
        # Indentify and shift signals to approximately minimize global l2 distance   
        amin_Xx = -(np.argmin( conv_X21 + conv_x21 - 2*conv_Xx, axis = 1) - window*res)
        high_res_X = strided_indexing_roll(high_res_X, amin_Xx)
        np.nan_to_num(high_res_X, copy = False)
        
        return (np.mean(high_res_X[:,::res], axis = 0), (min_index, max_index))

    
def all_pairwise(X_, Y_):
    """
    Compute pairwise difference signals between matrices with different number of rows
    X_: np.array
    Y_: np.array

    output: np.array
    """
    _, num_electrodes, num_samples = X_.shape
    X = X_[:,None,:,:]
    Y = Y_[None,:,:,:]
    
    return (X-Y).reshape((-1,num_electrodes,num_samples))


def get_shifts_2(signals,event_labels, m, n):
    """
    Compute difference signal shifts
    signals: np.array
    event_labels: list<int>
    m, n: int

    output: np.array
    """
    
    # Compute pairwise differences between (updating) event signal cliques
    _, num_electrodes, num_samples = signals.shape
    X = signals[event_labels == m]
    Y = np.mean(signals[event_labels == n], axis = 0, keepdims = True)
    Z = all_pairwise(X, Y)
    Z = Z.reshape((len(X), len(Y), num_electrodes, num_samples))
    Z = np.argmin(Z, axis = 3)
    Z = np.median(Z, axis = 1)
    shifts = Z.T
    
    return shifts

def shift_sig_2(sig, shifts):
    """
    Shift difference signal sig according to shifts
    
    sig: np.array
    shifts: np.array
    
    output: np.array
    """
    peak_times = np.argmin(sig, axis=1)
    offsets = shifts - peak_times[:, None]
    num_electrodes, num_relevant_trials = shifts.shape
    shifted_signals = np.einsum('ijk->jik', np.array([np.array([shift(sig[i],s) for s in offset]) for i, offset in enumerate(offsets)]))
    return shifted_signals

def remove_multi_paths(edge_to_cell_and_fit_info, edge_to_loss, signal_len = 25):
    """
    Remove edges according to approximate maximum likelihood such that there are no multipaths, i.e. for u, v, u != v, there is at most one unique path from u to v.
    Note: This version is not the best approximation that can be written. This is because the signals produced by multipaths are compared to the difference signals of the median event clique signals . 
    This is different than what is done in the earlier methods: the mean of the chi-square aligned pairwise differences between event clique signals.
    This is especially relevant in cases of highly variant latencies and cell superpositions. However, considering the performance of this approximate version, improvements to this do not seem like a priority. 


    electrode_list: list<int>
    median_sig_from_event_clique: np.array
    event_clique_labels: list<int> 
    edge_to_cell_and_fit_info: dict<key: tuple<int, int>, value: list<tuple<int, float, int> > >


    output: dict<key: tuple<int, int>, value: list<tuple<int, float, int> >
    """

    
    
    kept_edges = []
    end_nodes = []
    stop = False
    edges = list(edge_to_cell_and_fit_info.keys())
    loss = list([edge_to_loss[e] for e in edges])
    sorted_edges = [x for _, x in sorted(zip(loss, edges), key=lambda pair: pair[0])]
    
    for e in sorted_edges:
        if e in end_nodes:
            continue
        back_adds = []
        front_adds = []
        stop = False
        for e_star in end_nodes:
            if stop:
                break
            if e_star[1] == e[0]:
                if (e_star[0], e[1]) in end_nodes:
                    stop = True
                    
                    break
                else:
                    back_adds += [(e_star[0], e[1])]
            if e_star[0] == e[1]:
                if (e[0], e_star[1]) in end_nodes:
                    stop=True
                    break
                else:
                    front_adds += [(e[0], e_star[1])]
                    
        if stop:
            continue
        else:
            end_nodes+=back_adds
            end_nodes+=front_adds
            end_nodes+=[e]
            kept_edges +=[e]
    
    tmp_edge_to_cell_and_fit_info = {}

    for edge in kept_edges:
        tmp_edge_to_cell_and_fit_info[edge] = edge_to_cell_and_fit_info[edge]
    
    return tmp_edge_to_cell_and_fit_info
    


def connected_path(path):
    """
    Return true/false whether set of edges forms a connected path
    path: list<tuple>
    output: bool
    """
    if len(path) == 1:
        return True
    
    return np.array([path[i][1] == path[i+1][0] for i in range(len(path) - 1)]).all()

def strided_indexing_roll(A, r):
    """
    Roll each row of a matrix A according to shifts r. 
    A: np.array
    r: np.array
    output: np.array
    """
    # Concatenate with sliced to cover all rolls
    p = np.full((A.shape[0],A.shape[1]-1),np.nan)
    A_ext = np.concatenate((p,A,p),axis=1)

    # Get sliding windows; use advanced-indexing to select appropriate ones
    n = A.shape[1]
    return viewW(A_ext,(1,n))[np.arange(len(r)), -r + (n-1),0]

def find_candidate_edges_raw(n, electrode_list, data_on_cells, difference_signals, difference_signal_ordering, mask, event_labels, noise,
                               constrained = 'gof',  p = 0.99, max_electrodes_oor = 2, valid_shift_range = 35, artifact_estimate = None, plot = True):
    '''
    Generate potential edges in graph. 
    
    n: int
    electrode_list: list<int>
    data_on_cells: tuple<list<int>, np.array, np.array, np.array >
    difference_signals: np.array,
    difference_signal_ordering: np.array,
    state_to_cell: dict<key: int, value: list<tuple<int, int> > >
    mask: np.array
    event_labels: list<int>
    noise: list<float>

    output: dict<key: tuple<int, int>, value: list<tuple<int, float, int> > >, dict<key: tuple<int, int>, value: np.array >, dict<key: tuple<int, int>, value: np.array >, dict<key: tuple<int, int>, value: float >
    '''
    num_difference_signals = len(difference_signal_ordering)
    
    edge_cell_to_fit_info = {}
    edge_cell_to_sig = {}
    edge_cell_to_error = {}
    edge_cell_to_loss = {}
    edge_cell_to_noise_loss = {}
    
    if len(difference_signals) == 0:
        return edge_cell_to_fit_info, edge_cell_to_sig, edge_cell_to_error, edge_cell_to_loss, edge_cell_to_noise_loss

    # Chi-squared determined goodness-of-fit threshold for noise-normalized MSE between templates and difference signals
    goodness_of_fit_threshold = sum([scipy.stats.chi2.ppf(p, sum(mask[i])) for i in range(len(mask)) if sum(mask[i]) != 0])

    # Load up template information
    cell_ids, cell_eis, cell_variance, peak_st = data_on_cells
    num_cells = len(cell_ids)
    central_time = peak_st[0,0]
     
    # Compute noise-normalize mse, the best shift, and negative log-likehood
    event_counts = Counter(event_labels)
   
    event_pop = np.array([[1/event_counts[e[0]],1/event_counts[e[1]]] for e in difference_signal_ordering])
    normalized_mse, best_match_idx, negative_log_likelihood, noise_negative_log_likelihood = direct_similarity_raw_wueric(n, electrode_list, data_on_cells, difference_signals,  mask, event_pop, noise)
#     print("difference_signal_ordering",difference_signal_ordering)
#     print("cell_ids", cell_ids)
#     print("negative_log_likelihood", negative_log_likelihood)
    # Choose different signal orientation based on maxmimum likehood
    max_likelihood = np.min(negative_log_likelihood, axis = 1)
    if artifact_estimate != None:
        negative_log_likelihood[difference_signal_ordering[:, 0] == artifact_estimate] = np.inf
        normalized_mse[difference_signal_ordering[:, 0] == artifact_estimate] = np.inf
    
    difference_signal_ordering_rep = list(map(tuple, np.repeat(difference_signal_ordering, len(cell_ids), axis = 0)))
#     print("difference_signal_ordering_rep",difference_signal_ordering_rep)
    
    normalized_mse = normalized_mse.flatten()
#     print("normalized_mse",normalized_mse)
    best_match_idx = {k:v.flatten() for k, v in best_match_idx.items()}
    negative_log_likelihood = negative_log_likelihood.flatten()
    noise_negative_log_likelihood = noise_negative_log_likelihood.flatten()
#     print("normalized mse",normalized_mse)
    
    
    num_samples = difference_signals.shape[2]
    best_match_signals = np.zeros((len(negative_log_likelihood), len(electrode_list), num_samples))
    best_match_full = np.zeros((len(negative_log_likelihood), len(electrode_list), cell_eis.shape[2]))
    best_match_variance = np.zeros((len(negative_log_likelihood), len(electrode_list), num_samples))
    
    # Take out the best fitting slice from the template for the corresponding difference signal
    # For each electrode, 
    cell_indices = np.tile(np.arange(len(cell_ids)), num_difference_signals)
    for i, e in enumerate(electrode_list):
        relevant_indices_start = np.array(best_match_idx[e])
        
        cell_subset = cell_eis[cell_indices, i, :]
        variance_subset = cell_variance[cell_indices, i, :]
        
        # Slice out the relevant points from the template and template error according to the shift information from 'direct_similarity'
        best_match_signals[:, i, :] = np.array([[cell_subset[j, i] for i in range(l, l +num_samples)] for j, l in enumerate(relevant_indices_start)])
        best_match_variance[:, i, :] = np.array([[variance_subset[j, i] for i in range(l , l+num_samples)] for j, l in enumerate(relevant_indices_start)])   

        best_match_full[:, i, :] = cell_subset
                     
    # For each matched cell, 
    for i, pair in enumerate(difference_signal_ordering_rep):
               
        edge_cell_to_fit_info[(pair, cell_ids[cell_indices[i]])] = (negative_log_likelihood[i], noise_negative_log_likelihood[i], normalized_mse[i],goodness_of_fit_threshold)
        edge_cell_to_sig[(pair, cell_ids[cell_indices[i]])] = best_match_signals[i]
        edge_cell_to_error[(pair, cell_ids[cell_indices[i]])] = best_match_variance[i]
        edge_cell_to_loss[(pair, cell_ids[cell_indices[i]])] = negative_log_likelihood[i]
        edge_cell_to_noise_loss[(pair, cell_ids[cell_indices[i]])] = noise_negative_log_likelihood[i]
        
            
           
    return edge_cell_to_fit_info, edge_cell_to_sig, edge_cell_to_error, edge_cell_to_loss, edge_cell_to_noise_loss



def find_candidate_edges(n, electrode_list, data_on_cells, difference_signals, difference_signal_ordering, mask, event_labels, noise,
                               constrained = 'gof',  p = 0.99, max_electrodes_oor = 2, valid_shift_range = 35, artifact_estimate = None, plot = True, no_time = False):
    '''
    Generate potential edges in graph. 
    
    n: int
    electrode_list: list<int>
    data_on_cells: tuple<list<int>, np.array, np.array, np.array >
    difference_signals: np.array,
    difference_signal_ordering: np.array,
    state_to_cell: dict<key: int, value: list<tuple<int, int> > >
    mask: np.array
    event_labels: list<int>
    noise: list<float>

    output: dict<key: tuple<int, int>, value: list<tuple<int, float, int> > >, dict<key: tuple<int, int>, value: np.array >, dict<key: tuple<int, int>, value: np.array >, dict<key: tuple<int, int>, value: float >
    '''
    num_difference_signals = len(difference_signal_ordering)
    
    edge_cell_to_fit_info = {}
    edge_cell_to_sig = {}
    edge_cell_to_error = {}
    edge_cell_to_loss = {}
    edge_cell_to_noise_loss = {}
    
    if len(difference_signals) == 0:
        return edge_cell_to_fit_info, edge_cell_to_sig, edge_cell_to_error, edge_cell_to_loss, edge_cell_to_noise_loss

    # Chi-squared determined goodness-of-fit threshold for noise-normalized MSE between templates and difference signals
    goodness_of_fit_threshold = sum([scipy.stats.chi2.ppf(p, sum(mask[i])) for i in range(len(mask)) if sum(mask[i]) != 0])

    # Load up template information
    cell_ids, cell_eis, cell_variance, peak_st = data_on_cells
    num_cells = len(cell_ids)
    central_time = peak_st[0,0]
     
    # Compute noise-normalize mse, the best shift, and negative log-likehood
    event_counts = Counter(event_labels)
   
    event_pop = np.array([[1/event_counts[e[0]],1/event_counts[e[1]]] for e in difference_signal_ordering])
    if not no_time:
        normalized_mse, best_match_idx, negative_log_likelihood, noise_negative_log_likelihood = direct_similarity_wueric(n, electrode_list, data_on_cells, difference_signals,  mask, event_pop, noise)
    else:
        cell_eis_space_ind = np.argmin(cell_eis, axis = 2)[:,:,None]
        cell_eis_space = np.take_along_axis(cell_eis, cell_eis_space_ind, axis =2)
        cell_variance_space = np.take_along_axis(cell_variance, cell_eis_space_ind, axis =2)
        data_on_cells_space = (cell_ids, cell_eis_space, cell_variance_space, peak_st)
        difference_signals_space = np.min(difference_signals, axis = 2)[:,:,None]
        normalized_mse, best_match_idx, negative_log_likelihood, noise_negative_log_likelihood = direct_similarity_wueric(n, electrode_list, data_on_cells_space, difference_signals_space,  np.ones((len(mask), 1)), event_pop, noise)
        

    # Choose different signal orientation based on maxmimum likehood
    max_likelihood = np.min(negative_log_likelihood, axis = 1)
    if artifact_estimate != None:
        negative_log_likelihood[difference_signal_ordering[:, 0] == artifact_estimate] = np.inf
        normalized_mse[difference_signal_ordering[:, 0] == artifact_estimate] = np.inf
    
    difference_signal_ordering_rep = list(map(tuple, np.repeat(difference_signal_ordering, len(cell_ids), axis = 0)))
#     print("difference_signal_ordering_rep",difference_signal_ordering_rep)
    
    normalized_mse = normalized_mse.flatten()
#     print("normalized_mse",normalized_mse)
    best_match_idx = {k:v.flatten() for k, v in best_match_idx.items()}
    negative_log_likelihood = negative_log_likelihood.flatten()
    noise_negative_log_likelihood = noise_negative_log_likelihood.flatten()
#     print("normalized mse",normalized_mse)
    
    
    num_samples = difference_signals.shape[2]
    best_match_signals = np.zeros((len(negative_log_likelihood), len(electrode_list), num_samples))
    best_match_full = np.zeros((len(negative_log_likelihood), len(electrode_list), cell_eis.shape[2]))
    best_match_variance = np.zeros((len(negative_log_likelihood), len(electrode_list), num_samples))
    
    # Take out the best fitting slice from the template for the corresponding difference signal
    # For each electrode, 
    cell_indices = np.tile(np.arange(len(cell_ids)), num_difference_signals)
    for i, e in enumerate(electrode_list):
        relevant_indices_start = np.array(best_match_idx[e])
        
        cell_subset = cell_eis[cell_indices, i, :]
        variance_subset = cell_variance[cell_indices, i, :]
        
        # Slice out the relevant points from the template and template error according to the shift information from 'direct_similarity'
        best_match_signals[:, i, :] = np.array([[cell_subset[j, i] for i in range(l, l +num_samples)] for j, l in enumerate(relevant_indices_start)])
        best_match_variance[:, i, :] = np.array([[variance_subset[j, i] for i in range(l , l+num_samples)] for j, l in enumerate(relevant_indices_start)])   

        best_match_full[:, i, :] = cell_subset
                     
    # For each matched cell, 
    for i, pair in enumerate(difference_signal_ordering_rep):
               
        edge_cell_to_fit_info[(pair, cell_ids[cell_indices[i]])] = (negative_log_likelihood[i], noise_negative_log_likelihood[i], normalized_mse[i],goodness_of_fit_threshold)
        edge_cell_to_sig[(pair, cell_ids[cell_indices[i]])] = best_match_signals[i]
        edge_cell_to_error[(pair, cell_ids[cell_indices[i]])] = best_match_variance[i]
        edge_cell_to_loss[(pair, cell_ids[cell_indices[i]])] = negative_log_likelihood[i]
        edge_cell_to_noise_loss[(pair, cell_ids[cell_indices[i]])] = noise_negative_log_likelihood[i]
        
            
           
    return edge_cell_to_fit_info, edge_cell_to_sig, edge_cell_to_error, edge_cell_to_loss, edge_cell_to_noise_loss


def check_graph(G):
    relevant_nodes = list(set([a for e in G.edges for a in e]))
    edge_to_cell = nx.get_edge_attributes(G, 'cell')
    issa_problem = False
    issa_problem_edges = []
    resolved_v_structure_nodes = []
    
    roots = [n for n in G.nodes if G.in_degree(n)==0 and G.out_degree(n)>0]
    for n in [n for n in G.nodes if G.in_degree(n)>1]:
        for r in roots:
            end_edges = [tuple(p[-2:]) for p in list(nx.all_simple_paths(G, r, n))]
            if all([e in end_edges for e in G.in_edges(n)]):
                resolved_v_structure_nodes += [n]
    
    for ui, vi in list(itertools.combinations(np.arange(len(relevant_nodes)), 2 )):
        u =relevant_nodes[ui]
        v =relevant_nodes[vi]
        
        
        
        paths = nx.all_simple_paths(G, u, v)
        
        simple_node_paths = list(nx.all_simple_paths(G, u, v))
#         if G.in_degree(v) > 1 and len(simple_node_paths) > 1 and len(set([p[-2] for p in simple_node_paths])) > 1:
            
#             resolved_v_structure_nodes += [v]
        cells_on_path = []
        for p in map(nx.utils.pairwise, paths):
            
            
            cells_on_path += [set([edge_to_cell[x] for x in p])]
        
        if len(cells_on_path) > 1:
            if not all([cells_on_path[0] == c for c in cells_on_path]):
                issa_problem = True
                issa_problem_edges += [e for p in map(nx.utils.pairwise,nx.all_simple_paths(G, u, v)) for e in list(p) ]
                
        paths = nx.all_simple_paths(G, v, u)
        simple_node_paths = list(nx.all_simple_paths(G, v, u))
#         if G.in_degree(u) > 1 and len(simple_node_paths) > 1 and len(set([p[-2] for p in simple_node_paths])) > 1:
                
#             resolved_v_structure_nodes += [u]
        cells_on_path = []
        for p in map(nx.utils.pairwise, paths):
            
            cells_on_path += [set([edge_to_cell[x] for x in p])]
        
        if len(cells_on_path) > 1:
            if not all([cells_on_path[0] == c for c in cells_on_path]):
                issa_problem = True
                issa_problem_edges += [e for p in map(nx.utils.pairwise,nx.all_simple_paths(G, v, u)) for e in list(p) ]
                
        
    return issa_problem_edges, resolved_v_structure_nodes
                
            
        
            
            
                
            
def compute_difference_signals_from_virtual(signals, virtual_signals, event_labels, virtual_event_labels):
    """
    Compute all potential difference signals between event cliques and store the subtraction ordering
    signals: np.array
    event_labels: list<int>

    output: np.array, np.array
    """
    num_trials, num_electrodes, num_samples = signals.shape
    
    event_set = list(set(event_labels))
    num_event_signals = len(event_set)
    virtual_event_set = list(set(virtual_event_labels))
    num_virtual_event_signals = len(virtual_event_set)
    
    pairwise_signals = []
    signal_ordering = []
    

    for pair_of_event_cliques in list(product(np.arange(num_event_signals),  np.arange(num_virtual_event_signals))):
        # Compute set of pairwise difference signals between event cliques
        X = signals[event_labels == event_set[pair_of_event_cliques[0]]]
        Y = virtual_signals[virtual_event_labels == virtual_event_set[pair_of_event_cliques[1]]]
        Z = np.einsum('ijk->jik',all_pairwise(X,Y)).reshape((num_electrodes, -1))

        # Align and take the average as the exemplar difference signal between the two event cliques
        list_of_tuples = [align_group_main(Z[i], sample_len = num_samples) for i in range(num_electrodes)]
        diff_signal = np.array([t__[0] for t__ in list_of_tuples])

       
        pairwise_signals += [diff_signal]
        pairwise_signals += [-diff_signal]
        signal_ordering += [(event_set[pair_of_event_cliques[0]], virtual_event_set[pair_of_event_cliques[1]])]
        signal_ordering += [(virtual_event_set[pair_of_event_cliques[1]], event_set[pair_of_event_cliques[0]])]
        
        
    pairwise_signals = np.array(pairwise_signals)
    signal_ordering = np.array(signal_ordering)
    
    return pairwise_signals, signal_ordering
            
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    
def resort_clusters_by_noise(electrode_list, signals, event_labels_tmp,  mask, significant_electrodes,  noise):

    event_labels = copy.copy(event_labels_tmp)
   
    # Compute noise distance threshold
    _, _, num_samples = signals.shape
    baseline_variance =[noise[e]**2 for e in electrode_list]
    noise_threshold = sum([baseline_variance[e]*num_samples for e in significant_electrodes])
    
    mean_cluster_signal_array = np.zeros((len(set(event_labels)), num_samples*len(electrode_list)))
    kicked_out_signals = []
    update_ball_radius = np.zeros(len(set(event_labels)))
    num_in_cluster_array = np.zeros(len(set(event_labels)))
    update_signals_in_cluster = []
    for li, l in enumerate(set(event_labels)):
        signals_in_cluster = [i for i, x in enumerate(event_labels==l) if x]
        num_in_cluster = sum(event_labels==l)
        cluster_signals = signals[event_labels==l].reshape((num_in_cluster, -1))
        if num_in_cluster == 1:
            update_ball_radius[li] = 2*noise_threshold
            num_in_cluster_array[li] = 1
            mean_cluster_signal_array[li] = cluster_signals
        
        mean_cluster_signal = np.mean(cluster_signals, axis = 0, keepdims=True)
        
        
        signal_distance_from_center = np.sum((cluster_signals - mean_cluster_signal)**2, axis = 1)
        out_of_noise_ball_bool = signal_distance_from_center > (1+1/num_in_cluster) * noise_threshold
        out_of_noise_ball_idx = [i for i, x in enumerate(out_of_noise_ball_bool) if x]
        trials_out_of_ball = [signals_in_cluster[i] for i in out_of_noise_ball_idx]
        in_noise_ball_idx = [i for i, x in enumerate(out_of_noise_ball_bool) if not x]
        
        if len(in_noise_ball_idx) > 0:
            update_ball_radius[li] = (1+1/len(in_noise_ball_idx))*noise_threshold

            num_in_cluster_array[li] = len(in_noise_ball_idx)
            mean_cluster_signal_array[li] = np.mean([cluster_signals[i] for i in in_noise_ball_idx], axis = 0)
            update_signals_in_cluster += [[signals_in_cluster[i] for i in in_noise_ball_idx]]
        else:
            update_ball_radius[li] = float("nan")
            num_in_cluster_array[li] = float("nan")
            mean_cluster_signal_array[li] = float("nan")
        
        kicked_out_signals += [(t, cluster_signals[i:i+1]) for t, i in zip(trials_out_of_ball, out_of_noise_ball_idx)]
    
    update_ball_radius = update_ball_radius[~np.isnan(update_ball_radius)].flatten()
    num_in_cluster = num_in_cluster[~np.isnan(num_in_cluster).any()]
    mean_cluster_signal_array =  mean_cluster_signal_array[~np.isnan(mean_cluster_signal_array).any(axis=1)]
    
    for t, k in kicked_out_signals:
        
        k_distance = np.sum( (mean_cluster_signal_array - k)**2 , axis = 1)
        k_in_cluster = (k_distance < update_ball_radius).flatten()
        
        k_in_cluster_idx = [i for i, x in enumerate(k_in_cluster) if x]
        if len(k_in_cluster_idx) > 0:
            
            new_cluster = k_in_cluster_idx[np.argmin(k_distance[k_in_cluster])]
            
            mean_cluster_signal_array[new_cluster] = (mean_cluster_signal_array[new_cluster]*num_in_cluster_array[new_cluster] + k)/(num_in_cluster_array[new_cluster]+1)
            num_in_cluster_array[new_cluster] = num_in_cluster_array[new_cluster]+1
            
            
            update_ball_radius[new_cluster] = (1+1/num_in_cluster_array[new_cluster])*noise_threshold
            update_signals_in_cluster[new_cluster] += [t]
        else:
            
            mean_cluster_signal_array = np.append(mean_cluster_signal_array, k, axis=0)
            num_in_cluster_array = np.append(num_in_cluster_array, 1)
            update_ball_radius = np.append(update_ball_radius, 2*noise_threshold)
            update_signals_in_cluster += [[t]]
    
    
    new_event_labels = np.zeros(len(event_labels), dtype=int)
    looked_up_cluster = []
    index = 0
    for u in update_signals_in_cluster:
        if u not in looked_up_cluster:
            for l in u:
                new_event_labels[l] = np.int64(index)
            index += 1
            looked_up_cluster += [u]
            
                
        
    

    return new_event_labels

def get_probabilities(G, event_labels, num_trials = None):
    G_mod = G.copy()
    for n in [n for n in G_mod.nodes if G_mod.in_degree(n) > 1]:
        edge_list = list(G_mod.in_edges(n))[1:]
        G_mod.remove_edges_from(edge_list)

    cells_on_graph = nx.get_edge_attributes(G_mod, 'cell').values()
    edge_to_cells = nx.get_edge_attributes(G_mod, 'cell')
    cells_to_edges = {v: [ j for j, l in edge_to_cells.items() if l == v] for k, v in edge_to_cells.items()}
    probabilities = Counter()
    cell_in_clusters = {}
    event_counter = Counter(event_labels)
    for c in set(list(cells_on_graph)):
        cell_in_clusters[c] = []
        for u, v in cells_to_edges[c]:
            probabilities[c] += event_counter[v]
            cell_in_clusters[c] += [v]
            for n in nx.descendants(G_mod, v):
                probabilities[c] += event_counter[n]
                cell_in_clusters[c] += [n]
        if num_trials:
            probabilities[c] /= num_trials
        else:
            probabilities[c] /= len(event_labels)
    return probabilities, cell_in_clusters



