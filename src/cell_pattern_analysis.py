def run_movie(n, p, ks, preloaded_data , store_signals = False):
    
    start_time_limit, end_time_limit, estim_analysis_path


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

    electrode_list, data_on_cells = preloaded_data
    cell_ids, cell_eis, cell_error, cell_spk_times = data_on_cells
    significant_electrodes = np.arange(len(electrode_list))
    num_trials = len(signal)

    for k in range(ks):
    
        try:
            signal = get_oldlabview_pp_data(estim_analysis_path , p, k)[:,:,:]
        except:
            break


        
        
        
       
        raw_signal = signal[:, electrode_list, shift_window[0]:shift_window[1]].astype(float) 

        finished, G, (initial_event_labels, initial_signals), (event_labels_with_virtual, signals_with_virtual), (final_event_labels, final_signals), edge_to_matched_signals, tracker, mask, note = spike_sorter_EA(n, significant_electrodes, electrode_list, raw_signal, 1, 1000, noise, data_on_cells, artifact_cluster_estimate=None)



        total_p, cell_in_clusters = get_probabilities(G, event_labels_with_virtual)


        cosine_similarity = 0.7
        graph_signal = edge_to_matched_signals
        edge_to_cell = {k:v for k, v in graph_signal.keys()}
        signal_index = np.argwhere([k[1] ==n for k in  graph_signal.keys()]).flatten()
        g_signal_error = 0
        clustering = event_labels_with_virtual
        if len(signal_index) != 0:

            edges = np.array([list(graph_signal.keys())[k][0] for k in signal_index])  
            low_power = [np.sum(graph_signal[list(graph_signal.keys())[k]][0]*graph_signal[list(graph_signal.keys())[k]][1])/np.sqrt(np.sum(graph_signal[list(graph_signal.keys())[k]][1]**2))/np.sqrt(np.sum(graph_signal[list(graph_signal.keys())[k]][0]**2))<=cosine_similarity for k in signal_index]
            g_signal_error += sum([sum(clustering == e[1])/len(clustering) for e in edges[low_power]])

            g_signal_error += sum([sum(clustering == n)/len(clustering) for e in edges[low_power] for n in nx.descendants(G, e[1])])
            cosine_probs += [total_p[n]-g_signal_error]
        else:
            cosine_probs += [total_p[n]]
   
        probs += [total_p[n]]
        notes += [note]
        num_trials_list += [len(signal)]
        clustering_list += [list(event_labels_with_virtual)]
        cluster_label_list += [list(set(event_labels_with_virtual))]
        root_list += [final_event_labels[0]]
        root_signals += [np.median(raw_signal[event_labels_with_virtual==final_event_labels[0]], axis = 0, keepdims = True)]
        all_signals += [np.vstack([np.median(raw_signal[event_labels_with_virtual==l], axis = 0, keepdims = True) for l in set(event_labels_with_virtual)])]

    return (p, [i for i in range(len(probs))], cosine_probs, probs, (central_diff).astype(int), (upwind + downwind).astype(int))