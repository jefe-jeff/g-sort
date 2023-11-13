import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys

from src.g_sort_core_scripts import *
import argparse
from scipy.io import loadmat
from itertools import product
import tqdm
import logging
import re

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, help='Dataset in format YYYY-MM-DD-P.')
parser.add_argument('-w', '--wnoise', type=str, help='White noise run in format dataXXX (or streamed/dataXXX or kilosort_dataXXX/dataXXX, etc.).')
parser.add_argument('-e', '--estim', type=str, nargs = '+', help='Estim run in format dataXXX.')
parser.add_argument('-o', '--output', type=str, help='/path/to/output/directory.')
parser.add_argument('-t', '--threads', type=int, help='Number of threads to use in computation.')
parser.add_argument('-i', '--interest', type=float, help="Noise threshold for cells of interest")
parser.add_argument('-x', '--excluded_types', type=str, nargs='+', help="Cells to exclude in templates.")
parser.add_argument('-c', '--cell_types', type=str, nargs='+', help="Cell types on which to perform gsort.")
parser.add_argument('-p', '--power_threshold', type=str, help="Competing template power threshold.")
parser.add_argument('-sl', '--start_time_limit', type=int, help="Signal window start sample.")
parser.add_argument('-el', '--end_time_limit', type=int, help="Signal window end sample.")
parser.add_argument('-cd', '--cluster_delay', type=int, help="Delay after which clustering begins.")
parser.add_argument('-wb', '--window_buffer', type=int, help="Buffer samples on edge of template.")

parser.add_argument('-cmp', '--compartments', type=str, nargs='+',  help="Cell compartments.")
parser.add_argument('-m', '--mode', type=str,  help="Old or new labview.")

parser.add_argument('-eb', '--estim_base', type=str,  help="Estim base.")
parser.add_argument('-vb', '--vstim_base', type=str,  help="Vstim base.")

parser.add_argument('-a', '--all', help="Run gsort on all cell types.", action="store_true")
parser.add_argument('-sasi', '--sasi', help="Exclude crap in all considerations", action="store_true")
parser.add_argument('-sc', '--specific_cell', type=int, help='Cell id number.')
parser.add_argument('-mt', '--mutual_threshold', type=float, help="Overlap threshold for cells to be considered together")
args = parser.parse_args()

dataset = args.dataset
vstim_datarun = args.wnoise
estim_datarun = args.estim[0]
filepath = args.output
noise_thresh = args.interest
all_types = args.all
sasi = args.sasi
mode = args.mode

DEFAULT_THREADS = 24
if args.threads is not None:
    threads = args.threads
else:
    threads = DEFAULT_THREADS
    
if (args.end_time_limit is not None) :
    end_time_limit = args.end_time_limit
else:
    end_time_limit = 55
    

if (args.start_time_limit is not None):
    start_time_limit = args.start_time_limit
else:
    start_time_limit = 0

time_limit = end_time_limit - start_time_limit
    

if args.cluster_delay is not None:
    cluster_delay = args.cluster_delay
else:
    cluster_delay = 0
    
if args.window_buffer is not None:
    window_buffer = args.window_buffer
else:
    window_buffer = 0

if args.power_threshold is not None:
    pt = float(args.power_threshold)
else:
    pt = 1.5

if args.excluded_types is not None:
    excluded_types = args.excluded_types
else:
    excluded_types = ['bad', 'dup']
    
if args.cell_types is not None:
    cell_types = args.cell_types
else:
    cell_types = ['parasol', 'midget']
    
if args.compartments is not None:
    compartments = args.compartments
else:
    compartments = ['soma', 'mixed']

if args.mutual_threshold is not None:
    mutual_threshold = args.mutual_threshold
else:
    mutual_threshold = 1

if args.specific_cell is not None:
    specific_cell = args.specific_cell 
else:
    specific_cell = None

VISUAL_ANALYSIS_BASE = args.vstim_base if args.vstim_base!= None else '/Volumes/Analysis'
vstim_analysis_path = os.path.join(VISUAL_ANALYSIS_BASE, dataset, vstim_datarun)

print(vstim_analysis_path)
ESTIM_ANALYSIS_BASE = args.estim_base if args.estim_base!= None else '/Volumes/Analysis'
estim_analysis_path = []
for estim in args.estim:
    estim_analysis_path.append(os.path.join(ESTIM_ANALYSIS_BASE, dataset, estim))
print(estim_analysis_path)

pattern_path = os.path.join(estim_analysis_path[0], 'pattern_files')

vstim_data = vl.load_vision_data(vstim_analysis_path,
                                 vstim_datarun.rsplit('/')[-1],
                                 include_params=True,
                                 include_ei=True,
                                 include_noise=True,
                                 include_neurons=True)

vstim_data.update_cell_type_classifications_from_text_file(os.path.join(vstim_analysis_path, 'classification_deduped.txt'))

if all_types:
    all_cell_types = vstim_data.get_all_present_cell_types()
    
    cell_types = []
    for type_ in all_cell_types:
        if 'dup' in type_ or 'bad' in type_ or ('crap' in type_ and sasi):
            continue
        else:
            cell_types.append(type_)

noise = vstim_data.channel_noise

if not sasi:
    allowed_types = cell_types + ['crap']
else:
    allowed_types = cell_types
duplicates, cell_ei = compute_duplicates_new(vstim_data, allowed_types)  
print(duplicates)
NUM_CHANNELS = len(cell_ei)

amplitudes = np.array([0.10053543, 0.11310236, 0.11938583, 0.13195276, 0.14451969,
                       0.16337008, 0.17593701, 0.1947874 , 0.2136378 , 0.23877165,
                       0.25762205, 0.2780315 , 0.30330709, 0.35385827, 0.37913386,
                       0.42968504, 0.45496063, 0.50551181, 0.55606299, 0.60661417,
                       0.68244094, 0.73299213, 0.8088189 , 0.88464567, 0.98574803,
                       1.10433071, 1.20472441, 1.30511811, 1.40551181, 1.60629921,
                       1.70669291, 1.90748031, 2.10826772, 2.30905512, 2.50984252,
                       2.81102362, 3.11220472, 3.41338583, 3.71456693])

cell_spike_window = 25
rat = 2
max_electrodes_considered = 30
    
cellids = np.array(sorted(vstim_data.get_cell_ids()))



def get_collapsed_ei_thr(cell_no, thr_factor):
    # Read the EI for a given cell
    cell_ei = vstim_data.get_ei_for_cell(cell_no).ei
    
    # Collapse into maximum value
    collapsed_ei = np.amin(cell_ei, axis=1)
    
    # Threshold the EI to pick out only electrodes with large enough values
    good_inds = np.argwhere(np.abs(collapsed_ei) > thr_factor * noise).flatten()
    
    return good_inds, np.abs(collapsed_ei)

cellids = sorted(vstim_data.get_cell_ids())

if __name__ == "__main__":
    
    if not os.path.exists(os.path.join(filepath, dataset, estim_datarun, vstim_datarun)):
            os.makedirs(os.path.join(filepath, dataset, estim_datarun, vstim_datarun))
    
    logging.basicConfig(filename=os.path.join(filepath, dataset, estim_datarun, vstim_datarun, 'run.log'), level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    
    logging.info('Set up')
    logging.info('dataset: ' + dataset)
    logging.info('white noise datarun: ' + vstim_datarun)
    logging.info('electrical stim datarun: ' + estim_datarun)
    logging.info('electrode selection noise threshold: ' + str(noise_thresh))
    logging.info('threads: ' + str(threads))
    logging.info('cell spike window: ' + str(cell_spike_window))
    logging.info('time limit: ' + str(time_limit))
    logging.info('max electrode considered: ' + str(max_electrodes_considered))
    logging.info('electrode inclusion ratio' + str(rat))
    logging.info('included cell types: ' + str(cell_types))
    logging.info('excluded cell types: ' + str(excluded_types))
    logging.info('compartment: ' + str(compartments))
    logging.info('start time limit: ' + str(start_time_limit))
    logging.info('end time limit: ' + str(end_time_limit))
    logging.info('cluster delay: ' + str(cluster_delay))
    logging.info('window_buffer ' + str(window_buffer))
    
    
    
    patterns = []
    if mode == "oldlv":

        for filename in os.listdir(pattern_path):
                if filename.startswith('p') and filename.endswith('.mat'): 
                    pattern_movie = re.findall('\d+', filename)
                    patterns.append(int(pattern_movie[0]))
                    

        patterns, num_amps = np.unique(np.array(patterns), return_counts=True)
        
        NUM_ELECTRODES = np.max(patterns)
        NUM_AMPS = np.max(num_amps)
    elif mode == "newlv":
        stim_elecs = []
        num_amps = []
        for filename in os.listdir(pattern_path):
                if filename.startswith('p') and filename.endswith('.mat'): 
                    pattern = int(re.findall('\d+', filename)[0])
                    patterns.append(pattern)

                    pattern_file = loadmat(os.path.join(pattern_path, 'p' + str(pattern) + '.mat'), squeeze_me=True, struct_as_record=False)
                    # Add case to handle single amplitude (AJP 03/28/23)
                    if np.isscalar(pattern_file['patternStruct'].amplitudes):
                        num_amps.append(1)
                    else:
                        num_amps.append(len(pattern_file['patternStruct'].amplitudes))
                    stim_elecs.append(pattern_file['patternStruct'].stimElecs)

        patterns = np.array(patterns)
        stim_elecs = np.array(stim_elecs, dtype=object)
        num_amps = np.array(num_amps)
        NUM_ELECTRODES = np.max(patterns)
        NUM_AMPS = np.max(num_amps)
    else:
        assert  1==0, "Specify new or oldlv data"
            

    gsorted_cells = []
    
    data_tensor = np.zeros((len(cellids), NUM_ELECTRODES, NUM_AMPS))
    filtered_data_tensor = np.zeros((len(cellids), NUM_ELECTRODES, NUM_AMPS))
    run_data_tensor = np.zeros((len(cellids), NUM_ELECTRODES))
    relevant_cells = {k:[] for k in patterns}

   
    outpath = os.path.join(filepath, dataset, estim_datarun, vstim_datarun)
    all_cell_types = [ct for ct in vstim_data.get_all_present_cell_types() if 'bad' not in ct and 'dup' not in ct]
    total_electrode_list, total_cell_to_electrode_list, mutual_cells, array_id = get_cell_info(all_cell_types, vstim_data, compartments, noise, mutual_threshold=mutual_threshold)
    n_to_data_on_cells = {}
    # running_cells = []
    running_cells_ind = []
    for type_ in cell_types:
        print("Loading data for cell type %s" %type_)
        
        for cell in tqdm.tqdm(vstim_data.get_all_cells_similar_to_type(type_)):
            # if cell != 124:
            #     continue
            if specific_cell is not None:
                if cell != specific_cell:
                    continue

            cell_ind = cellids.index(cell)
            
            if mode == "oldlv":
                good_inds, EI = get_collapsed_ei_thr(cell, noise_thresh)
            elif mode == "newlv":
                relevant_patterns, EI = get_collapsed_ei_thr(cell, noise_thresh)
                good_inds = []
                for i in range(len(stim_elecs)):
                    if np.any(np.in1d(stim_elecs[i], relevant_patterns + 1)):
                        good_inds.append(patterns[i])
                good_inds = np.array(good_inds)-1
            else:
                assert 1==0, "Specify new or oldlv data"
                   
            if len(good_inds)==0:
                continue
    
            
            # print("good_inds",good_inds)
            logging.info('\ncell considered: ' + str(cell))
            
            electrode_list =  list(set([e for c in mutual_cells[cell] for e in total_cell_to_electrode_list[c]]))
            cell_to_electrode_list = {k:v for k,v in total_cell_to_electrode_list.items() if k in mutual_cells[cell]}
            if len(electrode_list) == 0:
                logging.info('no significant electrodes')
                # print('No significant electrodes.')
                continue
            running_cells_ind += [cell_ind]
            
            data_on_cells = get_center_eis(cell, electrode_list, ap = (vstim_analysis_path[:-7], vstim_datarun.rsplit('/')[-1]), excluded_types = excluded_types, excluded_cells = list(duplicates), power_threshold=pt, array_id = array_id, sample_len_left = time_limit +window_buffer,sample_len_right = time_limit+window_buffer)
            n_to_data_on_cells[cell]=data_on_cells
            for gi_ in good_inds+1:
                relevant_cells[gi_]+=[cell]

    running_cells_ind = np.array(running_cells_ind)
    # results = pool.starmap(run_pattern_movie, product(patterns, [i for i in range(NUM_AMPS)], [(cellids, relevant_cells, mutual_cells,total_cell_to_electrode_list,start_time_limit,end_time_limit,estim_analysis_path, noise,outpath,n_to_data_on_cells)]))
    # savemat(outpath + '/' + 'gsort_full_data_tensor.mat', {'cells': cellids, 'gsorted_cells': np.sort(np.array(gsorted_cells)),'probs': data_tensor, 'filtered_probs':filtered_data_tensor, 'run':run_data_tensor})
    
    patterns_run = 0
  
    file_exists = os.path.exists( os.path.join(outpath, 'init_probs.dat'))
    print("file_exists", file_exists)
    print("NUM_ELECTRODES", NUM_ELECTRODES)
    print("NUM_CHANNELS",NUM_CHANNELS )
    print("end_time_limit-start_time_limit", end_time_limit-start_time_limit)
    if file_exists:
        init_probs_fp = np.memmap(os.path.join(outpath, 'init_probs.dat'), dtype='float32', mode='r+', shape=(len(cellids), NUM_ELECTRODES,NUM_AMPS))
        run_fp = np.memmap(os.path.join(outpath, 'run.dat'), dtype='int16', mode='r+', shape=(NUM_ELECTRODES,NUM_AMPS))
        trials_fp = np.memmap(os.path.join(outpath, 'trial.dat'), dtype='int16', mode='r+', shape=(NUM_ELECTRODES,NUM_AMPS))
        artifact_fp = np.memmap(os.path.join(outpath, 'artifact.dat'), dtype='float32', mode='r+', shape=(NUM_ELECTRODES,NUM_AMPS, NUM_CHANNELS, end_time_limit-start_time_limit))
    
    else: 
        init_probs_fp = np.memmap(os.path.join(outpath, 'init_probs.dat'), dtype='float32', mode='w+', shape=(len(cellids), NUM_ELECTRODES,NUM_AMPS))
        init_probs_fp[:] = np.nan
        run_fp = np.memmap(os.path.join(outpath, 'run.dat'), dtype='int16', mode='w+', shape=(NUM_ELECTRODES,NUM_AMPS))
        trials_fp = np.memmap(os.path.join(outpath, 'trial.dat'), dtype='int16', mode='w+', shape=(NUM_ELECTRODES,NUM_AMPS))
        artifact_fp = np.memmap(os.path.join(outpath, 'artifact.dat'), dtype='float32', mode='w+', shape=(NUM_ELECTRODES,NUM_AMPS, NUM_CHANNELS, end_time_limit-start_time_limit))
        artifact_fp[:] = np.nan

    def listener(max_index, q):
        '''listens for messages on the q, writes to file. '''

        count = 0
        with tqdm.tqdm(total=max_index) as pbar:
            while True:
                p, k, init_probs, artifact, trials, m = q.get()
                if p != -1:
                    init_probs_fp[:,p-1, k] = init_probs
                    artifact_fp[p-1, k,:,:] = artifact
            
                    run_fp[p-1, k] = 1
                    trials_fp[p-1, k] = trials
                    
                    init_probs_fp.flush()
                    artifact_fp.flush()
                    run_fp.flush()
                    trials_fp.flush()
                pbar.update(1)
                count = count + 1
                if count == max_index:
                    break
            return 

    
    manager = mp.Manager()
    q = manager.Queue()    

    pool = mp.Pool(processes = threads)
    
    total_jobs = 0
    for p in patterns:
        for k in range(NUM_AMPS):
            if (run_fp[p-1, k] == 0):
                total_jobs += 1
    print("total jobs", total_jobs)
                
    watcher = pool.apply_async(listener, (total_jobs, q))

    savemat(os.path.join(outpath,'parameters.mat'), {'cells': cellids,'patterns':patterns, 'movies':NUM_AMPS, 'gsorted_cells': np.sort(running_cells_ind)})
            
    preloaded_data = (cellids, running_cells_ind, relevant_cells, mutual_cells,total_cell_to_electrode_list,start_time_limit,end_time_limit,estim_analysis_path, noise,outpath,n_to_data_on_cells,NUM_CHANNELS, cluster_delay)
    # import pickle
    # f = open('preloaded_data_exhaustive.pkl', 'wb')
    # pickle.dump(preloaded_data, f)
    # f.close()
    arguments = [(p, k, preloaded_data, q) for p in patterns for k in range(NUM_AMPS) if run_fp[p-1, k] == 0]
    result = pool.starmap_async(run_pattern_movie, arguments)

    
    print("Finished")

    pool.close()
    pool.join()
    logging.info('finished')
