import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys


from src.run_gsort_v2_wuericmod import *
import src.utilities.electrode_map as electrode_map
import argparse
from scipy.io import loadmat
from itertools import product
import tqdm
import logging
import re

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, help='Dataset in format YYYY-MM-DD-P.')
parser.add_argument('-w', '--wnoise', type=str, help='White noise run in format dataXXX (or streamed/dataXXX or kilosort_dataXXX/dataXXX, etc.).')
parser.add_argument('-e', '--estim', type=str, help='Estim run in format dataXXX.')
parser.add_argument('-o', '--output', type=str, help='/path/to/output/directory.')
parser.add_argument('-t', '--threads', type=int, help='Number of threads to use in computation.')
parser.add_argument('-i', '--interest', type=float, help="Noise threshold for cells of interest")
parser.add_argument('-x', '--excluded_types', type=str, nargs='+', help="Cells to exclude in templates.")
parser.add_argument('-c', '--cell_types', type=str, nargs='+', help="Cell types on which to perform gsort.")
parser.add_argument('-p', '--power_threshold', type=str, help="Competing template power threshold.")
parser.add_argument('-st', '--stim_type', type=str, help="Type of electrical stimulation.")
parser.add_argument('-sl', '--start_time_limit', type=int, help="Signal window start sample.")
parser.add_argument('-el', '--end_time_limit', type=int, help="Signal window end sample.")
parser.add_argument('-cmp', '--compartments', type=str, nargs='+',  help="Cell compartments.")
parser.add_argument('-m', '--mode', type=str,  help="Old/new labview.")

parser.add_argument('-eb', '--estim_base', type=str,  help="Estim base.")
parser.add_argument('-vb', '--vstim_base', type=str,  help="Vstim base.")


parser.add_argument('-a', '--all', help="Run gsort on all cell types.", action="store_true")
parser.add_argument('-ov', '--overwrite', help="Overwrite pickle files", action="store_true")
parser.add_argument('-sasi', '--sasi', help="Exclude crap in all considerations", action="store_true")
parser.add_argument('-sc', '--specific_cell', type=int, help='Cell id number.')
parser.add_argument('-mt', '--mutual_threshold', type=float, help="Overlap threshold for cells to be considered together")

parser.add_argument('-ms', '--num_movies', type=int, help="Pre-specified number of movies.")

args = parser.parse_args()

dataset = args.dataset
vstim_datarun = args.wnoise
estim_datarun = args.estim
filepath = args.output
noise_thresh = args.interest
all_types = args.all
overwrite = args.overwrite
sasi = args.sasi
mode = args.mode

DEFAULT_THREADS = 24
if args.threads is not None:
    threads = args.threads
else:
    threads = DEFAULT_THREADS
if args.stim_type is not None:
    stim_type = args.stim_type
else:
    stim_type = "single"
    
if (args.end_time_limit is not None) and (args.start_time_limit is not None):
    end_time_limit = args.end_time_limit
    start_time_limit = args.start_time_limit
    time_limit = end_time_limit - start_time_limit
else:
    end_time_limit = 55
    start_time_limit = 0
    time_limit = 55
    
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
estim_analysis_path = os.path.join(ESTIM_ANALYSIS_BASE, dataset, estim_datarun)
pattern_path = os.path.join(estim_analysis_path, 'pattern_files')

vstim_data = vl.load_vision_data(vstim_analysis_path,
                                 vstim_datarun.rsplit('/')[-1],
                                 include_params=True,
                                 include_ei=True,
                                 include_noise=True,
                                 include_neurons=True)

if all_types:
    all_cell_types = vstim_data.get_all_present_cell_types()
    
    cell_types = []
    for type_ in all_cell_types:
        if 'dup' in type_ or 'bad' in type_ or ('crap' in type_ and sasi):
            continue
        else:
            cell_types.append(type_)

noise = vstim_data.channel_noise

# duplicates, cell_ei = compute_duplicates(vstim_data, noise)  
duplicates=set()


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

            

    gsorted_cells = []
    
    
    relevant_cells = {}

   
    outpath = os.path.join(filepath, dataset, estim_datarun, vstim_datarun)

    all_cell_types = [ct for ct in vstim_data.get_all_present_cell_types() if 'bad' not in ct and 'dup' not in ct]
    total_electrode_list, total_cell_to_electrode_list, mutual_cells, array_id = get_cell_info(all_cell_types, vstim_data, compartments, noise, mutual_threshold=mutual_threshold)
    n_to_data_on_cells = {}
    # running_cells = []
    running_cells_ind = []
    print("cell_types", cell_types)
    NUM_ELECTRODES = len(electrode_map.get_litke_array_triplet_mat_by_array_id(array_id)) if stim_type == "triplet" else len(electrode_map.get_litke_array_adj_mat_by_array_id(array_id))
                
    relevant_cells = {se:[] for se in range(1, NUM_ELECTRODES+1)} 
    for type_ in cell_types:
        print("Running for cell type %s" %type_)
        for cell in vstim_data.get_all_cells_similar_to_type(type_):
            # if cell != 124:
            #     continue
            if specific_cell is not None:
                if cell != specific_cell:
                    continue

            cell_ind = cellids.index(cell)
            
          
            relevant_patterns, EI = get_collapsed_ei_thr(cell, noise_thresh)
            good_inds = []

            if args.stim_type == "triplet":
                for se_i, stim_elec in enumerate(electrode_map.get_litke_array_triplet_mat_by_array_id(array_id)):
                    # relevant_cells[se_i] = []
                    
                    if np.any(np.in1d(stim_elec, relevant_patterns )):
                        good_inds.append(se_i+1)
                good_inds = np.array(good_inds)-1
                # print("cell good ind", cell, good_inds)
            elif args.stim_type == "single":
                # relevant_cells = {k:[] for k in range(1, NUM_ELECTRODES+1)} 
                good_inds =relevant_patterns
            else:
                raise RuntimeError
                
            
            
            
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
            
            data_on_cells = get_center_eis(cell, electrode_list, ap = (vstim_analysis_path[:-7], vstim_datarun.rsplit('/')[-1]), excluded_types = excluded_types, excluded_cells = list(duplicates), power_threshold=pt, array_id = array_id, sample_len_left = time_limit ,sample_len_right = time_limit)
            n_to_data_on_cells[cell]=data_on_cells
            for gi_ in good_inds+1:
                
                relevant_cells[gi_]+=[cell]


    NUM_CHANNELS = len(EI)
    running_cells_ind = np.array(running_cells_ind)
   
    patterns_run = 0
    NUM_AMPS = args.num_movies
    NUM_ELECTRODES = len(electrode_map.get_litke_array_triplet_mat_by_array_id(array_id)) if stim_type == "triplet" else len(electrode_map.get_litke_array_adj_mat_by_array_id(array_id))
    assert NUM_AMPS != None, "Must specify max number of movies"
    num_amps = np.array([NUM_AMPS]*NUM_ELECTRODES)
    file_exists = os.path.exists( os.path.join(outpath, 'init_probs.dat'))
    print("file_exists", file_exists)
    if file_exists:
        init_probs_fp = np.memmap(os.path.join(outpath, 'init_probs.dat'), dtype='float32', mode='r+', shape=(len(cellids), NUM_ELECTRODES,NUM_AMPS))
        run_fp = np.memmap(os.path.join(outpath, 'run.dat'), dtype='float32', mode='r+', shape=(NUM_ELECTRODES,NUM_AMPS))
        trials_fp = np.memmap(os.path.join(outpath, 'trial.dat'), dtype='float32', mode='r+', shape=(NUM_ELECTRODES,NUM_AMPS))
    else: 
        init_probs_fp = np.memmap(os.path.join(outpath, 'init_probs.dat'), dtype='float32', mode='w+', shape=(len(cellids), NUM_ELECTRODES,NUM_AMPS))
        run_fp = np.memmap(os.path.join(outpath, 'run.dat'), dtype='float32', mode='w+', shape=(NUM_ELECTRODES,NUM_AMPS))
        trials_fp = np.memmap(os.path.join(outpath, 'trial.dat'), dtype='float32', mode='w+', shape=(NUM_ELECTRODES,NUM_AMPS))
    
    def listener(max_index, q):
        '''listens for messages on the q, writes to file. '''

        count = 0
        print("listening!")
        while True:
            # print("q", q.qsize())
            p, k, init_probs, artifact, final_probs, trials, m = q.get()
            # print("p", p)
            if p != -1:
                # print("p-1", p-1)
                # print("k", k)
                # print("init_probs_fp", init_probs_fp.shape)
                # print("run_fp", run_fp.shape)
                # print("trials_fp", trials_fp.shape)
                
                init_probs_fp[:,p-1, k] = init_probs
                # artifact_fp[p-1, k,:,:] = artifact
                # final_probs_fp[:,p-1, k] = init_probs

                run_fp[p-1, k] = 1
                trials_fp[p-1, k] = trials
                init_probs_fp.flush()
                # artifact_fp.flush()
                # final_probs_fp.flush()
                run_fp.flush()
                trials_fp.flush()
            count = count + 1
            print(p,k,count)
            if count == max_index:
                break
        return 

    
    manager = mp.Manager()
    q = manager.Queue()    

    pool = mp.Pool(processes = threads)

    
    savemat(os.path.join(outpath,'parameters.mat'), {'cells': cellids,'patterns':np.array([i+1 for i in range(NUM_ELECTRODES)]), 'movies':NUM_AMPS, 'gsorted_cells': np.sort(running_cells_ind)})
            
    preloaded_data = (cellids, running_cells_ind, relevant_cells, mutual_cells,total_cell_to_electrode_list,start_time_limit,end_time_limit,estim_analysis_path, noise,outpath,n_to_data_on_cells,NUM_CHANNELS)
    
    
    
    run_patterns = []
   
    while True:
        new_patterns = []
        all_patterns = []
        print("pattern_path",pattern_path)
        for filename in os.listdir(pattern_path):
                if filename.startswith('p') and filename.endswith('.mat'): 
                    # print(filename)
                    pattern = int(re.findall('\d+', filename)[0])
                    
                    try:
                        pattern_file = loadmat(os.path.join(pattern_path, filename), squeeze_me=True, struct_as_record=False)
                    except:
                        continue
                    amps = pattern_file['patternStruct'].amplitudes
                    ampsReady = np.argwhere(np.any(amps, axis = 1)).flatten() if len(amps.shape) == 2 else np.argwhere(amps).flatten()
                    all_patterns += [(pattern, movie) for movie in ampsReady ]
                    
                    new_patterns += [(pattern, movie) for movie in ampsReady if (pattern, movie) not in run_patterns and not run_fp[pattern-1, movie]]
                    # patterns.append(pattern)
        print("len(all_patterns)", len(all_patterns))
        print("len(new_patterns)", len(new_patterns))
        # print("new_patterns",new_patterns[:10])
        if len(new_patterns):
            watcher = pool.apply_async(listener, (len(new_patterns), q,))
            jobs = []
            for (p,k) in new_patterns:
                
                job = pool.apply_async(run_pattern_movie_live, (p, k, preloaded_data, q))
                jobs.append(job)
                run_patterns += [(p,k)]
            watcher.get()
            for job in jobs: 
                job.get()
    
    print("Die")

    pool.close()
    pool.join()
    logging.info('finished')