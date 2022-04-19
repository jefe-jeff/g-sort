import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import sys

from src.run_gsort_v2_wuericmod import *
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
parser.add_argument('-v', '--verbose', help="increase output verbosity", action="store_true")
parser.add_argument('-i', '--interest', type=float, help="Noise threshold for cells of interest")
parser.add_argument('-x', '--excluded_types', type=str, nargs='+', help="Cells to exclude in templates.")
parser.add_argument('-c', '--cell_types', type=str, nargs='+', help="Cell types on which to perform gsort.")
parser.add_argument('-u', '--clustering', type=str, help="Clustering scheme.")
parser.add_argument('-p', '--power_threshold', type=str, help="Competing template power threshold.")
parser.add_argument('-sl', '--start_time_limit', type=int, help="Signal window start sample.")
parser.add_argument('-el', '--end_time_limit', type=int, help="Signal window end sample.")
parser.add_argument('-cmp', '--compartments', type=str, nargs='+',  help="Cell compartments.")
parser.add_argument('-a', '--all', help="Run gsort on all cell types.", action="store_true")
parser.add_argument('-so', '--space_only', help="Disregard temporal information.", action="store_true")
parser.add_argument('-sm', '--small', help="Save subset of data", action="store_true")
parser.add_argument('-ov', '--overwrite', help="Overwrite pickle files", action="store_true")
parser.add_argument('-sasi', '--sasi', help="Exclude crap in all considerations", action="store_true")

args = parser.parse_args()

dataset = args.dataset
vstim_datarun = args.wnoise
estim_datarun = args.estim
filepath = args.output
verbose = args.verbose
noise_thresh = args.interest
all_types = args.all
clustering = args.clustering
space_only = args.space_only
overwrite = args.overwrite
sasi = args.sasi
small = args.small

DEFAULT_THREADS = 12
if args.threads is not None:
    threads = args.threads
else:
    threads = DEFAULT_THREADS
    
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

ANALYSIS_BASE = '/Volumes/Analysis'
vstim_analysis_path = os.path.join(ANALYSIS_BASE, dataset, vstim_datarun)

print(vstim_analysis_path)
estim_analysis_path = os.path.join(ANALYSIS_BASE, dataset, estim_datarun)
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

MIN_CORR = .975
duplicates = set()
cellids = vstim_data.get_cell_ids()
for cell in cellids:
    cell_ei = vstim_data.get_ei_for_cell(cell).ei
    cell_ei_error = vstim_data.get_ei_for_cell(cell).ei_error
    cell_ei_max = np.abs(np.amin(cell_ei,axis=1))
    cell_ei_power = np.sum(cell_ei**2,axis=1)
    celltype = vstim_data.get_cell_type_for_cell(cell).lower()
    if "dup" in celltype or "bad" in celltype:
        continue 
    if "parasol" in celltype:
        celltype = 'parasol'
    elif "midget" in celltype:
        celltype = 'midget'
    elif "sbc" in celltype:
        celltype = 'sbc'
    else:
        celltype = 'other'
    for other_cell in cellids:
        other_celltype = vstim_data.get_cell_type_for_cell(other_cell).lower()
        if cell == other_cell or cell in duplicates or other_cell in duplicates:
            continue
        if "dup" in other_celltype or "bad" in other_celltype:
            continue
        if "parasol" in other_celltype:
            other_celltype = 'parasol'
        elif "midget" in other_celltype:
            other_celltype = 'midget'
        elif "sbc" in other_celltype:
            other_celltype = 'sbc'
        else:
            other_celltype = 'other'
        # Quit out if both cell types are in the big five.
        if celltype in ['parasol','midget','sbc'] and other_celltype in ['parasol','midget','sbc']:
            continue
        other_cell_ei = vstim_data.get_ei_for_cell(other_cell).ei
        other_cell_ei_max = np.abs(np.amin(other_cell_ei,axis=1))
        other_cell_ei_power = np.sum(other_cell_ei**2,axis=1)
        # Compute the correlation and figure out if we have duplicates: take the larger number of spikes.
        corr = np.corrcoef(cell_ei_power,other_cell_ei_power)[0,1]
        if corr >= MIN_CORR:
            n_spikes_cell = vstim_data.get_spike_times_for_cell(cell).shape[0]
            n_spikes_other_cell = vstim_data.get_spike_times_for_cell(other_cell).shape[0]
            # Take the larger number of spikes, unless the one with fewer is a light responsive type.
            if celltype in ['parasol','midget','sbc'] or n_spikes_cell > n_spikes_other_cell:
                duplicates.add(other_cell)
            else:
                duplicates.add(cell)
                
noise = vstim_data.channel_noise

for cell in set(cellids).difference(duplicates):
    cell_ei_error = vstim_data.get_ei_for_cell(cell).ei_error[noise != 0]
    
    if np.any(cell_ei_error == 0):
        duplicates.add(cell)       

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


NUM_ELECTRODES = len(cell_ei)
NUM_AMPS = 40
    
cellids = np.array(sorted(vstim_data.get_cell_ids()))
gsorted_cells = []
data_tensor = np.zeros((len(cellids), NUM_ELECTRODES, NUM_AMPS))
filtered_data_tensor = np.zeros((len(cellids), NUM_ELECTRODES, NUM_AMPS))
run_data_tensor = np.zeros((len(cellids), NUM_ELECTRODES))



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
    
    patterns = []
    movies = []
    for filename in os.listdir(pattern_path):
            if filename.startswith('p') and filename.endswith('.mat'): 
                pattern_movie = re.findall('\d+', filename)
                patterns.append(int(pattern_movie[0]))
                

    patterns, counts = np.unique(np.array(patterns), return_counts=True)
    
    NUM_ELECTRODES = len(patterns)
    NUM_AMPS = np.max(counts)
    gsorted_cells = []
    data_tensor = np.zeros((len(cellids), NUM_ELECTRODES, NUM_AMPS))
    filtered_data_tensor = np.zeros((len(cellids), NUM_ELECTRODES, NUM_AMPS))
    run_data_tensor = np.zeros((len(cellids), NUM_ELECTRODES))
    
    pool = mp.Pool(processes = threads)
   
    outpath = os.path.join(filepath, dataset, estim_datarun, vstim_datarun)

    for type_ in cell_types:
        print("Running for cell type %s" %type_)
        for cell in tqdm.tqdm(vstim_data.get_all_cells_similar_to_type(type_)):
            cell_ind = cellids.index(cell)
            
            good_inds, EI = get_collapsed_ei_thr(cell, noise_thresh)
            if len(good_inds)==0:
                continue
            print(good_inds)

            logging.info('\ncell considered: ' + str(cell))
            
            ei = vstim_data.get_ei_for_cell(cell).ei
            num_electrodes = ei.shape[0]
            if num_electrodes == 519:
                array_id = 1502
            else:
                array_id = 502
                
            cell_power = ei**2
            
            e_sorted = np.argsort(np.sum(ei**2, axis = 1))[::-1]
            
            e_sorted = [e for e in e_sorted if eil.axonorsomaRatio(ei[e,:]) in compartments]
            
            
            cell_power = ei**2
            power_ordering = np.argsort(cell_power, axis = 1)[:,::-1]
            significant_electrodes = np.argwhere(np.sum(np.take_along_axis(cell_power[e_sorted], power_ordering[e_sorted,:cell_spike_window], axis = 1), axis = 1) >= rat * cell_spike_window * np.array(noise[e_sorted])**2).flatten()
            
            electrode_list = list(np.array(e_sorted)[significant_electrodes][:max_electrodes_considered])

            if len(electrode_list) == 0:
                logging.info('no significant electrodes')
                print('No significant electrodes.')
                continue
            
            data_on_cells = get_center_eis(cell, electrode_list, ap = (vstim_analysis_path[:-7], vstim_datarun.rsplit('/')[-1]), excluded_types = excluded_types, excluded_cells = list(duplicates), power_threshold=pt, array_id = array_id, sample_len_left = time_limit ,sample_len_right = time_limit)
            good_patterns = (good_inds + 1).tolist()
            
            logging.info('electrodes considered: ' + str(np.array(electrode_list) + 1))
            logging.info('patterns considered: ' + str(good_patterns))
            
            results = pool.starmap(run_movie, product([cell], good_patterns,[i for i in range(len(good_patterns))], [max(counts)], [(electrode_list,data_on_cells,start_time_limit,end_time_limit,estim_analysis_path, noise)]))
            
            ps = np.array([r[0] for r in results for i in range(len(r[1])) if len(r[1])>0]).astype(int)
            ks = np.array([i for r in results for i in r[1] if len(r[1])>0]).astype(int)
            cprobs = [i for r in results for i in r[2] if len(r[1])>0]
            probs = [i for r in results for i in r[3] if len(r[1])>0]
            
            data_tensor[cell_ind, ps, ks] = probs
            filtered_data_tensor[cell_ind, ps, ks] = cprobs
            run_data_tensor[cell_ind,ps] = 1
            
            if cell_ind not in gsorted_cells:
                gsorted_cells.append(cell_ind)

            savemat(outpath + '/' + 'gsort_full_data_tensor.mat', {'cells': cellids, 'gsorted_cells': np.sort(np.array(gsorted_cells)),'probs': data_tensor, 'filtered_probs':filtered_data_tensor, 'run':run_data_tensor})
                
    pool.close()
    logging.info('finished')