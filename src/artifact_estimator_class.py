import numpy as np
import sys
#sys.path.append('/home/visitor/jeffbrown/artificial-retina-software-pipeline/utilities/')
sys.path.append('/Volumes/Lab/Users/jeffbrown/artificial-retina-software-pipeline/utilities')
# sys.path.append('/Volumes/Lab/Users/jeffbrown/artificial-retina-software-pipeline/utilities')
import visionloader as vl
print(vl.__file__)


from old_labview_data_reader import *
import collections
from scipy.optimize import curve_fit
import copy
import electrode_map
import time
from scipy.io import loadmat,savemat
import pickle
import math
import scipy.special
import itertools

class TemplateLoader:
    """
    Load templates for electrical spike sorting. This base class includes various methods
    to store, preprocess, and subselect templates based on relevance. The templates are 
    sourced from the Python Vision API.
    
    Example usage:
    tl = TemplateLoader('/Volumes/Analysis/', '2015-11-09-3/', 'full_resolution/', 'data000')
    tl = TemplateLoader('/Volumes/Analysis/', '2015-11-09-3/', 'full_resolution/', 'data000', include_noise = False); relevant for spike sorting data from Softwares like Kilosort (which don't automatically compute the noise parameter); use the first version otherwise.
    
    After instantiating a TemplateLoader per above, you have to load the relevant EIs. 
    You can do this with 'store_cells_from_list' or 'store_cell_of_types'. For simplicity,
    this only adds to the data already loaded into the TemplateLoader object; so, for instance,
    if you call 'store_cells_from_list' twice on the same list of cell ids, you will store two 
    copies of the same template data. This would be silly of you to do and potentially dangerous
    depending on the e-stim spikesorting scheme you are using. 
    
    If you want to remove some of the templates according to a criteria, like significant signal
    on a selected electrode, you can use 'remove_templates_by_snr' or 'remove_templates_by_threshold.'
    
    If you want to remove all the templates loaded, 'reset_templates' does that for you. 

    If you want to mask your templates (remove electrodes irrelevant for e-stim spike sorting), 
    call 'mask_non_main_channels,' 'mask_by_snr,' or 'mask_by_threshold.'
    
    """
    def __init__(self, analysis_path, subfolders, datarun, array_id = 502, include_noise = True):
        spikesorting = vl.load_vision_data(analysis_path + subfolders + datarun,
                                  datarun,
                                  include_params=True,
                                  include_ei=True,
                                  include_neurons = True,
                                  include_noise=include_noise)
        self.analysis_path = analysis_path
        self.subfolders = subfolders
        self.datarun = datarun
        self.full_analysis_path = analysis_path + subfolders + datarun
        self.include_noise = include_noise

        self.spikesorting = spikesorting
        
        self.cellids = []
        self.celltypes = []
        self.templates = []
        self.templates_variance = []
        
        self.noise = self.spikesorting.channel_noise
        
        self.groups = {}
        array_map = electrode_map.get_litke_array_adj_mat_by_array_id(array_id)
        for center, surround in enumerate(array_map):
            self.groups[center] = [center] + surround
        
        
    def store_cells_from_list(self, celllist):
        '''
        Store templates from a list of cell IDs
        '''
        self.cellids += celllist
        error = np.ones_like(self.spikesorting.get_ei_for_cell(self.cellids[0]).ei)*self.noise[:,None]
        
        for cellid in celllist:
            self.templates += [self.spikesorting.get_ei_for_cell(cellid).ei]
            self.templates_variance += [error]
            
            self.celltypes += [self.spikesorting.get_cell_type_for_cell(cellid)]
            
        
    def store_cells_of_types(self, celltypes):
        '''
        Store templates from all cells of types in list of cell types.
        '''
        for celltype in celltypes:
            
            cellids = self.spikesorting.get_all_cells_similar_to_type(celltype)
            self.cellids += cellids
            self.celltypes += [celltype]*len(cellids)
            for cellid in cellids:
                self.templates += [self.spikesorting.get_ei_for_cell(cellid).ei]
                self.templates_variance += [self.spikesorting.get_ei_for_cell(cellid).ei_error]
                
    def store_good_cells(self):
        '''
        Store templates from all cells of types in list of cell types.
        '''
        
        cellids = list(set(self.spikesorting.get_cell_ids()) - set(self.spikesorting.get_all_cells_similar_to_type('bad')) - set(self.spikesorting.get_all_cells_similar_to_type('duplicate')))
        self.cellids += cellids
        for cellid in cellids:
            self.celltypes += [self.spikesorting.get_cell_type_for_cell(cellid)]
            self.templates += [self.spikesorting.get_ei_for_cell(cellid).ei]  
            self.templates_variance += [self.spikesorting.get_ei_for_cell(cellid).ei_error]
            
    def store_all_cells_except(self, excluded_types):
        '''
        Store templates from all cells of types in list of cell types.
        '''
        
        exclude_cellids = [self.spikesorting.get_all_cells_similar_to_type(type_) for type_ in excluded_types]
        exclude_cellids = itertools.chain(*exclude_cellids)
        
        
        cellids = list(set(self.spikesorting.get_cell_ids()) - set(exclude_cellids) )
        self.cellids += cellids
        for cellid in cellids:
            self.celltypes += [self.spikesorting.get_cell_type_for_cell(cellid)]
            self.templates += [self.spikesorting.get_ei_for_cell(cellid).ei]  
            self.templates_variance += [self.spikesorting.get_ei_for_cell(cellid).ei_error]
            
    def store_all_cells_except_with_noise(self, excluded_types):
        '''
        Store templates from all cells of types in list of cell types.
        '''
        
        exclude_cellids = [self.spikesorting.get_all_cells_similar_to_type(type_) for type_ in excluded_types]
        exclude_cellids = itertools.chain(*exclude_cellids)
        
        
        cellids = list(set(self.spikesorting.get_cell_ids()) - set(exclude_cellids) )
        error = np.ones_like(self.spikesorting.get_ei_for_cell(cellids[0]).ei)*self.noise[:,None]
        
        self.cellids += cellids
        for cellid in cellids:
            self.celltypes += [self.spikesorting.get_cell_type_for_cell(cellid)]
            self.templates += [self.spikesorting.get_ei_for_cell(cellid).ei]  
            self.templates_variance += [error]


    def store_good_cells_no_crap(self):
        '''
        Store templates from all cells of types in list of cell types.
        '''
        
        cellids = list(set(self.spikesorting.get_cell_ids()) - set(self.spikesorting.get_all_cells_similar_to_type('bad')) - set(self.spikesorting.get_all_cells_similar_to_type('duplicate'))-set(self.spikesorting.get_all_cells_similar_to_type('crap')))
        self.cellids += cellids
        for cellid in cellids:
            self.celltypes += [self.spikesorting.get_cell_type_for_cell(cellid)]
            self.templates += [self.spikesorting.get_ei_for_cell(cellid).ei]  
            self.templates_variance += [self.spikesorting.get_ei_for_cell(cellid).ei_error]
    
    def reset_templates(self):
        '''
        Remove all templates and associated data
        '''
        self.cellids = []
        self.celltypes = []
        self.templates = []
        self.templates_variance = []
    
    def get_template_from_cell(self, cell):
        return self.templates[cell == self.cellids]
        
    
    def mask_non_main_channels(self):
        '''
        For each template, mask electrodes that are not either the cells identifier 
        electrode (computed in Vision) or its neighbors
        ''' 
        with vl.NeuronsReader(self.full_analysis_path,self.datarun) as nr:
            for i, cellid in enumerate(self.cellids):
                seed_elec_ind = nr.get_identifier_electrode_for_neuron(cellid)-1
                group = self.groups[seed_elec_ind]
                mask = np.array([bool(i in group) for i in range(self.templates[0].shape[0])])
                self.templates[i][~mask, :] = 0
                
                
    def mask_non_significant_channels(self):
        '''
        For each template, mask electrodes that are not either the cells largest electrode
        or its neighbors
        ''' 

        for i, cellid in enumerate(self.cellids):
            template = self.templates[i]
            largest_electrode = np.argmax(np.max(np.abs(template), axis = 1)).item()
            group = self.groups[largest_electrode]
            mask = np.array([bool(i in group) for i in range(self.templates[0].shape[0])])
            self.templates[i][~mask, :] = 0
            
    def mask_from_list(self, electrode_list):
        '''
        For each template, mask electrodes that are not either the cells largest electrode
        or its neighbors
        ''' 

        for i, cellid in enumerate(self.cellids):
            template = self.templates[i]
            group = electrode_list
            mask = np.array([bool(i in group) for i in range(self.templates[0].shape[0])])
            self.templates[i][~mask, :] = 0
    
    def mask_by_snr(self, ratio):
        
        if self.include_noise:
            noise = self.spikesorting.channel_noise
        else:
            noise = np.ones((self.templates[0].shape[0],))
            
        for i, cellid in enumerate(self.cellids):
            template = self.templates[i]
            
            self.templates[i][np.max(np.abs(template), axis = 1) < noise*ratio, :] = 0
            
            
    def mask_by_latency(self, latency = 1):
        noise = self.spikesorting.channel_noise
        for i, cellid in enumerate(self.cellids):
            templates = self.templates[i]
       
            peak_samples = np.argmin(templates, axis = 1)
            peak_vals = templates[np.arange(len(peak_samples)), peak_samples]
            peak_elec = np.argmax(np.abs(peak_vals))
            
            
            self.templates[i][((np.abs(peak_samples - peak_samples[peak_elec]) >= latency+1) | (np.abs(peak_vals) < noise)), :]= 0
            
    def mask_by_threshold(self, threshold):
        '''
        For each template, mask electrodes whos channel max signal < threshold.
        Ex. threshold = 30
        '''
        for i, cellid in enumerate(self.cellids):
            template = self.templates[i]
            max_signals = np.max(np.abs(template), axis = 1)
            self.templates[i][max_signals < threshold, :] = 0
            
   
            
    def remove_templates_by_snr(self, r, n, add_surround = True):
        '''
        For each template, remove it if signal on channel r < n * noise on channel r 
        '''
        try:
            noise = self.spikesorting.channel_noise
        except UnboundLocalError:
            print('Noise parameters not loaded! Set include_noise to True or try remove_templates_by_threshold.')
            raise
           
        r -= 1
        if add_surround:
            r = self.groups[r]
        
            
        new_cellids = []
        new_templates = []
        new_celltypes = []
        new_templates_variance = []
        for i, cellid in enumerate(self.cellids):
            template = self.templates[i]
            
            if add_surround:
                signal = np.max(np.abs(template[r, :]), axis = 1)
            else:
                signal = np.max(np.abs(template[r, :]))
            celltype = self.celltypes[i]
            if (signal > n*noise[r]).any():
                new_cellids += [cellid]
                new_templates += [template]
                new_celltypes += [celltype]
                new_templates_variance += [self.templates_variance[i]]

        self.cellids = new_cellids
        self.templates = new_templates
        self.celltypes = new_celltypes
        self.templates_variance = new_templates_variance
        
    def remove_templates_with_zero_variance(self, electrode_list):
        '''
        For each template, remove it if signal on channel r < n * noise on channel r 
        '''
        
            
        new_cellids = []
        new_templates = []
        new_celltypes = []
        new_templates_variance = []
        for i, cellid in enumerate(self.cellids):
            templates_variance = self.templates_variance[i]
            
            if (templates_variance[electrode_list]!=0).all():
                new_cellids += [cellid]
                new_templates += [self.templates[i]]
                new_celltypes += [self.celltypes[i]]
                new_templates_variance += [self.templates_variance[i]]

        self.cellids = new_cellids
        self.templates = new_templates
        self.celltypes = new_celltypes
        self.templates_variance = new_templates_variance
        
    def remove_templates_with_excess_variance(self, electrode_list):
        '''
        For each template, remove it if signal on channel r < n * noise on channel r 
        '''
            
        new_cellids = []
        new_templates = []
        new_celltypes = []
        new_templates_variance = []
        for i, cellid in enumerate(self.cellids):
            templates_variance = self.templates_variance[i][electrode_list]
            template = self.templates[i][electrode_list]
            
            abs_template = np.abs(template)
            template_power = np.sum(abs_template[abs_template >= 1])
            error_power = np.sum((templates_variance)[abs_template >= 1])
            
            if template_power > error_power:
                new_cellids += [cellid]
                new_templates += [self.templates[i]]
                new_celltypes += [self.celltypes[i]]
                new_templates_variance += [self.templates_variance[i]]

        self.cellids = new_cellids
        self.templates = new_templates
        self.celltypes = new_celltypes
        self.templates_variance = new_templates_variance
        
    def remove_templates_by_elec_power(self, electrode_list, n, num_samples):

        try:
            noise = self.spikesorting.channel_noise
        except UnboundLocalError:
            print('Noise parameters not loaded! Set include_noise to True or try remove_templates_by_threshold.')
            raise
           
        
        
            
        new_cellids = []
        new_templates = []
        new_celltypes = []
        new_templates_variance = []
        
        
        for i, cellid in enumerate(self.cellids):
            template = self.templates[i]
            
            signal_power = np.sum(template[electrode_list, :]**2)
            celltype = self.celltypes[i]
            if (signal_power > (n-1)*sum([noise[e]**2 for e in electrode_list])*num_samples):
                new_cellids += [cellid]
                new_templates += [template]
                new_celltypes += [celltype]
                new_templates_variance += [self.templates_variance[i]]
                

        self.cellids = new_cellids
        self.templates = new_templates
        self.celltypes = new_celltypes
        self.templates_variance = new_templates_variance
            
    def remove_templates_by_threshold(self, r, threshold, add_surround = True):
        '''
        For each template, remove it if signal on channel r < threshold
        '''
        r -= 1
        if add_surround:
            r = self.groups[r]
        
            
        new_cellids = []
        new_templates = []
        new_celltypes = []
        new_templates_variance = []
        for i, cellid in enumerate(self.cellids):
            template = self.templates[i]
            if add_surround:
                signal = np.max(np.abs(template[r, :]), axis = 1)
            else:
                signal = np.max(np.abs(template[r, :]))
            celltype = self.celltypes[i]
            if (signal >= threshold).any():
                new_cellids += [cellid]
                new_templates += [template]
                new_celltypes += [celltype]
                new_templates_variance += [self.templates_variance[i]]
        
        self.cellids = new_cellids
        self.templates = new_templates
        self.celltypes = new_celltypes
        self.templates_variance = new_templates_variance
        
    def remove_templates_by_list(self, cell_list):
        '''
        For each template, remove it if signal on channel r < threshold
        '''

        new_cellids = []
        new_templates = []
        new_celltypes = []
        new_templates_variance = []
        for i, cellid in enumerate(self.cellids):
            template = self.templates[i]
            celltype = self.celltypes[i]
            if cellid not in cell_list:
                new_templates_variance += [self.templates_variance[i]]
                new_cellids += [cellid]
                new_templates += [template]
                new_celltypes += [celltype]
        
        self.cellids = new_cellids
        self.templates = new_templates
        self.celltypes = new_celltypes
        self.templates_variance = new_templates_variance
            

class AritifactEstimator:
    
    '''
    Base class for any electrical stimulation artifact estimation method. 
    The purpose of this class is to be inheritable for any artifact estimation scheme. To 
    use it, the class has the following arguments:
        - analysis_path: path to dataset (e.g. '/Volumes/Analysis/2019-06-20-0/')
        - datarun: datarun identifier of electrical stimulation data (e.g. data001)
        - p: stimulating electrode 
        - tl:template loader instance
        - OPTIONAL: stim_recorded_samples: # of samples in EI  
        - OPTIONAL: array_id: 1502 for 519 array; 502 for 512 array
     
    This class facilitates doing simple things, like getting the artifact of a particular
    stimulation level (see get_artifact), the response probablity for a particular cell
    and amplitude (see get_response_probability_for_cell), the response probability 
    array for a cell (see get_sigmoid), the cell stimulable >0.5 efficiency 
    (see get_responsive_cells), the sigmoid parameters of responsive cells 
    (see get_responsive_cells_data). You can also reset all of the artifact estimations
    with reset_artifacts.
    
    '''
    
    def __init__(self, analysis_path, datarun, p, tl, stim_recorded_samples = 55, array_id = 502):
        self.tl = tl
        self.array_id = array_id
        array_map = electrode_map.get_litke_array_adj_mat_by_array_id(array_id)
        self.num_electrodes = len(array_map)
        self.artifacts = {}
        self.artifacts[0] = np.zeros((stim_recorded_samples*self.num_electrodes,))
        self.sigmoids = {}
        self.sorted_cells_and_spikes = {}
        self.stim_recorded_samples = stim_recorded_samples
        self.analysis_path = analysis_path
        self.datarun = datarun
        self.p = p
        Y_low = get_oldlabview_pp_data(analysis_path + datarun, p, 0)
        self.mu = np.mean(Y_low, axis = 0)[:,:stim_recorded_samples].reshape((-1,))
    
        self.num_trials = [] # To avoid complications that come with uneven trials 
                             # during stimulation-recording sessions, this is computed
                             # when autosort is run. sigh
                
        
    
    def get_artifact(self, j):
        # Return artifact estimate for particular cell and stimulation level
        try:
            return self.artifacts[j].reshape((self.num_electrodes, -1))
        except KeyError:
            raise KeyError('No artifact estimate for stimulation level %s' %j)
                        
    def get_all_artifacts(self):
        return self.artifacts
    
    def get_response_probability_for_cell(self, j, cell):
        # Return response probability for particular cell and stimulation level
        return self.sigmoids[j][cell]/self.num_trials[j]
    
    def get_sigmoid(self, cell):
        # Return cell response vector accross stimulation levels
        sigmoid = []
        for j in list(self.sigmoids.keys()):
            sigmoid += [self.sigmoids[j][cell]]
        return np.array(sigmoid)/self.num_trials
    
    def get_responsive_cells(self):
        # cells with >50% response efficiency. 
        responsive_cells = []
        for j in list(self.sigmoids.keys()):
            for cell in self.sigmoids[j]:
                if self.sigmoids[j][cell]/self.num_trials[j] >= 0.5:
                    responsive_cells += [cell]
        return list(set(responsive_cells))
    
    def get_responsive_cells_data(self, amplitudes, noisy_monotonicity = 0.8):
        # Return sigmoid parameters for cells with >50% response efficiency. 
        # Given the noisiness in spike sorting but known monotonicity of spike 
        # probability vector, choose a toleration band for including
        # data in sigmoid estimation 
        def fsigmoid(x, a, b):
            return 1.0 / (1.0 + np.exp(-a*(x-b)))
        
       

        responsive_cells = self.get_responsive_cells()
        responsive_cell_data = {}
        for cell in responsive_cells:
            sigmoid = self.get_sigmoid(cell)
            if noisy_monotonicity:
                finalized = self.enforce_noisy_monotonicity( sigmoid, noisy_monotonicity)
            else:
                finalized = np.ones(len(amplitudes)).astype(np.int16)

                
                
            try:
                popt, pcov = curve_fit(fsigmoid, amplitudes[finalized==1], sigmoid[finalized==1])
            except RuntimeError:
                responsive_cell_data[cell] = (None,None, finalized)
                continue
            responsive_cell_data[cell] = (popt[0]/4, popt[1], finalized)
        
        return responsive_cell_data
    def enforce_noisy_monotonicity(self, sigmoid, noise_limit):
        # Function to force monotonicity while allowing some noise 
        # tolerance. Works by finding the threshold where spike
        # response probability first exceeds 0.5, then tracks how the 
        # response probability changes, storing the high spike response probability
        # through stimulation levels. For each subsequent response probability,
        # if it is lower than noise_limit*maximum_response probability,
        # don't include it in the sigmoid estimate.
        thr = np.argwhere(sigmoid >= 0.5).flatten()[0]
        J_array = [1 for _ in range(thr+1)]
        max_value = sigmoid[thr]
        for i in range(thr+1, len(sigmoid)):
            if sigmoid[i] >= max_value*noise_limit:
                max_value = sigmoid[i]
                J_array += [1]
            else:
                J_array += [0]
        return np.array(J_array).astype(np.int16)
                
            
                    
    def reset_artifacts(self):
        self.artifacts = []
        self.artifacts[0] = np.zeros((self.stim_recorded_samples*self.num_electrodes,))
        
    
