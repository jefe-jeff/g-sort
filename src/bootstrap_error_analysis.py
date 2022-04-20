import os
import numpy as np
import pickle
import tqdm
import src.multielec_utils as mutils
import re 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_bootstrap_information(ANALYSIS_BASE, gsort_path,dataset, estim, wnoise, p, n):
    electrical_path = os.path.join(ANALYSIS_BASE, dataset, estim)
    filepath = os.path.join(gsort_path, 
                            dataset, estim, wnoise, "p" + str(p))

    relevant_movies = [f for f in os.listdir(filepath) if f"_n{n}_" in f]
    num_movies = len(relevant_movies)

    total_probs = np.zeros(num_movies)
    edge_probs = []
    clusters = []
    edges = []
    for k, movie in enumerate(tqdm.tqdm(relevant_movies)):
        # if k%1000 == 0:
        #     print(f"{k} out of {num_movies}")
        with open(os.path.join(filepath,movie), "rb") as f:
            prob_dict = pickle.load(f)
        total_probs[k] = prob_dict["cosine_prob"][0]
        if total_probs[k]>0:
            edge_probs+=[prob_dict['edge_probs']]
        else:
            edge_probs+=[0]
        clusters += [prob_dict['initial_clustering_with_virtual']]

        edge_set = []
        for e,c in prob_dict['graph_info'][2].items():
            if c == n:
                edge_set += [e]
        edges += [edge_set]
    return relevant_movies, total_probs, edge_probs, clusters, edges

def make_surface_plot(relevant_movies, total_probs, ANALYSIS_BASE,dataset, estim, p, n):
    electrical_path = os.path.join(ANALYSIS_BASE, dataset, estim)
    
    triplet_elecs = mutils.get_stim_elecs_newlv(electrical_path, p)
    amplitudes = mutils.get_stim_amps_newlv(electrical_path, p)

    movies = [int(re.findall('\d+', m)[-1]) for m in relevant_movies]
    movie_sorting = np.argsort(movies)
    sorted_probs = total_probs[movie_sorting]

    p_thr = 2/19
    p_upper = 1

    good_inds = np.where((sorted_probs > p_thr) & (sorted_probs < p_upper))[0]

    fig = plt.figure()
    ax = Axes3D(fig)
    plt.xlabel(r'$I_1$')
    plt.ylabel(r'$I_2$')
    ax.set_zlabel(r'$I_3$')

    scat = ax.scatter(amplitudes[:, 0][good_inds], 
                amplitudes[:, 1][good_inds],
                amplitudes[:, 2][good_inds], marker='o', s=20, c=sorted_probs[good_inds], alpha=0.8)
    ax.set_title(f"n={n}, p={p}")
    clb = plt.colorbar(scat)
    clb.set_label('Activation Probability')
    plt.show()