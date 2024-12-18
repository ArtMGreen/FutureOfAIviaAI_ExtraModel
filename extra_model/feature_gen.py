from utils import NUM_OF_VERTICES
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


def sparse_to_csr(full_dynamic_graph_sparse):
    # The concatenation is used to produce a symmetric adjacency matrix
    data_rows = np.concatenate([full_dynamic_graph_sparse[:, 0], full_dynamic_graph_sparse[:, 1]])
    data_cols = np.concatenate([full_dynamic_graph_sparse[:, 1], full_dynamic_graph_sparse[:, 0]])
    data_ones = np.ones(len(data_rows), np.uint32)
    # efficient row slicing here;
    # if multiple edges between the same nodes are present, they are converted to a single one with weight = #(edges)
    adjM_csr = sp.csr_matrix((data_ones, (data_rows, data_cols)), shape=(NUM_OF_VERTICES, NUM_OF_VERTICES))
    return adjM_csr


def preferential_attachment(full_dynamic_graph_sparse, unconnected_vertex_pairs, return_ranking=False):
    adjM_csr = sparse_to_csr(full_dynamic_graph_sparse)
    degree_vector = adjM_csr.sum(1).A1

    pred_degree_0 = degree_vector[unconnected_vertex_pairs[:,0]]
    pred_degree_1 = degree_vector[unconnected_vertex_pairs[:,1]]

    score_list_pa = pred_degree_0 * pred_degree_1
    if return_ranking:
        sorted_predictions_eval = np.argsort(-1.0*score_list_pa)
        return sorted_predictions_eval
    else:
        return score_list_pa


def clustering_coefficient(full_dynamic_graph_sparse, unconnected_vertex_pairs, return_ranking=False):
    adjM_csr = sparse_to_csr(full_dynamic_graph_sparse)
    cluster_coeffs_vector = np.zeros(NUM_OF_VERTICES)

    for node in tqdm(range(NUM_OF_VERTICES), desc="Calculating clustering coefficients"):
        neighbors = adjM_csr[node].nonzero()[1]
        degree = len(neighbors)
        if degree < 2:
            cluster_coeffs_vector[node] = 0
        else:
            submatrix = adjM_csr[neighbors, :][:, neighbors]
            triangles = submatrix.nnz  # already multiplied by 2 (!!!) submatrix is symmetric
            cluster_coeffs_vector[node] = triangles / (degree * (degree - 1))
    # cluster_coeffs_vector now contains the clustering coefficient for each node

    pred_degree_0 = cluster_coeffs_vector[unconnected_vertex_pairs[:,0]]
    pred_degree_1 = cluster_coeffs_vector[unconnected_vertex_pairs[:,1]]

    score_list_cc = pred_degree_0 * pred_degree_1
    if return_ranking:
        sorted_predictions_eval = np.argsort(score_list_cc)
        return sorted_predictions_eval
    else:
        return score_list_cc


def ruzicka_similarity(full_dynamic_graph_sparse, unconnected_vertex_pairs, return_ranking=False):
    adjM_csr = sparse_to_csr(full_dynamic_graph_sparse)
    score_list_ruzicka = np.zeros(len(unconnected_vertex_pairs))

    for i, (u, v) in tqdm(enumerate(unconnected_vertex_pairs), desc="Calculating Ruzicka similarity"):
        minimum_vector = adjM_csr[u,:].minimum(adjM_csr[v,:])
        maximum_vector = adjM_csr[u,:].maximum(adjM_csr[v,:])

        numerator = minimum_vector.sum()
        denominator = maximum_vector.sum()
        if denominator != 0:
            score_list_ruzicka[i] = numerator / denominator

    if return_ranking:
        sorted_predictions_eval = np.argsort(-1.0 * score_list_ruzicka)
        return sorted_predictions_eval
    else:
        return score_list_ruzicka


def weighted_overlap_score(full_dynamic_graph_sparse, unconnected_vertex_pairs, return_ranking=False):
    adjM_csr = sparse_to_csr(full_dynamic_graph_sparse)
    degree_vector = adjM_csr.sum(1).A1
    score_list_wos = np.zeros(len(unconnected_vertex_pairs))

    for i, (u, v) in tqdm(enumerate(unconnected_vertex_pairs), desc="Calculating Weighted Overlap scores"):
        minimum_vector = adjM_csr[u,:].minimum(adjM_csr[v,:])

        numerator = minimum_vector.sum()
        denominator = min(degree_vector[u], degree_vector[v])

        if denominator != 0:
            score_list_wos[i] = numerator / denominator

    if return_ranking:
        sorted_predictions_eval = np.argsort(-1.0 * score_list_wos)
        return sorted_predictions_eval
    else:
        return score_list_wos



def adamic_adar_index(full_dynamic_graph_sparse, unconnected_vertex_pairs, return_ranking=False):
    adjM_csr = sparse_to_csr(full_dynamic_graph_sparse)
    degree_vector = adjM_csr.sum(axis=1).A1
    score_list_aa = np.zeros(len(unconnected_vertex_pairs))

    for idx, (u, v) in tqdm(enumerate(unconnected_vertex_pairs), desc="Calculating Adamic-Adar scores"):
        neighbors_u = set(adjM_csr[u, :].indices)
        neighbors_v = set(adjM_csr[v, :].indices)
        common_neighbors = neighbors_u & neighbors_v

        # only deg(a) > 1 are eligible
        valid_degrees = degree_vector[list(common_neighbors)]
        valid_degrees = valid_degrees[valid_degrees > 1]  # Exclude invalid degrees
        score_list_aa[idx] = np.sum(1 / np.log(valid_degrees)) if len(valid_degrees) > 0 else 0.0

    if return_ranking:
        sorted_predictions_eval = np.argsort(-1.0 * score_list_aa)
        return sorted_predictions_eval
    else:
        return score_list_aa


def compute_and_save_scores(full_dynamic_graph_sparse, unconnected_vertex_pairs, output_file="features.npz"):
    print("Computing Preferential Attachment scores...")
    pa_scores = preferential_attachment(full_dynamic_graph_sparse, unconnected_vertex_pairs)

    print("Computing Clustering Coefficients...")
    cc_scores = clustering_coefficient(full_dynamic_graph_sparse, unconnected_vertex_pairs)

    print("Computing Ruzicka Similarity scores...")
    rz_scores = ruzicka_similarity(full_dynamic_graph_sparse, unconnected_vertex_pairs)

    print("Computing Weighted Overlap scores...")
    wos_scores = weighted_overlap_score(full_dynamic_graph_sparse, unconnected_vertex_pairs)

    print("Computing Adamic-Adar scores...")
    aa_scores = adamic_adar_index(full_dynamic_graph_sparse, unconnected_vertex_pairs)

    np.savez_compressed(output_file, pa_scores=pa_scores, cc_scores=cc_scores, aa_scores=aa_scores, rz_scores=rz_scores, wos_scores=wos_scores)
    print(f"Scores saved to {output_file}")


def compute_pa_scores(args):
    return preferential_attachment(*args)

def compute_cc_scores(args):
    return clustering_coefficient(*args)

def compute_aa_scores(args):
    return adamic_adar_index(*args)

def compute_rz_scores(args):
    return ruzicka_similarity(*args)

def compute_wos_scores(args):
    return weighted_overlap_score(*args)

def worker_function(func, args):
    return func(args)

def compute_and_save_scores_multiprocess(full_dynamic_graph_sparse, unconnected_vertex_pairs, output_file="features.npz"):
    print("Starting score computations with multiprocessing...")
    args = (full_dynamic_graph_sparse, unconnected_vertex_pairs, False)
    tasks = [
        (compute_pa_scores, args),
        (compute_cc_scores, args),
        (compute_aa_scores, args),
        (compute_rz_scores, args),
        (compute_wos_scores, args),
    ]

    with Pool(processes=5) as pool:
        results = pool.starmap(worker_function, tasks)

    pa_scores, cc_scores, aa_scores, rz_scores, wos_scores = results

    print("Saving scores to file...")
    np.savez_compressed(output_file, pa_scores=pa_scores, cc_scores=cc_scores, aa_scores=aa_scores, rz_scores=rz_scores, wos_scores=wos_scores)
    print(f"Scores saved to {output_file}")


import pickle
import os
from multiprocessing import Pool


current_min_edges = 3
curr_vertex_degree_cutoff = 25
current_delta = 5

data_source="../SemanticGraph_delta_"+str(current_delta)+"_cutoff_"+str(curr_vertex_degree_cutoff)+"_minedge_"+str(current_min_edges)+".pkl"

if os.path.isfile(data_source):
    with open(data_source, "rb") as pkl_file:
        full_dynamic_graph_sparse, unconnected_vertex_pairs, unconnected_vertex_pairs_solution, year_start, years_delta, vertex_degree_cutoff, min_edges = pickle.load(pkl_file)
        compute_and_save_scores_multiprocess(full_dynamic_graph_sparse, unconnected_vertex_pairs)
