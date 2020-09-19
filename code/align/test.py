import gc
import multiprocessing
import time

import numpy as np
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist


def greedy_alignment(embed1, embed2, top_k, nums_threads, metric, normalize, csls_k, accurate):
    """
    Search align with greedy strategy.

    Parameters
    ----------
    embed1 : matrix_like
        An embedding matrix of size n1*d, where n1 is the number of embeddings and d is the dimension.
    embed2 : matrix_like
        An embedding matrix of size n2*d, where n2 is the number of embeddings and d is the dimension.
    top_k : list of integers
        Hits@k metrics for evaluating results.
    nums_threads : int
        The number of threads used to search align.
    metric : string
        The distance metric to use. It can be 'cosine', 'euclidean' or 'inner'.
    normalize : bool, true or false.
        Whether to normalize the input embeddings.
    csls_k : int
        K value for csls. If k > 0, enhance the similarity by csls.

    Returns
    -------
    alignment_rest :  list, pairs of aligned entities
    hits1 : float, hits@1 values for align results
    mr : float, MR values for align results
    mrr : float, MRR values for align results
    """
    t = time.time()
    sim_mat = sim(embed1, embed2, metric=metric, normalize=normalize, csls_k=csls_k)
    num = sim_mat.shape[0]
    if nums_threads > 1:
        hits = [0] * len(top_k)
        mr, mrr = 0, 0
        alignment_rest = set()
        rests = list()
        search_tasks = task_divide(np.array(range(num)), nums_threads)
        pool = multiprocessing.Pool(processes=len(search_tasks))
        for task in search_tasks:
            mat = sim_mat[task, :]
            rests.append(pool.apply_async(calculate_rank, (task, mat, top_k, accurate, num)))
        pool.close()
        pool.join()
        for rest in rests:
            sub_mr, sub_mrr, sub_hits, sub_hits1_rest = rest.get()
            mr += sub_mr
            mrr += sub_mrr
            hits += np.array(sub_hits)
            alignment_rest |= sub_hits1_rest
    else:
        mr, mrr, hits, alignment_rest = calculate_rank(list(range(num)), sim_mat, top_k, accurate, num)
    assert len(alignment_rest) == num
    hits = np.array(hits) / num * 100
    for i in range(len(hits)):
        hits[i] = round(hits[i], 3)
    cost = time.time() - t
    if accurate:
        if csls_k > 0:
            print("accurate results with csls, hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}, time = {:.3f} s ".
                  format(top_k, hits, mr, mrr, cost))
        else:
            print("accurate results: hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}, time = {:.3f} s ".
                  format(top_k, hits, mr, mrr, cost))
    else:
        if csls_k > 0:
            print("quick results with csls, hits@{} = {}%, time = {:.3f} s ".format(top_k, hits, cost))
        else:
            print("quick results: hits@{} = {}%, time = {:.3f} s ".format(top_k, hits, cost))
    hits1 = hits[0]
    del sim_mat
    gc.collect()
    return alignment_rest, hits1, mr, mrr


def arg_sort(idx, sim_mat, prefix1, prefix2):
    candidates = dict()
    for i in range(len(idx)):
        x_i = prefix1 + str(idx[i])
        rank = (-sim_mat[i, :]).argsort()
        y_j = [prefix2 + str(r) for r in rank]
        candidates[x_i] = y_j
    return candidates


def calculate_rank(idx, sim_mat, top_k, accurate, total_num):
    assert 1 in top_k
    mr = 0
    mrr = 0
    hits = [0] * len(top_k)
    hits1_rest = set()
    for i in range(len(idx)):
        gold = idx[i]
        if accurate:
            rank = (-sim_mat[i, :]).argsort()
        else:
            rank = np.argpartition(-sim_mat[i, :], np.array(top_k) - 1)
        hits1_rest.add((gold, rank[0]))
        assert gold in rank
        rank_index = np.where(rank == gold)[0][0]
        mr += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                hits[j] += 1
    mr /= total_num
    mrr /= total_num
    return mr, mrr, hits, hits1_rest


def sim(embed1, embed2, metric='inner', normalize=False, csls_k=0):
    """
    Compute pairwise similarity between the two collections of embeddings.

    Parameters
    ----------
    embed1 : matrix_like
        An embedding matrix of size n1*d, where n1 is the number of embeddings and d is the dimension.
    embed2 : matrix_like
        An embedding matrix of size n2*d, where n2 is the number of embeddings and d is the dimension.
    metric : str, optional, inner default.
        The distance metric to use. It can be 'cosine', 'euclidean', 'inner'.
    normalize : bool, optional, default false.
        Whether to normalize the input embeddings.
    csls_k : int, optional, 0 by default.
        K value for csls. If k > 0, enhance the similarity by csls.

    Returns
    -------
    sim_mat : An similarity matrix of size n1*n2.
    """
    if normalize:
        embed1 = preprocessing.normalize(embed1)
        embed2 = preprocessing.normalize(embed2)
    if metric == 'inner':
        sim_mat = np.matmul(embed1, embed2.T)  # numpy.ndarray, float32
    elif metric == 'cosine' and normalize:
        sim_mat = np.matmul(embed1, embed2.T)  # numpy.ndarray, float32
    elif metric == 'euclidean':
        sim_mat = 1 - euclidean_distances(embed1, embed2)
        print(type(sim_mat), sim_mat.dtype)
        sim_mat = sim_mat.astype(np.float32)
    elif metric == 'cosine':
        sim_mat = 1 - cdist(embed1, embed2, metric='cosine')  # numpy.ndarray, float64
        sim_mat = sim_mat.astype(np.float32)
    elif metric == 'manhattan':
        sim_mat = 1 - cdist(embed1, embed2, metric='cityblock')
        sim_mat = sim_mat.astype(np.float32)
    else:
        sim_mat = 1 - cdist(embed1, embed2, metric=metric)
        sim_mat = sim_mat.astype(np.float32)
    if csls_k > 0:
        sim_mat = csls_sim(sim_mat, csls_k)
    return sim_mat


def csls_sim(sim_mat, k):
    """
    Compute pairwise csls similarity based on the input similarity matrix.

    Parameters
    ----------
    sim_mat : matrix-like
        A pairwise similarity matrix.
    k : int
        The number of nearest neighbors.

    Returns
    -------
    csls_sim_mat : A csls similarity matrix of n1*n2.
    """

    nearest_values1 = calculate_nearest_k(sim_mat, k)
    nearest_values2 = calculate_nearest_k(sim_mat.T, k)
    csls_sim_mat = 2 * sim_mat - nearest_values1 - nearest_values2.T
    return csls_sim_mat


def calculate_nearest_k(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    return np.mean(nearest_k, axis=1, keepdims=True)


def csls_sim_multi_threads(sim_mat, k, nums_threads):
    tasks = task_divide(np.array(range(sim_mat.shape[0])), nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    rests = list()
    for task in tasks:
        rests.append(pool.apply_async(calculate_nearest_k, (sim_mat[task, :], k)))
    pool.close()
    pool.join()
    sim_values = None
    for res in rests:
        val = res.get()
        if sim_values is None:
            sim_values = val
        else:
            sim_values = np.append(sim_values, val)
    assert sim_values.shape[0] == sim_mat.shape[0]
    return sim_values


def task_divide(idx, n):
    total = len(idx)
    if n <= 0 or 0 == total:
        return [idx]
    if n > total:
        return [idx]
    elif n == total:
        return [[i] for i in idx]
    else:
        j = total // n
        tasks = []
        for i in range(0, (n - 1) * j, j):
            tasks.append(idx[i:i + j])
        tasks.append(idx[(n - 1) * j:])
        return tasks