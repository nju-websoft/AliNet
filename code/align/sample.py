import multiprocessing
import gc
import numpy as np
from align.util import task_divide, merge_dic


# def generate_neighbours(entity_embeds1, entity_list1, entity_embeds2, entity_list2, neighbors_num, threads_num=4):
#     ent_frags = task_divide(np.array(entity_list1), threads_num)
#     ent_frag_indexes = task_divide(np.array(range(len(entity_list1))), threads_num)

#     pool = multiprocessing.Pool(processes=len(ent_frags))
#     results = list()
#     for i in range(len(ent_frags)):
#         results.append(pool.apply_async(find_neighbours,
#                                         args=(ent_frags[i],
#                                               entity_embeds1[ent_frag_indexes[i], :],
#                                               np.array(entity_list2),
#                                               entity_embeds2,
#                                               neighbors_num)))

#     pool.close()
#     pool.join()

#     dic = dict()
#     for res in results:
#         dic = merge_dic(dic, res.get())

#     del results
#     gc.collect()
#     return dic


def generate_neighbours(entity_embeds1, entity_list1, entity_embeds2, entity_list2, neighbors_num, threads_num=4):
    ent_frags = task_divide(np.array(entity_list1), threads_num)
    ent_frag_indexes = task_divide(np.array(range(len(entity_list1))), threads_num)
    dic = dict()
    for i in range(len(ent_frags)):
        res = find_neighbours(ent_frags[i], entity_embeds1[ent_frag_indexes[i], :], np.array(entity_list2),
                              entity_embeds2, neighbors_num)
        dic = merge_dic(dic, res)
    return dic


def find_neighbours(frags, sub_embed1, entity_list2, embed2, k):
    dic = dict()
    sim_mat = np.matmul(sub_embed1, embed2.T)
    for i in range(sim_mat.shape[0]):
        sort_index = np.argpartition(-sim_mat[i, :], k)
        neighbors_index = sort_index[0:k]
        neighbors = entity_list2[neighbors_index].tolist()
        dic[frags[i]] = neighbors
    del sim_mat
    return dic
