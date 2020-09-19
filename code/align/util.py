import time
import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import pickle
from .input import read_dbp15k_input
from .preprocess import enhance_triples, generate_3hop_triples, generate_2hop_triples, \
    remove_unlinked_triples


def merge_dic(dic1, dic2):
    return {**dic1, **dic2}


def read_relation_triples(file_path):
    print("read relation triples:", file_path)
    if file_path is None:
        return set(), set(), set()
    triples = set()
    entities, relations = set(), set()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 3
        h = params[0].strip()
        r = params[1].strip()
        t = params[2].strip()
        triples.add((h, r, t))
        entities.add(h)
        entities.add(t)
        relations.add(r)
    return triples, entities, relations


def read_links(file_path):
    print("read links:", file_path)
    links = list()
    refs = list()
    reft = list()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        e1 = params[0].strip()
        e2 = params[1].strip()
        refs.append(e1)
        reft.append(e2)
        links.append((e1, e2))
    assert len(refs) == len(reft)
    return links


def read_dict(file_path):
    file = open(file_path, 'r', encoding='utf8')
    ids = dict()
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        ids[params[0]] = int(params[1])
    file.close()
    return ids


def read_pair_ids(file_path):
    file = open(file_path, 'r', encoding='utf8')
    pairs = list()
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        pairs.append((int(params[0]), int(params[1])))
    file.close()
    return pairs


def pair2file(file, pairs):
    if pairs is None:
        return
    with open(file, 'w', encoding='utf8') as f:
        for i, j in pairs:
            f.write(str(i) + '\t' + str(j) + '\n')
        f.close()


def dict2file(file, dic):
    if dic is None:
        return
    with open(file, 'w', encoding='utf8') as f:
        for i, j in dic.items():
            f.write(str(i) + '\t' + str(j) + '\n')
        f.close()
    print(file, "saved.")


def line2file(file, lines):
    if lines is None:
        return
    with open(file, 'w', encoding='utf8') as f:
        for line in lines:
            f.write(line + '\n')
        f.close()
    print(file, "saved.")


def radio_2file(radio, folder):
    path = folder + str(radio).replace('.', '_')
    if not os.path.exists(path):
        os.makedirs(path)
    return path + '/'


def save_results(folder, rest_12):
    if not os.path.exists(folder):
        os.makedirs(folder)
    pair2file(folder + 'alignment_results_12', rest_12)
    print("Results saved!")


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


def generate_out_folder(out_folder, training_data_path, div_path, method_name):
    params = training_data_path.strip('/').split('/')
    path = params[-1]
    folder = out_folder + method_name + '/' + path + "/" + div_path + str(time.strftime("%Y%m%d%H%M%S")) + "/"
    print("results output folder:", folder)
    return folder


def save_embeddings(folder, kgs, ent_embeds, rel_embeds, attr_embeds, mapping_mat=None):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if ent_embeds is not None:
        np.save(folder + 'ent_embeds.npy', ent_embeds)
    if rel_embeds is not None:
        np.save(folder + 'rel_embeds.npy', rel_embeds)
    if attr_embeds is not None:
        np.save(folder + 'attr_embeds.npy', attr_embeds)
    if mapping_mat is not None:
        np.save(folder + 'mapping_mat.npy', mapping_mat)
    dict2file(folder + 'kg1_ent_ids', kgs.kg1.entities_id_dict)
    dict2file(folder + 'kg2_ent_ids', kgs.kg2.entities_id_dict)
    dict2file(folder + 'kg1_rel_ids', kgs.kg1.relations_id_dict)
    dict2file(folder + 'kg2_rel_ids', kgs.kg2.relations_id_dict)
    dict2file(folder + 'kg1_attr_ids', kgs.kg1.attributes_id_dict)
    dict2file(folder + 'kg2_attr_ids', kgs.kg2.attributes_id_dict)
    print("Embeddings saved!")


# ***************************adj & sparse**************************
def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN gnn and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))
    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])
    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))
    return sparse_to_tuple(t_k)


def func(triples):
    head = {}
    cnt = {}
    for tri in triples:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            head[tri[1]] = {tri[0]}
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
    r2f = {}
    for r in cnt:
        r2f[r] = len(head[r]) / cnt[r]
    return r2f


def ifunc(triples):
    tail = {}
    cnt = {}
    for tri in triples:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            tail[tri[1]] = {tri[2]}
        else:
            cnt[tri[1]] += 1
            tail[tri[1]].add(tri[2])
    r2if = {}
    for r in cnt:
        r2if[r] = len(tail[r]) / cnt[r]
    return r2if


def get_weighted_adj(e, triples):
    r2f = func(triples)
    r2if = ifunc(triples)
    M = {}
    for tri in triples:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = max(r2if[tri[1]], 0.3)
        else:
            M[(tri[0], tri[2])] += max(r2if[tri[1]], 0.3)
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = max(r2f[tri[1]], 0.3)
        else:
            M[(tri[2], tri[0])] += max(r2f[tri[1]], 0.3)
    row = []
    col = []
    data = []
    for key in M:
        row.append(key[1])
        col.append(key[0])
        data.append(M[key])
    data = np.array(data, dtype='float32')
    return sp.coo_matrix((data, (row, col)), shape=(e, e))


def generate_rel_ht(triples):
    rel_ht_dict = dict()
    for h, r, t in triples:
        hts = rel_ht_dict.get(r, list())
        hts.append((h, t))
        rel_ht_dict[r] = hts
    return rel_ht_dict


def gcn_load_data(input_folder, is_two=False, is_three=False, is_four=False):
    kg1, kg2, sup_ent1, sup_ent2, ref_ent1, ref_ent2, total_tri_num, total_e_num, total_r_num, rel_id_mapping = \
        read_dbp15k_input(input_folder)
    linked_ents = set(sup_ent1 + sup_ent2 + ref_ent1 + ref_ent2)
    enhanced_triples1, enhanced_triples2 = enhance_triples(kg1, kg2, sup_ent1, sup_ent2)
    ori_triples = kg1.triple_list + kg2.triple_list
    triples = remove_unlinked_triples(ori_triples + list(enhanced_triples1) + list(enhanced_triples2), linked_ents)
    rel_ht_dict = generate_rel_ht(triples)

    saved_data_path = input_folder + 'alinet_saved_data.pkl'
    if os.path.exists(saved_data_path):
        print('load saved adj data from', saved_data_path)
        adj = pickle.load(open(saved_data_path, 'rb'))
    else:
        one_adj, _ = no_weighted_adj(total_e_num, triples, is_two_adj=False)
        adj = [one_adj]
        two_hop_triples1, two_hop_triples2 = None, None
        three_hop_triples1, three_hop_triples2 = None, None
        if is_two:
            two_hop_triples1 = generate_2hop_triples(kg1, linked_ents=linked_ents)
            two_hop_triples2 = generate_2hop_triples(kg2, linked_ents=linked_ents)
            triples = two_hop_triples1 | two_hop_triples2
            two_adj, _ = no_weighted_adj(total_e_num, triples, is_two_adj=False)
            adj.append(two_adj)
        if is_three:
            three_hop_triples1 = generate_3hop_triples(kg1, two_hop_triples1, linked_ents=linked_ents)
            three_hop_triples2 = generate_3hop_triples(kg2, two_hop_triples2, linked_ents=linked_ents)
            triples = three_hop_triples1 | three_hop_triples2
            three_adj, _ = no_weighted_adj(total_e_num, triples, is_two_adj=False)
            adj.append(three_adj)
        if is_four:
            four_hop_triples1 = generate_3hop_triples(kg1, three_hop_triples1, linked_ents=linked_ents)
            four_hop_triples2 = generate_3hop_triples(kg2, three_hop_triples2, linked_ents=linked_ents)
            triples = four_hop_triples1 | four_hop_triples2
            four_adj, _ = no_weighted_adj(total_e_num, triples, is_two_adj=False)
            adj.append(four_adj)
        print('save adj data to', saved_data_path)
        pickle.dump(adj, open(saved_data_path, 'wb'))
        
    return adj, kg1, kg2, sup_ent1, sup_ent2, ref_ent1, ref_ent2, total_tri_num, \
           total_e_num, total_r_num, rel_id_mapping, rel_ht_dict


def diag_adj(adj):
    d = np.array(adj.sum(1)).flatten()
    d_inv = 1. / d
    d_inv[np.isinf(d_inv)] = 0
    d_inv = sp.diags(d_inv)
    return sparse_to_tuple(d_inv.dot(adj))


def rgcn_adj_list(kg1, kg2, adj_number, all_rel_num, all_ent_num):
    # *****************test two adj**********************************
    # adj_list = list()
    # adj1 = list()
    # adj2 = list()
    # for item in kg1.triple_list:
    #     adj1.append([item[0], item[2]])
    #     adj1.append([item[2], item[0]])
    # for item in kg2.triple_list:
    #     adj2.append([item[0], item[2]])
    #     adj2.append([item[2], item[0]])
    # kg1_pos = np.array(adj1)
    # row, col = np.transpose(kg1_pos)
    # data = np.ones(row.shape[0])
    #
    # # n_row = np.hstack((row, col))
    # # n_col = np.hstack((col, row))
    # # adj = sp.coo_matrix((data, (n_row, n_col)), shape=(all_ent_num, all_ent_num))
    #
    # adj = sp.coo_matrix((data, (row, col)), shape=(all_ent_num, all_ent_num))
    # adj = diag_adj(adj)
    # # adj = preprocess_adj(adj)
    # #
    # # adj_rev = sp.coo_matrix((data, (col, row)), shape=(all_ent_num, all_ent_num))
    # # adj_rev = diag_adj(adj_rev)
    # # adj = sp.coo_matrix((data, (row, col)), shape=(all_ent_num, all_ent_num))
    # adj_list.append(adj)
    # # adj_list.append(adj_rev)
    #
    # kg2_pos = np.array(adj2)
    # row, col = np.transpose(kg2_pos)
    # data = np.ones(row.shape[0])
    # adj = sp.coo_matrix((data, (row, col)), shape=(all_ent_num, all_ent_num))
    # # adj = preprocess_adj(adj)
    # adj = diag_adj(adj)
    # #
    # # adj_rev = sp.coo_matrix((data, (col, row)), shape=(all_ent_num, all_ent_num))
    # # adj_rev = diag_adj(adj_rev)
    # # adj = sp.coo_matrix((data, (row, col)), shape=(all_ent_num, all_ent_num))
    #
    # # data = np.ones(row.shape[0] * 2)
    # #
    # # n_row = np.hstack((row, col))
    # # n_col = np.hstack((col, row))
    # # adj = sp.coo_matrix((data, (n_row, n_col)), shape=(all_ent_num, all_ent_num))
    # # adj = diag_adj(adj)
    # adj_list.append(adj)
    # # adj_list.append(adj_rev)
    # return adj_list

    # *****************************************************************

    adj_list = list()
    triple_list = kg1.triple_list + kg2.triple_list
    edge = dict()
    edge_length = np.zeros(all_rel_num)

    for item in triple_list:
        if item[1] not in edge.keys():
            edge[item[1]] = list()
        edge[item[1]].append([item[0], item[2]])
        edge[item[1]].append([item[2], item[0]])
        edge_length[item[1]] += 2
    sort_edge_length = np.argsort(-edge_length)
    # ****************************将剩余的关系构造成一个adj********************
    left_len = int(edge_length[sort_edge_length[adj_number]])
    pos = np.array(edge[sort_edge_length[adj_number]])
    first_row, first_col = np.transpose(pos)
    init_row = first_row
    init_col = first_col
    # init_row = np.hstack((first_row, first_col))
    # init_col = np.hstack((first_col, first_row))
    for i in range(adj_number + 1, len(edge.keys())):
        pos = np.array(edge[sort_edge_length[i]])
        row, col = np.transpose(pos)
        init_row = np.hstack((init_row, row))
        # init_row = np.hstack((init_row, col))
        init_col = np.hstack((init_col, col))
        # init_col = np.hstack((init_col, row))
        left_len += int(edge_length[sort_edge_length[i]])
    data = np.ones(left_len)
    left_adj = sp.coo_matrix((data, (init_row, init_col)), shape=(all_ent_num, all_ent_num))
    left_adj = diag_adj(left_adj)
    # left_adj = preprocess_adj(left_adj)
    left_rev_adj = sp.coo_matrix((data, (init_col, init_row)), shape=(all_ent_num, all_ent_num))
    left_rev_adj = diag_adj(left_rev_adj)
    adj_list.append(left_adj)
    adj_list.append(left_rev_adj)
    # **********************************************************************
    for i in range(adj_number):
        pos = np.array(edge[sort_edge_length[i]])
        row, col = np.transpose(pos)
        # *********************************构造对称adj*************************
        # new_row = np.hstack((row, col))
        # new_col = np.hstack((col, row))
        # data = np.ones(shape=int(edge_length[sort_edge_length[i]]*2))
        # adj = sp.coo_matrix((data, (new_row, new_col)), shape=(all_ent_num, all_ent_num))
        # adj = diag_adj(adj)
        # # adj = preprocess_adj(adj)
        # adj_list.append(adj)
        # ********************************************************************
        data = np.ones(shape=int(edge_length[sort_edge_length[i]]))
        adj = sp.coo_matrix((data, (row, col)), shape=(all_ent_num, all_ent_num))
        adj = diag_adj(adj)
        #
        adj_rev = sp.coo_matrix((data, (col, row)), shape=(all_ent_num, all_ent_num))
        adj_rev = diag_adj(adj_rev)
        # # adj = sp.coo_matrix((data, (row, col)), shape=(all_ent_num, all_ent_num))
        adj_list.append(adj)
        adj_list.append(adj_rev)
    return adj_list


def test_no_weighted_adj(total_ent_num, kg1_triple_list, kg2_triple_list):
    adj = list()
    for triple_list in [kg1_triple_list, kg2_triple_list]:
        edge = dict()
        for item in triple_list:
            if 0 <= item[0] < 10500:
                item_first = item[0]
            elif 10500 <= item[0] < 21000:
                item_first = item[0] - 10500
            elif item[0] < 25500:
                item_first = item[0] - 10500
            else:
                item_first = item[0] - 15000
            if 0 <= item[2] < 10500:
                item_second = item[2]
            elif 10500 <= item[2] < 21000:
                item_second = item[2] - 10500
            elif item[2] < 25500:
                item_second = item[2] - 10500
            else:
                item_second = item[2] - 15000
            if item_first not in edge.keys():
                edge[item_first] = set()
            if item_second not in edge.keys():
                edge[item_second] = set()
            edge[item_first].add(item_second)
            edge[item_second].add(item_first)
        row = list()
        col = list()
        for i in range(int(total_ent_num / 2)):
            if i not in edge.keys():
                continue
            key = i
            value = edge[key]
            add_key_len = len(value)
            add_key = (key * np.ones(add_key_len)).tolist()
            row.extend(add_key)
            col.extend(list(value))
        data_len = len(row)
        data = np.ones(data_len)
        one_adj = sp.coo_matrix((data, (row, col)), shape=(int(total_ent_num / 2), int(total_ent_num / 2)))
        one_adj = preprocess_adj(one_adj)
        adj.append(one_adj)
    return adj


def no_weighted_adj(total_ent_num, triple_list, is_two_adj=False):
    start = time.time()
    edge = dict()
    for item in triple_list:
        if item[0] not in edge.keys():
            edge[item[0]] = set()
        if item[2] not in edge.keys():
            edge[item[2]] = set()
        edge[item[0]].add(item[2])
        edge[item[2]].add(item[0])
    row = list()
    col = list()
    for i in range(total_ent_num):
        if i not in edge.keys():
            continue
        key = i
        value = edge[key]
        add_key_len = len(value)
        add_key = (key * np.ones(add_key_len)).tolist()
        row.extend(add_key)
        col.extend(list(value))
    data_len = len(row)
    data = np.ones(data_len)
    one_adj = sp.coo_matrix((data, (row, col)), shape=(total_ent_num, total_ent_num))
    one_adj = preprocess_adj(one_adj)
    print('generating one-adj costs time: {:.4f}s'.format(time.time() - start))
    if not is_two_adj:
        return one_adj, None
    expend_edge = dict()
    row = list()
    col = list()
    temp_len = 0
    for key, values in edge.items():
        if key not in expend_edge.keys():
            expend_edge[key] = set()
        for value in values:
            add_value = edge[value]
            for item in add_value:
                if item not in values and item != key:
                    expend_edge[key].add(item)
                    no_len = len(expend_edge[key])
                    if temp_len != no_len:
                        row.append(key)
                        col.append(item)
                    temp_len = no_len
    data = np.ones(len(row))
    two_adj = sp.coo_matrix((data, (row, col)), shape=(total_ent_num, total_ent_num))
    two_adj = preprocess_adj(two_adj)
    print('generating one- and two-adj costs time: {:.4f}s'.format(time.time() - start))
    return one_adj, two_adj


def temp_weighted_two_adj(total_ent_num, triple_list, is_two_adj=False):
    start = time.time()
    edge = dict()
    for item in triple_list:
        if item[0] not in edge.keys():
            edge[item[0]] = set()
        if item[2] not in edge.keys():
            edge[item[2]] = set()
        edge[item[0]].add(item[2])
        edge[item[2]].add(item[0])
    row = list()
    col = list()
    for i in range(total_ent_num):
        if i not in edge.keys():
            continue
        key = i
        value = edge[key]
        add_key_len = len(value)
        add_key = (key * np.ones(add_key_len)).tolist()
        row.extend(add_key)
        col.extend(list(value))
    data_len = len(row)
    data = (np.ones(data_len)) * 0.5
    one_adj = sp.coo_matrix((data, (row, col)), shape=(total_ent_num, total_ent_num))
    one_adj = preprocess_adj(one_adj)
    print('generating one-adj costs time: {:.4f}s'.format(time.time() - start))

    print('generating one- and two-adj costs time: {:.4f}s'.format(time.time() - start))
    return one_adj


def relation_adj_list(kg1, kg2, adj_number, all_rel_num, all_ent_num, linked_ents, rel_id_mapping):
    rel_dict = rel_id_mapping
    adj_list = list()
    triple_list = kg1.triple_list + kg2.triple_list
    edge = dict()
    edge_length = np.zeros(all_rel_num)
    # for item in triple_list:
    #     if item[1] not in edge.keys():
    #         edge[item[1]] = list()
    #     edge[item[1]].append([item[0], item[2]])
    #     edge_length[item[1]] += 1
    # sort_edge_length = np.argsort(-edge_length)

    for item in triple_list:
        if rel_dict[item[1]] is not None and rel_dict[item[1]] != "":
            edge_id = rel_dict[item[1]]
        else:
            edge_id = item[1]
        if edge_id not in edge.keys():
            edge[edge_id] = list()
        edge[edge_id].append([item[0], item[2]])
        edge_length[edge_id] += 1
    sort_edge_length = np.argsort(-edge_length)

    # **********************************************************************
    adj_len = list()
    for i in range(adj_number):
        pos = np.array(edge[sort_edge_length[i]])
        row, col = np.transpose(pos)
        data = np.ones(shape=int(edge_length[sort_edge_length[i]]))

        adj_len.append(int(edge_length[sort_edge_length[i]]))

        adj = sp.coo_matrix((data, (row, col)), shape=(all_ent_num, all_ent_num))
        adj = sparse_to_tuple(adj)
        adj_list.append(adj)
    # r1_count = 0
    # r2_count = 0
    # count = 0
    # r1_adj_number = adj_number / 2
    # r2_adj_number = adj_number / 2
    # while r1_count <= r1_adj_number and r2_count <= r2_adj_number:
    #     r_id = sort_edge_length[count]
    #     i = count
    #     count += 1
    #     if r_id > 1700:
    #         r2_count += 1
    #         if r2_count > adj_number / 2:
    #             continue
    #     else:
    #         r1_count += 1
    #         if r1_count > adj_number / 2:
    #             continue
    #     pos = np.array(edge[sort_edge_length[i]])
    #     row, col = np.transpose(pos)
    #     data = np.ones(shape=int(edge_length[sort_edge_length[i]]))
    #
    #     adj_len.append(int(edge_length[sort_edge_length[i]]))
    #
    #     adj = sp.coo_matrix((data, (row, col)), shape=(all_ent_num, all_ent_num))
    #     adj = sparse_to_tuple(adj)
    #     adj_list.append(adj)

    return adj_list


def transloss_add2hop(kg1, kg2, sup_ent1, sup_ent2, ref_ent1, ref_ent2, total_e_num):
    linked_ents = set(sup_ent1 + sup_ent2 + ref_ent1 + ref_ent2)
    enhanced_triples1 = generate_2hop_triples(kg1, linked_ents=linked_ents)
    enhanced_triples2 = generate_2hop_triples(kg2, linked_ents=linked_ents)
    triples = enhanced_triples1 | enhanced_triples2
    edge = dict()
    for item in triples:
        if item[0] not in edge.keys():
            edge[item[0]] = set()
        if item[2] not in edge.keys():
            edge[item[2]] = set()
        edge[item[0]].add(item[2])
        edge[item[2]].add(item[0])
    row = list()
    col = list()
    for i in range(total_e_num):
        if i not in edge.keys():
            continue
        key = i
        value = edge[key]
        add_key_len = len(value)
        add_key = (key * np.ones(add_key_len)).tolist()
        row.extend(add_key)
        col.extend(list(value))
    data_len = len(row)
    data = np.ones(data_len)
    one_adj = sp.coo_matrix((data, (row, col)), shape=(total_e_num, total_e_num))
    one_adj = sparse_to_tuple(one_adj)
    return one_adj
