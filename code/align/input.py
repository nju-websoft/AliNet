import os

import numpy as np
import time

from align.kg import KG


def read_input(folder):
    triples_set1 = read_triples(folder + 'triples_1')
    triples_set2 = read_triples(folder + 'triples_2')
    triples1 = KG(triples_set1)
    triples2 = KG(triples_set2)
    total_ent_num = len(triples1.ents | triples2.ents)
    total_rel_num = len(triples1.props | triples2.props)
    total_triples_num = len(triples1.triple_list) + len(triples2.triple_list)
    print('total ents:', total_ent_num)
    print('total rels:', len(triples1.props), len(triples2.props), total_rel_num)
    print('total triples: %d + %d = %d' % (len(triples1.triples), len(triples2.triples), total_triples_num))
    ref_ent1, ref_ent2 = read_references(folder + 'ref_ent_ids')
    assert len(ref_ent1) == len(ref_ent2)
    print("To aligned entities:", len(ref_ent1))
    sup_ent1, sup_ent2 = read_references(folder + 'sup_ent_ids')
    return triples1, triples2, sup_ent1, sup_ent2, ref_ent1, ref_ent2, total_triples_num, total_ent_num, total_rel_num


def read_dbp15k_input(folder):
    triples_set1 = read_triples(folder + 'triples_1')
    triples_set2 = read_triples(folder + 'triples_2')
    kg1 = KG(triples_set1)
    kg2 = KG(triples_set2)
    total_ent_num = len(kg1.ents | kg2.ents)
    total_rel_num = len(kg1.props | kg2.props)
    total_triples_num = len(kg1.triple_list) + len(kg2.triple_list)
    print('total ents:', total_ent_num)
    print('total rels:', len(kg1.props), len(kg2.props), total_rel_num)
    print('total triples: %d + %d = %d' % (len(kg1.triples), len(kg2.triples), total_triples_num))
    if os.path.exists(folder + 'ref_pairs'):
        ref_ent1, ref_ent2 = read_references(folder + 'ref_pairs')
    else:
        ref_ent1, ref_ent2 = read_references(folder + 'ref_ent_ids')
    assert len(ref_ent1) == len(ref_ent2)
    print("To aligned entities:", len(ref_ent1))
    if os.path.exists(folder + 'sup_pairs'):
        sup_ent1, sup_ent2 = read_references(folder + 'sup_pairs')
    else:
        sup_ent1, sup_ent2 = read_references(folder + 'sup_ent_ids')
    #     ****************************id mapping*************************
    rel_id_mapping = get_id_mapping(folder)
    return kg1, kg2, sup_ent1, sup_ent2, ref_ent1, ref_ent2, total_triples_num, total_ent_num, total_rel_num, rel_id_mapping


def read_triples(file):
    triples = set()
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            params = line.strip('\n').split('\t')
            assert len(params) == 3
            h = int(params[0])
            r = int(params[1])
            t = int(params[2])
            triples.add((h, r, t))
        f.close()
    return triples


def read_references(file):
    ref1, ref2 = list(), list()
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            params = line.strip('\n').split('\t')
            assert len(params) == 2
            e1 = int(params[0])
            e2 = int(params[1])
            ref1.append(e1)
            ref2.append(e2)
        f.close()
        assert len(ref1) == len(ref2)
    return ref1, ref2


def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return [ls]
    if n > ls_len:
        return [ls]
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return


def triples2ht_set(triples):
    ht_set = set()
    for h, r, t in triples:
        ht_set.add((h, t))
    print("the number of ht: {}".format(len(ht_set)))
    return ht_set


def merge_dic(dic1, dic2):
    return {**dic1, **dic2}


def generate_adjacency_mat(triples1, triples2, ent_num, sup_ents):
    adj_mat = np.mat(np.zeros((ent_num, len(sup_ents)), dtype=np.int32))
    ht_set = triples2ht_set(triples1) | triples2ht_set(triples2)
    for i in range(ent_num):
        for j in sup_ents:
            if (i, j) in ht_set:
                adj_mat[i, sup_ents.index(j)] = 1
    print("shape of adj_mat: {}".format(adj_mat.shape))
    print("the number of 1 in adjacency matrix: {}".format(np.count_nonzero(adj_mat)))
    return adj_mat


def generate_adj_input_mat(adj_mat, d):
    W = np.random.randn(adj_mat.shape[1], d)
    M = np.matmul(adj_mat, W)
    print("shape of input adj_mat: {}".format(M.shape))
    return M


def generate_ent_attrs_sum(ent_num, ent_attrs1, ent_attrs2, attr_embeddings):
    t1 = time.time()
    ent_attrs_embeddings = None
    for i in range(ent_num):
        attrs_index = list(ent_attrs1.get(i, set()) | ent_attrs2.get(i, set()))
        assert len(attrs_index) > 0
        attrs_embeds = np.sum(attr_embeddings[attrs_index,], axis=0)
        if ent_attrs_embeddings is None:
            ent_attrs_embeddings = attrs_embeds
        else:
            ent_attrs_embeddings = np.row_stack((ent_attrs_embeddings, attrs_embeds))
    print("shape of ent_attr_embeds: {}".format(ent_attrs_embeddings.shape))
    print("generating ent features costs: {:.3f} s".format(time.time() - t1))
    return ent_attrs_embeddings


def get_id_mapping(folder):
    rel_ids_1 = folder + "rel_ids_1"
    rel_ids_2 = folder + "rel_ids_2"
    kg1_id_dict = dict()
    kg2_id_dict = dict()
    with open(rel_ids_1, 'r', encoding='utf8') as f:
        for line in f:
            params = line.strip('\n').split('\t')
            kg1_id_dict[params[1]] = int()
            kg1_id_dict[params[1]] = int(params[0])
        f.close()
    with open(rel_ids_2, 'r', encoding='utf8') as f:
        for line in f:
            params = line.strip('\n').split('\t')
            kg2_id_dict[params[1]] = int()
            kg2_id_dict[params[1]] = int(params[0])
        f.close()
    rt_dict = dict()
    fold = folder.split("/")[-2]
    new_dir = folder.split("mtranse")[0] + fold + "/"
    if os.path.exists(new_dir):
        new_ids_1 = new_dir + "rel_ids_1"
        new_ids_2 = new_dir + "rel_ids_2"
        with open(new_ids_1, "r", encoding="utf8") as f:
            for line in f:
                params = line.strip("\n").split("\t")
                if kg1_id_dict[params[1]] not in rt_dict.keys():
                    rt_dict[kg1_id_dict[params[1]]] = int()
                rt_dict[kg1_id_dict[params[1]]] = int(params[0])
            f.close()
        with open(new_ids_2, "r", encoding="utf8") as f:
            for line in f:
                params = line.strip("\n").split("\t")
                if kg2_id_dict[params[1]] not in rt_dict.keys():
                    rt_dict[kg2_id_dict[params[1]]] = int()
                rt_dict[kg2_id_dict[params[1]]] = int(params[0])
            f.close()
    else:
        for value in kg1_id_dict.values():
            if value not in rt_dict.keys():
                rt_dict[value] = int()
                rt_dict[value] = value
        for value in kg2_id_dict.values():
            if value not in rt_dict.keys():
                rt_dict[value] = int()
                rt_dict[value] = value

    return rt_dict
