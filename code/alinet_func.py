from align.kg import KG


def get_neighbor_dict(out_dict, in_dict):
    dic = dict()
    for key, value in out_dict.items():
        dic[key] = value
    for key, value in in_dict.items():
        values = dic.get(key, set())
        values |= value
        dic[key] = values
    return dic


def get_neighbor_counterparts(neighbors, alignment_dic):
    neighbor_counterparts = set()
    for n in neighbors:
        if n in alignment_dic:
            neighbor_counterparts.add(alignment_dic.get(n))
    return neighbor_counterparts


def compute_overlap_of_one_hop(hits1_results, sup_ent1, sup_ent2, ref_ent1, ref_ent2, kg1: KG, kg2: KG):
    right_hits1_results = set()
    for i, j in hits1_results:
        if i == j:
            right_hits1_results.add((i, j))
    right_aligned_ents = [(ref_ent1[i], ref_ent2[j]) for (i, j) in right_hits1_results]
    ents1 = sup_ent1 + ref_ent1
    ents2 = sup_ent2 + ref_ent2
    alignment_dict = dict(zip(ents1, ents2))
    kg1_neighbors_dic = get_neighbor_dict(kg1.in_related_ents_dict, kg1.out_related_ents_dict)
    kg2_neighbors_dic = get_neighbor_dict(kg2.in_related_ents_dict, kg2.out_related_ents_dict)
    radio1, radio2 = 0.0, 0.0
    dsc = 0.0
    jsc = 0.0
    for i, j in right_aligned_ents:
        i_neighbors = kg1_neighbors_dic.get(i, set())
        j_neighbors = kg2_neighbors_dic.get(j, set())
        i_neighbor_counterparts = get_neighbor_counterparts(i_neighbors, alignment_dict)
        overlap = i_neighbor_counterparts & j_neighbors
        radio1 += len(overlap) / len(i_neighbors)
        radio2 += len(overlap) / len(j_neighbors)
        dsc += 2 * (len(overlap)) / (len(i_neighbors) + len(j_neighbors))
        jsc += len(overlap) / (len(i_neighbors) + len(j_neighbors) - len(overlap))
    if len(right_aligned_ents) > 0:
        print("radio of overlap in KG1:", radio1 / len(right_aligned_ents))
        print("radio of overlap in KG2:", radio2 / len(right_aligned_ents))
        print("dsc score:", dsc / len(right_aligned_ents))
        print("jsc score:", jsc / len(right_aligned_ents))


def check_new_alignment(aligned_pairs, context="check align"):
    if aligned_pairs is None or len(aligned_pairs) == 0:
        print("{}, empty aligned pairs".format(context))
        return
    num = 0
    for x, y in aligned_pairs:
        if x == y:
            num += 1
    print("{}, right align: {}/{}={:.3f}".format(context, num, len(aligned_pairs), num / len(aligned_pairs)))


def update_labeled_alignment_x(pre_labeled_alignment, curr_labeled_alignment, sim_mat):
    check_new_alignment(pre_labeled_alignment, context="before editing (<-)")
    labeled_alignment_dict = dict(pre_labeled_alignment)
    n1, n2 = 0, 0
    for i, j in curr_labeled_alignment:
        if labeled_alignment_dict.get(i, -1) == i and j != i:
            n2 += 1
        if i in labeled_alignment_dict.keys():
            pre_j = labeled_alignment_dict.get(i)
            if pre_j == j:
                continue
            pre_sim = sim_mat[i, pre_j]
            new_sim = sim_mat[i, j]
            if new_sim >= pre_sim:
                if pre_j == i and j != i:
                    n1 += 1
                labeled_alignment_dict[i] = j
        else:
            labeled_alignment_dict[i] = j
    print("update wrongly: ", n1, "greedy update wrongly: ", n2)
    pre_labeled_alignment = set(zip(labeled_alignment_dict.keys(), labeled_alignment_dict.values()))
    check_new_alignment(pre_labeled_alignment, context="after editing (<-)")
    return pre_labeled_alignment


def update_labeled_alignment_y(labeled_alignment, sim_mat):
    labeled_alignment_dict = dict()
    updated_alignment = set()
    for i, j in labeled_alignment:
        i_set = labeled_alignment_dict.get(j, set())
        i_set.add(i)
        labeled_alignment_dict[j] = i_set
    for j, i_set in labeled_alignment_dict.items():
        if len(i_set) == 1:
            for i in i_set:
                updated_alignment.add((i, j))
        else:
            max_i = -1
            max_sim = -10
            for i in i_set:
                if sim_mat[i, j] > max_sim:
                    max_sim = sim_mat[i, j]
                    max_i = i
            updated_alignment.add((max_i, j))
    check_new_alignment(updated_alignment, context="after editing (->)")
    return updated_alignment
