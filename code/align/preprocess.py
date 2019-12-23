import random

import numpy as np
import pandas as pd


def remove_unlinked_triples(triples, linked_ents):
    print("before removing unlinked triples:", len(triples))
    new_triples = set()
    for h, r, t in triples:
        if h in linked_ents and t in linked_ents:
            new_triples.add((h, r, t))
    print("after removing unlinked triples:", len(new_triples))
    return list(new_triples)


def enhance_triples(kg1, kg2, ents1, ents2):
    assert len(ents1) == len(ents2)
    print("before enhanced:", len(kg1.triples), len(kg2.triples))
    enhanced_triples1, enhanced_triples2 = set(), set()
    links1 = dict(zip(ents1, ents2))
    links2 = dict(zip(ents2, ents1))
    for h1, r1, t1 in kg1.triples:
        h2 = links1.get(h1, None)
        t2 = links1.get(t1, None)
        if h2 is not None and t2 is not None and t2 not in kg2.out_related_ents_dict.get(h2, set()):
            enhanced_triples2.add((h2, r1, t2))
    for h2, r2, t2 in kg2.triples:
        h1 = links2.get(h2, None)
        t1 = links2.get(t2, None)
        if h1 is not None and t1 is not None and t1 not in kg1.out_related_ents_dict.get(h1, set()):
            enhanced_triples1.add((h1, r2, t1))
    print("after enhanced:", len(enhanced_triples1), len(enhanced_triples2))
    return enhanced_triples1, enhanced_triples2


def generate_3hop_triples(kg, two_hop_triples, linked_ents=None):

    two_triple_df = np.array([[tr[0], tr[1], tr[2]] for tr in two_hop_triples])
    two_triple_df = pd.DataFrame(two_triple_df, columns=['h', 'r', 't'])

    triples = kg.triples
    if linked_ents is not None:
        triples = remove_unlinked_triples(triples, linked_ents)
    triple_df = np.array([[tr[0], tr[1], tr[2]] for tr in triples])
    triple_df = pd.DataFrame(triple_df, columns=['h', 'r', 't'])
    # print(triple_df)
    two_hop_triple_df = pd.merge(two_triple_df, triple_df, left_on='t', right_on='h')
    # print(two_hop_triple_df)
    two_step_quadruples = set()
    relation_patterns = dict()
    for index, row in two_hop_triple_df.iterrows():
        head = row["h_x"]
        tail = row["t_y"]
        r_x = row["r_x"]
        r_y = row['r_y']
        if tail not in kg.out_related_ents_dict.get(head, set()) and \
                head not in kg.in_related_ents_dict.get(tail, set()):
            relation_patterns[(r_x, r_y)] = relation_patterns.get((r_x, r_y), 0) + 1
            two_step_quadruples.add((head, r_x, r_y, tail))
    print("total 3-hop neighbors:", len(two_step_quadruples))
    print("total 3-hop relation patterns:", len(relation_patterns))
    relation_patterns = sorted(relation_patterns.items(), key=lambda x: x[1], reverse=True)
    p = 0.05
    num = int(p * len(relation_patterns))
    selected_patterns = set()
    # for i in range(20, num):
    for i in range(5, len(relation_patterns)):
        pattern = relation_patterns[i][0]
        selected_patterns.add(pattern)
    print("selected relation patterns:", len(selected_patterns))
    two_step_triples = set()
    for head, rx, ry, tail in two_step_quadruples:
        if (rx, ry) in selected_patterns:
            two_step_triples.add((head, 0, head))
            two_step_triples.add((head, rx + ry, tail))
    print("selected 3-hop neighbors:", len(two_step_triples))
    return two_step_triples


def generate_2hop_triples(kg, linked_ents=None):
    triples = kg.triples
    if linked_ents is not None:
        triples = remove_unlinked_triples(triples, linked_ents)
    triple_df = np.array([[tr[0], tr[1], tr[2]] for tr in triples])
    triple_df = pd.DataFrame(triple_df, columns=['h', 'r', 't'])
    # print(triple_df)
    two_hop_triple_df = pd.merge(triple_df, triple_df, left_on='t', right_on='h')
    # print(two_hop_triple_df)
    two_step_quadruples = set()
    relation_patterns = dict()
    for index, row in two_hop_triple_df.iterrows():
        head = row["h_x"]
        tail = row["t_y"]
        r_x = row["r_x"]
        r_y = row['r_y']
        if tail not in kg.out_related_ents_dict.get(head, set()) and \
                head not in kg.in_related_ents_dict.get(tail, set()):
            relation_patterns[(r_x, r_y)] = relation_patterns.get((r_x, r_y), 0) + 1
            two_step_quadruples.add((head, r_x, r_y, tail))
    print("total 2-hop neighbors:", len(two_step_quadruples))
    print("total 2-hop relation patterns:", len(relation_patterns))
    relation_patterns = sorted(relation_patterns.items(), key=lambda x: x[1], reverse=True)
    p = 0.05
    num = int(p * len(relation_patterns))
    selected_patterns = set()
    # for i in range(20, num):
    for i in range(5, len(relation_patterns)):
        pattern = relation_patterns[i][0]
        selected_patterns.add(pattern)
    print("selected relation patterns:", len(selected_patterns))
    two_step_triples = set()
    for head, rx, ry, tail in two_step_quadruples:
        if (rx, ry) in selected_patterns:
            two_step_triples.add((head, 0, head))
            two_step_triples.add((head, rx + ry, tail))
    print("selected 2-hop neighbors:", len(two_step_triples))
    return two_step_triples


def generate_2steps_path(triples):
    tr = np.array([[tr[0], tr[2], tr[1]] for tr in triples])
    tr = pd.DataFrame(tr, columns=['h', 't', 'r'])
    """
               h      t    r
        0      21860   8837   18
        1       2763  25362   42
        2        158  22040  130
    """
    sizes = tr.groupby(['h', 'r']).size()
    sizes.name = 'size'
    tr = tr.join(sizes, on=['h', 'r'])
    train_raw_df = tr[['h', 'r', 't', 'size']]
    two_step_df = pd.merge(train_raw_df, train_raw_df, left_on='t', right_on='h')
    print("total 2-hop triples:", two_step_df.shape[0])
    """
              h_x  r_x    t_x  size_x    h_y  r_y    t_y  size_y
        0       21860   18   8837       5   8837   18   1169       7
        1       21860   18   8837       5   8837   18  24618       7
        2       21860   18   8837       5   8837  216   1899       1
        3       21860   18   8837       5   8837   18    523       7
    """
    two_hop_relations = two_step_df[['r_x', 'r_y']]
    """
            r_x  r_y
        0        18   18
        1        18   18
        2        18  216
    """
    freq = two_hop_relations.groupby(['r_x', 'r_y']).size()
    freq.name = 'freq'
    freq_two_hop_relations = two_hop_relations.join(freq, on=['r_x', 'r_y']).drop_duplicates().dropna(axis=0)
    freq_two_hop_relations = freq_two_hop_relations.sort_values('freq', axis=0, ascending=False)
    """
            r_x  r_y     freq
        0        18   18  34163.0
        90980   103   18  34163.0
    """
    # print(freq_two_hop_relations)
    total_lines = freq_two_hop_relations.shape[0]
    print("total relation paths:", total_lines)
    p = 0.1
    num = int(p * total_lines)
    print("choose top", num)
    freq_two_hop_relations = freq_two_hop_relations.head(num)[['r_x', 'r_y']].values.tolist()
    freq_two_hop_relations = [(x, y) for x, y in freq_two_hop_relations]
    freq_two_hop_relations = set(freq_two_hop_relations)
    two_step_triples = set()
    for index, row in two_step_df.iterrows():
        head = row["h_x"]
        tail = row["t_y"]
        r_x = row["r_x"]
        r_y = row['r_y']
        if (r_x, r_y) in freq_two_hop_relations:
            two_step_triples.add((head, r_x + r_y, tail))
    print("new two hop neighbors:", len(two_step_triples))
    return set(two_step_triples)
