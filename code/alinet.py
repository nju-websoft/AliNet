import random
import time
import gc

import tensorflow as tf
import numpy as np
import scipy

from gnn.gcn.layers import GraphConvolution, InputLayer
from alinet_layer import AliNetGraphAttentionLayer, HighwayLayer
from alinet_func import update_labeled_alignment_x, update_labeled_alignment_y
from align.test import greedy_alignment, sim
from align.semi_align import find_alignment
from align.util import no_weighted_adj
from align.preprocess import enhance_triples, remove_unlinked_triples
from align.sample import generate_neighbours


class AliNet:
    def __init__(self, adj, kg1, kg2, sup_ent1, sup_ent2, ref_ent1, ref_ent2, tri_num, ent_num, rel_num,
                 rel_ht_dict, args):
        self.one_hop_layers = None
        self.two_hop_layers = None
        self.layers_outputs = None

        self.adj_mat = adj
        self.kg1 = kg1
        self.kg2 = kg2
        self.sup_ent1 = sup_ent1
        self.sup_ent2 = sup_ent2
        self.ref_ent1 = ref_ent1
        self.ref_ent2 = ref_ent2
        self.tri_num = tri_num
        self.ent_num = ent_num
        self.rel_num = rel_num

        self.rel_ht_dict = rel_ht_dict
        self.rel_win_size = args.batch_size // len(rel_ht_dict)
        if self.rel_win_size <= 1:
            self.rel_win_size = args.min_rel_win

        self.neg_multi = args.neg_multi
        self.neg_margin = args.neg_margin
        self.neg_param = args.neg_param
        self.rel_param = args.rel_param
        self.truncated_epsilon = args.truncated_epsilon
        self.learning_rate = args.learning_rate

        self.layer_dims = args.layer_dims
        self.layer_num = len(args.layer_dims) - 1
        self.num_features_nonzero = args.num_features_nonzero
        self.dropout_rate = args.dropout_rate
        self.activation = args.activation

        self.eval_metric = args.eval_metric
        self.hits_k = args.hits_k
        self.eval_threads_num = args.eval_threads_num
        self.eval_normalize = args.eval_normalize
        self.eval_csls = args.eval_csls

        self.new_edges1, self.new_edges2 = set(), set()
        self.new_links = set()
        self.pos_link_batch = None
        self.neg_link_batch = None
        self.sim_th = args.sim_th
        self.start_augment = args.start_augment
        self.sup_links_set = set()
        for i in range(len(sup_ent1)):
            self.sup_links_set.add((self.sup_ent1[i], self.sup_ent2[i]))
        self.new_sup_links_set = set()
        self.linked_ents = set(sup_ent1 + sup_ent2 + ref_ent1 + ref_ent2)

        sup_ent1 = np.array(self.sup_ent1).reshape((len(self.sup_ent1), 1))
        sup_ent2 = np.array(self.sup_ent2).reshape((len(self.sup_ent1), 1))
        weight = np.ones((len(self.sup_ent1), 1), dtype=np.float)
        self.sup_links = np.hstack((sup_ent1, sup_ent2, weight))

        enhanced_triples1, enhanced_triples2 = enhance_triples(self.kg1, self.kg2, self.sup_ent1, self.sup_ent2)
        triples = self.kg1.triple_list + self.kg2.triple_list + list(enhanced_triples1) + list(enhanced_triples2)
        triples = remove_unlinked_triples(triples, self.linked_ents)
        one_adj, _ = no_weighted_adj(self.ent_num, triples, is_two_adj=False)
        self.ori_adj = [one_adj]

        self.model = self.define_model()
        self.input_embeds, self.output_embeds_list = None, None
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)

    def define_model(self):
        print('Getting AliNet model...')
        layer_num = len(self.layer_dims) - 1
        pos_link = tf.keras.Input(shape=[2, ])
        neg_link = tf.keras.Input(shape=[2, ])
        input_embeds = InputLayer(shape=[self.ent_num, self.layer_dims[0]])(pos_link)

        output_embeds = input_embeds
        one_layers = list()
        two_layers = list()
        layers_outputs = list()
        for i in range(layer_num):
            gcn_layer = GraphConvolution(input_dim=self.layer_dims[i],
                                         output_dim=self.layer_dims[i + 1],
                                         activations='tanh',
                                         adj=[self.adj_mat[0]],
                                         num_features_nonzero=self.num_features_nonzero,
                                         dropout_rate=0.0)
            one_layers.append(gcn_layer)
            one_output_embeds = gcn_layer(output_embeds)

            if i < layer_num - 1:
                gat_layer = AliNetGraphAttentionLayer(input_dim=self.layer_dims[i],
                                                      output_dim=self.layer_dims[i + 1],
                                                      adj=[self.adj_mat[1]],
                                                      nodes_num=self.ent_num,
                                                      num_features_nonzero=self.num_features_nonzero,
                                                      alpha=0.0,
                                                      activations='tanh',
                                                      dropout_rate=self.dropout_rate)
                two_layers.append(gat_layer)
                two_output_embeds = gat_layer(output_embeds)

                highway_layer = HighwayLayer(self.layer_dims[i + 1], self.layer_dims[i + 1],
                                             dropout_rate=self.dropout_rate)
                output_embeds = highway_layer([two_output_embeds, one_output_embeds])
            else:
                output_embeds = one_output_embeds

            layers_outputs.append(output_embeds)

        self.one_hop_layers = one_layers
        self.two_hop_layers = two_layers
        self.layers_outputs = layers_outputs
        model = tf.keras.Model(inputs=(pos_link, neg_link), outputs=(input_embeds, layers_outputs))
        return model

    def compute_loss(self, pos_links, neg_links, only_pos=False):
        index1 = pos_links[:, 0]
        index2 = pos_links[:, 1]
        neg_index1 = neg_links[:, 0]
        neg_index2 = neg_links[:, 1]
        embeds_list = list()
        for output_embeds in self.output_embeds_list + [self.input_embeds]:
            output_embeds = tf.nn.l2_normalize(output_embeds, 1)
            embeds_list.append(output_embeds)
        output_embeds = tf.concat(embeds_list, axis=1)
        output_embeds = tf.nn.l2_normalize(output_embeds, 1)

        embeds1 = tf.nn.embedding_lookup(output_embeds, tf.cast(index1, tf.int32))
        embeds2 = tf.nn.embedding_lookup(output_embeds, tf.cast(index2, tf.int32))
        pos_loss = tf.reduce_sum(tf.reduce_sum(tf.square(embeds1 - embeds2), 1))

        embeds1 = tf.nn.embedding_lookup(output_embeds, tf.cast(neg_index1, tf.int32))
        embeds2 = tf.nn.embedding_lookup(output_embeds, tf.cast(neg_index2, tf.int32))
        neg_distance = tf.reduce_sum(tf.square(embeds1 - embeds2), 1)
        neg_loss = tf.reduce_sum(tf.keras.activations.relu(self.neg_margin - neg_distance))

        return pos_loss + self.neg_param * neg_loss

    def compute_rel_loss(self, hs, ts):
        embeds_list = list()
        for output_embeds in self.output_embeds_list + [self.input_embeds]:
            output_embeds = tf.nn.l2_normalize(output_embeds, 1)
            embeds_list.append(output_embeds)
        output_embeds = tf.concat(embeds_list, axis=1)
        output_embeds = tf.nn.l2_normalize(output_embeds, 1)
        h_embeds = tf.nn.embedding_lookup(output_embeds, tf.cast(hs, tf.int32))
        t_embeds = tf.nn.embedding_lookup(output_embeds, tf.cast(ts, tf.int32))
        r_temp_embeds = tf.reshape(h_embeds - t_embeds, [-1, self.rel_win_size, output_embeds.shape[-1]])
        r_temp_embeds = tf.reduce_mean(r_temp_embeds, axis=1, keepdims=True)
        r_embeds = tf.tile(r_temp_embeds, [1, self.rel_win_size, 1])
        r_embeds = tf.reshape(r_embeds, [-1, output_embeds.shape[-1]])
        r_embeds = tf.nn.l2_normalize(r_embeds, 1)
        return tf.reduce_sum(tf.reduce_sum(tf.square(h_embeds - t_embeds - r_embeds), 1)) * self.rel_param

    @staticmethod
    def early_stop(flag1, flag2, flag):
        if flag <= flag2:
            return flag2, flag, True
        if flag < flag2 < flag1:
            return flag2, flag, True
        else:
            return flag2, flag, False

    def eval_embeds(self):
        self.reset_neighborhood()
        input_embeds, output_embeds = self.model((self.pos_link_batch, self.neg_link_batch), training=False)
        return input_embeds, output_embeds

    def augment(self):
        _, output_embeds_list = self.eval_embeds()
        embeds1 = tf.nn.embedding_lookup(output_embeds_list[-1], self.ref_ent1)
        embeds2 = tf.nn.embedding_lookup(output_embeds_list[-1], self.ref_ent2)
        embeds1 = tf.nn.l2_normalize(embeds1, 1)
        embeds2 = tf.nn.l2_normalize(embeds2, 1)
        embeds1 = embeds1.numpy()
        embeds2 = embeds2.numpy()
        print("calculate sim mat...")
        sim_mat = sim(embeds1, embeds2, csls_k=self.eval_csls)
        sim_mat = scipy.special.expit(sim_mat)
        th = self.sim_th
        print("sim th:", th)
        pair_index = find_alignment(sim_mat, th, 1)
        return pair_index, sim_mat

    def augment_neighborhood(self):
        pair_index, sim_mat = self.augment()
        if pair_index is None or len(pair_index) == 0:
            return
        self.new_links = update_labeled_alignment_x(self.new_links, pair_index, sim_mat)
        self.new_links = update_labeled_alignment_y(self.new_links, sim_mat)
        new_sup_ent1 = [self.ref_ent1[i] for i, _, in self.new_links]
        new_sup_ent2 = [self.ref_ent2[i] for _, i, in self.new_links]
        self.new_sup_links_set = set([(new_sup_ent1[i], new_sup_ent2[i]) for i in range(len(new_sup_ent1))])
        if new_sup_ent1 is None or len(new_sup_ent1) == 0:
            return
        enhanced_triples1, enhanced_triples2 = enhance_triples(self.kg1, self.kg2, self.sup_ent1 + new_sup_ent1,
                                                               self.sup_ent2 + new_sup_ent2)
        self.new_edges1 = enhanced_triples1
        self.new_edges2 = enhanced_triples2
        triples = self.kg1.triple_list + self.kg2.triple_list + list(self.new_edges1) + list(self.new_edges2)
        triples = remove_unlinked_triples(triples, self.linked_ents)
        one_adj, _ = no_weighted_adj(self.ent_num, triples, is_two_adj=False)
        adj = [one_adj]
        for layer in self.one_hop_layers:
            layer.update_adj(adj)
        del sim_mat
        gc.collect()

    def reset_neighborhood(self):
        for layer in self.one_hop_layers:
            layer.update_adj(self.ori_adj)

    def valid(self):
        embeds_list1, embeds_list2 = list(), list()
        input_embeds, output_embeds_list = self.eval_embeds()
        for output_embeds in [input_embeds] + output_embeds_list:
            output_embeds = tf.nn.l2_normalize(output_embeds, 1)
            embeds1 = tf.nn.embedding_lookup(output_embeds, self.ref_ent1)
            embeds2 = tf.nn.embedding_lookup(output_embeds, self.ref_ent2)
            embeds1 = tf.nn.l2_normalize(embeds1, 1)
            embeds2 = tf.nn.l2_normalize(embeds2, 1)
            embeds1 = embeds1.numpy()
            embeds2 = embeds2.numpy()
            embeds_list1.append(embeds1)
            embeds_list2.append(embeds2)
        embeds1 = np.concatenate(embeds_list1, axis=1)
        embeds2 = np.concatenate(embeds_list2, axis=1)
        alignment_rest, hits1_12, mr_12, mrr_12 = greedy_alignment(embeds1, embeds2, self.hits_k, self.eval_threads_num,
                                                                   self.eval_metric, False, 0, False)
        return hits1_12

    def test(self):
        embeds_list1, embeds_list2 = list(), list()
        input_embeds, output_embeds_list = self.eval_embeds()
        for output_embeds in [input_embeds] + output_embeds_list:
            output_embeds = tf.nn.l2_normalize(output_embeds, 1)
            embeds1 = tf.nn.embedding_lookup(output_embeds, self.ref_ent1)
            embeds2 = tf.nn.embedding_lookup(output_embeds, self.ref_ent2)
            embeds1 = tf.nn.l2_normalize(embeds1, 1)
            embeds2 = tf.nn.l2_normalize(embeds2, 1)
            embeds1 = embeds1.numpy()
            embeds2 = embeds2.numpy()
            embeds_list1.append(embeds1)
            embeds_list2.append(embeds2)
        embeds1 = np.concatenate(embeds_list1, axis=1)
        embeds2 = np.concatenate(embeds_list2, axis=1)
        alignment_rest, _, _, _ = greedy_alignment(embeds1, embeds2, self.hits_k, self.eval_threads_num,
                                                   self.eval_metric, False, 0, True)
        alignment_rest, _, _, _ = greedy_alignment(embeds1, embeds2, self.hits_k, self.eval_threads_num,
                                                   self.eval_metric, False, self.eval_csls, True)

    def generate_input_batch(self, batch_size, neighbors1=None, neighbors2=None):
        if batch_size > len(self.sup_ent1):
            batch_size = len(self.sup_ent1)
        index = np.random.choice(len(self.sup_ent1), batch_size)
        pos_links = self.sup_links[index,]
        neg_links = list()
        if neighbors1 is None:
            neg_ent1 = list()
            neg_ent2 = list()
            for i in range(self.neg_multi):
                neg_ent1.extend(random.sample(self.sup_ent1 + self.ref_ent1, batch_size))
                neg_ent2.extend(random.sample(self.sup_ent2 + self.ref_ent2, batch_size))
            neg_links.extend([(neg_ent1[i], neg_ent2[i]) for i in range(len(neg_ent1))])
        else:
            for i in range(batch_size):
                e1 = pos_links[i, 0]
                candidates = random.sample(neighbors1.get(e1), self.neg_multi)
                neg_links.extend([(e1, candidate) for candidate in candidates])
                e2 = pos_links[i, 1]
                candidates = random.sample(neighbors2.get(e2), self.neg_multi)
                neg_links.extend([(candidate, e2) for candidate in candidates])
        neg_links = set(neg_links) - self.sup_links_set
        neg_links = neg_links - self.new_sup_links_set
        neg_links = np.array(list(neg_links))
        return pos_links, neg_links

    def generate_rel_batch(self):
        hs, rs, ts = list(), list(), list()
        for r, hts in self.rel_ht_dict.items():
            hts_batch = [random.choice(hts) for _ in range(self.rel_win_size)]
            for h, t in hts_batch:
                hs.append(h)
                ts.append(t)
                rs.append(r)
        return hs, rs, ts

    def find_neighbors(self):
        if self.truncated_epsilon <= 0.0:
            return None, None
        start = time.time()
        input_embeds, output_embeds_list = self.eval_embeds()
        ents1 = self.sup_ent1 + self.ref_ent1
        ents2 = self.sup_ent2 + self.ref_ent2
        embeds1 = tf.nn.embedding_lookup(output_embeds_list[-1], ents1)
        embeds2 = tf.nn.embedding_lookup(output_embeds_list[-1], ents2)
        embeds1 = tf.nn.l2_normalize(embeds1, 1)
        embeds2 = tf.nn.l2_normalize(embeds2, 1)
        embeds1 = embeds1.numpy()
        embeds2 = embeds2.numpy()
        num = int((1-self.truncated_epsilon) * len(ents1))
        print("neighbors num", num)
        neighbors1 = generate_neighbours(embeds1, ents1, embeds2, ents2, num, threads_num=self.eval_threads_num)
        neighbors2 = generate_neighbours(embeds2, ents2, embeds1, ents1, num, threads_num=self.eval_threads_num)
        print('finding neighbors for sampling costs time: {:.4f}s'.format(time.time() - start))
        del embeds1, embeds2
        gc.collect()
        return neighbors1, neighbors2

    def train(self, batch_size, max_epochs=1000, start_valid=10, eval_freq=10):
        flag1 = 0
        flag2 = 0
        steps = len(self.sup_ent2) // batch_size
        neighbors1, neighbors2 = None, None
        if steps == 0:
            steps = 1
        for epoch in range(1, max_epochs + 1):
            start = time.time()
            epoch_loss = 0.0
            for step in range(steps):
                self.pos_link_batch, self.neg_link_batch = self.generate_input_batch(batch_size,
                                                                                     neighbors1=neighbors1,
                                                                                     neighbors2=neighbors2)
                with tf.GradientTape() as tape:
                    self.input_embeds, self.output_embeds_list = self.model((self.pos_link_batch, self.neg_link_batch),
                                                                            training=True)
                    batch_loss = self.compute_loss(self.pos_link_batch, self.neg_link_batch)
                    if self.rel_param > 0.0:
                        hs, _, ts = self.generate_rel_batch()
                        rel_loss = self.compute_rel_loss(hs, ts)
                        batch_loss += rel_loss
                    grads = tape.gradient(batch_loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                    epoch_loss += batch_loss
            print('epoch {}, loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))
            if epoch % eval_freq == 0 and epoch >= start_valid:
                flag = self.valid()
                flag1, flag2, is_stop = self.early_stop(flag1, flag2, flag)
                if is_stop:
                    print("\n == training stop == \n")
                    break
                neighbors1, neighbors2 = self.find_neighbors()
                if epoch >= self.start_augment * eval_freq:
                    if self.sim_th > 0.0:
                        self.augment_neighborhood()
