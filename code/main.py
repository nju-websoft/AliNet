import argparse

from alinet import AliNet
from align.util import gcn_load_data

parser = argparse.ArgumentParser(description='AliNet')
parser.add_argument('--input', type=str, default="../data/DBP15K/zh_en/mtranse/0_3/")
parser.add_argument('--output', type=str, default='../output/results/')

parser.add_argument('--embedding_module', type=str, default='AliNet')
parser.add_argument('--layer_dims', type=list, default=[500, 400, 300])  # [300, 200, 100] for two layers
parser.add_argument('--num_features_nonzero', type=float, default=0.0)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--activation', type=str, default="tanh")

parser.add_argument('--neg_multi', type=int, default=10)  # for negative sampling
parser.add_argument('--neg_margin', type=float, default=1.5)  # margin value for negative loss
parser.add_argument('--neg_param', type=float, default=0.1)  # weight for negative loss
parser.add_argument('--truncated_epsilon', type=float, default=0.98)  # epsilon for truncated negative sampling

parser.add_argument('--batch_size', type=int, default=4500)
parser.add_argument('--max_epoch', type=int, default=1000)

parser.add_argument('--is_save', type=bool, default=False)
parser.add_argument('--start_valid', type=int, default=10)
parser.add_argument('--start_augment', type=int, default=4)

parser.add_argument('--eval_metric', type=str, default='inner')
parser.add_argument('--hits_k', type=list, default=[1, 5, 10, 50])
parser.add_argument('--eval_threads_num', type=int, default=10)
parser.add_argument('--eval_normalize', type=bool, default=True)
parser.add_argument('--eval_csls', type=int, default=2)
parser.add_argument('--eval_freq', type=int, default=5)
parser.add_argument('--adj_number', type=int, default=1)
parser.add_argument('--sim_th', type=float, default=0.5)


class ModelFamily(object):
    AliNet = AliNet


def get_model(model_name):
    return getattr(ModelFamily, model_name)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    is_two = True
    adj, kg1, kg2, sup_ent1, sup_ent2, ref_ent1, ref_ent2, tri_num, ent_num, rel_num, rel_id_mapping = \
        gcn_load_data(args.input, is_two=is_two)
    linked_entities = set(sup_ent1 + sup_ent2 + ref_ent1 + ref_ent2)
    gcn_model = get_model(args.embedding_module)(adj, kg1, kg2, sup_ent1, sup_ent2, ref_ent1, ref_ent2,
                                                 tri_num, ent_num, rel_num, args)
    gcn_model.train(args.batch_size, max_epochs=args.max_epoch, start_valid=args.start_valid, eval_freq=args.eval_freq)
    gcn_model.test()

