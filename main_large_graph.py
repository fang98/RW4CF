

import argparse
from config import const_args
from utils import setup_seed,read_dataset,generate_node2vec_embeddings,test_in_large_graph
import warnings


# ------------ argument ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='epinions', dest = 'dataset', help='dataset name')#epinions,twitter
parser.add_argument('--seed', type=int, default=0, dest = 'seed', help='data seed')
parser.add_argument('--learning_rate', type=float, default=1e-2)
parser.add_argument('--batch_size',type=int, default=256)
parser.add_argument('--epochs',type=int, default=10)
parser.add_argument('--emb_dim',type=int, default=64)
parser.add_argument('--walk_length',type=int, default=25)
parser.add_argument('--context_size',type=int, default=5)
parser.add_argument('--walks_per_node',type=int, default=10)
parser.add_argument('--num_negative_samples',type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.8)
args = parser.parse_args()

args_dict = vars(args)
args_dict.update(const_args)
args = argparse.Namespace(**args_dict)


# ------------- training & evaluation ---------------
def run(args):
    warnings.filterwarnings('ignore')
    print(args)
    setup_seed(seed=args.seed)
    
    num_nodes, adj_train = read_dataset(args.dataset,args.seed)
    z = generate_node2vec_embeddings(args,num_nodes,adj_train)
    auc_all,acc_all = test_in_large_graph(args,z,num_nodes)
    
    return auc_all,acc_all


if __name__ == '__main__':
    auc_all,acc_all = run(args)
    print('Existence Prediction (AUC) =',auc_all[0])
    print('Existence Prediction (acc) =',acc_all[0])
    print('Direction Prediction (AUC) =',auc_all[1])
    print('Direction Prediction (acc) =',acc_all[1])
    print('Four-type Classification (acc) =',acc_all[2])