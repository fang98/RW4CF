

import argparse
from config import const_args
from utils import setup_seed,read_nonid_dataset,generate_node2vec_embeddings,test
import warnings


# ------------ argument ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='GNU', dest = 'dataset', help='dataset name')#GNU,Wiki,JUNG,Ciao
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
parser.add_argument('--split_type', type=str, default='in')#in,out
parser.add_argument('--target', type=str, default='u')#u,b

args = parser.parse_args()

args_dict = vars(args)
args_dict.update(const_args)
args = argparse.Namespace(**args_dict)


# ------------- training & evaluation ---------------
def run(args):
    warnings.filterwarnings('ignore')
    print(args)
    
    setup_seed(seed=args.seed)
    num_nodes, adj_train, train_edges, test_edges = \
        read_nonid_dataset(args.dataset,args.seed,args.split_type,args.target)
    z = generate_node2vec_embeddings(args,num_nodes,adj_train)
    res = test(args,1,z,train_edges, test_edges, None)
    
    return res[0]


if __name__ == '__main__':
    auc = run(args)
    print('Link Prediction Result (AUC) =',auc)