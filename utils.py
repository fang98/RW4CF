
import random
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from model import Node2Vec, classifier
import sys
from torch.utils.data import TensorDataset,DataLoader
from torch import nn
from torch import optim
import os


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def read_dataset(dataset,seed):
    traindi = np.loadtxt('./Dataset/large/edge_data/'+dataset+'/'+str(seed)+'/train_di.txt', dtype=int, delimiter=',')
    
    num_nodes = np.max(traindi[:,:2])+1
    G2 = nx.DiGraph()
    G2.add_nodes_from(list(range(num_nodes)))
    G2.add_edges_from(traindi[:,:2])
    adj_train = nx.adjacency_matrix(G2)
    
    return num_nodes, adj_train
    

def generate_node2vec_embeddings(args,num_nodes,adj_train):
    adj_train.eliminate_zeros()
    adj_coo = adj_train.tocoo()
    edge_index = np.hstack([np.array([adj_coo.row,adj_coo.col+num_nodes]),
                            np.array([adj_coo.col+num_nodes,adj_coo.row])])
    edge_index = torch.tensor(edge_index,dtype=torch.long).to(args.device)
    
    model = Node2Vec(
        edge_index=edge_index,
        embedding_dim=args.emb_dim,
        walk_length=args.walk_length,
        context_size=args.context_size,
        device = args.device,
        walks_per_node=args.walks_per_node,
        p=1.0,
        q=1.0,
        num_negative_samples=args.num_negative_samples,
        num_nodes=num_nodes*2,
        alpha=args.alpha).to(args.device)
    
    num_workers = 4 if sys.platform == 'linux' else 0
    loader = model.loader(batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    
    for epoch in tqdm(range(args.epochs), desc="node2vec training", unit="epoch"):
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(args.device), neg_rw.to(args.device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # avg_loss = total_loss / len(loader)
    
    model.eval()
    z = model()
    z = z.detach().cpu().numpy()
    return z


def read_train_test_val_edges(dataset,i_loop):
    train_edges1 = np.loadtxt('./Dataset/large/edge_data/'+dataset+'/'+str(i_loop)+'/train_1.txt', dtype=int, delimiter=',')
    train_edges2 = np.loadtxt('./Dataset/large/edge_data/'+dataset+'/'+str(i_loop)+'/train_2.txt', dtype=int, delimiter=',')
    train_edges3 = np.loadtxt('./Dataset/large/edge_data/'+dataset+'/'+str(i_loop)+'/train_3.txt', dtype=int, delimiter=',')
    
    test_edges1 = np.loadtxt('./Dataset/large/edge_data/'+dataset+'/'+str(i_loop)+'/test_1.txt', dtype=int, delimiter=',')
    test_edges2 = np.loadtxt('./Dataset/large/edge_data/'+dataset+'/'+str(i_loop)+'/test_2.txt', dtype=int, delimiter=',')
    test_edges3 = np.loadtxt('./Dataset/large/edge_data/'+dataset+'/'+str(i_loop)+'/test_3.txt', dtype=int, delimiter=',')
    
    val_edges1 = np.loadtxt('./Dataset/large/edge_data/'+dataset+'/'+str(i_loop)+'/val_1.txt', dtype=int, delimiter=',')
    val_edges2 = np.loadtxt('./Dataset/large/edge_data/'+dataset+'/'+str(i_loop)+'/val_2.txt', dtype=int, delimiter=',')
    val_edges3 = np.loadtxt('./Dataset/large/edge_data/'+dataset+'/'+str(i_loop)+'/val_3.txt', dtype=int, delimiter=',')
    
    train_edges = [train_edges1,train_edges2,train_edges3]
    test_edges = [test_edges1,test_edges2,test_edges3]
    val_edges = [val_edges1,val_edges2,val_edges3]
    
    return train_edges, test_edges, val_edges



def test(args,task,z,train_edges, test_edges, val_edges):
    if not os.path.exists('./classifier_model'):
        os.makedirs('./classifier_model')
    epochs_mlp = 10
    bs_mlp = 512
    num_nodes = int(z.shape[0]/2)
    
    train_features = np.hstack([z[:num_nodes,:][train_edges[:,0],:],
                                z[:num_nodes,:][train_edges[:,1],:],
                                z[num_nodes:,:][train_edges[:,0],:],
                                z[num_nodes:,:][train_edges[:,1],:]])
    test_features = np.hstack([z[:num_nodes,:][test_edges[:,0],:],
                               z[:num_nodes,:][test_edges[:,1],:],
                               z[num_nodes:,:][test_edges[:,0],:],
                               z[num_nodes:,:][test_edges[:,1],:]])
    train_labels = train_edges[:,2]
    test_labels = test_edges[:,2]
    
    train_features = torch.tensor(train_features,dtype=torch.float).to(args.device)
    train_labels = torch.LongTensor(train_labels).to(args.device)
    train_data = TensorDataset(train_features,train_labels)
    train_loader = DataLoader(train_data, batch_size=bs_mlp, shuffle=True)
    
    test_features = torch.tensor(test_features,dtype=torch.float).to(args.device)
    test_labels = torch.LongTensor(test_labels).to(args.device)
    test_data = TensorDataset(test_features,test_labels)
    test_loader = DataLoader(test_data, batch_size=bs_mlp, shuffle=False)

    if val_edges is not None:
        val_features = np.hstack([z[:num_nodes,:][val_edges[:,0],:],
                                  z[:num_nodes,:][val_edges[:,1],:],
                                  z[num_nodes:,:][val_edges[:,0],:],
                                  z[num_nodes:,:][val_edges[:,1],:]])
        val_labels = val_edges[:,2]
        val_features = torch.tensor(val_features,dtype=torch.float).to(args.device)
        val_labels = torch.LongTensor(val_labels).to(args.device)
        val_data = TensorDataset(val_features,val_labels)
        val_loader = DataLoader(val_data, batch_size=bs_mlp, shuffle=False)
    
    lr = 0.001
    upd = 1
    fea_dim = train_features.shape[1]
    if task==3:
        out_dim = 4
    else:
        out_dim = 2
    mlp = classifier(fea_dim, 32, 16,out_dim).to(args.device)
    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    
    best_res = 0
    for epoch in range(epochs_mlp):
        mlp.train()
        total_loss = []
        n_samples = 0
        
        for features,labels in train_loader:
            out = mlp(features)
            loss = loss_func(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append( loss.item() * len(labels))
            n_samples += len(labels)
        total_loss = np.array(total_loss)
        # avg_loss = np.sum(total_loss, 0) / n_samples
        if (epoch + 1) % upd == 0:
            upt_res = evaluate(mlp,test_loader,args.device,task)
            if val_edges is not None:
                upt_res = evaluate(mlp,val_loader,args.device,task)
                if upt_res[0]+upt_res[1] > best_res:
                    torch.save(obj=mlp.state_dict(), f='classifier_model/mlp.pth')
                    best_res = upt_res[0]+upt_res[1]
    if val_edges is not None:
        new_model = classifier(fea_dim, 32, 16,out_dim).to(args.device)
        new_model.load_state_dict(torch.load('classifier_model/mlp.pth'))
        upt_res = evaluate(new_model,test_loader,args.device,task)
    else:
        upt_res = evaluate(mlp,test_loader,args.device,task)
    
    return upt_res


def test_in_large_graph(args,z,num_nodes):
    auc_all = np.zeros(3)
    acc_all = np.zeros(3)
    train_edges, test_edges, val_edges = read_train_test_val_edges(args.dataset,args.seed)
    for task in [1,2,3]:
        upt_res = test(args,task,z,train_edges[task-1], test_edges[task-1], val_edges[task-1])
        auc_all[task-1] = upt_res[0]
        acc_all[task-1] = upt_res[1]
    return auc_all,acc_all



def evaluate(model,loader,device,task):
    model.eval()
    all_targets = []
    all_scores = []
    
    if task in [1,2]:
        for features,labels in loader:
            out = model(features)
            all_scores.append(F.softmax(out,dim=1)[:, 1].cpu().detach())
            all_targets.extend(labels.tolist())
        all_scores = torch.cat(all_scores).cpu().numpy()
        pred_labels = np.zeros(np.size(all_scores))
        pred_labels[np.where(all_scores>0.5)] = 1
        auc = roc_auc_score(all_targets,all_scores)
        # ap = average_precision_score(all_targets,all_scores)
        acc = accuracy_score(all_targets,pred_labels)
    else:
        for features,labels in loader:
            out = model(features)
            all_scores.append(F.softmax(out,dim=1)[:, :4].cpu().detach())
            all_targets.extend(labels.tolist())
        all_scores = torch.cat(all_scores).cpu().numpy()
        pred_labels = all_scores.argmax(1)
        auc = 0.0
        # auc = roc_auc_score(all_targets,all_scores)
        # ap = average_precision_score(all_targets,all_scores)
        acc = accuracy_score(all_targets,pred_labels)
        
    return auc,acc


def str_list_to_int(str_list):
    return [int(item) for item in str_list]


def read_edges_from_file(filename): # all columns
    with open(filename, "r") as f: 
        lines = f.readlines()
        edges = [str_list_to_int(line.split()) for line in lines]
    return edges


def read_nonid_dataset(dataset,seed,split_type,target):
    dict_dataname = {"GNU":'p2p-Gnutella08',"Wiki":'wiki-vote',"JUNG":'jung',"Ciao":'ciaodvd'}
    dataset = dict_dataname[dataset]
    fold_id = 'u'+str(seed+1)
    split_type = 'noniid-'+split_type+'-barrier_1.0'
    file_path = "./Dataset/nonid/"+dataset+"/"+split_type+"/"+fold_id+"/"+fold_id
    train_filename = file_path+".edgelist"
    test_filename = file_path + "_test.edgelist"
    if target == 'u':
        target = 'LP-uniform'
    elif target == 'b':
        target = 'LP-biased'
    else:
        target = 'LP-mixed'

    if "LP-uniform" == target:
        train_uncon_filename = file_path + "_unconnected_ulp_train_1times.edgelist"
        test_uncon_filename = file_path + "_unconnected_ulp_test_1times.edgelist"
    elif "LP-mixed" == target:
        train_uncon_filename = file_path + "_unconnected_mlp_train_1times.edgelist"
        test_uncon_filename = file_path + "_unconnected_mlp_test_1times.edgelist"
    elif "LP-biased" == target:
        train_uncon_filename = file_path + "_unconnected_blp_train_1times.edgelist"
        test_uncon_filename = file_path + "_unconnected_blp_test_1times.edgelist"
    
    test_edges = read_edges_from_file(test_filename)
    train_edges = read_edges_from_file(train_filename)
    train_non_edges = read_edges_from_file(train_uncon_filename)
    test_non_edges = read_edges_from_file(test_uncon_filename)
    
    train_non_edges = [[u,v] for u,v,l in train_non_edges]
    test_non_edges = [[u,v] for u,v,l in test_non_edges]
    
    nodes = set()
    for edge in train_edges:#+test_edges+train_non_edges+test_non_edges
        nodes = nodes.union(set(edge[:2]))
    num_nodes = len(nodes)
    
    G2 = nx.DiGraph()
    G2.add_nodes_from(nodes)
    G2.add_edges_from(train_edges)
    adj_train = nx.adjacency_matrix(G2)
    
    train_edges = [[u,v,1] for u,v in train_edges]+[[u,v,0] for u,v in train_non_edges]
    test_edges = [[u,v,1] for u,v in test_edges]+[[u,v,0] for u,v in test_non_edges]
    train_edges = np.array(train_edges)
    test_edges = np.array(test_edges)
    
    return num_nodes, adj_train, train_edges, test_edges


