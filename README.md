# Random Walk based Hierarchical Collaborative Filtering for Directed Network Embedding

Implementation of [Random Walk based Hierarchical Collaborative Filtering for Directed Network Embedding](https://ieeexplore.ieee.org/document/11410006/).

## Requirements
- Python 3.7
- PyTorch 1.8.1
- CUDA 11.1
- dgl 0.9.1
- torch_geometric 2.0.2
- torch_cluster 1.5.9


## Small-scale Networks

Instructions for reproducing the experiments reported in GNU, Wiki, JUNG and Ciao datasets.

### Training & Evaluation

To conduct experiments for link prediction in small-scale directed networks, execute `python main_small_graph.py`.

Example command:
```bash
python main_small_graph.py --dataset=GNU --epochs=10
``` 


## Large-scale Networks

Instructions for reproducing the experiments reported in Epinions and Twitter networks.

### Data Preparation

To preprocess data, follow these steps:

1. Execute `python preprocessing.py` to create the train/validation/test sets for the link prediction task.

### Training & Evaluation

To conduct experiments for link prediction in large-scale directed networks, execute `python main_large_graph.py`.

Example command:
```bash
python main_large_graph.py --dataset=epinions
``` 


## Acknowlegdements

Part of code borrow from [https://github.com/alipay/DUPLEX](https://github.com/alipay/DUPLEX), [https://github.com/hsyoo32/odin](https://github.com/hsyoo32/odin) and [https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/node2vec.py](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/node2vec.py). Thanks for their excellent work!
