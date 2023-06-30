import dgl
import numpy as np
import torch


from dgl.nn.pytorch.conv import GATConv, GraphConv, TAGConv, GINConv,TWIRLSConv



from dgl.nn.pytorch.glob import MaxPooling, GlobalAttentionPooling


#这里将邻接矩阵的源节点和目标节点分别拿出来，最后给出节点数量（这个参数在所有节点都在源节点和目标节点集合里面的时候可以省略）
g = dgl.graph(([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]), num_nodes=6)
# Equivalently, PyTorch LongTensors also work.
g = dgl.graph((torch.LongTensor([0, 0, 0, 0, 0]), torch.LongTensor([1, 2, 3, 4, 5])), num_nodes=6)

# You can omit the number of nodes argument if you can tell the number of nodes from the edge list alone.
g = dgl.graph(([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]))
# Print the source and destination nodes of every edge.
print(g.edges())
