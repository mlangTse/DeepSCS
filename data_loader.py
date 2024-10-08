import utils
import dgl
import torch
import scipy.sparse as sp

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import  AmazonCoBuyPhotoDataset,CoauthorCSDataset,CoauthorPhysicsDataset



def get_dataset(dataset, pe_dim):
    file_path = "dataset/"+dataset+"_pyg.pt"

    data_list = torch.load(file_path)
    
    adj = data_list[0]
    print(len(adj))
    features = data_list[1]
    
    adj_scipy = utils.torch_adj_to_scipy(adj)
    graph = dgl.from_scipy(adj_scipy)
    lpe = utils.laplacian_positional_encoding(graph, pe_dim)
    # print(len(features), len(lpe), len(features[0]))
#        features = torch.cat((features, lpe), dim=1)
    # print(len(features[0]))

    # print(type(adj), type(features))
    
    return adj.cpu().type(torch.LongTensor), features.long()




