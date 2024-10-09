
import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GCNConv

def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)



def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
    """
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

class LGCN(nn.Module):
    def __init__(
        self,
        hops, 
        input_dim,
        n_layers=6,
        num_heads=8,
        hidden_dim=64,
        dropout_rate=0.0,
        attention_dropout_rate=0.1
    ):
        super().__init__()

        self.seq_len = hops+1
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = 2 * hidden_dim
        self.num_heads = num_heads
        
        self.n_layers = n_layers

        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.att_embeddings_nope = nn.Linear(self.input_dim, self.hidden_dim)
        
        encoders = [GCNConv(self.hidden_dim, self.hidden_dim)  for _ in range(self.n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

        self.out_proj = nn.Linear(self.hidden_dim, int(self.hidden_dim/2))

        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)

        # self.Linear1 = nn.Linear(int(self.hidden_dim/2), self.n_class)

        self.scaling = nn.Parameter(torch.ones(1) * 0.5)


        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):

        # print(batched_data.shape)
        tensor = self.att_embeddings_nope(batched_data)
        
        # Reshape the 3D tensor to 2D for GCNConv: (num_graphs * num_nodes_per_graph, num_features_per_node)
        node_features_2d = tensor.view(-1, tensor.shape[2])

        # Repeat the edge_index for each graph in the batch, adjusting node indices
        edge_indices = []
        for i in range(tensor.shape[1]):
           offset = i * tensor.shape[0]
           adjusted_edge_index = edge_index + offset  # Shift the node indices for each graph
           edge_indices.append(adjusted_edge_index)

        # Concatenate all edge indices for the batched graph
        batched_edge_index = torch.cat(edge_indices, dim=1)
        
        for enc_layer in self.layers:
           gcn_output = enc_layer(node_features_2d, batched_edge_index)
        
        tensor = gcn_output.view(tensor.shape[0], tensor.shape[1], -1)
        
        for enc_layer in self.layers:
            tensor = enc_layer(tensor)

        output = self.final_ln(tensor)

        # print(output.shape)
        target = output[:,0,:].unsqueeze(1).repeat(1,self.seq_len-1,1)
        #print(output.shape)
        # target = output.repeat(1, self.seq_len-1,1)
        split_tensor = torch.split(output, [1,self.seq_len-1], dim=1)

        node_tensor = split_tensor[0]
        neighbor_tensor = split_tensor[1]
    
        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))
        
        layer_atten = F.softmax(layer_atten, dim=1)
    
        neighbor_tensor = neighbor_tensor * layer_atten
        
        return node_tensor, neighbor_tensor
    