import os 
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
from scipy import stats
from torch_geometric.nn import GCNConv, GINConv, GATConv
from torch_geometric.data import Data 


class InputEncoder(nn.Module):
    def __init__(self, 
                 num_states=3, 
                 num_stations=16, 
                 embedding_dim=256, 
                 noise_std=0.1, 
                 use_noise=True):
        super().__init__()
        self.expr_embedding = nn.Embedding(num_states, embedding_dim)
        self.pos_embedding = nn.Embedding(num_stations, embedding_dim)
        self.noise_std = noise_std 
        self.use_noise = use_noise 
        self.num_stations = num_stations

    def forward(self, X):
        expr_embedded = self.expr_embedding((X + 1).long())
        pos_embedded = self.pos_embedding(torch.arange(self.num_stations).to(X.device))
        X_embedded = expr_embedded + pos_embedded
        if self.use_noise and self.training:
            noise = torch.randn_like(X_embedded) * self.noise_std
            X_embedded = X_embedded + noise 
        return X_embedded
    
class TransformerEncoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 num_encoder_layers=6, 
                 hidden_dim=256, 
                 num_heads=8,
                 dropout=0.1):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_encoder_layers
        )
        
    def forward(self, X, M):
        """
        X: (B, N, C)
        M: (B, N)
        """
        mask = M == 0  
        encoded_X = self.transformer_encoder(X, src_key_padding_mask=mask)
        return encoded_X  # (B, N, C)

class SelfAttentionDecoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim=256,
                 num_heads=8,
                 num_decoder_layers=6,
                 dropout=0.1):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=input_dim,
                                                         num_heads=num_heads,
                                                         dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.num_decoder_layers = num_decoder_layers

    def forward(self, X):
        for _ in range(self.num_decoder_layers):
            attn_output, _ = self.multihead_attention(X, X, X)
            X = self.layer_norm1(X + self.dropout(attn_output))
            ffn_output = self.ffn(X) 
            X = self.layer_norm2(X + self.dropout(ffn_output))
        return X

class TransformerDecoder(nn.Module):
    def __init__(self,
                 input_dim,
                 num_decoder_layers=6,
                 hidden_dim=256,
                 num_heads=8,
                 dropout=0.1):
        super().__init__()
        self.decoder = SelfAttentionDecoder(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            num_heads=num_heads, 
            num_decoder_layers=num_decoder_layers, 
            dropout=dropout)
        
    def forward(self, X):
        decoder_X = self.decoder(X)
        return decoder_X
    
class DKiT(nn.Module):
    def __init__(self, 
                 num_states=3, 
                 num_stations=16, 
                 embedding_dim=256, 
                 noise_std=0.1, 
                 use_noise=True, 
                 num_encoder_layers=6, 
                 num_decoder_layers=6,
                 hidden_dim=256, 
                 num_heads=8,
                 dropout=0.1,
                 num_classes=2):
        super().__init__()
        self.encoder = InputEncoder(num_states, num_stations, embedding_dim, noise_std, use_noise)
        self.transformer_encoder = TransformerEncoder(embedding_dim, num_encoder_layers, hidden_dim, num_heads, dropout=dropout)
        self.decoder = TransformerDecoder(embedding_dim, num_decoder_layers, hidden_dim, num_heads, dropout=dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, X):
        M = (X >= 0).float()
        X_encoded = self.encoder(X)
        X_encoded = self.transformer_encoder(X_encoded, M)
        X_decoded = self.decoder(X_encoded)
        pred = self.classifier(X_decoded)
        return pred 
    
class GCN(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, mid_channels) 
        self.conv2 = GCNConv(mid_channels, out_channels) 

    def forward(self, data):
        x, edge_index = data.x, data.edge_index 
        x = self.conv1(x, edge_index) 
        x = torch.relu(x)  
        x = self.conv2(x, edge_index) 
        return x
    
class GIN(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, mid_channels)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(mid_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.conv1 = GINConv(self.mlp1)
        self.conv2 = GINConv(self.mlp2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
class GAT(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, heads, dropout):
        super().__init__()
        self.conv1 = GATConv(in_channels, mid_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(mid_channels * heads, out_channels, heads=1, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
class GraphGCN(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(in_channels, mid_channels))
            elif i == num_layers - 1:
                self.convs.append(GCNConv(mid_channels, out_channels))
            else:
                self.convs.append(GCNConv(mid_channels, mid_channels))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
        return x
    
class GraphModel(nn.Module):
    def __init__(self, 
                 num_states=3, 
                 num_stations=16, 
                 embedding_dim=256, 
                 noise_std=0.1, 
                 use_noise=True, 
                 num_heads=8,
                 dropout=0.1,
                 num_classes=2,
                 gnn_name='GCN',
                 gnn_adjmatrix=None,
                 gnn_in_channels=256,
                 gnn_mid_channels=256,
                 gnn_out_channels=256):
        super().__init__()
        self.encoder = InputEncoder(num_states, num_stations, embedding_dim, noise_std, use_noise)
        self.classifier = nn.Linear(gnn_out_channels, num_classes)
        assert "+" not in gnn_adjmatrix, "Only one graph is supported"
        A = pd.read_csv(os.path.join("./graphs", gnn_adjmatrix+".csv")).values[:, 1:] # (N, N)
        A = torch.FloatTensor(A.astype(np.float32)).nonzero(as_tuple=False).t()
        self.A = A 
        if gnn_name == "GAT":
            self.gnn = eval(f"Graph{gnn_name}")(gnn_in_channels, gnn_mid_channels, gnn_out_channels, 6, num_heads, dropout)
        else:
            self.gnn = eval(f"Graph{gnn_name}")(gnn_in_channels, gnn_mid_channels, gnn_out_channels, 6)

    def forward_graph(self, X_encoded):
        data_list = []
        edge_index = self.A.to(X_encoded.device)
        for b in range(X_encoded.shape[0]):
            x = X_encoded[b]
            data_list.append(Data(x=x, edge_index=edge_index))
        gnn_X = []
        for data in data_list:
            gnn_X.append(self.gnn(data))
        gnn_X = torch.stack(gnn_X, dim=0)  # (B, N, C)
        return gnn_X 

    def forward(self, X):
        X_encoded = self.encoder(X)
        X_encoded = self.forward_graph(X_encoded)
        pred = self.classifier(X_encoded)
        return pred

class GraphDecoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim=256,
                 num_heads=8,
                 num_decoder_layers=6,
                 dropout=0.1,
                 num_graphs=1):
        super().__init__()
        self.self_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=input_dim,
                                  num_heads=num_heads,
                                  dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.cross_attention = nn.ModuleList([
            nn.ModuleList([ 
                nn.MultiheadAttention(embed_dim=input_dim,
                                      num_heads=num_heads,
                                      dropout=dropout)
                for _ in range(num_decoder_layers)
            ])
            for _ in range(num_graphs)
        ])
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.layer_norm3 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.num_decoder_layers = num_decoder_layers

        self.sigmoid_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )
        self.tanh_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
            nn.Tanh(),
        )

    def forward(self, X, G_lst):
        # X: (B, N, C) | G: [(B, N, C),]
        for i in range(self.num_decoder_layers):
            cross_attn_output = []
            for j, G in enumerate(G_lst):
                cross_attn_output_j, _ = self.cross_attention[j][i](X, G, G)
                cross_attn_output.append(cross_attn_output_j)
            cross_attn_output = torch.stack(cross_attn_output, dim=0) # (G, B, N, C)
            sigmoid_fc = self.sigmoid_fc(cross_attn_output).squeeze(-1) # (G, B, N)
            tanh_fc = self.tanh_fc(cross_attn_output).squeeze(-1) # (G, B, N)
            exp_score = torch.exp(sigmoid_fc * tanh_fc) # (G, B, N)
            exp_score_norm = exp_score / exp_score.sum(dim=0, keepdim=True) # (G, B, N)
            cross_attn_output = (exp_score_norm.unsqueeze(-1) * cross_attn_output).sum(dim=0) # (B, N, C)

            X = self.layer_norm1(X + cross_attn_output)
            
            self_attn_output, _ = self.self_attention[i](X, X, X)
            X = self.layer_norm2(X + self.dropout(self_attn_output))
        
            ffn_output = self.ffn(X)
            X = self.layer_norm3(X + self.dropout(ffn_output))
        return X
    
class DKiT_GNN(DKiT):
    def __init__(self, 
                 num_states=3, 
                 num_stations=16, 
                 embedding_dim=256, 
                 noise_std=0.1, 
                 use_noise=True, 
                 num_encoder_layers=6, 
                 num_decoder_layers=6,
                 hidden_dim=256, 
                 num_heads=8,
                 dropout=0.1,
                 num_classes=2,
                 gnn_name='GCNModel',
                 gnn_adjmatrix=None,
                 gnn_in_channels=256,
                 gnn_mid_channels=256,
                 gnn_out_channels=256):
        super().__init__()
        self.encoder = InputEncoder(num_states, num_stations, embedding_dim, noise_std, use_noise)
        self.transformer_encoder = TransformerEncoder(embedding_dim, num_encoder_layers, hidden_dim, num_heads, dropout=dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.A_lst = []
        for graph in gnn_adjmatrix.split("+"):
            A = pd.read_csv(os.path.join("./graphs", graph+".csv")).values[:, 1:] # (N, N)
            A = torch.FloatTensor(A.astype(np.float32)).nonzero(as_tuple=False).t()
            self.A_lst.append(A)
        self.decoder = GraphDecoder(embedding_dim, hidden_dim, num_heads, num_decoder_layers, dropout=dropout, num_graphs=len(self.A_lst))
        if gnn_name == "GAT":
            self.gnn = nn.ModuleList([
                eval(gnn_name)(gnn_in_channels, gnn_mid_channels, gnn_out_channels, num_heads, dropout) for _ in range(len(self.A_lst))
            ])
        else:
            self.gnn = nn.ModuleList([
                eval(gnn_name)(gnn_in_channels, gnn_mid_channels, gnn_out_channels) for _ in range(len(self.A_lst))
            ])

    def forward_graph(self, X_encoded, A, gnn):
        data_list = []
        edge_index = A.to(X_encoded.device)
        for b in range(X_encoded.shape[0]):
            x = X_encoded[b]
            data_list.append(Data(x=x, edge_index=edge_index))
        gnn_X = []
        for data in data_list:
            gnn_X.append(gnn(data))
        gnn_X = torch.stack(gnn_X, dim=0)  # (B, N, C)
        return gnn_X 

    def forward(self, X):
        M = (X >= 0).float()
        X_encoded = self.encoder(X)
        X_encoded = self.transformer_encoder(X_encoded, M) # (B, N, C)
        gnn_X_lst = [self.forward_graph(X_encoded, self.A_lst[i], self.gnn[i]) for i in range(len(self.A_lst))]
        X_decoded = self.decoder(X_encoded, gnn_X_lst)
        pred = self.classifier(X_decoded)
        return pred 
    
class StatsMode(nn.Module):
    def forward(self, X):
        B, N = X.shape 
        output = X.clone()
        for b in range(B):
            vector = X[b]
            valid_values = vector[vector != -1]
            if valid_values.size(0) > 0:
                mode_value = float(stats.mode(valid_values.cpu().numpy()).mode)
                output[b][vector == -1] = mode_value
            else:
                output[b] = 1
        output = torch.stack([1.0-output, output], dim=-1) # (B, N, 2)
        return output
    
class StatsMean(nn.Module):
    def forward(self, X):
        B, N = X.shape 
        output = X.clone()
        for b in range(B):
            vector = X[b]
            valid_values = vector[vector != -1]
            if valid_values.size(0) > 0:
                mean_value = valid_values.mean()
                output[b][vector == -1] = mean_value
            else:
                output[b] = 1
        output = torch.stack([1.0-output, output], dim=-1) # (B, N, 2)
        return output
    
class StatsPercentile(nn.Module):
    def __init__(self, p=50):
        super().__init__()
        self.p = p

    def forward(self, X):
        B, N = X.shape 
        output = X.clone()
        for b in range(B):
            vector = X[b]
            valid_values = vector[vector != -1]
            if valid_values.size(0) > 0:
                percentile_value = np.percentile(valid_values.cpu().numpy(), self.p)
                output[b][vector == -1] = percentile_value
            else:
                output[b] = 1
        output = torch.stack([1.0-output, output], dim=-1) # (B, N, 2)
        return output

if __name__ == "__main__":
    pass

