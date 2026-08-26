import torch
import torch.nn as nn
import math
import copy

class TimeSeriesEmbedder(nn.Module):
    def __init__(self, L, D):
        super().__init__()
        self.L = L
        self.D = D
        self.proj = nn.Linear(L * 3, D)

    def forward(self, x):
        B, T, _ = x.shape 
        N = T // self.L    # window count
        x = x[:, :N * self.L, :]    # get rid of leftover data
        windows = x.reshape(B, N, self.L * 3)    # create flattened data
        embeddings = self.proj(windows)   # embed with linear layer
        return embeddings 

class ContinuousPositionalEmbedding(nn.Module):
    def __init__(self, D, max_freq=10000):
        super().__init__()
        div_term = torch.exp(torch.arange(0,D,2).float() * (-math.log(max_freq) / D))
        self.register_buffer("w", div_term)

    def forward(self, t):
        PE = t.unsqueeze(-1) * self.w
        PE = torch.cat([torch.sin(PE), torch.cos(PE)], dim=-1)
        return PE

class ChuaJEPA(nn.Module):
    def __init__(self, L, D, n_heads=8, num_layers=6):
        super().__init__()
        self.ts_embedder = TimeSeriesEmbedder(L, D)
        self.pos_embedder = ContinuousPositionalEmbedding(D)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model= D, nhead=n_heads, batch_first=True) ## idk if 8 is appropriate
        self.context_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers) # idk if 6 is good for this either
        self.target_encoder = copy.deepcopy(self.context_encoder)

        for param in self.target_encoder.parameters():
            param.requires_grad = False

        predictor_layer = nn.TransformerEncoderLayer(d_model=D, nhead=n_heads, batch_first=True)
        self.prediction = nn.TransformerEncoder(predictor_layer, num_layers=num_layers // 2)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, D))

    def forward(self, x, t):
        h = self.ts_embedder(x)
        t_embed = self.pos_embedder(t)
        v = h + t_embed

        B, N, D = v.shape
        split_idx = N // 2

        v_c = v[:, :split_idx, :] 
        v_t = v[:, split_idx:, :]

        context_pos = t_embed[:, :split_idx, :]
        s_x = self.context_encoder(v_c) + context_pos
        s_y = self.target_encoder(v_t)

        mask_tokens = self.mask_token.expand(B, N - split_idx, D)
        target_pos = t_embed[:, split_idx:, :]
        target_setup = mask_tokens + target_pos

        pred_in = torch.cat([s_x, target_setup], dim=1)
        pred_out = self.prediction(pred_in)

        s_y_hat = pred_out[:, split_idx:, :]

        return s_y_hat, s_y


class ReadoutMLP(nn.Module):
    def __init__(self, D, L, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.D = D
        self.L = L
        self.mlp = nn.Sequential(
            nn.Linear(D, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, L*3)
        )

    def forward(self, s):
        B, N, D = s.shape
        mlp_out = self.mlp(s)
        coords = mlp_out.view(B,N,self.L,3)
        return coords
