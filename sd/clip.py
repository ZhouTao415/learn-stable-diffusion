import torch 
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIP_Embedding(nn.Module):

    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):
        super().__init__()
        
        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens):
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        x = self.token_embedding(tokens)

        x += self.position_embedding

        return x

class CLIP_Layer(nn.Module):

    def __init__(self, n_heads: int, n_embed: int):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_heads, n_embed)
        self.layer_norm_2 = nn.LayerNorm(n_embed)
        
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch_Size, Seq_Len, Dim)
        residual = x

        # SELF_ATTENTION
        x = self.layer_norm_1(x)

        x = self.attention(x, causal_mask=True)

        x += residual

        # FEED_FORWARD

        residual = x

        x = self.layer_norm_2(x)

        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x)

        x = self.linear_2(x)

        x += residual

        return x

class CLIP(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = CLIP_Embedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIP_Layer(12, 768) for i in range(12)
        ])

        self.layer_norm = nn.LayerNorm(768)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        tokens = x.type(torch.long)

        # (Batch_Size, Sequence_Length) -> (Batch_Size, Sequence_Length, 768)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)
        
        # (Batch_Size, Sequence_Length, 768)
        output = self.layer_norm(state)

        return output