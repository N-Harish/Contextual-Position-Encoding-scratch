from torch import nn
import torch
import math
from cope import CoPE


def generate_cope_encodings(query, attn_logits, cop_encoding):
    n = attn_logits.shape[1]

    matrices = [cop_encoding(query[0,i], attn_logits[0, i]) for i in range(n)]
    print(f"\n matrices :- {matrices} \n\n")
    # Stack the list into a single tensor of shape (n, 8, 8)
    tensor = torch.stack(matrices)
    
    # Add an extra dimension to match the desired shape (1, n, 8, 8)
    tensor = tensor.unsqueeze(0)
    pass


# Multihead attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float, npos_max = 8):
        """MultiHeadAttention layer

        Args:
            d_model (int): Embedding size
            h (int): number of attention heads needed
            dropout (float): dropout value
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert self.d_model % self.h == 0, "d_model should be divisible by h"

        self.d_k = self.d_model // self.h # divide emb by no of heads
        self.cope = CoPE(npos_max, self.d_k)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(self.d_k * self.h, d_model)

        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout, cope):
        d_k = query.shape[-1]
        print(f'Shape of query:- {query.shape}')
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        ## Using torch.bmm (bmm only works in 3d so reshape to 3 dim and after bmm again revert to og shape)
        # b, h, seq_len, feat_dim = query.shape
        # attention_score = torch.bmm(query.reshape(b*h, seq_len,feat_dim), key.reshape(b*h, seq_len,feat_dim).transpose(-2, -1))
        # attention_score = attention_score.contiguous().view(b, h, seq_len, seq_len) / math.sqrt(d_k)

        print(f"\nattention score shape :- {attention_score.shape}\n")
        print(f"attention first matrix :- {attention_score.shape[1]}")
        # print(f"CoPE ;- {cope(query, attention_score)} \n")
        print(f"generate cope encodings :- {generate_cope_encodings(query, attention_score, cope)}")
        print(f"cope shape :- {cope(query, attention_score).shape}")
        # apply mask if provided
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)
    
        attention_score = attention_score.softmax(dim=-1) # (batch, no_head, seq_len, seq_len)
        if dropout is not None:
            attention_score = dropout(attention_score)
        
        return (attention_score @ value), attention_score
        # return torch.bmm(attention_score, value), attention_score
    

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq-len, d-model) --> (batch, seq-len, d-model)
        print(f"query shape :- {query.shape} \n")
        key = self.w_k(k) # (batch, seq-len, d-model) --> (batch, seq-len, d-model)
        value = self.w_v(v) # (batch, seq-len, d-model) --> (batch, seq-len, d-model)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k)  # (batch, seq-len, d-model) --> (batch, seq-len, no_head, d-k)
        query = query.transpose(1,2)  # (batch, seq-len,no_head, d-k) --> (batch, no_h, seq-len, d-k)

        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k)  # (batch, seq-len, d-model) --> (batch, seq-len, no_head, d-k)
        key = key.transpose(1,2)  # (batch, seq-len,no_head, d-k) --> (batch, no_h, seq-len, d-k)

        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k)  # (batch, seq-len, d-model) --> (batch, seq-len, no_head, d-k)
        value = value.transpose(1,2)  # (batch, seq-len,no_head, d-k) --> (batch, no_h, seq-len, d-k)
        cope = self.cope
        x, self.attention_score = self.attention(query, key, value, mask, self.dropout, cope)
        x = x.transpose(1,2) # (batch, no_h, seq-len, d_k) --> (batch, seq-len, no_h, d_k)
        x = x.contiguous().view(x.shape[0], x.shape[1], -1) # (batch, seq-len, no_h, d_k) --> (batch, seq-len, d-model)

        return self.w_o(x), self.attention_score


lyr = MultiHeadAttention(8, 4, 0)
print(lyr(torch.rand(1, 8, 8), torch.rand(1, 8, 8), torch.rand(1, 8, 8), None)[0].shape)