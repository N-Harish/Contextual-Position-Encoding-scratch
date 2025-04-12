import torch.nn as nn
import torch
import math


class CoPE (nn . Module ) :
    def __init__ (self, npos_max, head_dim ) :
        super () . __init__ ()
        self.npos_max = npos_max
        self.pos_emb = nn.parameter.Parameter(torch.zeros(1 , head_dim , npos_max))
    
    def forward(self, query, attn_logits):
        gates = torch.sigmoid(attn_logits)
        print(f"gates.shape :- {gates.shape}")
        print(f"\n gates :- {gates} \n")
        print(f"gates flip -1 :- {gates.flip(-1)} \n")
        print(f"gates flip -1 and cumsum dim -1 :- {gates.flip(-1).cumsum(dim=-1)} \n")
        pos = gates.flip(-1).cumsum(dim = -1).flip(-1)

        print(f"\n pos :- {pos}")
        pos = pos.clamp(max = self.npos_max - 1)

        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        print(f"pos ceil :- {pos_ceil}\n")
        logits_int = torch.matmul(query, self.pos_emb)
        print(f"logits :- {logits_int.shape}")
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor
        return logits_ceil * w + logits_floor * (1 - w )
    

class SelfAttn (nn.Module):
    def __init__ (self, npos_max, head_dim ) :
        super () . __init__ ()
        self.cope = CoPE(npos_max, head_dim)
        self.head_dim = head_dim

    def forward (self, query, key, val, mask ) :
        # q, k, v have dimensions batch x seq_len x head_dim
        attn_logits = torch.bmm( query, key.transpose(-1 ,-2))
        attn_logits = attn_logits / math . sqrt ( self . head_dim )
        attn_logits += mask.log()
        print(f"attn_logits shape :- {attn_logits.shape}")
        
        print(f"cope shape :- {self.cope(query, attn_logits).shape}")
        attn_logits += self.cope(query, attn_logits)
        attn = torch.softmax(attn_logits, dim = -1)
        out = torch.bmm(attn, val)
        return out

lyr = SelfAttn(8, 16)
lyr(torch.rand(1,8,16), torch.rand(1,8,16), torch.rand(1,8,16), torch.rand(1,8,8))