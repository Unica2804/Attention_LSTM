import torch.nn as nn
import torch.nn.functional as F
import torch

class SelfAttention(nn.Module):
    def __init__(self,d_x):
        super().__init__()

        self.d_x=d_x
        self.scale=torch.sqrt(torch.tensor(d_x, dtype=torch.float32))

        self.q=nn.Linear(d_x,d_x)
        self.k=nn.Linear(d_x,d_x)
        self.v=nn.Linear(d_x,d_x)
    
    def forward(self,x,mask=None):
        Q,K,V=self.q(x),self.k(x),self.v(x)

        attn_score=torch.matmul(Q,K.transpose(-2,-1))/self.scale
        if mask is not None:
            # mask shape: (batch, seq) -> (batch, 1, seq) for broadcasting
            mask=mask.unsqueeze(1)
            attn_score=attn_score.masked_fill(mask==0,float('-inf'))

        weights=F.softmax(attn_score,dim=-1)
        context=torch.matmul(weights,V)
        return context,weights