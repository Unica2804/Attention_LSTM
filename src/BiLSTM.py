import torch.nn as nn
from .Attention import SelfAttention
class SpamAttentionClassifier(nn.Module):
    def __init__(self,embedding_matrix,hidden_dim=128):
        super().__init__()
        vocab_size,embed_dim=embedding_matrix.shape
        self.embedding=nn.Embedding.from_pretrained(embedding_matrix,freeze=False)
        self.lstm=nn.LSTM(embed_dim,hidden_dim,batch_first=True,bidirectional=True)
        self.attention=SelfAttention(hidden_dim*2)
        self.fc=nn.Linear(hidden_dim*2,1)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        # Create padding mask: 1 for real tokens, 0 for pad (index 0)
        mask=(x!=0).float()  # shape: (batch, seq_len)
        x=self.embedding(x)
        out, (hn,cn)=self.lstm(x)
        context_seq,weights=self.attention(out,mask=mask)
        # Masked mean pooling: average only over real tokens
        mask_expanded=mask.unsqueeze(-1)  # (batch, seq, 1)
        context=(context_seq*mask_expanded).sum(dim=1)/mask_expanded.sum(dim=1).clamp(min=1)
        return self.fc(context),weights