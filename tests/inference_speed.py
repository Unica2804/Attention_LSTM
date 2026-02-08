import torch
import torch.nn as nn
import time
import os
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from collections import Counter
import json
import re

# ==========================================
# 1. SETUP & RE-DEFINE BI-LSTM ARCHITECTURE
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⚔️  Running Showdown on: {device}")

# Self Attention Class

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

# We must re-define the class to load the saved weights
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

# vocab class init
class SMSvocab:
    def __init__(self,min_freq=1):
        self.itos={0:"<pad>",1:"<unk>"}
        self.stoi={"<pad>":0,"<unk>":1}
        self.min_freq=min_freq

    def build_voabulary(self,sentence_list):
        frequencies=Counter()
        idx=2
        for sentence in sentence_list:
            for word in str(sentence).lower().split():
                frequencies[word]+=1
        for word, count in frequencies.items():
            if count >= self.min_freq:
                self.stoi[word]=idx
                self.itos[idx]=word
                idx+=1
    def encode(self,text):
        return [self.stoi.get(w.lower(), self.stoi["<unk>"]) for w in str(text).split()]
    def __len__(self):
        return len(self.stoi)
    
    def save_vocab(self, filepath):
        """
        Saves the stoi dictionary to a JSON file.
        """
        with open(filepath, 'w') as f:
            json.dump(self.stoi, f, indent=4)
        print(f"✅ Vocabulary saved to {filepath}")

    @classmethod
    def load_vocab(cls, filepath):
        """
        Creates an instance of SMSvocab from a saved JSON file.
        Useful for inference.
        """
        with open(filepath, 'r') as f:
            stoi = json.load(f)
        
        # Create a new instance
        vocab_instance = cls()
        vocab_instance.stoi = stoi
        # Reconstruct itos from stoi
        vocab_instance.itos = {v: k for k, v in stoi.items()}
        return vocab_instance
# =========================================
# Prepare Data
# =========================================





# ==========================================
# 2. LOAD MODELS
# ==========================================
print("\n--- Loading Models ---")

# --- Load Bi-LSTM ---

def load_your_model(path, device):
    """
    Since your model requires an 'embedding_matrix' to init, 
    we need to create a dummy one with the correct shape to build the class,
    then overwrite it with the state_dict weights.
    """
    if not os.path.exists(path):
        print(f"❌ Error: {path} not found.")
        return None, 0
    
    checkpoint = torch.load(path, map_location=device)
    
    # We try to guess shapes from the state_dict if we don't have the original matrix
    # The key usually looks like 'embedding.weight'
    vocab_size, embed_dim = checkpoint['embedding.weight'].shape
    
    # Create dummy matrix just to initialize the class
    dummy_matrix = torch.zeros((vocab_size, embed_dim))
    
    model = SpamAttentionClassifier(dummy_matrix)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model,vocab_size

print("✅ Architecture definitions updated.")

lstm_model,vocab_len=load_your_model('./Data/Spam_Classifier.pth',device)

# --- Load DistilBERT ---
MODEL_DIR = './Data/spam_classifier_model' # Where we saved it yesterday
try:
    bert_model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    bert_tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    bert_model.to(device)
    bert_model.eval()
    print("✅ DistilBERT Model Loaded")
except OSError:
    print("❌ DistilBERT files not found. (Using base model for demo)")
    bert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    bert_model.to(device)
    bert_model.eval()


# ==========================================
# 3. HELPER FUNCTIONS (Size & Speed)
# ==========================================

def get_model_size_mb(model_path):
    """Returns file size in MB"""
    try:
        size = os.path.getsize(model_path)
        return size / (1024 * 1024)
    except FileNotFoundError:
        return 0.0

def benchmark_model(model, inputs, tokenizer_fn, model_type="lstm"):
    model.eval()
    start_time = time.time()
    
    with torch.no_grad():
        for text in inputs:
            if model_type == "lstm":
                # Tokenizer returns (1, seq_len)
                input_tensor = tokenizer_fn(text).to(device)
                _ = model(input_tensor)
            elif model_type == "bert":
                encoded = tokenizer_fn(text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
                _ = model(**encoded)

    total_time = time.time() - start_time
    if total_time == 0: total_time = 0.001
    return len(inputs) / total_time

def mock_lstm_tokenizer(text):
    tokens = text.split()
    # Hash word to an index between 1 and vocab_size-1
    indices = [abs(hash(w)) % (vocab_len - 1) + 1 for w in tokens]
    # Pad to length 50
    max_len = 50
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return torch.tensor([indices], dtype=torch.long)

# ==========================================
# 4. THE SHOWDOWN
# ==========================================
test_emails = ["Free money now!", "Hey, how are you?", "Urgent reply needed", "Meeting at 5pm"] * 25 # 100 emails

print("\n🚀 STARTING SPEED TEST...")

if lstm_model:
    lstm_speed = benchmark_model(lstm_model, test_emails, mock_lstm_tokenizer, "lstm")
    print(f"👉 Custom LSTM Speed:   {lstm_speed:.2f} emails/sec")

bert_speed = benchmark_model(bert_model, test_emails, bert_tokenizer, "bert")
print(f"👉 DistilBERT Speed:    {bert_speed:.2f} emails/sec")

print("\n✅ Done.")