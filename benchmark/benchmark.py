import torch
import torch.nn as nn
import time
import os
import json
import numpy as np
import torch.nn.functional as F
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re
from sklearn.model_selection import train_test_split
import pandas as pd

# ==========================================
# 1. SETUP & RESOURCES
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⚖️  Running Efficiency Benchmark on: {device}")

VOCAB_PATH = './Data/vocab.json'
LSTM_PATH = './Data/Spam_Classifier.pth'
BERT_PATH = './Data/spam_classifier_model'

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

# ==========================================
# 2. "GOLDEN" TEST SET (Ground Truth)
# ==========================================
# In production, replace this with your full X_test and y_test
# texts = [
#     "Urgent! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!", # Spam
#     "Hey man, are we still on for the meeting tomorrow?", # Ham
#     "Your account has been compromised. Click here to reset.", # Spam
#     "I'll be late for dinner, start without me.", # Ham
#     "SIX chances to win CASH! From 100 to 20,000 pounds txt CSH11", # Spam
#     "Can you send me the report by 5pm?", # Ham
#     "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005", # Spam
#     "Nah I don't think he goes to usf, he lives around here though", # Ham
#     "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!", # Spam
#     "I'm forcing myself to eat a slice. I'm really not hungry tho. This sucks." # Ham
# ]
# labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] # 1=Spam, 0=Ham

def clean_text(text):
    text = text.lower()
    # Remove URLs (common spam vector)
    text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text)
    # Handle email addresses (spam indicator)
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)    
    # Keep digits but normalize excessive repetition
    text = re.sub(r'(\d)\1{3,}', r'\1\1\1', text)      
    # Remove punctuation except $ and % (spam signals)
    text = re.sub(r'[^\w\s$%]', ' ', text)    
    # Normalize currency
    text = re.sub(r'\$\s*(\d+)', r'$\1', text)    
    # Collapse whitespace
    text = " ".join(text.split())   
    return text

def prepare_sms_data(file_path:str):
    df= pd.read_csv(file_path,sep='\t', names=['label', 'message'])
    df['label']=df['label'].map({'ham':0,'spam':1})
    df['message']=df['message'].apply(clean_text)
    df = df[df['message'].str.strip().astype(bool)]
    train_texts,test_texts,train_labels,test_labels=train_test_split(df['message'].values,df['label'].values,test_size=0.3,random_state=42)
    return train_texts,test_texts,train_labels,test_labels

tr_text,texts,tr_label,labels=prepare_sms_data("./Data/SMSSpamCollection")
# ==========================================
# 3. METRIC CALCULATOR
# ==========================================
def evaluate_model(model, tokenizer_fn, texts, labels, model_type,max_len=50):
    model.eval()
    predictions = []
    start_time = time.time()
    
    with torch.no_grad():
        for text in texts:
            # A. Inference
            if model_type == 'lstm':
                # 1. Get indices from your vocab object
                indices = tokenizer_fn.encode(text)  # This returns a list
                
                # Pad/truncate to consistent length (same as your SMSvocab should do)
                if len(indices) < max_len:
                    indices += [0] * (max_len - len(indices))
                else:
                    indices = indices[:max_len]
                
                input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
                output, _ = model(input_tensor)
                pred = 1 if torch.sigmoid(output).item() > 0.5 else 0
                
            elif model_type == 'bert':
                encoded = tokenizer_fn(text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
                output = model(**encoded)
                # Argmax of logits is prediction
                pred = torch.argmax(output.logits, dim=1).item()
            
            predictions.append(pred)

    end_time = time.time()
    
    # B. Metrics
    latency_ms = ((end_time - start_time) / len(texts)) * 1000  # ms per email
    accuracy = accuracy_score(labels, predictions) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    return {
        "accuracy": accuracy,
        "latency_ms": latency_ms,
        "f1": f1 * 100,
        "predictions": predictions
    }

# ==========================================
# 4. LOAD EVERYTHING (Classes hidden for brevity, assume defined as before)
# ==========================================
# ... [Paste the Class Definitions for SpamAttentionClassifier, SelfAttention, SMSvocab here] ...
# (I am skipping pasting the classes again to keep this readable. 
#  Ensure the classes are defined in your cell before running this!)

# Load Vocab
vocab = SMSvocab.load_vocab(VOCAB_PATH)

# Load LSTM
checkpoint = torch.load(LSTM_PATH, map_location=device)
vocab_size, embed_dim = checkpoint['embedding.weight'].shape
dummy_matrix = torch.zeros((vocab_size, embed_dim))
lstm_model = SpamAttentionClassifier(dummy_matrix).to(device)
lstm_model.load_state_dict(checkpoint)

# Load BERT
try:
    bert_model = DistilBertForSequenceClassification.from_pretrained(BERT_PATH).to(device)
    bert_tokenizer = DistilBertTokenizer.from_pretrained(BERT_PATH)
except: 
    bert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)
    bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# ==========================================
# 5. THE RESULTS
# ==========================================
print("\n📊 RUNNING COMPOSITE BENCHMARK...")

# Evaluate LSTM
lstm_metrics = evaluate_model(lstm_model, vocab, texts, labels, 'lstm')

# Evaluate BERT
bert_metrics = evaluate_model(bert_model, bert_tokenizer, texts, labels, 'bert')

# Calculate Efficiency Score (Accuracy / Latency)
lstm_score = lstm_metrics['accuracy'] / lstm_metrics['latency_ms']
bert_score = bert_metrics['accuracy'] / bert_metrics['latency_ms']

print("\n" + "="*65)
print(f"{'METRIC':<20} | {'CUSTOM LSTM':<20} | {'DISTILBERT':<20}")
print("="*65)
print(f"{'Accuracy':<20} | {lstm_metrics['accuracy']:<20.1f}% | {bert_metrics['accuracy']:<20.1f}%")
print(f"{'F1 Score':<20} | {lstm_metrics['f1']:<20.1f} | {bert_metrics['f1']:<20.1f}")
print("-" * 65)
print(f"{'Latency (ms/email)':<20} | {lstm_metrics['latency_ms']:<20.2f} ms | {bert_metrics['latency_ms']:<20.2f} ms")
print("-" * 65)
print(f"{'EFFICIENCY SCORE':<20} | {lstm_score:<20.2f} | {bert_score:<20.2f}")
print(f"{'(Acc / Latency)':<20} | {'(Higher is Better)':<20} |")
print("="*65)

if lstm_score > bert_score:
    print(f"\n🏆 WINNER: Custom LSTM is {lstm_score/bert_score:.1f}x more efficient.")
else:
    print(f"\n🏆 WINNER: DistilBERT is {bert_score/lstm_score:.1f}x more efficient.")