import torch
import time
import os
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re

from src.vocab import SMSvocab
from src.BiLSTM import SpamAttentionClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VOCAB_PATH = './Data/vocab.json'
LSTM_PATH = './Data/Spam_Classifier.pth'
BERT_PATH = './Data/spam_classifier_model'

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

def prediction(text):
    max_len=50
    if not text:
        return "N/A","N/A","N/A","N/A","N/A","N/A"
    
    start_lstm=time.time()
    with torch.no_grad():
        indices = vocab.encode(text)
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
        output, _ = lstm_model(input_tensor)
        prob_lstm=torch.sigmoid(output).item()
    time_lstm=(time.time()-start_lstm)*1000
    label_lstm="SPAM" if prob_lstm > 0.5 else "HAM"
    conf_lstm=f"{prob_lstm:.2%}" if prob_lstm>0.5 else f"{(1-prob_lstm):.2%}"

    start_bert=time.time()
    with torch.no_grad():
        encoded=bert_tokenizer(text,return_tensors='pt',padding=True,truncation=True, max_length=128).to(device)
        output=bert_model(**encoded)
        probs=F.softmax(output.logits,dim=1)
        prob_spam=probs[0][1].item()
    time_bert=(time.time()-start_bert)*1000
    label_bert="SPAM" if prob_spam > 0.5 else "HAM"
    conf_bert = f"{prob_spam:.2%}" if prob_spam > 0.5 else f"{(1-prob_spam):.2%}"
    return (
        label_lstm, conf_lstm, f"{time_lstm:.2f} ms",
        label_bert, conf_bert, f"{time_bert:.2f} ms"
    )
