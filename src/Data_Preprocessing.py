import pandas as pd
import re
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

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
    train_texts,test_texts,train_labels,test_labels=train_test_split(df['message'].values,df['label'].values,test_size=0.1,random_state=42)
    return train_texts,test_texts,train_labels,test_labels

class SMSdataset(Dataset):
    def __init__(self,texts,labels,vocab,max_len=50):
        self.labels = labels
        self.max_len = max_len
        
        # Pre-calculate sequences once to save CPU time during training
        self.sequences = []
        for text in texts:
            tokens = vocab.encode(text)
            
            # Efficient padding/truncating
            if len(tokens) < max_len:
                tokens += [0] * (max_len - len(tokens))
            else:
                tokens = tokens[:max_len]
            
            self.sequences.append(tokens)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,index):
        x = torch.tensor(self.sequences[index], dtype=torch.long)
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        return x, y