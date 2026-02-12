from gensim.models import FastText
from .Data_Preprocessing import clean_text
import pandas as pd
import torch
import numpy as np

def get_embedding_matrix(model_path:str,vocab:dict):
    """ 
    Loads gensim FastText and creates a pytorch embedding matrix from FastText 
    """

    try: 
        ft_model=FastText.load(model_path)
        embed_dim=ft_model.vector_size
        matrix_len = len(vocab)+1
        weight_matrix=np.zeros((matrix_len,embed_dim))

        for words, i in vocab.items():
            weight_matrix[i]=ft_model.wv[words]
        return torch.tensor(weight_matrix,dtype=torch.float32)
    except FileNotFoundError:
        print("could not find the.npy files")
        raise

def create_embedding(Data_path:str):
    df=pd.read_csv(Data_path,sep='\t', names=['label', 'message'])
    df['message']=df['message'].apply(clean_text)
    sentences=[sentences.split() for sentences in df['message']]
    model=FastText(vector_size=100,window=5,min_count=2)
    model.build_vocab(corpus_iterable=sentences)
    model.train(corpus_iterable=sentences,total_examples=len(sentences),epochs=10)
    print("Model trained succesfully!!")
    model.save("./Data/spam_fasttext_gensim.model")
    print("Model Saved succesfully!!")