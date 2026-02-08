from gensim.models import FastText
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