import json
from collections import Counter
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