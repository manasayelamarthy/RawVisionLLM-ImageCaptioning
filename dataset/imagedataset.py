import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import re
from collections import Counter

class captions_dataingestion:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path

    def read_csv(self):
        df = pd.read_csv(self.csv_file_path)
        return df

    def data_cleaning(self, df) -> list:
        df['caption'] = df['caption'].apply(lambda x: x.lower())
        df['caption'] = df['caption'].apply(lambda x: re.sub(r"[^a-z\s]", "", x))
        df['caption'] = df['caption'].apply(lambda x: " ".join([word for word in x.split() if len(word) > 5]))

        captions = df['caption'].tolist()
        return captions

    def convert_tokens(self, captions: list):
        all_words = [word for caption in captions for word in caption.split()]
        word_count = Counter(all_words)

        vocab = {} 
        for word, count in word_count.items():
            if count >= 5:
                vocab[word] = len(vocab) + 1  

        vocab["<PAD>"] = 0
        vocab["<UNK>"] = len(vocab) + 1
        vocab["<START>"] = len(vocab) + 2
        vocab["<END>"] = len(vocab) + 3

        tokenized_captions = [[vocab.get(word, vocab["<UNK>"]) for word in caption.split()] for caption in captions]

        max_len = max(len(seq) for seq in tokenized_captions)

    
        padded_captions = [seq + [vocab["<PAD>"]] * (max_len - len(seq)) for seq in tokenized_captions]

        tokens_tensor = torch.tensor(padded_captions, dtype=torch.long)
        return tokens_tensor

if __name__ == "__main__":
    csv_file_path = "data/captions.csv"
    
    data_ingestor = captions_dataingestion(csv_file_path)
    df = data_ingestor.read_csv()
    captions = data_ingestor.data_cleaning(df)
    tokens_tensor = data_ingestor.convert_tokens(captions)

    print("Tokenized Captions Tensor Shape:", tokens_tensor.shape)
