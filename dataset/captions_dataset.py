import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import re
from collections import Counter

class captions_dataingestion:
    def __init__(self, csv_file_path):
        self.data = pd.read_csv(csv_file_path)

        self.data_cleaning()
        self.vocab_size = self.convert_tokens()

    def data_cleaning(self) -> list:
        """
          done data cleaning for csv file retunrn it as a list
        """
        self.data['caption'] = self.data['caption'].apply(lambda x: x.lower())
        self.data['caption'] = self.data['caption'].apply(lambda x: re.sub(r"[^a-z\s]", "", x))
        self.data['caption'] = self.data['caption'].apply(lambda x: " ".join([word for word in x.split() if len(word) > 5]))

    def get_random_caption(self, image_name : str):
        tokens_list = self.data[self.data['image'] == image_name]['tokens'].tolist()
        caption_tokens = random.choice(tokens_list)
        return caption_tokens

    def convert_tokens(self):
        """
          Convert words into token with padding
        """
        captions = self.data['caption'].tolist()
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

        all_tokens = torch.tensor(padded_captions, dtype=torch.long)
        
        self.data['tokens'] = [t.tolist() for t in all_tokens]

        return len(vocab)


    
    
