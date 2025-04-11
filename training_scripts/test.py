import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models
from typing import List, Tuple
import json
import os

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dataset import captions_dataingestion
from models import all_models

class Config:
    # Initialize data ingestion first
    data_ingestion = captions_dataingestion(csv_file_path="data/captions.csv")
    vocab = data_ingestion.vocab
    
    # Adjust vocabulary indices to start from 1
    # Move special tokens to the end of the vocabulary
    vocab_size = len(vocab) - 4  # Subtract special tokens
    special_tokens = {
        '<PAD>': 0,  # PAD token at index 0
        '<UNK>': vocab_size + 1,
        '<START>': vocab_size + 2,
        '<END>': vocab_size + 3
    }
    
    # Update vocabulary with new indices
    for token, idx in special_tokens.items():
        vocab[token] = idx
    
    # Shift all word indices by 1 to avoid index 0
    new_vocab = {}
    for word, idx in vocab.items():
        if word not in special_tokens:
            new_vocab[word] = idx + 1
        else:
            new_vocab[word] = idx
    vocab = new_vocab
    
    embed_size = 512
    hidden_size = 512
    model = "Lstm"
    checkpoint_dir = "checkpoint/"
    model_name = "densenet201" 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Tester:
    def __init__(self, model, checkpoint_path: str):
        self.device = Config.device
        self.model = model.to(self.device)
        self.model.eval()
        self.vocab = Config.vocab
        self.idx2word = {idx: word for word, idx in self.vocab.items()}
        self.feature_extractor = self.load_model()
        
        # Print vocabulary information
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Model embedding size: {self.model.embedding.num_embeddings}")
        print(f"Special tokens: {[k for k in self.vocab.keys() if k.startswith('<')]}")
        print(f"Start token index: {self.vocab['<START>']}")
        print(f"End token index: {self.vocab['<END>']}")
        
        # Load checkpoint and verify vocabulary size
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        
        # Verify vocabulary size matches
        if self.model.embedding.num_embeddings != len(self.vocab):
            raise ValueError(f"Vocabulary size mismatch: model expects {self.model.embedding.num_embeddings}, but vocab has {len(self.vocab)} words")

    def sample_word(self, logits, temperature=1.0, top_k=5):
        """Sample a word from the logits using temperature sampling."""
        # Apply temperature
        logits = logits / temperature
        
        # Get top k predictions
        top_values, top_indices = torch.topk(logits, top_k)
        
        # Convert to probabilities
        probs = torch.softmax(top_values, dim=-1)
        
        # Sample from the top k
        sampled_idx = torch.multinomial(probs, 1).item()
        return top_indices[sampled_idx].item()

    def test(self, images: List[str], image_size: Tuple[int, int] = (224, 224), max_length: int = 20):
        """Generate captions for a list of images."""
        captions = []
        with torch.no_grad():
            for image_path in images:
                # Get image features
                features = self.get_image_features(image_path, image_size)
                # Add batch dimension if not present
                if len(features.shape) == 1:
                    features = features.unsqueeze(0)
                
                # Initialize caption with start token
                start_token = self.vocab['<START>']
                if start_token >= len(self.vocab):
                    raise ValueError(f"Start token index {start_token} is out of range (vocab size: {len(self.vocab)})")
                
                caption = [start_token]
                print(f"Starting caption with token: {start_token}")
                
                # Track previous predictions to avoid repetition
                previous_words = set()
                temperature = 1.0  # Start with high temperature for diversity
                
                # Generate caption word by word
                for i in range(max_length):
                    # Prepare input sequence with proper dimensions
                    input_seq = torch.tensor(caption).unsqueeze(0).to(self.device)  # [1, seq_len]
                    print(f"Input sequence at step {i}: {input_seq}")
                    
                    # Ensure features have correct dimensions
                    if len(features.shape) == 2:
                        features = features.unsqueeze(0)  # Add batch dimension if needed
                    
                    output = self.model(features, input_seq)
                    
                    # Get logits for the last word
                    logits = output[0, -1]
                    
                    # Get top k predictions for debugging
                    top_k = 5
                    top_values, top_indices = torch.topk(logits, top_k)
                    predictions = [(idx.item(), self.idx2word.get(idx.item(), 'UNK'), val.item()) for idx, val in zip(top_indices, top_values)]
                    print(f"Top {top_k} predictions: {predictions}")
                    
                    # Sample a word using temperature sampling
                    predicted = self.sample_word(logits, temperature)
                    
                    # If we get a special token or repeated word, try again with higher temperature
                    max_attempts = 3
                    attempt = 0
                    while (predicted in [self.vocab['<PAD>'], self.vocab['<UNK>'], self.vocab['<START>']] or 
                           self.idx2word.get(predicted, '') in previous_words) and attempt < max_attempts:
                        temperature *= 1.5  # Increase temperature
                        predicted = self.sample_word(logits, temperature)
                        attempt += 1
                    
                    # If we still get an invalid word, use END token
                    if predicted in [self.vocab['<PAD>'], self.vocab['<UNK>'], self.vocab['<START>']]:
                        predicted = self.vocab['<END>']
                    
                    print(f"Predicted index: {predicted}")
                    
                    # Add word to previous words set
                    word = self.idx2word.get(predicted, '')
                    if word:
                        previous_words.add(word)
                    
                    caption.append(predicted)
                    
                    # Stop if end token is generated
                    if predicted == self.vocab['<END>']:
                        break
                    
                    # Decrease temperature as we generate more words
                    temperature = max(0.5, temperature * 0.9)
                
                # Convert indices to words, handling unknown indices
                caption_words = []
                for idx in caption:
                    if idx in self.idx2word:
                        caption_words.append(self.idx2word[idx])
                    else:
                        caption_words.append('<UNK>')
                        print(f"Warning: Unknown index {idx} in caption")
                
                captions.append(' '.join(caption_words[1:-1]))  # Remove <START> and <END> tokens
        
        return captions

    def load_model(self):
        """Load and prepare the feature extractor model."""
        model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        features = model.features
        
        feature_extractor = nn.Sequential(
            features,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        feature_extractor.eval()
        feature_extractor.to(self.device)
        return feature_extractor

    def get_image_features(self, image_path: str, image_size: Tuple[int, int]):
        """Extract features from an image."""
        # Read and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, image_size)
        image = image.astype('float32') / 255.0
        
        # Convert to tensor and normalize
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
        image = image.to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(image)
            features = features.squeeze(-1).squeeze(-1)
        
        return features

def main():
    # Print configuration
    print("Configuration:")
    print(f"Vocabulary size: {len(Config.vocab)}")
    print(f"Model: {Config.model}")
    print(f"Embed size: {Config.embed_size}")
    print(f"Hidden size: {Config.hidden_size}")
    
    # Initialize model
    model = all_models[Config.model](
        embed_size=Config.embed_size,
        hidden_size=Config.hidden_size,
        vocab_size=len(Config.vocab)  # Use actual vocabulary size
    )
    
    # Create tester
    checkpoint_path = os.path.join(Config.checkpoint_dir, f"{Config.model}.pth")
    print(f"Loading checkpoint from: {checkpoint_path}")
    tester = Tester(model, checkpoint_path)
    
    # Test images
    test_images = [
        "data/images/23445819_3a458716c1.jpg",
        "data/images/47871819_db55ac4699.jpg"
    ]
    
    # Generate captions
    captions = tester.test(test_images)
    
    # Print results
    for img_path, caption in zip(test_images, captions):
        print(f"Image: {img_path}")
        print(f"Caption: {caption}\n")

if __name__ == "__main__":
    main()

