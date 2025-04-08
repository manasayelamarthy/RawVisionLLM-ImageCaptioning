import torch
import torch.nn as nn
import torch.nn.functional as F

class LstmModel_1(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256):
        super(LstmModel_1, self).__init__()

        # Linear layer to map image features to embedding dimension
        self.img_fc = nn.Linear(1920, embed_dim)

        # Embedding layer for text tokens
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM to process sequence of embeddings
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)

        # Dropout and projection layers
        self.dropout1 = nn.Dropout(0.5)
        self.add_fc = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(0.5)

        # Final classification layers
        self.fc = nn.Linear(embed_dim, 128)
        self.output = nn.Linear(128, vocab_size)

    def forward(self, img_features, captions):
        """
        img_features: [batch_size, 1920] - image features
        captions: [batch_size, T]        - tokenized captions
        returns: [batch_size, vocab_size] - logits for next word prediction
        """

        # Project image features to embedding space and add time dimension
        img_embed = F.relu(self.img_fc(img_features)).unsqueeze(1)  # [B, 1, embed_dim]

        # Embed captions
        embedded_captions = self.embedding(captions)  # [B, T, embed_dim]

        # Concatenate image as the first "word" in the caption
        merged = torch.cat((img_embed, embedded_captions), dim=1)  # [B, T+1, embed_dim]

        # LSTM output
        lstm_out, _ = self.lstm(merged)  # [B, T+1, hidden_dim]

        # Use last time step's output as sentence representation
        sentence_vector = lstm_out[:, -1, :]  # [B, hidden_dim]

        # Combine with image embedding
        x = self.dropout1(sentence_vector)
        x = self.add_fc(x) + img_embed.squeeze(1)  # [B, embed_dim]

        # Pass through dense layers
        x = F.relu(self.fc(x))
        x = self.dropout2(x)
        output = self.output(x)  # [B, vocab_size]

        return output
