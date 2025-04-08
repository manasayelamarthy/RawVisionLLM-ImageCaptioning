import torch
import torch.nn as nn

class LstmModel_2(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(LstmModel_2, self).__init__()

        # Embedding layer for text input
        self.embed = nn.Embedding(vocab_size, embed_size)

        # Linear layer to project image features (e.g., DenseNet: 1920) to embedding size
        self.img_fc = nn.Linear(1920, embed_size)

        # LSTM to learn temporal dependencies
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

        # Final output layer to predict word from vocab
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, image_features, captions):
        """
        image_features: [B, 1920]         # extracted from DenseNet
        captions: [B, T]                  # tokenized input captions (excluding <END>)
        returns: [B, T+1, vocab_size]     # predicted token logits (including image token)
        """

        # Project image features and reshape to [B, 1, embed_size]
        img_embed = self.img_fc(image_features).unsqueeze(1)

        # Embed input captions: [B, T, embed_size]
        captions_embed = self.embed(captions)

        # Concatenate image embedding at the beginning of caption embeddings: [B, T+1, embed_size]
        inputs = torch.cat((img_embed, captions_embed), dim=1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(inputs)  # [B, T+1, hidden_size]

        # Final vocabulary scores
        outputs = self.linear(lstm_out)  # [B, T+1, vocab_size]

        return outputs
