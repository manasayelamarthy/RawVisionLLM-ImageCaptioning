import torch
import torch.nn as nn

class LstmModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(LstmModel, self).__init__()


        self.image_fc = nn.Linear(1920, embed_size)

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.dropout1 = nn.Dropout(0.5)
        self.fc_add = nn.Linear(hidden_size, embed_size)
        self.relu = nn.ReLU()

        # Final classifier
        self.fc1 = nn.Linear(embed_size, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, vocab_size)

    def forward(self, image_features, captions):
        
        img_embed = self.relu(self.image_fc(image_features))  
        
        caption_embed = self.embedding(captions)  # [B, T, embed_size]
        
        sequence_input = torch.cat((img_embed, caption_embed), dim=1)  # [B, T+1, embed_size]

        
        lstm_out, _ = self.lstm(sequence_input)  # [B, T+1, hidden_size]
        
        lstm_out = self.dropout1(lstm_out)
        combined = self.fc_add(lstm_out)  # [B, T+1, embed_size]
        
       
        x = self.relu(self.fc1(combined))
        x = self.dropout2(x)
        output = self.fc2(x)  # [B, T+1, vocab_size]
        
        output = output[:, 1:, :]  # [B, T, vocab_size]
        
        return output
