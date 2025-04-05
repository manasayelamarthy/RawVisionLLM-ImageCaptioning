import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256):
        super(ImageCaptioningModel, self).__init__()
        self.img_fc = nn.Linear(1920, embed_dim) 
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        
        self.dropout1 = nn.Dropout(0.5)
        self.add_fc = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(0.5)

        self.fc = nn.Linear(embed_dim, 128)
        self.output = nn.Linear(128, vocab_size)

    def forward(self, img_features, captions):
        """
        img_features: shape [batch_size, 1920] (from DenseNet201)
        captions: shape [batch_size, max_length]
        """
        
        img_embed = F.relu(self.img_fc(img_features))  
        img_embed = img_embed.unsqueeze(1)  
        
        embedded_captions = self.embedding(captions)  
        
        merged = torch.cat((img_embed, embedded_captions), dim=1)  

        lstm_out, _ = self.lstm(merged)  
        sentence_features = lstm_out[:, -1, :]  

        x = self.dropout1(sentence_features)
        x = x + img_embed.squeeze(1) 
        x = F.relu(self.fc(x))
        x = self.dropout2(x)
        output = self.output(x)  

        return output
