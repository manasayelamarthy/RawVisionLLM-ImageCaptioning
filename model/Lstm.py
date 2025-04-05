import torch
import torch.nn as nn

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(ImageCaptioningModel, self).__init__()

        self.image_fc = nn.Linear(1920, embed_size)  
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

        self.dropout1 = nn.Dropout(0.5)
        self.add = nn.Linear(hidden_size, embed_size) 
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(embed_size, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, vocab_size)

    def forward(self, image_features, captions):
        
        img_features = self.relu(self.image_fc(image_features))  
        img_features_reshaped = img_features.unsqueeze(1)  

        embedded_captions = self.embedding(captions)  

        merged = torch.cat((img_features_reshaped, embedded_captions), dim=1)  

        lstm_out, _ = self.lstm(merged)  
        sentence_vector = lstm_out[:, -1, :] 

        sentence_vector = self.dropout1(sentence_vector)
        added = self.add(sentence_vector) + img_features  

        x = self.relu(self.fc1(added))
        x = self.dropout2(x)
        output = self.fc2(x)  

        return output
