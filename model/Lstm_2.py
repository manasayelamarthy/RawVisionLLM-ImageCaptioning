import torch.nn as nn
import torch

class LstmModel_2(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(LstmModel_2, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        

    def forward(self, image_features, captions):
        
        img_embed = self.img_fc(image_features).unsqueeze(1) 
        captions_embed = self.embed(captions)  
        inputs = torch.cat((img_embed, captions_embed), dim=1)

       
        lstm_out, _ = self.lstm(inputs)

        outputs = self.linear(lstm_out)

        return outputs
