import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import time

from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
# print("PYTHONPATH:", sys.path)

from dataset import get_dataloader, ImageCaptionDataset
from all_config import train_config 
from models import all_models
from validation import validate

from utils import trainLogging,save_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type = int, help = 'version of the config')
    parser.add_argument('--mode', choices = ['train', 'validation', 'test'], help = 'choose between train, validation, test')

    parser.add_argument('--csv_file_path', type = str, help = 'path to the captions dataset dir')
    parser.add_argument('--image-dir', type = str, help = 'path to the dataset dir')
    parser.add_argument('--image-size', type = int, nargs = 2, help = 'dimensions of image') 
    parser.add_argument('--batch-size', type = int, help = 'batch size')


    parser.add_argument('--model', choices = ['Lstm', 'Lstm_1', 'Lstm_2'], help = 'choose between models')
    parser.add_argument('--embed-size', type = int, help = 'embedding size')
    parser.add_argument('--hidden-size', type = int, help = 'hidden size')
    parser.add_argument('--optimizer', type = str, help = 'optimizer')
    parser.add_argument('--learning_rate', type = int, help = 'learning_rate')
    parser.add_argument('--loss', type = int, help = 'loss')
    parser.add_argument('--epochs', type = int, help = 'number of epochs')
    parser.add_argument('--checkpoint-dir', type = str, help = 'path to the checkpoint dir')
    parser.add_argument('--log-dir', type = str, help = 'path to the log dir')

    return parser.parse_args()

args = arg_parse().__dict__
Config = train_config(**args)


dataset = ImageCaptionDataset(img_dir=Config.image_dir,
                            image_size=Config.image_size,
                            csv_file_path=Config.csv_file_path)

print("Dataset Loaded")

train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

print("Train and Valid Datasets Split")

train_dataloader, val_dataloader = get_dataloader(train_dataset, valid_dataset, Config)

print("Train and Valid Dataloaders Loaded")

model = all_models[Config.model](
    embed_size=Config.embed_size,
    hidden_size=Config.hidden_size,
    vocab_size = dataset.captions_dataingestion.vocab_size
).to(device)

print("Model Loaded")

optimizer_class = getattr(torch.optim, Config.optimizer)
optimizer = optimizer_class(model.parameters(), lr=Config.learning_rate)


print("Optimizer Loaded")

criterion = nn.CrossEntropyLoss()

num_epochs = Config.epochs

train_logs = {
    'loss' : 0
}

best_loss = float('inf')
train_logger = trainLogging(Config)
start_time = time.time()

for epoch in range(num_epochs):
    
    model.train()

    train_iterator = tqdm(train_dataloader, total = len(train_dataloader), desc = f'Epoch-{epoch+1}:')

    
    for features, input_tensor, target_tensor in train_iterator:
        inputs = features.to(device)
        input_captions = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, input_captions)
       
        
        outputs = outputs.reshape(-1, outputs.size(-1)) 
        target_tensor = target_tensor.view(-1)  
       
        loss = criterion(outputs, target_tensor.long())

        loss.backward()
        optimizer.step()

        train_logs['loss'] +=loss.item()
    train_logs['loss'] /= len(train_dataloader)


    val_logs = validate(model, val_dataloader, criterion, device)
    print("Train : ", train_logs)
    print("Validation : " , val_logs)

    train_logger.add_logs(epoch + 1, train_logs, val_logs)

    if val_logs['loss'] < best_loss:
        filename = Config.checkpoint_dir + f'{Config.model}.pth'
        checkpoint = save_checkpoint(model, filename)
        best_loss = val_logs['loss']

filename = Config.log_dir + Config.model + '.csv'
train_logger.save_logs(filename)

total_training_time = time.time() - start_time

print(f"training completed in {total_training_time:.2f}s")
