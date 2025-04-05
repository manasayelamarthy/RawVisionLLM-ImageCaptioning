import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import time

from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import get_dataset, get_dataloader
from config import train_Config
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
    parser.add_argument('--optimizer', type = str, help = 'optimizer')
    parser.add_argument('--learning_rate', type = int, help = 'learning_rate')
    parser.add_argument('--loss', type = int, help = 'loss')
    parser.add_argument('--epochs', type = int, help = 'number of epochs')
    parser.add_argument('--checkpoint-dir', type = str, help = 'path to the checkpoint dir')
    parser.add_argument('--log-dir', type = str, help = 'path to the log dir')

    return parser.parse_args()

args = arg_parse().__dict__
Config = train_Config(**args)


Dataset = get_dataset(Config)
train_dataset, valid_dataset = train_test_split(Dataset)

train_dataloader, val_dataloader = get_dataloader(train_dataset, valid_dataset, Config)

model = all_models[Config.model]
optimizer = torch.optim.Config.optimizer(model.parameters(), learning_rate = Config.learning_rate)
criterion = nn.CrossEntropyLoss()

num_epochs = Config.epochs

train_logs = {
    'loss' : 0
}

best_metric = 0
train_logger = trainLogging(Config)
start_time = time.time()

for epoch in range(num_epochs):
    
    model.train()

    train_iterator = tqdm(train_dataloader, total = len(train_dataloader), desc = f'Epoch-{epoch+1}:')

    
    for features,input_tensor,target_tensor in train_iterator:
        inputs = features.to(device)
    
        input_captions= input_tensor.to(device)
        target_tensor= target_tensor.to(device)

        optimizer.zero_grad()
        outputs = model(inputs,input_captions)
        loss = criterion(outputs, target_tensor.long())

        loss.backward()
        optimizer.step()

        train_logs['loss'] +=loss.item()
    train_logs['loss'] /= len(train_dataloader)


    val_logs = validate(model, val_dataloader, criterion, device)
    print("Train : ", train_logs)
    print("Validation : " , val_logs)

    train_logger.add_logs(epoch + 1, train_logs, val_logs)

    if val_logs['accuracy'] > best_accuracy:
        filename = Config.checkpoint_dir + f'{Config.model}.pth'
        checkpoint = save_checkpoint(model, filename)
        best_accuracy = val_logs['accuracy']

filename = Config.log_dir + Config.model + '.csv'
train_logger.save_logs(filename)

total_training_time = time.time() - start_time

print(f"training completed in {total_training_time:.2f}s")
