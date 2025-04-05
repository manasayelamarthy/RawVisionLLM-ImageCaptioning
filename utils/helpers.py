import torch
import torch.nn as nn

def save_checkpoint(model, filename):
    checkpoint = model.state_dict()
    torch.save(checkpoint,filename)

def load_checkpoint(model:nn.Module, checkpoint_dir):
    checkpoint_file= torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint_file)

    return model
