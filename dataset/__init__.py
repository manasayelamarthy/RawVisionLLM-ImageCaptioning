
from .captions_dataset import captions_dataingestion
from .image_dataset import ImageCaptionDataset, FeatureExtraction

from all_config import train_config


from torch.utils.data import DataLoader


def get_dataloader(train_dataset, valid_dataset, config = train_config()):
    if config.mode == 'train':
        train_dataloader = DataLoader(train_dataset,
                            batch_size = config.batch_size,
                            shuffle = True)

        val_dataloader = DataLoader (valid_dataset,
                                     batch_size = config.batch_size,
                                     shuffle = False)
        return train_dataloader, val_dataloader
        
    else :
        val_dataloader = DataLoader (valid_dataset,
                                     batch_size = config.batch_size,
                                     shuffle = False)
        
        return val_dataloader