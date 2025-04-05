
from .captions_dataset import captions_dataingestion
from .image_dataset import ImageCaptionDataset, FeatureExtraction

from config import train_config


from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def get_dataset(config = train_config):

    data_ingestor = captions_dataingestion(config.csv_file_path)
    df = data_ingestor.read_csv()
    cleaned_captions = data_ingestor.data_cleaning(df)
    tokens_tensor = data_ingestor.convert_tokens(cleaned_captions)

    fe = FeatureExtraction(config.image_dir)
    image_features = fe.get_image_features(config.image_size)

    dataset = ImageCaptionDataset(image_features=image_features,
                                captions_tokenized=tokens_tensor,)

    return dataset



def get_dataloader(train_dataset, valid_dataset, config = train_config()):
    if config.mode == 'train':
        train_dataloader = DataLoader(train_dataset,
                            batch_size = config.batch_size,
                            shuffle = True)

        val_dataloader = DataLoader (valid_dataset,
                                     batch_size = config.batch_size,
                                     shuffle = False)
        
    else :
        val_dataloader = DataLoader (valid_dataset,
                                     batch_size = config.batch_size,
                                     shuffle = False)
        
    return train_dataloader, val_dataloader
