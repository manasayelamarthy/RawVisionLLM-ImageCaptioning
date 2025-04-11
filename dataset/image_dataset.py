import os
import cv2
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset

from .captions_dataset import captions_dataingestion

class FeatureExtraction:
    def __init__(self):  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = self.load_model().to(self.device)

    def load_model(self):
        model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
      
        features = model.features
        
        feature_extractor = nn.Sequential(
            features,
            nn.AdaptiveAvgPool2d((1, 1))  
        )
        feature_extractor.eval()
        return feature_extractor

    def get_image_features(self, image_path: str, image_size: tuple[int, int]):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, image_size)
        image = image.astype('float32') / 255.0
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.feature_extractor(image)
            features = features.squeeze(-1).squeeze(-1)

        return features


class ImageCaptionDataset(Dataset):
    def __init__(self, img_dir : str, image_size : tuple[int, int], csv_file_path : str):
        self.feature_extractor = FeatureExtraction()
        self.image_size = image_size
        self.img_dir = img_dir
        self.image_names = os.listdir(img_dir)
        self.captions_dataingestion = captions_dataingestion(csv_file_path)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.img_dir, image_name)
        features = self.feature_extractor.get_image_features(image_path, self.image_size)

        caption_tokens = self.captions_dataingestion.get_random_caption(image_name)
        
        input_caption = torch.tensor(caption_tokens[:-1], dtype=torch.long)
        target_caption = torch.tensor(caption_tokens[1:], dtype=torch.long)
      

        return features, input_caption, target_caption
