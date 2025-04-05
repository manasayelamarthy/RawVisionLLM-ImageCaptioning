import os
import cv2
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset

class FeatureExtraction:
    def __init__(self, img_dir):
        self.img_dir = img_dir  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = self.load_model().to(self.device)

    def load_model(self):
        model = models.densenet201(pretrained=True)
        feature_extractor = nn.Sequential(*list(model.features.children()))
        feature_extractor.eval()
        return feature_extractor

    def get_image_features(self, image_size: tuple[int, int]):
        image_features = {}
        image_names = os.listdir(self.img_dir)

        for image_name in image_names:
            img_path = os.path.join(self.img_dir, image_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, image_size)
            image = image.astype('float32') / 255.0
            image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.feature_extractor(image)
                features = torch.flatten(features, 1)

            image_features[image_name] = features.cpu()

        return image_features


class ImageCaptionDataset(Dataset):
    def __init__(self,image_features: dict, captions_tokenized: list):
    
        self.image_features_dict = image_features  
        self.captions_tokenized = captions_tokenized

    def __len__(self):
        return len(self.captions_tokenized)

    def __getitem__(self, idx):
        image_name = list(self.image_features_dict.keys())[idx]
        featured_image = self.image_features_dict[image_name].squeeze(0)

        caption = self.captions_tokenized[idx]
        input_caption = torch.tensor(caption[:-1], dtype=torch.long)
        target_caption = torch.tensor(caption[1:], dtype=torch.long)

        return featured_image, input_caption, target_caption
