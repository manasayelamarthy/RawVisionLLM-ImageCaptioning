import os
import cv2
import torch
import torch.nn as nn
from torchvision import models

class FeatureExtraction:
    def __init__(self, df):
        self.df = df
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = self.load_model().to(self.device)

    def load_model(self):
        model = models.densenet201(pretrained=True)

        feature_extractor = nn.Sequential(*list(model.features.children()))
        feature_extractor.eval()
        return feature_extractor

    def get_image_features(self, img_dir, image_size: tuple[int, int]):
        image_features = {}
        image_names = os.listdir(img_dir)

        for image_name in image_names:
            img_path = os.path.join(img_dir, image_name)

           
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
