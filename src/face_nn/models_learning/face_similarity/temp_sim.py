import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
from tqdm import tqdm
from torch import nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.backbone = models.vgg16(pretrained=True).features

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2622)
        )

    def forward_once(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def forward(self, anchor, positive, negative):
        anchor_embedding = self.forward_once(anchor)
        positive_embedding = self.forward_once(positive)
        negative_embedding = self.forward_once(negative)
        return anchor_embedding, positive_embedding, negative_embedding

model = SiameseNetwork().to(device)