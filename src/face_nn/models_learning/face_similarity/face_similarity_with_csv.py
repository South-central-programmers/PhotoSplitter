import os

from tqdm import tqdm

from PIL import Image
import pandas as pd

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        efficientnet_b0 = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.features = nn.Sequential(*list(efficientnet_b0.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return x


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
        )

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class SiameseNetworkDataset(Dataset):
    def __init__(self, image_folder, pairs_file, transform=None):
        self.image_folder = image_folder
        self.pairs_df = pd.read_csv(pairs_file)
        self.transform = transform

    def __getitem__(self, index):
        row = self.pairs_df.iloc[index]
        img1_path, img2_path, label = row["image1"], row["image2"], row["label"]
        img1 = Image.open(os.path.join(self.image_folder, img1_path))
        img2 = Image.open(os.path.join(self.image_folder, img2_path))

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.pairs_df)


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output1, output2, label):
        cos_sim = F.cosine_similarity(output1, output2)
        label = 2 * label - 1
        loss = F.mse_loss(cos_sim, label)
        return loss


class ModifiedSiameseNetwork(SiameseNetwork):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()


def calculate_metrics(output1, output2, labels):
    distances = torch.nn.functional.pairwise_distance(output1, output2)
    threshold = 0.5
    predictions = (distances < threshold).float()
    accuracy = accuracy_score(labels.cpu(), predictions.cpu())
    f1 = f1_score(labels.cpu(), predictions.cpu())
    return accuracy, f1


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total_accuracy = 0.0
    total_f1 = 0.0

    for batch_idx, (img0, img1, label) in enumerate(
        tqdm(train_loader, desc="Training Epoch")
    ):
        img0, img1, label = img0.to(device), img1.to(device), label.to(device)

        optimizer.zero_grad()
        output1, output2 = model(img0, img1)
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        accuracy, f1 = calculate_metrics(output1, output2, label)
        total_accuracy += accuracy
        total_f1 += f1

    average_loss = running_loss / len(train_loader)
    average_accuracy = total_accuracy / len(train_loader)
    average_f1 = total_f1 / len(train_loader)

    return average_loss, average_accuracy, average_f1


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_accuracy = 0.0
    total_f1 = 0.0

    with torch.no_grad():
        for img0, img1, label in tqdm(val_loader, desc="Validation Epoch"):
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)

            output1, output2 = model(img0, img1)
            loss = criterion(output1, output2, label)

            running_loss += loss.item()

            accuracy, f1 = calculate_metrics(output1, output2, label)
            total_accuracy += accuracy
            total_f1 += f1

    average_loss = running_loss / len(val_loader)
    average_accuracy = total_accuracy / len(val_loader)
    average_f1 = total_f1 / len(val_loader)

    return average_loss, average_accuracy, average_f1


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

TRAIN_CSV = "data/rebuilded_face_simm/train_pairs.csv"
VAL_CSV = "data/rebuilded_face_simm/val_pairs.csv"
IMAGE_FOLDER = "data/rebuilded_face_simm/images/"

train_dataset = SiameseNetworkDataset(
    image_folder=IMAGE_FOLDER, pairs_file=TRAIN_CSV, transform=transform
)
val_dataset = SiameseNetworkDataset(
    image_folder=IMAGE_FOLDER, pairs_file=VAL_CSV, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

device = torch.device("cuda")

model = ModifiedSiameseNetwork().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = CosineSimilarityLoss()

num_epochs = 50
for epoch in tqdm(range(num_epochs)):
    print(f"-----Epoch {epoch + 1}/{num_epochs}------")

    train_loss, train_accuracy, train_f1 = train_epoch(
        model, train_loader, optimizer, criterion, device
    )
    val_loss, val_accuracy, val_f1 = validate(model, val_loader, criterion, device)

    print(
        f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Train F1: {train_f1}\n"
        f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, Validation F1: {val_f1}"
    )

torch.save(
    model.state_dict(),
    "training_results/weights/nude_classifier/siamese_network_model.pth",
)
