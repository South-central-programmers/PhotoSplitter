import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
from tqdm import tqdm

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        efficientnet = models.efficientnet_b0(pretrained=True)
        self.features = nn.Sequential(*list(efficientnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return x

class TripletNetwork(nn.Module):
    def __init__(self):
        super(TripletNetwork, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        )

    def forward(self, anchor, positive, negative):
        anchor_output = self.fc(self.feature_extractor(anchor))
        positive_output = self.fc(self.feature_extractor(positive))
        negative_output = self.fc(self.feature_extractor(negative))
        return anchor_output, positive_output, negative_output

class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_triplets=10000):
        self.root_dir = root_dir
        self.transform = transform
        self.identities = os.listdir(root_dir)
        self.anchor_positive_pairs = []
        
        total_triplets = 0

        for identity in tqdm(self.identities, desc="Preparing dataset"):
            identity_path = os.path.join(root_dir, identity)
            images = os.listdir(identity_path)
            
            images = random.sample(images, min(len(images), 10))
            
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    if total_triplets >= max_triplets:
                        break
                    self.anchor_positive_pairs.append((identity, images[i], images[j]))
                    total_triplets += 1
                if total_triplets >= max_triplets:
                    break
            if total_triplets >= max_triplets:
                break

    def __len__(self):
        return len(self.anchor_positive_pairs)

    def __getitem__(self, idx):
        anchor_identity, anchor_img, positive_img = self.anchor_positive_pairs[idx]
        negative_identity = random.choice([i for i in self.identities if i != anchor_identity])
        negative_img = random.choice(os.listdir(os.path.join(self.root_dir, negative_identity)))
        anchor_path = os.path.join(self.root_dir, anchor_identity, anchor_img)
        positive_path = os.path.join(self.root_dir, anchor_identity, positive_img)
        negative_path = os.path.join(self.root_dir, negative_identity, negative_img)
        anchor_image = Image.open(anchor_path).convert("RGB")
        positive_image = Image.open(positive_path).convert("RGB")
        negative_image = Image.open(negative_path).convert("RGB")
        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)
        return anchor_image, positive_image, negative_image

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.abs(anchor - positive).sum(dim=1)
        distance_negative = torch.abs(anchor - negative).sum(dim=1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

def calculate_metrics(anchor_output, positive_output, negative_output, threshold):
    positive_distances = torch.abs(anchor_output - positive_output).sum(dim=1)
    negative_distances = torch.abs(anchor_output - negative_output).sum(dim=1)
    true_labels = torch.cat([torch.ones_like(positive_distances), torch.zeros_like(negative_distances)])
    predictions = torch.cat([(positive_distances < threshold).float(), (negative_distances >= threshold).float()])
    accuracy = accuracy_score(true_labels.cpu().numpy(), predictions.cpu().numpy())
    f1 = f1_score(true_labels.cpu().numpy(), predictions.cpu().numpy())
    return accuracy, f1

def train_epoch_with_metrics(model, data_loader, criterion, optimizer, device, threshold):
    model.train()
    running_loss = 0.0
    total_accuracy = 0.0
    total_f1 = 0.0
    for batch_idx, (anchor, positive, negative) in enumerate(tqdm(data_loader, desc="Training")):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad()
        anchor_output, positive_output, negative_output = model(anchor, positive, negative)
        loss = criterion(anchor_output, positive_output, negative_output)
        loss.backward()
        optimizer.step()
        accuracy, f1 = calculate_metrics(anchor_output, positive_output, negative_output, threshold)
        running_loss += loss.item()
        total_accuracy += accuracy
        total_f1 += f1
    avg_loss = running_loss / len(data_loader)
    avg_accuracy = total_accuracy / len(data_loader)
    avg_f1 = total_f1 / len(data_loader)
    print(f'Train Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, F1: {avg_f1:.4f}')

def validate_with_metrics(model, data_loader, criterion, device, threshold):
    model.eval()
    running_loss = 0.0
    total_accuracy = 0.0
    total_f1 = 0.0
    with torch.no_grad():
        for batch_idx, (anchor, positive, negative) in enumerate(tqdm(data_loader, desc="Validation")):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_output, positive_output, negative_output = model(anchor, positive, negative)
            loss = criterion(anchor_output, positive_output, negative_output)
            accuracy, f1 = calculate_metrics(anchor_output, positive_output, negative_output, threshold)
            running_loss += loss.item()
            total_accuracy += accuracy
            total_f1 += f1
    avg_loss = running_loss / len(data_loader)
    avg_accuracy = total_accuracy / len(data_loader)
    avg_f1 = total_f1 / len(data_loader)
    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, F1: {avg_f1:.4f}')

if __name__ == "__main__":

    device = torch.device("mps")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = TripletFaceDataset(root_dir='data/face_simm/train', transform=transform)
    val_dataset = TripletFaceDataset(root_dir='data/face_simm/val', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = TripletNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = TripletLoss(margin=1.0).to(device)
    threshold = 1.0

    num_epochs = 10
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_epoch_with_metrics(model, train_loader, criterion, optimizer, device, threshold)
        validate_with_metrics(model, val_loader, criterion, device, threshold)
