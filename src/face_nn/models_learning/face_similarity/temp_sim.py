import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
from tqdm import tqdm
from torch import nn
from torchvision import models
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LFWDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.people = [
            d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
        ]
        self.image_paths = {
            person: [
                os.path.join(root_dir, person, img)
                for img in os.listdir(os.path.join(root_dir, person))
            ]
            for person in self.people
        }

    def __len__(self):
        return len(self.people)

    def __getitem__(self, idx):
        person = self.people[idx]
        if len(self.image_paths[person]) > 1:
            positive_pair = random.sample(self.image_paths[person], 2)

        negative_person = random.choice([p for p in self.people if p != person])
        negative_image = random.choice(self.image_paths[negative_person])

        anchor_image = Image.open(positive_pair[0]).convert("RGB")
        positive_image = Image.open(positive_pair[1]).convert("RGB")
        negative_image = Image.open(negative_image).convert("RGB")

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image


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
            nn.Linear(4096, 2622),
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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

lfw_train_dataset = LFWDataset(root_dir='', transform=transform)
lfw_val_dataset = LFWDataset(root_dir='', transform=transform)

train_loader = DataLoader(lfw_train_dataset, batch_size=16, shuffle=True, drop_last=True)
val_loader = DataLoader(lfw_val_dataset, batch_size=16, shuffle=False, drop_last=True)

model = SiameseNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.TripletMarginLoss(margin=1.0)

def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, (anchor, positive, negative) in enumerate(tqdm(data_loader)):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        
        optimizer.zero_grad()
        anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
        loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)

def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (anchor, positive, negative) in enumerate(tqdm(data_loader)):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def main(model, train_loader, val_loader, optimizer, criterion, device, epochs):
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

epochs = 100
main(model, train_loader, val_loader, optimizer, criterion, device, epochs)
