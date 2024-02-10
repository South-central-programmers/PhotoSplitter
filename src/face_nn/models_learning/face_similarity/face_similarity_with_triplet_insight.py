import torch
import insightface
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
from tqdm import tqdm
from torchvision import models

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


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

lfw_train_dataset = LFWDataset(
    root_dir="data/face_similiarity_v3/train", transform=transform
)
lfw_val_dataset = LFWDataset(
    root_dir="data/face_similiarity_v3/val", transform=transform
)

train_loader = DataLoader(
    lfw_train_dataset, batch_size=16, shuffle=True, drop_last=True
)
val_loader = DataLoader(lfw_val_dataset, batch_size=16, shuffle=False, drop_last=True)


class SiameseNetwork(torch.nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.backbone = insightface.app.FaceAnalysis(rec_name='arcface_r100_v1')
        self.backbone.prepare(ctx_id=0)

    def forward(self, anchor, positive, negative):
        anchor_np = self._tensor_to_numpy(anchor)
        positive_np = self._tensor_to_numpy(positive)
        negative_np = self._tensor_to_numpy(negative)

        anchor_embedding = self._get_embedding(anchor_np)
        positive_embedding = self._get_embedding(positive_np)
        negative_embedding = self._get_embedding(negative_np)

        return anchor_embedding, positive_embedding, negative_embedding

    def _tensor_to_numpy(self, tensor):
        return tensor.cpu().numpy().transpose(0, 2, 3, 1)

    def _get_embedding(self, images_np):
        embeddings = []
        for img_np in images_np:
            img = Image.fromarray((img_np * 255).astype(np.uint8))
            face = self.backbone.get(img)
            embedding = face.normed_embedding
            embeddings.append(embedding)
        embeddings = np.vstack(embeddings)
        return torch.tensor(embeddings).to(device)


model = SiameseNetwork().to(device)


def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for anchor, positive, negative in tqdm(data_loader, leave=True):
        anchor, positive, negative = (
            anchor.to(device),
            positive.to(device),
            negative.to(device),
        )

        optimizer.zero_grad()
        anchor_embedding, positive_embedding, negative_embedding = model(
            anchor, positive, negative
        )

        loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for anchor, positive, negative in data_loader:
            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )
            anchor_embedding, positive_embedding, negative_embedding = model(
                anchor, positive, negative
            )

            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.TripletMarginLoss(margin=0.9)


def main(model, train_loader, val_loader, optimizer, criterion, device, epochs):
    best_val_loss = float("inf")
    best_epoch = 0
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(
            f"Epoch [{epoch+1}/{epochs}]: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(
                model.state_dict(),
                f"temp_models_lol/best_model_state_dict_{epoch+1}.pth",
            )
            print(
                f"New best model saved at epoch {epoch+1} with Validation Loss: {val_loss:.4f}"
            )

    print(
        f"Training completed. Best model was at epoch {best_epoch} with Validation Loss: {best_val_loss:.4f}"
    )


main(model, train_loader, val_loader, optimizer, criterion, device, epochs=500)
