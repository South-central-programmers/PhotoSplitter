import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import os
import copy
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

Image.MAX_IMAGE_PIXELS = 100000000


def plot_loss(train_losses, val_losses, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"{model_name} Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"training_results/plots/nude_classifier/{model_name}_loss_plot.png")
    plt.close()


def plot_accuracy(train_accs, val_accs, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label="Training Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.title(f"{model_name} Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(
        f"training_results/plots/nude_classifier/{model_name}_accuracy_plot.png"
    )
    plt.close()


def initialize_model(model_name):
    if model_name == "efficientnet":
        model_ft = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        for param in model_ft.parameters():
            param.requires_grad = False

        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_ftrs, 2))

    elif model_name == "resnet":
        model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        for param in model_ft.parameters():
            param.requires_grad = False

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_ftrs, 2))

    else:
        raise ValueError("Invalid model name")

    return model_ft


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    dataloaders,
    dataset_sizes,
    device,
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-----------")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "train":
                train_losses.append(epoch_loss)
                train_accs.append(
                    epoch_acc.cpu().numpy()
                    if isinstance(epoch_acc, torch.Tensor)
                    else epoch_acc
                )
            else:
                val_losses.append(epoch_loss)
                val_accs.append(
                    epoch_acc.cpu().numpy()
                    if isinstance(epoch_acc, torch.Tensor)
                    else epoch_acc
                )

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    model.load_state_dict(best_model_wts)

    return model, train_losses, val_losses, train_accs, val_accs


def calculate_classification_report(model, dataloader, device):
    y_true = []
    y_pred = []

    model.eval()

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    class_names = dataloader.dataset.classes
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "data/nude_classification/nude_classification_images"

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in tqdm(["train", "val", "test"])
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=8, shuffle=True, num_workers=4
        )
        for x in tqdm(["train", "val", "test"])
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in tqdm(["train", "val", "test"])}

    class_names = image_datasets["train"].classes

    model_names = ["resnet", "efficientnet"]

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for model_name in model_names:
        print(f"Training {model_name} model")
        model_ft = initialize_model(model_name)
        model_ft = model_ft.to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer_ft = optim.Adam(
            filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.0001
        )
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

        trained_model, train_losses, val_losses, train_accs, val_accs = train_model(
            model_ft,
            criterion,
            optimizer_ft,
            exp_lr_scheduler,
            num_epochs=20,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            device=device,
        )

        torch.save(
            trained_model.state_dict(),
            f"training_results/weights/nude_classifier/model_{model_name}.pth",
        )

        plot_loss(train_losses, val_losses, model_name)
        plot_accuracy(train_accs, val_accs, model_name)

        calculate_classification_report(trained_model, dataloaders["test"], device)


if __name__ == "__main__":
    main()
