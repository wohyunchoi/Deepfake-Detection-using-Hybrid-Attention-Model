import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from safetensors.torch import save_file, load_file
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from torch.amp import autocast, GradScaler
import argparse
import os
import pickle

from model import (
    XceptionDeepfakeDetector, SwinDeepfakeDetector,
    HybridDeepfakeDetector_XS, HybridDeepfakeDetector_ES,
    XceptionCBAM_FFT, XceptionCBAM_FFT2,
    EfficientCBAM_FFT, ResCBAM_FFT,
    rgb_fft_magnitude
)


def get_model(model_class, device):
    if model_class == "X":
        return XceptionDeepfakeDetector(num_classes=2).to(device)
    elif model_class == "S":
        return SwinDeepfakeDetector(num_classes=2).to(device)
    elif model_class == "XS":
        return HybridDeepfakeDetector_XS(num_classes=2).to(device)
    elif model_class == "ES":
        return HybridDeepfakeDetector_ES(num_classes=2).to(device)
    elif model_class == "XCF":
        return XceptionCBAM_FFT(num_classes=2).to(device)
    elif model_class == "XCF2":
        return XceptionCBAM_FFT2(num_classes=2).to(device)
    elif model_class == "ECF":
        return EfficientCBAM_FFT(num_classes=2).to(device)
    elif model_class == "RCF":
        return ResCBAM_FFT(num_classes=2).to(device)
    else:
        raise ValueError("Invalid model class")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="5-Fold CV for Hybrid Deepfake Detector")
    parser.add_argument("-m", "--model-class", type=str, default="X", choices=["X", "S", "XS", "ES", "XCF", "XCF2", "ECF", "RCF"])
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Epochs per fold")
    args = parser.parse_args()

    # Config
    BATCH_SIZE = 16
    EPOCHS = args.epochs
    LR = 1e-4
    WD = 1e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    KFOLD_SPLITS = 5
    MODEL_CLASS = args.model_class
    LOG_DIR = f"{MODEL_CLASS}_cv_logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"cv_log.txt")

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Dataset
    train_dataset = datasets.ImageFolder("./archive/train", transform=transform)
    valid_dataset = datasets.ImageFolder("./archive/valid", transform=transform)
    dataset = ConcatDataset([train_dataset, valid_dataset])

    # Fold indices
    kfold = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=42)

    # Log header
    with open(log_file, 'w') as f:
        f.write("Fold, Epoch, Train_Loss, Val_Acc, TP, TN, FP, FN, Precision, Recall, F1\n")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n=== Fold {fold+1}/{KFOLD_SPLITS} ===")

        # Reinitialize model per fold
        model = get_model(MODEL_CLASS, DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler(device=DEVICE)

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                if MODEL_CLASS in ["XCF", "XCF2", "ECF", "RCF"]:
                    images = rgb_fft_magnitude(images)

                optimizer.zero_grad()
                with autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)

            # Validation
            model.eval()
            all_preds, all_labels = [], []

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                    if MODEL_CLASS in ["XCF", "XCF2", "ECF", "RCF"]:
                        images = rgb_fft_magnitude(images)

                    with autocast(device_type="cuda"):
                        outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_acc = accuracy_score(all_labels, all_preds)
            cm = confusion_matrix(all_labels, all_preds)
            tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
            precision = precision_score(all_labels, all_preds, zero_division=0)
            recall = recall_score(all_labels, all_preds, zero_division=0)
            f1 = f1_score(all_labels, all_preds, zero_division=0)

            print(f"Fold {fold+1}, Epoch [{epoch+1}/{EPOCHS}] Loss: {train_loss:.4f}, Val Acc: {val_acc:.2f}, F1: {f1:.4f}")

            with open(log_file, 'a') as f:
                f.write(f"{fold+1}, {epoch+1}, {train_loss:.4f}, {val_acc:.4f}, {tp}, {tn}, {fp}, {fn}, {precision:.4f}, {recall:.4f}, {f1:.4f}\n")
