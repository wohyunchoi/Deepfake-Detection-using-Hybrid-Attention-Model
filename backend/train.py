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
    DeepfakeDetector,
    rgb_fft_magnitude
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hybrid Deepfake Detector")
    parser.add_argument("-d", "--dataset", type=str, default="./dataset", help="Directory of Dataset")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Epochs per fold")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch Size")
    parser.add_argument("--resume_fold", type=int, default=None, help="Resume training from fold")
    parser.add_argument("--resume_epoch", type=int, default=None, help="Resume training from epoch")
    args = parser.parse_args()

    # Config
    BATCH_SIZE = 16
    EPOCHS = args.epochs
    LR = 1e-4
    WD = 1e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    KFOLD_SPLITS = 5
    LOG_DIR = f"logs"
    CKPT_DIR = f"checkpoints"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"train_log.txt")
    DATASET_DIR = args.dataset

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Dataset
    train_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, f"train"), transform=transform)
    valid_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, f"valid"), transform=transform)
    dataset = ConcatDataset([train_dataset, valid_dataset])

    # Fold indices
    kfold = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=42)
    fold_indices = list(kfold.split(dataset))  # 리스트로 저장

    # Log header
    with open(log_file, 'w') as f:
        f.write("Epoch, Train_Loss, Val_Acc, TP, TN, FP, FN, Precision, Recall, F1\n")

    model = DeepfakeDetector().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(device=DEVICE)
    
    # Resume ckpt
    start_fold, start_epoch = 0, 0
    if args.resume_fold is not None and args.resume_epoch is not None:
        ckpt_path = os.path.join(CKPT_DIR, f"fold{args.resume_fold}_epoch{args.resume_epoch}.safetensors")
        state_path = os.path.join(CKPT_DIR, f"fold{args.resume_fold}_epoch{args.resume_epoch}.pkl")

        if os.path.exists(ckpt_path) and os.path.exists(state_path):
            print(f"Resuming from Fold {args.resume_fold}, Epoch {args.resume_epoch}")

            state_dict = load_file(ckpt_path)
            model.load_state_dict(state_dict)
            
            with open(state_path, "rb") as f:
                state = pickle.load(f)
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            fold_indices = state["fold_indices"]
            start_fold, start_epoch = args.resume_fold - 1, args.resume_epoch
        else:
            print("Checkpoint not found, starting from scratch. (Fold : 0, Epoch : 0)")

    # Train
    print("Train Start")
    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        if fold < start_fold:
            continue
        print(f"\n=== Fold {fold+1}/{KFOLD_SPLITS} ===")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

        for epoch in range(EPOCHS):
            if fold == start_fold and (epoch + 1) < start_epoch:
                continue
            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
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

            print(f"Fold {fold+1}, Epoch [{epoch+1}/{EPOCHS}] "
                  f"Loss: {train_loss:.4f}, Val Acc: {val_acc:.2f}, F1: {f1:.4f}")

            with open(log_file, 'a') as f:
                f.write(f"{fold+1}, {epoch+1}, {train_loss:.4f}, {val_acc:.4f}, "
                        f"{tp}, {tn}, {fp}, {fn}, {precision:.4f}, {recall:.4f}, {f1:.4f}\n")

            # Save ckpt
            ckpt_path = os.path.join(CKPT_DIR, f"fold{fold+1}_epoch{epoch+1}.safetensors")
            state_path = os.path.join(CKPT_DIR, f"fold{fold+1}_epoch{epoch+1}.pkl")

            save_file(model.state_dict(), ckpt_path)
            with open(state_path, "wb") as f:
                pickle.dump({
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "fold": fold + 1,
                    "epoch": epoch + 1,
                    "fold_indices": fold_indices
                }, f)
