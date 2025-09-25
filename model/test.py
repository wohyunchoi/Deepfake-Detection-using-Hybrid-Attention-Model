import argparse
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.amp import autocast
import torch.nn.functional as F

from model import (
    XCFModel,
    rgb_fft_magnitude
)

def test(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            images = rgb_fft_magnitude(images)

            with autocast(device_type="cuda"):
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)[:, 1]  # probability of class 1 (deepfake)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    return acc, precision, recall, f1, auc, tp, tn, fp, fn

if __name__ == '__main__':
    # Argument Parse
    parser = argparse.ArgumentParser(description="Test Hybrid Deepfake Detector")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to model checkpoint (.safetensors)")
    args = parser.parse_args()

    # Hyperparameter
    BATCH_SIZE = 16
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    LOG_DIR = f"logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"test_log.txt")

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Dataset
    test_dataset = datasets.ImageFolder("./archive/test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = XCFModel(num_classes=2).to(DEVICE)

    # Load checkpoint
    state_dict = load_file(args.checkpoint)
    model.load_state_dict(state_dict)

    # Evaluate
    ckpt = args.checkpoint
    print(f"Test Start - Checkpoint:{ckpt}")
    acc, precision, recall, f1, auc, tp, tn, fp, fn = test(model, test_loader, DEVICE)

    # Print results
    print("===== Test Results =====")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")
    print(f"Confusion Matrix -> TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    # Save log
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("Accuracy, Precision, Recall, F1, AUC, TP, TN, FP, FN\n")
    with open(log_file, 'a') as f:
        f.write(f"{acc:.4f}, {precision:.4f}, {recall:.4f}, {f1:.4f}, {auc:.4f}, {tp}, {tn}, {fp}, {fn}\n")