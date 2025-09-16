import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os
from safetensors.torch import load_file

from model import (
    XceptionDeepfakeDetector, SwinDeepfakeDetector,
    HybridDeepfakeDetector_XS, HybridDeepfakeDetector_ES,
    XceptionCBAM_FFT, XceptionCBAM_FFT2,
    rgb_fft_magnitude
)


def load_model(model_name, weight_path, device):
    if model_name == "X":
        model = XceptionDeepfakeDetector(num_classes=2)
    elif model_name == "S":
        model = SwinDeepfakeDetector(num_classes=2)
    elif model_name == "XS":
        model = HybridDeepfakeDetector_XS(num_classes=2)
    elif model_name == "ES":
        model = HybridDeepfakeDetector_ES(num_classes=2)
    elif model_name == "XCF":
        model = XceptionCBAM_FFT(num_classes=2)
    elif model_name == "XCF2":
        model = XceptionCBAM_FFT2(num_classes=2)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    state_dict = load_file(weight_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # (1, 3, 224, 224)


def infer(model_class, model, image_tensor, device):
    if model_class in ["XCF", "XCF2"]:
        image_tensor = rgb_fft_magnitude(image_tensor)

    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
    return predicted_class, confidence


def get_image_paths(folder):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(valid_exts)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Inference for Deepfake Detection")
    parser.add_argument("-w", "--weights", type=str, required=True, help="Path to model weights (.safetensors)")
    parser.add_argument("-m", "--model", type=str, default="X", choices=["X", "S", "XS", "ES", "XCF", "XCF2"],
                        help="Model class to use")
    parser.add_argument("-d", "--image_dir", type=str, required=True, help="Path to directory of input images")
    parser.add_argument("-o", "--output", type=str, default="classification_results.txt", help="Output file name")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.weights, device)

    image_paths = get_image_paths(args.image_dir)

    with open(args.output, "w") as f:
        for path in sorted(image_paths):
            try:
                image_tensor = preprocess_image(path)
                label, conf = infer(args.model, model, image_tensor, device)
                label_name = "FAKE" if label == 0 else "REAL"
                f.write(f"{os.path.basename(path)}\t{label_name}\t{conf:.4f}\n")
                print(f"[✓] {os.path.basename(path)} -> {label_name} ({conf:.4f})")
            except Exception as e:
                print(f"[✗] Error processing {path}: {e}")