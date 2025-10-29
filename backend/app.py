from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import torch
from safetensors.torch import load_file
import torchvision.transforms as transforms
from model import DeepfakeDetector, rgb_fft_magnitude

app = FastAPI()

# CORS 설정: React 개발 서버에서 접근 가능하도록
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 초기화
model = DeepfakeDetector(num_classes=2)
model.to(device)
model.eval()

# safetensors 모델 로드
state_dict = load_file("model/model.safetensors")
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 학습 당시 입력 크기와 동일하게
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        tensor = rgb_fft_magnitude(tensor)  # FFT 채널 추가

        with torch.no_grad():
            output = model(tensor)
            pred = torch.softmax(output, dim=1)
            confidence, label = torch.max(pred, dim=1)

        return JSONResponse({
            "label": "deepfake" if label.item() == 0 else "real",
            "confidence": round(confidence.item(), 4)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)