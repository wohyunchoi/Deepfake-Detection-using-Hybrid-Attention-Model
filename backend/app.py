from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from PIL import Image
from io import BytesIO
import torch
from safetensors.torch import load_file
import torchvision.transforms as transforms

from database import Base, engine, get_db
from crud import create_image_result
from model import DeepfakeDetector, rgb_fft_magnitude

app = FastAPI()

# CORS Setting
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Create DB Table
Base.metadata.create_all(bind=engine)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DeepfakeDetector(num_classes=2)
model.to(device)
model.eval()

# Load Model
state_dict = load_file("model/model.safetensors")
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        tensor = rgb_fft_magnitude(tensor)

        with torch.no_grad():
            output = model(tensor)
            pred = torch.softmax(output, dim=1)
            confidence, label = torch.max(pred, dim=1)

        label_text = "deepfake" if label.item() == 0 else "real"

        create_image_result(
            db=db,
            filename=file.filename,
            label=label_text,
            confidence=confidence
        )

        return JSONResponse({
            "label": label_text,
            "confidence": round(confidence.item(), 4)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)