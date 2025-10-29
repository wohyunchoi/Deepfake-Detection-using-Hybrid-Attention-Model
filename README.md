# Deepfake-Detection-using-Hybrid-Attention-Model

## 목차 (Table of Contents)
- [Introduction](#introduction)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Useful for](#useful-for)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training / How to Run Experiments](#training--how-to-run-experiments)
- [Results / Performance](#results--performance)
- [Screenshots / Demo](#screenshots--demo)
- [Project Structure](#project-structure)
- [Setup and Running](#setup-and-running)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

- (소개영상 추후 업데이트 예정)
- This project aims to automatically detect deepfake images using a deep learning model trained on public datasets.
It provides an interactive web interface where users can upload an image and receive a classification result ("Real" or "Fake") with confidence visualization.

---

## Tech Stack
**AI / Backend**
- Python 3.10+
- FastAPI
- PyTorch
- PostgreSQL

**Frontend**
- React

---

## Features
- Upload an image (drag & drop support)
- Run inference using trained deepfake classifier
- Display prediction confidence using visual bars
- Store results in database
- FastAPI backend for REST API inference

---

## Useful for
- AI researchers exploring deepfake detection
- Students studying computer vision / attention mechanisms
- Developers building AI-driven web apps
- Educators teaching explainable AI and fake media detection

---

## Model Architecture

The model (DeepfakeDetector) integrates:
- Xception-like separable convolutions for efficient feature extraction
- CBAM (Convolutional Block Attention Module) for channel & spatial attention
- Frequency-domain features using RGB + FFT Magnitude
- Fully connected classifier for binary output (Real / Fake)

## Architecture Summary
<img width="1920" height="1080" alt="Untitled design" src="https://github.com/user-attachments/assets/48de1338-bc50-4973-aa17-808726773a69" />

---

## Dataset

- **Source:** [https://www.kaggle.com/datasets/tusharpadhy/deepfake-dataset-merged]
- **License:** CC0: Public Domain
- **Composition:** Real and synthetic deepfake faces from multiple benchmark datasets.
- **Usage:** Used solely for research and educational purposes.
Redistribution of original images is not included in this repository.

---
## Training / How to Run Experiments
This section is intended for developers or researchers who want to **train or fine-tune the DeepfakeDetector model**.

### 1. Clone the repository
```bash
git clone https://github.com/wohyunchoi/Deepfake-Detection-using-Hybrid-Attention-Model.git
```

### 2. Prepare the environment
```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Dataset
- Download dataset from Kaggle: [https://www.kaggle.com/datasets/tusharpadhy/deepfake-dataset-merged]
- Place dataset in a suitable folder, e.g., backend/dataset/

### 4. Training / Fine-tuning
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

python train.py --dataset <your_dataset_path> --epochs <epochs> --batch_size <batch_size> --resume_fold <resume_fold> --resume_epoch <resume_epoch>
```
- Default parameters: "./dataset", "10", "16", None, None
- Resume fold and resume epoch are required when you need to continue training after it has been interrupted.
- When resuming training or fine-tuning an existing model, the corresponding checkpoint files are required to continue from the previous state — specifically, checkpoints/fold<fold>_epoch<epoch>.safetensors and checkpoints/fold<fold>_epoch<epoch>.pkl.
- dataset directory and other directories related to training must be like this:
```plaintext
│
└── backend/
    ├── train.py              # Python code for train
    ├── dataset/              # Default path, you can modify the dataset path
    │   ├── train/
    |   |   ├── Fake/
    |   |   |   ├── a.jpg
    |   |   |   ├── b.jpg
    |   |   |   ├── ...
    |   |   └── Real/
    |   |       ├── a.jpg
    |   |       ├── b.jpg
    |   |       ├── ...
    │   ├── valid/             # Optional, the dataset in this directory will be merged with the training dataset and split into 5 folds for sequential fold-wise training
    |   |   ├── Fake/
    |   |   |   ├── a.jpg
    |   |   |   ├── b.jpg
    |   |   |   ├── ...
    |   |   └── Real/
    |   |       ├── a.jpg
    |   |       ├── b.jpg
    |   |       ├── ...
    │   └── test/
    |       ├── Fake/
    |       |   ├── a.jpg
    |       |   ├── b.jpg
    |       |   ├── ...
    |       └── Real/
    |       |   ├── a.jpg
    |       |   ├── b.jpg
    |       |   ├── ...
    ├── logs/
    |   ├── train_log.txt      # Training Information is logged into this file
    |   └── test_log.txt       # Test Information is logged into this file
    └── checkpoints/           
        ├── fold<fold>_epoch<epoch>.safetensors  # Model weights
        └── fold<fold>_epoch<epoch>.pkl      # Training metadata
```
- Epoch specifies the number of training epochs for each fold.
- For example, if you set this value to 10 and use 5-fold fold-wise sequential training, the model will be trained for a total of 50 epochs.
- Adjust hyperparameters as needed.
- Training information(loss, validation accuracy/precision/recall/f1 etc.) is logged to a file located at logs/train_log.txt.
- Trained model will be saved as checkpoints/fold<fold>_epoch<epoch>.safetensors.
- Training metadata such as optimizer state, scaler state, fold indices will be saved as fold<fold>_epoch<epoch>.pkl.
- After training is completed and use your trained model, rename the checkpoint file (e.g., fold<fold>_epoch<epoch>.safetensors) to model.safetensors and place it in backend/model/.

### 5. Testing
```bash
python test.py --dataset <your_dataset_path> --checkpoint <your_checpoint_path e.g. checkpoints/fold5_epoch10.safetensors>
```
- Test information(accuracy, precision, recall, f1, AUC etc.) is logged to a file located at logs/test_log.txt.

### 6. Optional: Database Logging
- If PostgreSQL is configured, training metrics can be logged via crud.py.
 
---

## Results / Performance
- **Accuracy:** 96.26%
- **Precision:** 98.69%
- **Recall:** 93.49%
- **AUC:** 0.9957%  

---

## Screenshots / Demo
> (시연영상 및 Demo 추가 예정)

---

## Project Structure
```plaintext
Deepfake-Detection-using-Hybrid-Attention-Model/
│
├── backend/
│   ├── app.py                # FastAPI app entry point
|   ├── model.py              # Model architecture used for inference
│   ├── model/                # Model files (.safetensors)
│   │   └── model.safetensors
│   ├── database.py           # PostgreSQL setup
│   ├── crud.py               
│   └── schemas.py            
│
├── frontend/
│   ├── src/                  # React component
│   │   ├── App.js
│   │   └── components/
│   ├── public/
│   └── package.json
│
├── .env
└── README.md
```

---

## Setup and Running
This section explains how to **set up and run the project** using the pre-trained model.

### 1. Clone the repository
```bash
git clone https://github.com/wohyunchoi/Deepfake-Detection-using-Hybrid-Attention-Model.git
```

### 2. Create virtual environment and install dependencies
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure PostgreSQL database
#### 1. Create PostgreSQL database:
```sql
CREATE DATABASE deepfake_db;
```
#### 2. Create .env in backend/:
```.env
DATABASE_URL=postgresql://username:password@localhost:5432/deepfake_db
```
- Replace username, password, localhost, and deepfake_db with your PostgreSQL settings.

### 4. Run FastAPI backend
```bash
uvicorn app:app --reload
```
- Server URL: http://127.0.0.1:8000
- /predict endpoint available for image upload

### 5. Run React frontend
```bash
cd ../frontend
npm install
npm run dev
```
- Open browser at: http://localhost:5173
- Drag & Drop images to see predictions and confidence bars
- Note: You can skip training and use the pre-trained model at backend/model/model.safetensors.

---

## Future Work
- Extend support for video-based deepfake detection
- Add Grad-CAM visualization for explainable AI
- Deploy backend with Docker or AWS
- Implement history analytics dashboard

---

## Contributing

---

## License
