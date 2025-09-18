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

(소개영상 추후 업데이트 예정)

딥페이크(Deepfake) 이미지 여부를 판별해주는 웹 애플리케이션입니다.
사용자가 업로드한 사진을 분석하여, AI 기반 모델로 진짜/가짜 여부를 예측하고 신뢰도 점수를 제공합니다.
이 프로젝트는 딥페이크로 인한 허위 정보 확산을 방지하고, 온라인상의 디지털 신뢰성 향상을 목표로 합니다.

---

## Tech Stack
- **Frontend**: React
- **Backend**: Node.js, Express  
- **Database**: MongoDB Atlas  
- **Deep Learning**: PyTorch
- **Model Weights**: safetensors  

---

## Features
- Hybrid Attention 기반 딥페이크 탐지 모델
- 실시간 이미지/영상 판별
- 사용자 친화적인 UI (웹 인터페이스 제공)

---

## Useful for
- 인공지능 보안 연구자
- 이미지 진위 판별이 필요한 서비스

---

## Model Architecture

이 프로젝트에서 사용한 **Hybrid Attention Model**은 Deepfake 영상의 특징을 효과적으로 추출하고 분류하기 위해 설계되었습니다. 모델의 전체 구조는 다음과 같습니다:

<img width="1001" height="574" alt="image" src="https://github.com/user-attachments/assets/ebd09844-28d2-4d95-a57c-e7ec3253f6a6" />


### 모델 구조 요약
1. **Input Layer**
   - 입력 영상: `(Batch, 4, 224, 224)` (RGBF 이미지)
2. **Entry Flow**
   - Xception Entry Flow + CBAM  
   - 특징 추출 및 초기 Attention 적용
3. **Middle Flow**
   - Xception Middle Flow 반복 수행 이후 CBAM 
   - 채널/공간별 중요한 패턴 강조
4. **Exit Flow**
   - Xception Exit Flow + CBAM  
   - 최종 특징 맵 생성
5. **Fully Connected Layer**
   
**핵심 특징**
- XceptionNet backbone으로 깊은 특징 추출
- CBAM 모듈 삽입으로 공간 + 채널 Attention 결합
- 기존 Xception 대비 Deepfake 검출 성능 향상

---

## Dataset

모델 학습에 사용한 데이터셋 정보입니다.

- **출처:** [https://www.kaggle.com/datasets/tusharpadhy/deepfake-dataset-merged]
- **클래스:** Real / Fake  
- **데이터 크기:**  
  - Train : Fake(약 12만장)+Real(약 13만장) 총 약 25만장
  - Valid : Fake(약 3만장)+Real(약 3만장) 총 약 6만장
  - Test : Fake(250장) + Real(250장) / Fake(약 1.6만장)+Real(약 1.5만장) 총 약 3만장

---
## Training / How to Run Experiments

모델 학습 환경 및 방법입니다.

- **환경**  
  - GPU: NVIDIA RTX 4060Ti
  - Pytorch 2.6.0+cu126, Python 3.13.1
  - 주요 라이브러리: torchvision, pytorch

- **학습 하이퍼파라미터**  
  - Optimizer: Adam  
  - Learning rate: 0.0001
  - Weight Decay : 0.00001
  - Dropout Rate : Convolution Layer는 0.2, Fully Connected Layer는 0.5
  - Batch size: 16  
  - Loss function: Binary Cross-Entropy
 
  - Sequential Fold-wise Training
     - To better utilize the entire dataset
     - The data were split into 5 folds, and a single model was trained sequentially on each fold for 10 epochs, resulting in a total of 50 epochs
     - Unlike traditional 5-fold cross-validation, the model was not reinitialized between folds, allowing it to retain knowledge across them
     - The final evaluation was conducted on a separate test set

- **학습 실행 방법**
(추가 예정)
 
---

## Results / Performance

(추가 예정)

**설명**  

---
## Screenshots / Demo
> 진행된 구현 부분만 우선 공개합니다. (추후 업데이트 예정)
<img width="1386" height="607" alt="image" src="https://github.com/user-attachments/assets/ee386a79-8713-47ad-8826-7640320de607" />

![시연real](https://github.com/user-attachments/assets/f773c27c-0894-46b6-ba4e-db59fae4a9b0)

---

## Project Structure
```plaintext
Deepfake-Detection-using-Hybrid-Attention-Model/
│
├── frontend/              # React 기반 UI
│   ├── public/
│   ├── src/
│   └── package.json
│
├── backend/               # Node.js + Express 서버
│   ├── routes/
│   ├── models/
│   ├── controllers/
│   └── server.js
│
├── model/                 # PyTorch 모델 관련 코드
│   ├── train.py
│   ├── inference.py
│   └── weights/           # safetensors 파일 저장 위치
│
└── README.md
```

---

## Setup and Running

### a. Requirements 설치

### b. 모델 다운로드 (Hugging Face)
(Hugging Face에서 safetensors 모델 가중치 다운로드)

### c. Web Localhost 실행

### d. Database 연동

### Developer / Researcher Guide

### e. 모델 재학습시키기
(PyTorch 기반 재학습 방법)

### f. 모델 교체하기
(새로운 safetensors 파일 교체 및 반영 방법)

---

## Future Work

- 여러 모델을 학습시킨 뒤 결과를 Ensemble하여 최종 결과를 출력하는 Ensemble Method 사용
- GAN 기반 딥페이크 생성 및 대응 연구

---

## Contributing

---

## License
