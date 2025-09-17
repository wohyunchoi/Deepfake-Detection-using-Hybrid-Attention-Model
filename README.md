# Deepfake-Detection-using-Hybrid-Attention-Model

## 목차 (Table of Contents)
- [Introduction](#introduction)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Useful for](#useful-for)
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

## Screenshots / Demo
> 진행된 구현 부분만 우선 공개합니다. (추후 업데이트 예정)
<img width="1386" height="607" alt="image" src="https://github.com/user-attachments/assets/ee386a79-8713-47ad-8826-7640320de607" />


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

## Setup and Running

## Future Work

- 여러 모델을 학습시킨 뒤 결과를 Ensemble하여 최종 결과를 출력하는 Ensemble Method 사용
- GAN 기반 딥페이크 생성 및 대응 연구

## Contributing

## License
