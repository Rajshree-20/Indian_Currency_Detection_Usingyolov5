
# 🪙 Indian Currency Detection Using YOLOv5

This repository provides a deep learning-based solution for real-time Indian currency and coin detection using the YOLOv5 object detection model. It includes a web interface integrated with FastAPI that allows users to detect and hear the value of Indian currency notes and coins in real-time.

## 📌 Overview

This project uses the YOLOv5 (You Only Look Once) object detection framework to identify and classify Indian currency denominations from images or live camera feeds. A custom-built web interface enables real-time detection with audio announcements of the detected denomination.

## 🧠 Model Architecture

- **Base Model**: YOLOv5 (Small or Medium)
- **Framework**: PyTorch
- **Classes Detected**: 
  - ₹1 coin
  - ₹2 coin
  - ₹5 coin
  - ₹10 coin
  - ₹10 note
  - ₹20 note
  - ₹50 note
  - ₹100 note
  - ₹200 note
  - ₹500 note

## 📁 Repository Structure

```
Indian_Currency_Detection_Usingyolov5/
├── yolov5_models/            # pretrained models
│   ├── best.pt/              # currency model
│   └── coin_best.pt/         # coin model
├── Prepare_Data.py          # data split
├── train_on_sagemaker.py    # Script to train model
├── website.html             # website to test the models
├── server.py                # fastapi server code to call the detection models from website frontend
└── requirements.txt         # Python dependencies
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Rajshree-20/Indian_Currency_Detection_Usingyolov5.git
cd Indian_Currency_Detection_Usingyolov5
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download or Prepare Dataset

Make sure the dataset is in YOLO format and placed under the `data/` directory. If not already annotated, you can use tools like [LabelImg](https://github.com/tzutalin/labelImg) or [Roboflow](https://roboflow.com/) to annotate and export in YOLO format.

### 4. Train the model

```bash
python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt --cache
```

### 5. Start the FastAPI server

Before using the website, start the FastAPI server:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Make sure the server is running before launching the frontend.

### 6. Launch the Website

Once the FastAPI server is running:

- Open `website.html` in your browser.
- Click the **Start Free Trial** button.
- The website will access your camera to detect currency and coins in real-time.
- Detected denominations will be displayed on screen and announced using text-to-speech.

## ✅ Features

- Real-time currency and coin detection via webcam
- Audio announcement of detected currency
- Seamless web interface with backend API
- Fast and accurate detection using YOLOv5
- Lightweight and optimized for performance

## 🔧 Future Improvements

- Add counterfeit detection
- Mobile-friendly UI and deployment
- Support for visually impaired users with better accessibility

## 🙌 Acknowledgements

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- Dataset from Roboflow 

---

Made with ❤️ by [Rajshree](https://github.com/Rajshree-20)
