import cv2
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
import logging
import numpy as np
from ultralytics import YOLO
import re
import torch
import pathlib
import sys
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fix PosixPath issue on Windows
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

# Mode descriptions for the unified API
MODES = {
    0: "Indian Currency Detection",
    1: "Indian Coin Detection"
}

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load models
logger.info("Loading YOLOv5 model for object detection...")
object_model = YOLO("yolov5su.pt")
logger.info("Object detection model loaded successfully.")


logger.info("Loading YOLOv5 model for currency detection...")
indian_currency_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5_models/best.pt', force_reload=True)
logger.info("Currency detection model loaded successfully.")

logger.info("Loading YOLOv5 model for coin detection...")
indian_coin_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5_models/coin_best.pt', force_reload=True)
logger.info("Currency detection model loaded successfully.")

# Global label_map for currency detection
indian_label_map = {
    0: '10 rupees',
    1: '100 rupees',
    2: '20 rupees',
    3: '200 rupees',
    4: '50 rupees',
    5: '500 rupees'
}

indian_coin_label_map = {
    0: '10RS',
    1: '1RS',
    2: '20rs',
    3: '2RS',
    4: '5RS',
    5: 'NEW',
    6: 'OLD',
}
        
def detect_indian_currency(frame):
    logger.info("Starting Indian currency detection...")

    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = indian_currency_model(frame_rgb)  # Run detection

        # ✅ Use .pandas().xyxy[0] to access detections
        detections_df = results.pandas().xyxy[0]

        if detections_df.empty:
            logger.info("No detections made.")
            return []

        detected_currencies = []

        for _, row in detections_df.iterrows():
            confidence = row['confidence'] * 100
            class_id = int(row['class'])

            if confidence > 75:
                currency_label = indian_label_map[class_id]  # get readable label
                detected_currencies.append({
                    "currency": currency_label,
                    "confidence": f"{confidence:.2f}%"
                })

                logger.info(f"Detected: {currency_label} (Class ID: {class_id}, Confidence: {confidence:.2f}%)")

        logger.info(f"Currency detection results: {detected_currencies}")
        return detected_currencies  # Make sure to return in the proper format

    except Exception as e:
        logger.error(f"Error during detection: {e}")
        return []

def detect_indian_coin(frame):
    logger.info("Starting Indian coin detection...")

    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = indian_coin_model(frame_rgb)  # Run detection

        # ✅ Use .pandas().xyxy[0] to access detections
        detections_df = results.pandas().xyxy[0]

        if detections_df.empty:
            logger.info("No detections made.")
            return []

        detected_currencies = []

        for _, row in detections_df.iterrows():
            confidence = row['confidence'] * 100
            class_id = int(row['class'])

            if confidence > 75:
                currency_label = indian_coin_label_map[class_id]  # get readable label
                detected_currencies.append({
                    "currency": currency_label,
                    "confidence": f"{confidence:.2f}%"
                })

                logger.info(f"Detected: {currency_label} (Class ID: {class_id}, Confidence: {confidence:.2f}%)")

        logger.info(f"Coin detection results: {detected_currencies}")
        return detected_currencies  # Make sure to return in the proper format

    except Exception as e:
        logger.error(f"Error during detection: {e}")
        return []

# API endpoint for processing
@app.post("/api/process")
async def process_request(
    mode: int = Query(..., ge=0, le=1, description="Mode (0:Indian Currency , 1:Indian Coin)"),
    file: UploadFile = File(...)
):
    try:
        # Read and decode the uploaded image
        content = await file.read()
        np_array = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Process the image based on the mode
        if mode == 0:
            logger.info("Processing mode: Indian Currency Detection")
            results = detect_indian_currency(frame)
        elif mode == 1:
            logger.info("Processing mode: Indian Coin Detection")
            results = detect_indian_coin(frame)
            
        else:
            raise HTTPException(status_code=400, detail="Invalid mode")

        # Return a unified response structure
        return {
            "status": "success",
            "mode": MODES[mode],  # Add mode information for clarity
            "results": results
        }

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
