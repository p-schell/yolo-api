from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("yolov8n.pt")

@app.post("/detect")
async def detect(request: Request):
    try:
        # App Inventor sendet das File ohne Key
        form = await request.form()
        file = None
        for key in form.keys():
            file = form[key]
            break
        if file is None:
            return {"error": "No file received"}

        # Temp-Pfad auf Render
        temp_path = "/tmp/temp.jpg" if os.name != "nt" else "temp.jpg"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # YOLO
        results = model(temp_path)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": int(box.cls[0]),
                    "class_name": model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                })

        return {"detections": detections}
    except Exception as e:
        # Fehler zurückgeben
        return {"error": str(e)}
