from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil

app = FastAPI()

# CORS erlauben (für App Inventor)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YOLO Modell laden
model = YOLO("yolov8n.pt")  # kleines, schnelles Modell

@app.post("/detect")
async def detect(request: Request):
    # Versuche, die hochgeladene Datei zu bekommen
    form = await request.form()
    file = None
    for key in form.keys():
        file = form[key]
        break  # nur das erste File verwenden

    if file is None:
        return {"error": "No file received"}

    # Bild temporär speichern
    with open("temp.jpg", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # YOLO Vorhersage
    results = model("temp.jpg")
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
