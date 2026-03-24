from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os

app = FastAPI()

# CORS erlauben
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YOLO Modell laden
model = YOLO("yolov8n.pt")

@app.post("/detect")
async def detect(request: Request):
    try:
        # Alle Form-Felder abfragen
        form = await request.form()
        file_field = None
        for key in form.keys():
            file_field = form[key]
            break

        if file_field is None:
            return {"error": "No file received"}

        # Prüfen, ob file_field ein UploadFile ist (Swagger) oder ein String (App Inventor)
        if hasattr(file_field, "file"):
            # UploadFile → Inhalt kopieren
            temp_path = "/tmp/temp.jpg" if os.name != "nt" else "temp.jpg"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file_field.file, buffer)
        else:
            # App Inventor PostFile liefert nur den Pfad → direkt verwenden
            temp_path = file_field
            if temp_path.startswith("file://"):
                temp_path = temp_path.replace("file://", "")  # Android Pfad anpassen

        # YOLO Vorhersage
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
        # Fehler sauber zurückgeben
        return {"error": str(e)}
