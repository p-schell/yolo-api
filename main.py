from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os
import uuid

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

        # Jeder Upload bekommt einen eindeutigen temporären Pfad
        temp_path = f"/tmp/{uuid.uuid4().hex}.jpg" if os.name != "nt" else f"temp_{uuid.uuid4().hex}.jpg"

        # Prüfen, ob UploadFile oder App Inventor PostFile
        if hasattr(file_field, "file"):
            # UploadFile (Swagger)
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file_field.file, buffer)
        else:
            # App Inventor liefert nur Pfad
            temp_path = file_field
            if temp_path.startswith("file://"):
                temp_path = temp_path.replace("file://", "")

        # YOLO-Vorhersage
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

        # Optional: temporäre Datei löschen
        try:
            if os.path.exists(temp_path) and os.name != "nt":  # Nicht auf Android löschen
                os.remove(temp_path)
        except:
            pass

        return {"detections": detections}

    except Exception as e:
        # Fehler sauber zurückgeben
        return {"error": str(e)}
