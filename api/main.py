from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
import nest_asyncio
nest_asyncio.apply()

app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:3000",
    ]

app.add_middleware(CORSMiddleware, 
                   allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )

MODEL = load_model("F:\\yedek\\00 AI-ML HER ŞEY\\0_PROJELER\\End-to-end\\plant_disease\\saved_models\\1")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return "Hello, I am working"


def read_file_as_image(data) -> np.ndarray:
        image = np.array(Image.open(BytesIO(data)))
        return image
        
@app.post("/predict")
async def prediction(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)
    
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return {
            "class": predicted_class,
            "confidence": float(confidence)
            }

if __name__ == "__main__":
    uvicorn.run(app, host = "localhost", port=8000)
    
