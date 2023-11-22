from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()
templates = Jinja2Templates(directory="templates")  # Create a templates directory in the same directory as your script

MODEL = tf.keras.models.load_model("../saved_models/1")
CLASS_NAMES = ["normal", "glaucoma", "diabetic_retinopathy", "cataract"]
endpoint = "your_endpoint_url"  # Replace with your actual endpoint URL

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
):
    image = read_file_as_image(await file.read())
    json_data = {
        "instances": image.tolist()
    }
    response = requests.post(endpoint, json=json_data)
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return templates.TemplateResponse("output.html", {"request": request, "class": predicted_class, "confidence": float(confidence)})

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
