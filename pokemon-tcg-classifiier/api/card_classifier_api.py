import os
import json
import base64
from io import BytesIO
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch.optim as optim
import threading
import uvicorn
import time
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Import training functions from card-classifier.py
from api.card_classifier_function import train_model, get_model
from api.download_sets import download_all_sets

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        np_img = np.array(img)
        noise = np.random.normal(self.mean, self.std, np_img.shape)
        noisy_img = np_img + noise
        noisy_img = np.clip(noisy_img, 0, 255)  
        return Image.fromarray(noisy_img.astype(np.uint8))
    
class CropTopHalf(object):
    def __call__(self, img):
        width, height = img.size  
        cropped_img = img.crop((0, 0, width, height // 2))  
        return cropped_img
    

class PredictRequest(BaseModel):
    image_base64: str
    model_name: str = "efficientnet_b0"
    
class TrainRequest(BaseModel):
    model_name: str = "efficientnet_b0"
    dataset_exists: bool = False  
    
class PokemonClassifier:
    def __init__(self, model_name:str, model_path: str, label_encoder_path: str, device: str = None):
       
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        #take from the file in label_encoder_path the number of classes
        num_classes = len(np.load(label_encoder_path, allow_pickle=True))
       
        self.model = get_model(model_name, model_path, num_classes, device)
        self.model.eval()

       
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.load(label_encoder_path, allow_pickle=True)

     
        self.transform = transforms.Compose([
        CropTopHalf(),
        transforms.Resize((128, 96)),  
        #transforms.Pad(padding=(16, 16), fill=0, padding_mode='constant'),  
        transforms.RandomRotation(degrees=5),  
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def predict_image(self, image_path: str) -> str:
        """
        Predict the class of an image.

        Args:
            image_path (str): Path to the image to predict.

        Returns:
            str: The predicted class label.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"The image at {image_path} does not exist.")
        
       
        image = Image.open(image_path).convert("RGB")
        self.transform = transforms.Compose([
            CropTopHalf(),
            transforms.Resize((128, 96)),  
            #transforms.Pad(padding=(16, 16), fill=0, padding_mode='constant'),  
           # transforms.RandomRotation(degrees=5),  
            #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  
            #transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = self.transform(image).unsqueeze(0)


        with torch.no_grad():
            image = image.to(self.device)
            output = self.model(image)
            pred_idx = output.argmax(dim=1).item()
        
      
        predicted_label = self.label_encoder.inverse_transform([pred_idx])[0]
        
        return predicted_label

app = FastAPI()

    
@app.post("/predict")
def predict(request: PredictRequest):
    classifier = None
    #check if the model and classes files exist
    if not os.path.exists("model/pokemon_classifier.pth"):
        print("Model file not found. Please upload the model file.")
    else:
        classifier = PokemonClassifier(model_name=request.model_name, model_path="model/pokemon_classifier.pth", label_encoder_path="classes/classes.npy")
    
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    if not request.image_base64:
        raise HTTPException(status_code=400, detail="No image data provided.")
    
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image data.")

    try:
       
        inference_transform = transforms.Compose([
            CropTopHalf(),
            transforms.Resize((128, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        image_tensor = inference_transform(image).unsqueeze(0).to(classifier.device)
        with torch.no_grad():
            output = classifier.model(image_tensor)
            pred_idx = output.argmax(dim=1).item()
        predicted_label = classifier.label_encoder.inverse_transform([pred_idx])[0]

        return {"predicted_class": predicted_label}
               # "json_info": json_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def async_background_training(model_name: str, dataset_exists: bool = False):
    """Funzione async per eseguire il training in background"""
    global training_status
    
    try:
        training_status["is_training"] = True
        training_status["status"] = "initializing"
        training_status["error"] = None
        training_status["accuracy"] = None
        training_status["progress"] = 0
        dataset_path = 'card_images'
        if(not dataset_exists):
            # Crea la directory se non esiste
            os.makedirs(dataset_path, exist_ok=True)
            training_status["progress"] = 10
            training_status["status"] = "downloading_data"
            print("Starting data download...")
            # Await della funzione async
            await download_all_sets()
        else:
            print("Dataset already exists, skipping download.")
            training_status["progress"] = 40
        
        image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) == 0:
            raise Exception(f"No images found in {dataset_path} after download")
        
        print(f"Found {len(image_files)} images in dataset")
        training_status["progress"] = 60
        training_status["status"] = "training_model"
        
        accuracy = train_model(model_name)
        
        training_status["is_training"] = False
        training_status["status"] = "completed"
        training_status["accuracy"] = accuracy
        training_status["progress"] = 100
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        training_status["is_training"] = False
        training_status["status"] = "error"
        training_status["error"] = str(e)
        training_status["progress"] = 0

def background_training(model_name: str, dataset_exists: bool = False):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(async_background_training(model_name, dataset_exists))
    finally:
        loop.close()

@app.delete("/reset")
def reset_model():
    model_path = "model/pokemon_classifier.pth"
    classes_path = "classes/classes.npy"
    
    if os.path.exists(model_path):
        os.remove(model_path)
    
    if os.path.exists(classes_path):
        os.remove(classes_path)
    
    return {"status": "Model and classes reset successfully"}

@app.post("/train")
def train_model_endpoint(request: TrainRequest):
    """Avvia il training in background"""
    global training_status
    
    if training_status["is_training"]:
        raise HTTPException(
            status_code=409, 
            detail="Training already in progress. Use /train/status to check progress."
        )
    
    # Avvia il training in un thread separato
    training_thread = threading.Thread(
        target=background_training, 
        args=(request.model_name, request.dataset_exists)
    )
    training_thread.daemon = True  
    training_thread.start()
    
    return {
        "status": "Training started in background",
        "message": "Use /train/status to check progress"
    }

@app.get("/train/status")
def get_training_status():
    return training_status

@app.get("/")
def read_root():
    return {"message": "Welcome to the Pokémon Card Classifier API!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}
    
training_status = {
    "is_training": False,
    "status": "idle",
    "accuracy": None,
    "error": None,
    "progress": 0
}
if __name__ == "__main__":
    # Crea le directory necessarie se non esistono
    os.makedirs("model", exist_ok=True)
    os.makedirs("classes", exist_ok=True)
    os.makedirs("card_images", exist_ok=True)
    #test path exists
    if not os.path.exists("model/pokemon_classifier.pth"):
        print("Model file not found. Please upload the model file.")
    
        
    if not os.path.exists("classes/classes.npy"):
        print("Classes file not found. Please upload the classes file.")

    # Run the FastAPI app on host 0.0.0.0 and port 8000
    uvicorn.run("api.card_classifier_api:app", host="0.0.0.0", port=8000, reload=True)