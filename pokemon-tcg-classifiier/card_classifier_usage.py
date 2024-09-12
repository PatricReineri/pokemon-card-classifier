import os
import json
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torchvision import models
import torch.optim as optim


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
     
class PokemonClassifier:
    def __init__(self, model_path: str, label_encoder_path: str, num_classes: int = 1289, device: str = None):
       
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        
        self.model = models.resnet18(weights=None)  # Do not preload weights here
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(self.device)
        
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.load(label_encoder_path, allow_pickle=True)

 
        
        self.transform =  transforms.Compose([
    transforms.Resize((128, 128)),
    CropTopHalf(),
   # transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomRotation(10),  # Simple and fast rotation
   # #transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Reduced range for faster processing
    #transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),  # Simplified affine transformation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        
        # Load and transform the image
        image = Image.open(image_path).convert("RGB")
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Optionally resize to a consistent size first
            CropTopHalf(),  # Custom crop the top half
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            #transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            #RandomBlur(p=0.3),  # Apply random blur with a probability of 30%
            #transforms.RandomResizedCrop((128, 128), scale=(0.7, 1.0)),  # Randomly crop and resize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #AddGaussianNoise(mean=0, std=0.1)  # Adjust noise levels as needed
        ])
        image = self.transform(image).unsqueeze(0)


        with torch.no_grad():
            image = image.to(self.device)
            output = self.model(image)
            pred_idx = output.argmax(dim=1).item()
        
      
        predicted_label = self.label_encoder.inverse_transform([pred_idx])[0]
        
        return predicted_label

    def print_json_info(self, json_path: str):
        """
        Extract and print the Pokémon name, series, artist, and price from a JSON file.

        Args:
            json_path (str): Path to the JSON file containing the information.
        """
        if not os.path.isfile(json_path):
            print(f"The JSON file at {json_path} does not exist.")
            return
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
          
            if not isinstance(data, dict):
                print("The JSON file content is not a valid dictionary.")
                return
            
       
            name = data.get("name", "N/A")
            series = data.get("set", {}).get("name", "N/A") if isinstance(data.get("set"), dict) else "N/A"
            artist = data.get("artist", "N/A")
            tcgplayer_prices = data.get("tcgplayer", {}).get("prices", {})
            price = tcgplayer_prices.get("normal", {}).get("market", "N/A") if isinstance(tcgplayer_prices.get("normal"), dict) else "N/A"
            
    
            print(f"Pokémon Name: {name}")
            print(f"Series: {series}")
            print(f"Artist: {artist}")
            print(f"Price: {price}")

        except json.JSONDecodeError:
            print("Error decoding JSON file.")
        except Exception as e:
            print(f"An error occurred: {e}")



if __name__ == "__main__":
    classifier = PokemonClassifier(model_path="pokemon_classifier.pth", label_encoder_path='classes.npy')
    
    image_path = '/home/patric/Desktop/pippo/files/tcg-classifier/base1_images/sv6-104.png'
    try:
        # Predict the class of the image
        prediction = classifier.predict_image(image_path)
        print(f'Predicted class: {prediction}')
        
        # Construct JSON file path based on the predicted label
        json_filename = f"{prediction}.json"
        json_path = os.path.join('base1', json_filename)
        
     
        classifier.print_json_info(json_path)
    except Exception as e:
        print(f"An error occurred: {e}")
