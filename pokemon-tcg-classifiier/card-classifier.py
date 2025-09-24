import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import random

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        np_img = np.array(img)
        if len(np_img.shape) == 2:  
            np_img = np_img[:, :, np.newaxis]
        noise = np.random.normal(self.mean, self.std, np_img.shape)
        noisy_img = np_img + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    
class CropTopHalf(object):
    def __call__(self, img):
        width, height = img.size  
        cropped_img = img.crop((0, 0, width, height // 2))  
        return cropped_img

class RandomBlur(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 3)))
        return img

class PokemonCardDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
        self.labels = [f.split('.')[0] for f in self.image_files]

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)
        self.label_map = {label: idx for idx, label in enumerate(self.label_encoder.classes_)}

        if not os.path.exists('classes.npy'):
            np.save('classes.npy', self.label_encoder.classes_)
            print(f"Classes saved in 'classes.npy': {self.label_encoder.classes_}")
        else:
            print("Classes are already saved in 'classes.npy'.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        label = self.label_map[label] 
        
        return image, torch.tensor(label, dtype=torch.long)

transform =  transforms.Compose([
    CropTopHalf(),
    transforms.Resize((128, 96)), 
    #transforms.Pad(padding=(16, 16), fill=0, padding_mode='constant'),  
    transforms.RandomRotation(degrees=5), 
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),    
    ], p=0.5),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_data_loaders(dataset_path, batch_size=32, num_workers=4):
    pokemon_dataset = PokemonCardDataset(root_dir=dataset_path, transform=transform)

    train_loader = DataLoader(pokemon_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)
    test_loader = DataLoader(pokemon_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True)
    num_classes = len(pokemon_dataset.label_encoder.classes_)
    return train_loader, test_loader, num_classes

def get_model(model_name, num_classes, device):
    match model_name:
        case 'efficientnet_b0':
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        case 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        case 'efficientnet_b1':
            model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        case 'efficientnet_b2':
            model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        case 'efficientnet_b3':
            model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        case 'efficientnet_b4':
            model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        case 'efficientnet_b5':
            model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT)
        case _:
            raise ValueError(f"Unknown model name: {model_name}")
        
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    model_path = "pokemon_classifier.pth"
    if os.path.exists(model_path):
        print(f"Found existing model file '{model_path}', loading weights...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print(f"No pre-trained model found, starting training from scratch.")
        
    return model

def setup_training(model_name):
    
    #Initialize Dataset and DataLoader
    dataset_path = 'base1_dimages'
    print(os.path.exists(dataset_path))
    print("Dataset path:", os.path.abspath(dataset_path))

    train_loader, test_loader, num_classes = get_data_loaders(dataset_path, batch_size=32, num_workers=6)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    #Initialize Model
    model = get_model(model_name, num_classes, device)
    
    #define loss function, scheduler and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-62)
    scheduler = CosineAnnealingLR(optimizer, T_max=25)
    
    return model, device, train_loader, test_loader, optimizer, criterion, scheduler
 

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]  Loss: {loss.item():.6f}')

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return accuracy



def train_model(model_name):
    accuracy = 0 
    model, device, train_loader, test_loader, optimizer, criterion, scheduler = setup_training(model_name)
    for epoch in range(1, 51):
        train(model, device, train_loader, optimizer, criterion, epoch)
        if epoch % 5 == 0:
            accuracy = test(model, device, test_loader, criterion)
        scheduler.step()

        if accuracy >= 99.9:
            print("Accuracy reached 99.9%, saving model...")
            torch.save(model.state_dict(), "pokemon_classifier.pth")
            break

    torch.save(model.state_dict(), "pokemon_classifier.pth")
    
    
    
if __name__ == '__main__':
    train_model(model_name="efficientnet_b0")
