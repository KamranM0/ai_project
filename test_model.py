import torch
from torchvision import transforms
from PIL import Image
from models.resnet18 import ResNet18
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import yaml
def load_model(device, num_classes):
    model = ResNet18(num_classes=num_classes) 
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "currency_model_Resnet_IMAGENET1K_V1.pth")))
    model.to(device)
    model.eval()  
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),      
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    image = Image.open(image_path).convert('RGB') 
    return transform(image).unsqueeze(0), image 

def predict(model, image_tensor, class_names, device):
    image_tensor = image_tensor.to(device)  
    with torch.no_grad(): 
        outputs = model(image_tensor) 
        _, predicted_idx = torch.max(outputs, 1)  
    return class_names[predicted_idx.item()]

def display_image_with_prediction(image, predicted_class):
    plt.imshow(image)
    plt.title(f"Predicted Class: {predicted_class}")
    plt.axis('off')
    plt.show()

def main():
    config = yaml.safe_load(open('config/config.yaml'))

    test_images_dir = "dataset/test_images" 
    val_dataset = datasets.ImageFolder(config['dataset']['val_path'], transform=None)


    class_names = val_dataset.classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading model...")
    model = load_model(device, num_classes=len(class_names))
    print("Model loaded successfully.")

    print("Testing images...")
    for image_name in os.listdir(test_images_dir):
        image_path = os.path.join(test_images_dir, image_name)
        image_tensor, original_image = preprocess_image(image_path)
        predicted_class = predict(model, image_tensor, class_names, device)
        print(f"Image: {image_name}, Predicted Class: {predicted_class}")
        display_image_with_prediction(original_image, predicted_class)

if __name__ == '__main__':
    main()
