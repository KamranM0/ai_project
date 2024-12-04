import torch
from torchvision import transforms
from PIL import Image
from models.resnet18 import ResNet18  # Import your ResNet18 model
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import yaml

# Define function to load the model
def load_model(device, num_classes):
    model = ResNet18(num_classes=num_classes)  # Initialize the model
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "currency_model_Resnet_IMAGENET1K_V1.pth")))
    model.to(device)  # Move model to device
    model.eval()  # Set model to evaluation mode
    return model

# Define function to preprocess images
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224
        transforms.ToTensor(),         # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize
    ])
    image = Image.open(image_path).convert('RGB')  # Open image and ensure it's RGB
    return transform(image).unsqueeze(0), image  # Return both tensor and original image

# Define function to predict the class of an image
def predict(model, image_tensor, class_names, device):
    image_tensor = image_tensor.to(device)  # Move image to device
    with torch.no_grad():  # Disable gradient computation
        outputs = model(image_tensor)  # Forward pass
        _, predicted_idx = torch.max(outputs, 1)  # Get the class index with the highest score
    return class_names[predicted_idx.item()]  # Return the class name

# Function to display image with prediction
def display_image_with_prediction(image, predicted_class):
    plt.imshow(image)
    plt.title(f"Predicted Class: {predicted_class}")
    plt.axis('off')
    plt.show()

# Main function
def main():
    # Configuration
    config = yaml.safe_load(open('config/config.yaml'))

    test_images_dir = "dataset/test_images"  # Directory containing test images
    val_dataset = datasets.ImageFolder(config['dataset']['val_path'], transform=None)


    class_names = val_dataset.classes  # Replace with your actual class names
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained model
    print("Loading model...")
    model = load_model(device, num_classes=len(class_names))
    print("Model loaded successfully.")

    # Loop through all images in the test_images directory
    print("Testing images...")
    for image_name in os.listdir(test_images_dir):
        image_path = os.path.join(test_images_dir, image_name)

        # Preprocess the image
        image_tensor, original_image = preprocess_image(image_path)

        # Predict the class of the image
        predicted_class = predict(model, image_tensor, class_names, device)

        # Display the result
        print(f"Image: {image_name}, Predicted Class: {predicted_class}")
        display_image_with_prediction(original_image, predicted_class)

if __name__ == '__main__':
    main()
