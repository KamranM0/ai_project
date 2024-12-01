import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.resnet18 import ResNet18
from models.vgg16 import VGG16
from models.model_training import train_model
from utils.optimizer import get_optimizer  # Correct import from utils/optimizer.py
from utils.dataset_splitter import split_dataset
from utils.evaluation import evaluate_model
import yaml
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.loss_function import FocalLoss
from torch.utils.tensorboard import SummaryWriter


def main():
    # Load configuration
    config = yaml.safe_load(open('config/config.yaml'))

    # Set device (GPU or CPU)
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')

    # Load and preprocess data
    print("Splitting dataset...")
    split_dataset(config['dataset']['raw_data_path'], config['dataset']['processed_data_path'])

    # Define transformations for data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),          # Convert images to tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize the images
    ])

    # Load datasets using ImageFolder (with the transform applied)
    train_dataset = datasets.ImageFolder(config['dataset']['train_path'], transform=transform)
    val_dataset = datasets.ImageFolder(config['dataset']['val_path'], transform=transform)
    test_dataset = datasets.ImageFolder(config['dataset']['test_path'], transform=transform)

    # Create DataLoader for each dataset
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # TensorBoard writer
    writer = SummaryWriter(log_dir="runs/currency_classification")

    # Choose model (ResNet18 or VGG16)
    model = ResNet18(num_classes=len(train_dataset.classes))  # You can switch to VGG16 if needed
    model.to(device)  # Move model to GPU if available

    # Choose optimizer
    optimizer = get_optimizer(model, optimizer_name=config['training']['optimizer'], lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Define loss function
    criterion = FocalLoss(alpha=0.75, gamma=2)  # For imbalanced datasets

    # Train the model with TensorBoard logging
    print("Training model...")
    train_model(
    model, 
    {'train': train_loader, 'val': val_loader}, 
    criterion, 
    optimizer, 
    device, 
    scheduler=scheduler,  # Pass the scheduler here
    epochs=config['training']['epochs'], 
    writer=writer
    )


    # Evaluate the model on the test set
    print("Evaluating model on test set...")
    accuracy = evaluate_model(model, test_loader, device)
    print(f"Test Set Accuracy: {accuracy * 100:.2f}%")

    # Log test set accuracy to TensorBoard
    writer.add_scalar("Test Accuracy", accuracy, 0)

    # Ensure checkpoints directory exists
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Optionally, save the trained model
    model_save_path = os.path.join(checkpoint_dir, "currency_model_Resnet_IMAGENET1K_V1.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Close the TensorBoard writer
    writer.close()

if __name__ == '__main__':
    main()
