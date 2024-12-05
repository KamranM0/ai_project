import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.resnet18 import ResNet18
from models.vgg16 import VGG16
from models.model_training import train_model
from utils.optimizer import get_optimizer  
from utils.dataset_splitter import split_dataset
from utils.evaluation import evaluate_model
import yaml
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter


def main():

    config = yaml.safe_load(open('config/config.yaml'))

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')

    print("Splitting dataset...")
    split_dataset(config['dataset']['raw_data_path'], config['dataset']['processed_data_path'])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(config['dataset']['train_path'], transform=transform)
    val_dataset = datasets.ImageFolder(config['dataset']['val_path'], transform=transform)
    test_dataset = datasets.ImageFolder(config['dataset']['test_path'], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    writer = SummaryWriter()

    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model = VGG16(num_classes=len(train_dataset.classes)) 
    
    model.to(device) 

    optimizer = get_optimizer(model, optimizer_name=config['training']['optimizer'], lr=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    criterion = torch.nn.CrossEntropyLoss() 

    print("Training model...")
    train_model(
    model, 
    {'train': train_loader, 'val': val_loader}, 
    criterion, 
    optimizer, 
    device, 
    scheduler=scheduler,
    epochs=config['training']['epochs'], 
    writer=writer
    )

    print("Evaluating model on test set...")
    accuracy,f1 = evaluate_model(model, test_loader, device)
    print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Set F1: {f1 * 100:.2f}%")

    writer.add_scalar("Test Accuracy", accuracy, 0)
    writer.add_scalar("Test F1 Score", f1, 0)

    model_save_path = os.path.join(checkpoint_dir, "currency_model_Resnet_IMAGENET1K_V1.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    writer.close()

if __name__ == '__main__':
    main()
