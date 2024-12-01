# models/model_training.py
import torch
from sklearn.metrics import f1_score
from utils.optimizer import get_optimizer  # Now importing from utils/optimizer.py

def train_model(model, dataloaders, criterion, optimizer, device, scheduler=None, epochs=25, writer=None):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Log metrics to TensorBoard
            if writer:
                writer.add_scalar(f"{phase.capitalize()} Loss", epoch_loss, epoch)
                writer.add_scalar(f"{phase.capitalize()} Accuracy", epoch_acc, epoch)

            # Step the scheduler if applicable
            if phase == 'val' and scheduler:
                scheduler.step(epoch_loss)

    print("Training complete")