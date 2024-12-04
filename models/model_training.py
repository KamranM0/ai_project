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
            all_preds = []
            all_labels = []

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

                # Accumulate loss
                running_loss += loss.item() * inputs.size(0)

                # Collect predictions and true labels for F1-score calculation
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Calculate loss and F1-score for the epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_f1 = f1_score(all_labels, all_preds, average='macro')  # Adjust average as needed

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} F1-Score: {epoch_f1:.4f}")

            # Log metrics to TensorBoard
            if writer:
                writer.add_scalar(f"{phase.capitalize()} F1 score", epoch_f1, epoch)
                writer.add_scalar(f"{phase.capitalize()} Loss", epoch_loss, epoch)

            # Step the scheduler if applicable
            if phase == 'val' and scheduler:
                scheduler.step(epoch_loss)

    print("Training complete")
