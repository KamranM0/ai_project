import torch
from sklearn.metrics import f1_score


def train_model(model, dataloaders, criterion, optimizer, device, scheduler=None, epochs=25, writer=None):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()  

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_f1 = f1_score(all_labels, all_preds, average='macro')  
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} F1-Score: {epoch_f1:.4f} Accuracy: {epoch_acc:.4f}")
            for name, param in model.named_parameters():
                if 'weight' in name:
                    print(f"{name}: {param.data.mean()}")
            if writer:
                writer.add_scalar(f"{phase.capitalize()} F1 score", epoch_f1, epoch)
                writer.add_scalar(f"{phase.capitalize()} Loss", epoch_loss, epoch)
                writer.add_scalar(f"{phase.capitalize()} Accuracy", epoch_acc, epoch)

            if phase == 'val' and scheduler:
                scheduler.step(epoch_loss)

    print("Training complete")
