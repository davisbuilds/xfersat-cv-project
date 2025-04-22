import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

def train_model(model, train_loader, val_loader, criterion=None, optimizer=None, 
                scheduler=None, num_epochs=10, device=None):
    """
    Train a PyTorch model
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train for
        device: Device to train on
        
    Returns:
        model: Trained model
        history: Training history
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model = model.to(device)
    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    # For tracking metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Stats
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            # Store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
                if scheduler:
                    scheduler.step()
            else:  # val
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Deep copy the model if best validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict().copy()
        
        print()
    
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

def evaluate_model(model, test_loader, class_names, device=None):
    """
    Evaluate a trained model on the test set
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        class_names: List of class names
        device: Device to evaluate on
        
    Returns:
        y_true: True labels
        y_pred: Predicted labels
        test_acc: Test accuracy
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # Calculate metrics
    test_acc = (np.array(y_true) == np.array(y_pred)).mean()
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print(f"Test accuracy: {test_acc:.4f}")
    
    return y_true, y_pred, test_acc

def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history: Dictionary containing training history
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

def save_model(model, model_name, freeze_backbone, class_to_idx, idx_to_class, 
            optimizer=None, scheduler=None, history=None, num_epochs=None, 
            batch_size=None, model_dir="../models"):
    """
    Save a trained model with timestamp to prevent overwriting
    
    Args:
        model: Trained PyTorch model
        model_name: Name of the model
        freeze_backbone: Whether backbone was frozen during training
        class_to_idx: Class to index mapping
        idx_to_class: Index to class mapping
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        history: Training history
        num_epochs: Number of epochs trained
        batch_size: Batch size used for training
        model_dir: Directory to save model
    """
    import datetime
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Add timestamp to prevent overwriting
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_save_path = os.path.join(
        model_dir, 
        f"{model_name}_{'frozen' if freeze_backbone else 'unfrozen'}_{timestamp}.pth"
    )
    
    # Create a comprehensive save dictionary with all relevant model information
    save_dict = {
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'model_name': model_name,
        'freeze_backbone': freeze_backbone,
        'timestamp': timestamp
    }
    
    # Add optional components if provided
    if optimizer:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    
    if history:
        save_dict['history'] = history
    
    if num_epochs:
        save_dict['num_epochs'] = num_epochs
    
    if batch_size:
        save_dict['batch_size'] = batch_size
        
    torch.save(save_dict, model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return model_save_path