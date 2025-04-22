"""
Simple script to train and evaluate a model with default params
"""
import os
import torch
import torch.optim as optim

from src.data.dataset import prepare_data
from src.models.model import create_model
from src.models.train import train_model, evaluate_model, save_model, plot_training_history

# Configuration
CONFIG = {
    "dataset_dir": "data/raw/EuroSAT_RGB",
    "output_dir": "data/processed",
    "model_dir": "models",
    "model_name": "resnet18",  # options: resnet18, resnet50, efficientnet_b0, efficientnet_b4
    "freeze_backbone": True,
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 0.001,
    "num_workers": 0  # Set to 0 for CPU
}

def main():
    # Create directories
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    train_loader, val_loader, test_loader, class_to_idx, idx_to_class = prepare_data(
        dataset_dir=CONFIG["dataset_dir"],
        output_dir=CONFIG["output_dir"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"]
    )
    
    # Create model
    num_classes = len(class_to_idx)
    model = create_model(
        model_name=CONFIG["model_name"],
        num_classes=num_classes,
        freeze_backbone=CONFIG["freeze_backbone"]
    )
    
    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train model
    print(f"Training {CONFIG['model_name']} with {'frozen' if CONFIG['freeze_backbone'] else 'unfrozen'} backbone")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=CONFIG["epochs"],
        device=device
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    save_model(
        model=model,
        model_name=CONFIG["model_name"],
        freeze_backbone=CONFIG["freeze_backbone"],
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        optimizer=optimizer,
        scheduler=scheduler,
        history=history,
        num_epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        model_dir=CONFIG["model_dir"]
    )
    
    # Evaluate model
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    evaluate_model(model, test_loader, class_names, device)

if __name__ == "__main__":
    main()