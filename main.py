"""
Main script to train and evaluate models
"""
import os
import argparse
import torch
import torch.optim as optim

from src.data.dataset import prepare_data
from src.models.model import create_model
from src.models.train import train_model, evaluate_model, save_model, plot_training_history

def main(args):
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    train_loader, val_loader, test_loader, class_to_idx, idx_to_class = prepare_data(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    num_classes = len(class_to_idx)
    model = create_model(
        model_name=args.model_name,
        num_classes=num_classes,
        freeze_backbone=args.freeze_backbone
    )
    
    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train model
    print(f"Training {args.model_name} with {'frozen' if args.freeze_backbone else 'unfrozen'} backbone")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    save_model(
        model=model,
        model_name=args.model_name,
        freeze_backbone=args.freeze_backbone,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        optimizer=optimizer,
        scheduler=scheduler,
        history=history,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        model_dir=args.model_dir
    )
    
    # Evaluate model
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    evaluate_model(model, test_loader, class_names, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models for satellite imagery classification")
    
    # Data arguments
    parser.add_argument("--dataset_dir", type=str, default="../data/raw/EuroSAT_RGB",
                        help="Directory containing EuroSAT dataset")
    parser.add_argument("--output_dir", type=str, default="../data/processed",
                        help="Directory to save processed data")
    parser.add_argument("--model_dir", type=str, default="../models",
                        help="Directory to save trained models")
    
    # Training arguments
    parser.add_argument("--model_name", type=str, default="resnet18",
                        choices=["resnet18", "resnet50", "efficientnet_b0", "efficientnet_b4"],
                        help="Model architecture to use")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone weights")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for data loading")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    
    args = parser.parse_args()
    main(args)