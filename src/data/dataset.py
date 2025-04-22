import os
import torch
import json
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from PIL import Image
from torchvision import transforms

class EuroSATDataset(Dataset):
    """
    Dataset class for EuroSAT
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # Load all image paths and labels
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpg'):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class TransformedSubset(Dataset):
    """
    Dataset wrapper that applies transforms to a subset
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.subset)

def get_transforms():
    """
    Return train and validation transforms
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def prepare_data(dataset_dir="../data/raw/EuroSAT_RGB", output_dir="../data/processed", batch_size=32, num_workers=0):
    """
    Prepare datasets and dataloaders

    Args:
        dataset_dir: Path to the dataset directory
        output_dir: Path to save processed data
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader, test_loader, class_to_idx, idx_to_class
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get class mappings
    full_dataset = EuroSATDataset(dataset_dir, transform=None)
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    
    # Save mappings
    with open(os.path.join(output_dir, "class_mappings.json"), "w") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f)
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - (train_size + val_size)
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Apply transforms
    train_dataset = TransformedSubset(train_dataset, train_transform)
    val_dataset = TransformedSubset(val_dataset, val_transform)
    test_dataset = TransformedSubset(test_dataset, val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Save dataset splits
    torch.save({
        'train_indices': train_dataset.subset.indices,
        'val_indices': val_dataset.subset.indices,
        'test_indices': test_dataset.subset.indices,
    }, os.path.join(output_dir, 'dataset_splits.pt'))
    
    print(f"Data preparation complete!")
    print(f"Total images: {total_size}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, class_to_idx, idx_to_class

if __name__ == "__main__":
    prepare_data()