import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
from torchcam.methods import GradCAM
from torchvision import transforms

def predict_image(image_path, model, transform, device=None):
    """
    Make a prediction on a single image
    
    Args:
        image_path: Path to the image file
        model: PyTorch model for inference
        transform: Preprocessing transformations
        device: Device to run inference on
        
    Returns:
        predicted_class: Class index with highest probability
        probabilities: List of probabilities for all classes
        image: Original PIL image
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    return predicted.item(), probabilities.cpu().tolist(), image

def visualize_prediction(image_path, model, transform, idx_to_class, device=None):
    """
    Visualize an image with its prediction and class probabilities
    
    Args:
        image_path: Path to the image file
        model: PyTorch model for inference
        transform: Preprocessing transformations
        idx_to_class: Dictionary mapping indices to class names
        device: Device to run inference on
    """
    predicted_class, probabilities, image = predict_image(image_path, model, transform, device)
    
    # Get class name
    class_name = idx_to_class[predicted_class]
    
    # Get top 5 probabilities
    top5_prob, top5_classes = torch.topk(torch.tensor(probabilities), 5)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show image
    ax1.imshow(image)
    ax1.set_title(f"Predicted: {class_name}")
    ax1.axis('off')
    
    # Show probabilities
    top5_labels = [idx_to_class[i.item()] for i in top5_classes]
    colors = ['green' if i == predicted_class else 'grey' for i in top5_classes]
    
    ax2.barh(range(5), top5_prob, color=colors)
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(top5_labels)
    ax2.set_xlabel('Probability')
    ax2.set_title('Top 5 Predictions')
    
    plt.tight_layout()
    plt.show()

def visualize_gradcam(image_path, model, transform, idx_to_class, device=None):
    """
    Visualize Class Activation Maps (GradCAM) for an image
    
    Args:
        image_path: Path to the image file
        model: PyTorch model for inference
        transform: Preprocessing transformations
        idx_to_class: Dictionary mapping indices to class names
        device: Device to run inference on
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Determine target layer based on model architecture
    model_name = model.__class__.__name__
    if 'ResNet' in model_name:
        target_layer = model.layer4[-1]
    elif 'EfficientNet' in model_name:
        target_layer = model.features[-1]
    else:
        print(f"GradCAM not implemented for {model_name}")
        return
    
    # Initialize GradCAM
    gradcam = GradCAM(model, target_layer)
    
    # Get activation map
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    predicted_class = torch.argmax(output, dim=1).item()
    
    # Generate heatmap
    try:
        activation_map = gradcam(input_tensor, [predicted_class])
        
        # Extract and process heatmap
        heatmap = activation_map[0].cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Activation map
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Class Activation Map')
        axes[1].axis('off')
        
        # Overlay
        from matplotlib.cm import jet
        heatmap_resized = Image.fromarray(heatmap).resize((image.width, image.height))
        heatmap_normalized = heatmap_resized / (heatmap_resized.max() + 1e-9)
        heatmap_colored = jet(np.array(heatmap_normalized))[:, :, :3]
        
        # Create overlay (blend original image with heatmap)
        overlay = 0.7 * np.array(image) / 255.0 + 0.3 * heatmap_colored
        overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
        
        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay: {idx_to_class[predicted_class]}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        print(f"Error generating GradCAM: {e}")
    finally:
        # Clean up by removing hooks
        gradcam.remove_hooks()

def get_random_images(dataset_dir, num_images=5):
    """
    Get random image paths from the dataset
    
    Args:
        dataset_dir: Path to the dataset directory
        num_images: Number of random images to select
        
    Returns:
        images: List of image file paths
    """
    images = []
    class_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    for _ in range(num_images):
        # Select random class
        class_name = random.choice(class_dirs)
        class_path = os.path.join(dataset_dir, class_name)
        
        # Select random image from class
        image_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
        if image_files:  # Make sure there are images in this class
            random_image = random.choice(image_files)
            image_path = os.path.join(class_path, random_image)
            images.append(image_path)
    
    return images

def analyze_misclassifications(model, dataset_dir, transform, class_to_idx, idx_to_class, num_samples=5, device=None):
    """
    Find and visualize misclassified examples
    
    Args:
        model: PyTorch model for inference
        dataset_dir: Path to the dataset directory
        transform: Preprocessing transformations
        class_to_idx: Dictionary mapping class names to indices
        idx_to_class: Dictionary mapping indices to class names
        num_samples: Maximum number of misclassifications to find
        device: Device to run inference on
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"\n--- Misclassification Analysis ---")
    print(f"Searching for up to {num_samples} misclassifications...")
    
    misclassified_images = []
    
    # Iterate through classes
    for class_name in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        true_class_idx = class_to_idx[class_name]
        
        # Only look at a random subset of images for efficiency
        image_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
        for image_file in random.sample(image_files, min(50, len(image_files))):
            image_path = os.path.join(class_path, image_file)
            
            # Predict
            pred_class, _, _ = predict_image(image_path, model, transform, device)
            
            # If misclassified
            if pred_class != true_class_idx:
                misclassified_images.append((image_path, true_class_idx, pred_class))
                
                # Break after finding enough misclassifications
                if len(misclassified_images) >= num_samples:
                    break
        
        if len(misclassified_images) >= num_samples:
            break
    
    # Visualize misclassifications
    if misclassified_images:
        print(f"Found {len(misclassified_images)} misclassifications")
        fig, axes = plt.subplots(1, len(misclassified_images), figsize=(5*len(misclassified_images), 5))
        if len(misclassified_images) == 1:
            axes = [axes]
        
        for i, (img_path, true_class, pred_class) in enumerate(misclassified_images):
            img = Image.open(img_path).convert('RGB')
            axes[i].imshow(img)
            axes[i].set_title(f"True: {idx_to_class[true_class]}\nPred: {idx_to_class[pred_class]}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print("No misclassifications found in the samples analyzed.")