# test_model.py
import unittest
import torch
from src.models.model import create_model

class TestModel(unittest.TestCase):
    
    def test_resnet18_creation(self):
        """Test that ResNet18 model is created correctly"""
        model = create_model(model_name='resnet18', num_classes=10)
        self.assertEqual(model.fc.out_features, 10, "Output layer should have 10 classes")
        
    def test_forward_pass(self):
        """Test that model forward pass works"""
        model = create_model(model_name='resnet18', num_classes=10)
        # Create a dummy input tensor
        dummy_input = torch.randn(1, 3, 64, 64)
        # Forward pass
        output = model(dummy_input)
        self.assertEqual(output.shape, torch.Size([1, 10]), "Output should have shape [batch_size, num_classes]")
        
    def test_efficientnet_creation(self):
        """Test that EfficientNet model is created correctly"""
        model = create_model(model_name='efficientnet_b0', num_classes=10)
        self.assertEqual(model.classifier[1].out_features, 10, "Output layer should have 10 classes")
        
if __name__ == '__main__':
    unittest.main()