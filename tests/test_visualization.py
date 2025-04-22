# test_visualization.py
import unittest
import torch
import numpy as np
from PIL import Image
import os
from src.models.model import create_model
from src.visualization.visualize import predict_image
from torchvision import transforms

class TestVisualization(unittest.TestCase):
    
    def setUp(self):
        self.model = create_model(model_name='resnet18', num_classes=10)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # Create a dummy image if needed for testing
        self.test_img_path = "tests/test_data/test_image.jpg"
        os.makedirs(os.path.dirname(self.test_img_path), exist_ok=True)
        if not os.path.exists(self.test_img_path):
            # Create a simple test image (green square)
            img = np.ones((64, 64, 3), dtype=np.uint8) * 100
            img[:, :, 1] = 200  # Make it greenish
            Image.fromarray(img).save(self.test_img_path)
    
    def test_prediction_output_format(self):
        """Test that prediction function returns expected format"""
        # This test passes even if predictions are meaningless (untrained model)
        class_idx, probs, _ = predict_image(self.test_img_path, self.model, self.transform)
        self.assertIsInstance(class_idx, int, "Class index should be an integer")
        self.assertEqual(len(probs), 10, "Should have probabilities for 10 classes")
        self.assertAlmostEqual(sum(probs), 1.0, places=5, msg="Probabilities should sum to 1")
        
if __name__ == '__main__':
    unittest.main()