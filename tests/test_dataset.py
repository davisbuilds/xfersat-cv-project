# test_dataset.py
import unittest
import os
import torch
from src.data.dataset import EuroSATDataset, get_transforms

class TestDataset(unittest.TestCase):
    
    def setUp(self):
        self.dataset_dir = "data/raw/EuroSAT_RGB"
        self.train_transform, self.val_transform = get_transforms()
        
    def test_dataset_loading(self):
        """Test that the dataset loads and has the expected number of classes"""
        dataset = EuroSATDataset(self.dataset_dir)
        self.assertEqual(len(dataset.classes), 10, "Dataset should have 10 classes")
        
    def test_image_shape(self):
        """Test that images have the correct shape after transform"""
        dataset = EuroSATDataset(self.dataset_dir, transform=self.train_transform)
        image, _ = dataset[0]
        self.assertEqual(image.shape, torch.Size([3, 64, 64]), "Image should be 3x64x64 after transform")
        
    def test_class_mapping(self):
        """Test that class to index mapping is consistent"""
        dataset = EuroSATDataset(self.dataset_dir)
        self.assertEqual(len(dataset.class_to_idx), 10, "Should have 10 class mappings")
        # Verify some expected classes exist
        self.assertIn("Forest", dataset.class_to_idx, "Forest class should exist")
        self.assertIn("River", dataset.class_to_idx, "River class should exist")
        
if __name__ == '__main__':
    unittest.main()