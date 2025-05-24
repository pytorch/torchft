"""
Utility functions and classes for TorchFT examples.
"""

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np


class SyntheticCIFAR10(data.Dataset):
    """
    Synthetic CIFAR10-like dataset for testing purposes.
    
    This dataset generates synthetic 32x32 RGB images with 10 classes,
    mimicking the structure and interface of torchvision.datasets.CIFAR10
    without requiring network downloads.
    
    Args:
        root (str): Not used, kept for interface compatibility with CIFAR10
        train (bool): Whether to generate training or test data
        download (bool): Not used, kept for interface compatibility
        transform: Optional transform to be applied on samples
        target_transform: Optional transform to be applied on targets
        size (int): Number of samples to generate (default: 1000)
    """
    
    def __init__(self, root=None, train=True, download=None, transform=None, 
                 target_transform=None, size=1000):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.size = size
        
        # Use fixed seed for deterministic generation
        np.random.seed(42 if train else 123)
        torch.manual_seed(42 if train else 123)
        
        # Generate synthetic data
        self.data = self._generate_images()
        self.targets = self._generate_labels()
        
        # Class names matching CIFAR10
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    def _generate_images(self):
        """Generate synthetic 32x32x3 images."""
        # Create synthetic images with some structure (not just random noise)
        images = []
        for i in range(self.size):
            # Create base pattern
            img = np.zeros((32, 32, 3), dtype=np.uint8)
            
            # Add some structured patterns based on index
            pattern_type = i % 4
            if pattern_type == 0:
                # Gradient pattern
                for row in range(32):
                    img[row, :, 0] = (row * 8) % 256
                    img[:, row, 1] = (row * 8) % 256
            elif pattern_type == 1:
                # Checkerboard pattern
                for row in range(32):
                    for col in range(32):
                        if (row // 4 + col // 4) % 2:
                            img[row, col] = [255, 255, 255]
            elif pattern_type == 2:
                # Circular pattern
                center = 16
                for row in range(32):
                    for col in range(32):
                        dist = ((row - center) ** 2 + (col - center) ** 2) ** 0.5
                        intensity = int((np.sin(dist / 3) + 1) * 127)
                        img[row, col] = [intensity, intensity // 2, intensity]
            else:
                # Random noise pattern
                img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            
            images.append(img)
        
        return np.array(images)
    
    def _generate_labels(self):
        """Generate synthetic labels (0-9)."""
        # Distribute labels evenly across classes
        labels = []
        for i in range(self.size):
            labels.append(i % 10)
        return labels
    
    def __getitem__(self, index):
        """Get a sample from the dataset."""
        img, target = self.data[index], self.targets[index]
        
        # Convert numpy array to PIL Image (matching CIFAR10 behavior)
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
    def __len__(self):
        """Return the size of the dataset."""
        return self.size


def get_cifar10_dataset(root="./cifar", train=True, download=True, transform=None, quick_run=False):
    """
    Get CIFAR10 dataset - either real or synthetic based on quick_run flag.
    
    Args:
        root (str): Root directory for real CIFAR10 data
        train (bool): Whether to get training or test data
        download (bool): Whether to download real CIFAR10 data
        transform: Transform to apply to the data
        quick_run (bool): If True, use synthetic data; if False, use real CIFAR10
    
    Returns:
        Dataset: Either SyntheticCIFAR10 or torchvision.datasets.CIFAR10
    """
    if quick_run:
        print("Using synthetic CIFAR10 dataset")
        return SyntheticCIFAR10(
            root=root, 
            train=train, 
            download=download, 
            transform=transform,
            size=1000  # Smaller dataset for quick testing
        )
    else:
        print("Using real CIFAR10 dataset")
        import torchvision.datasets
        return torchvision.datasets.CIFAR10(
            root=root, 
            train=train, 
            download=download, 
            transform=transform
        )
