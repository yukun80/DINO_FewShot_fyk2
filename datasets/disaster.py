
import os
import json
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset

class DisasterDataset(Dataset):
    """
    A PyTorch Dataset for the Exp_Disaster_Few-Shot dataset.

    This dataset class is designed to work with a JSON split file that defines
    the support and query sets for a few-shot learning task.
    """
    def __init__(self, root, split_file, mode='support', transforms=None):
        """
        Args:
            root (str): The project root directory. Paths in the split file are relative to this.
            split_file (str): Path to the JSON file containing the data splits.
            mode (str): 'support' or 'query' to specify which set to load.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.mode = mode
        self.transforms = transforms
        self.images = []
        self.labels = []

        # Load the splits from the JSON file
        with open(split_file, 'r') as f:
            splits = json.load(f)

        if mode not in splits:
            raise ValueError(f"Mode '{mode}' not found in split file. Available modes: {list(splits.keys())}")

        # Get the image and label paths for the specified mode
        image_paths = splits[mode]['images']
        label_paths = splits[mode]['labels']

        # Store absolute paths
        self.images = [os.path.join(self.root, p) for p in image_paths]
        self.labels = [os.path.join(self.root, p) for p in label_paths]

        print(f"Initialized DisasterDataset in '{mode}' mode. Found {len(self.images)} samples.")

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.

        Returns:
            tuple: (image, label, image_path) where image is the input tensor, 
                   label is the segmentation mask, and image_path is the original file path.
        """
        try:
            # Open image and label using rasterio
            with rasterio.open(self.images[idx]) as src:
                # Ensure reading 3 channels, and transpose to (C, H, W)
                image = src.read((1, 2, 3)).astype(np.float32)
            
            with rasterio.open(self.labels[idx]) as src:
                label = src.read(1).astype(np.int64)

            # Remap label values: 20 -> 1, keeping 0 as background
            label[label == 20] = 1

            # Convert numpy arrays to PyTorch tensors
            image_tensor = torch.from_numpy(image)
            label_tensor = torch.from_numpy(label)

            # Note: The original project's transforms in `utils/transforms.py` are PIL-based.
            # They are not compatible with the tensors produced here.
            # If augmentations are needed, new tensor-based transforms should be implemented.
            if self.transforms:
                # Placeholder for future tensor-based transformations
                # image_tensor, label_tensor = self.transforms(image_tensor, label_tensor)
                pass

            # Return the image path along with the tensors for the prediction script
            return image_tensor, label_tensor, self.images[idx]

        except Exception as e:
            print(f"Error loading sample at index {idx}: {self.images[idx]}")
            print(f"Error: {e}")
            # Return empty tensors or handle appropriately
            return torch.zeros((3, 512, 512)), torch.zeros((512, 512), dtype=torch.long), ""

