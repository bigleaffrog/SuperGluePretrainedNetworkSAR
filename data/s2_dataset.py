"""Dataset class for Sentinel-2 RGB PNG images (Scheme 3: B4/B3/B2)."""

import os
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class S2RGBPNGDataset(Dataset):
    """Dataset for Sentinel-2 RGB PNG images.
    
    This dataset reads PNG images containing Sentinel-2 B4/B3/B2 bands
    saved as RGB channels. It automatically handles RGBA by discarding
    the alpha channel, and normalizes pixel values to [0, 1].
    
    Args:
        image_dir: Directory containing PNG images
        image_glob: Glob pattern(s) for image files (default: ['*.png'])
        transform: Optional transform to apply to images
        return_path: If True, also return image path in the output dict
    """
    
    def __init__(self, image_dir, image_glob=None, transform=None, return_path=False):
        self.image_dir = Path(image_dir)
        if not self.image_dir.exists():
            raise ValueError(f"Image directory {image_dir} does not exist")
        
        if image_glob is None:
            image_glob = ['*.png']
        
        # Collect all image paths
        self.image_paths = []
        for pattern in image_glob:
            self.image_paths.extend(list(self.image_dir.glob(pattern)))
        
        self.image_paths = sorted(self.image_paths)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir} with patterns {image_glob}")
        
        self.transform = transform
        self.return_path = return_path
        
        # Probe first image to determine channel count
        self.channels = self._probe_channels()
        self.band_mapping = 'B4/B3/B2' if self.channels == 3 else f'{self.channels}ch'
        
        print(f"S2RGBPNGDataset: Found {len(self.image_paths)} images with {self.channels} channels")
    
    def _probe_channels(self):
        """Determine number of channels from first image."""
        img = self._load_image(self.image_paths[0])
        return img.shape[0]  # CHW format
    
    def _load_image(self, path):
        """Load a PNG image and convert to normalized CHW float32 tensor.
        
        Args:
            path: Path to PNG image
            
        Returns:
            Normalized image tensor in CHW format, shape (C, H, W), dtype float32
        """
        # Load image with all channels
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        
        # Handle different channel configurations
        if len(img.shape) == 2:
            # Grayscale image
            img = img[None, :, :]  # Add channel dimension: HW -> CHW
        else:
            # Multi-channel image (RGB or RGBA)
            # OpenCV loads as HWC, convert to CHW
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            
            # If RGBA, discard alpha channel
            if img.shape[0] == 4:
                img = img[:3, :, :]  # Keep only RGB
        
        # Normalize to [0, 1] based on dtype
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32)
        
        return img
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get a data sample.
        
        Returns:
            dict with keys:
                - image: Normalized image tensor (C, H, W)
                - channels: Number of channels
                - band_mapping: String describing band configuration
                - path: (optional) Image path if return_path=True
        """
        img_path = self.image_paths[idx]
        img = self._load_image(img_path)
        
        if self.transform:
            img = self.transform(img)
        
        # Convert to torch tensor if not already
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        
        sample = {
            'image': img,
            'channels': self.channels,
            'band_mapping': self.band_mapping,
        }
        
        if self.return_path:
            sample['path'] = str(img_path)
        
        return sample


class PairedS2Dataset(Dataset):
    """Dataset for pairs of Sentinel-2 images for matching/training.
    
    This dataset creates pairs of images for training SuperPoint/SuperGlue.
    It can be used for both supervised and self-supervised training.
    
    Args:
        image_dir: Directory containing PNG images
        pairs_file: Optional file containing image pairs (one pair per line)
        image_glob: Glob pattern(s) for image files
        transform: Optional transform to apply to images
        max_pairs: Maximum number of pairs to generate (default: unlimited)
    """
    
    def __init__(self, image_dir, pairs_file=None, image_glob=None, 
                 transform=None, max_pairs=None):
        self.base_dataset = S2RGBPNGDataset(image_dir, image_glob, transform)
        self.transform = transform
        
        if pairs_file and Path(pairs_file).exists():
            # Load pairs from file
            self.pairs = self._load_pairs_file(pairs_file)
        else:
            # Generate pairs from all images
            self.pairs = self._generate_pairs(max_pairs)
        
        print(f"PairedS2Dataset: Created {len(self.pairs)} image pairs")
    
    def _load_pairs_file(self, pairs_file):
        """Load image pairs from a text file."""
        pairs = []
        with open(pairs_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
        return pairs
    
    def _generate_pairs(self, max_pairs):
        """Generate pairs from available images."""
        n = len(self.base_dataset)
        pairs = []
        
        # Generate sequential pairs
        for i in range(n - 1):
            pairs.append((i, i + 1))
            if max_pairs and len(pairs) >= max_pairs:
                break
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """Get a pair of images.
        
        Returns:
            dict with keys for both images (image0, image1, channels, etc.)
        """
        if isinstance(self.pairs[idx][0], str):
            # Pairs are image paths
            img0_path = self.pairs[idx][0]
            img1_path = self.pairs[idx][1]
            # Find indices in base dataset
            idx0 = next(i for i, p in enumerate(self.base_dataset.image_paths) 
                       if str(p).endswith(img0_path))
            idx1 = next(i for i, p in enumerate(self.base_dataset.image_paths) 
                       if str(p).endswith(img1_path))
        else:
            # Pairs are indices
            idx0, idx1 = self.pairs[idx]
        
        sample0 = self.base_dataset[idx0]
        sample1 = self.base_dataset[idx1]
        
        return {
            'image0': sample0['image'],
            'image1': sample1['image'],
            'channels': sample0['channels'],
            'band_mapping': sample0['band_mapping'],
        }
