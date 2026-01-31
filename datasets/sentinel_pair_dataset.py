"""
SentinelPairDataset: Dataset for Sentinel-1 SAR and Sentinel-2 optical image pairs

Supports:
- Sentinel-1 VV+VH dual-channel input
- Sentinel-2 grayscale
- Three types of training pairs: SAR↔SAR, SAR↔S2, S2↔S2
- Geometric augmentation for SAR↔SAR pairs
- Aligned pixel grids (co-registered data)
"""

import os
import csv
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path


class SentinelPairDataset(Dataset):
    """Dataset for training SuperGlue with Sentinel-1 and Sentinel-2 data
    
    Args:
        data_root: Root directory containing the data
        csv_file: CSV file with columns: vv_path, vh_path, s2_path
                 Each row represents one scene
        patch_size: Size of patches to extract (height, width)
        pair_type: Type of pairs to generate ('sar_sar', 'sar_s2', 's2_s2')
        pair_ratios: Dict with ratios for each pair type, e.g., {'sar_sar': 0.6, 'sar_s2': 0.4}
        num_pairs_per_scene: Number of pairs to generate per scene
        sar_log_transform: Whether to apply log transformation to SAR data
        sar_percentile_clip: Percentile for robust normalization (e.g., (2, 98))
        augment_sar: Whether to apply geometric augmentation to SAR pairs
    """
    
    def __init__(self,
                 data_root,
                 csv_file,
                 patch_size=(480, 480),
                 pair_type='sar_sar',
                 pair_ratios=None,
                 num_pairs_per_scene=10,
                 sar_log_transform=True,
                 sar_percentile_clip=(2, 98),
                 augment_sar=True,
                 seed=None):
        
        self.data_root = Path(data_root)
        self.patch_size = patch_size
        self.pair_type = pair_type
        self.num_pairs_per_scene = num_pairs_per_scene
        self.sar_log_transform = sar_log_transform
        self.sar_percentile_clip = sar_percentile_clip
        self.augment_sar = augment_sar
        
        if pair_ratios is None:
            pair_ratios = {'sar_sar': 0.6, 'sar_s2': 0.4, 's2_s2': 0.0}
        self.pair_ratios = pair_ratios
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Load scene list from CSV
        self.scenes = self._load_scenes(csv_file)
        
        # Generate pair indices
        self.pairs = self._generate_pair_indices()
        
    def _load_scenes(self, csv_file):
        """Load scene information from CSV file"""
        scenes = []
        csv_path = self.data_root / csv_file
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                scenes.append({
                    'vv': self.data_root / row['vv_path'],
                    'vh': self.data_root / row['vh_path'],
                    's2': self.data_root / row['s2_path']
                })
        
        return scenes
    
    def _generate_pair_indices(self):
        """Generate indices for all training pairs"""
        pairs = []
        
        for scene_idx, scene in enumerate(self.scenes):
            # Determine pair types for this scene
            num_sar_sar = int(self.num_pairs_per_scene * self.pair_ratios.get('sar_sar', 0))
            num_sar_s2 = int(self.num_pairs_per_scene * self.pair_ratios.get('sar_s2', 0))
            num_s2_s2 = self.num_pairs_per_scene - num_sar_sar - num_sar_s2
            
            # Add pair indices
            for _ in range(num_sar_sar):
                pairs.append({'scene_idx': scene_idx, 'type': 'sar_sar'})
            for _ in range(num_sar_s2):
                pairs.append({'scene_idx': scene_idx, 'type': 'sar_s2'})
            for _ in range(num_s2_s2):
                pairs.append({'scene_idx': scene_idx, 'type': 's2_s2'})
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def _preprocess_sar(self, vv, vh):
        """Preprocess SAR data (log transform + robust normalization)"""
        # Stack VV and VH
        sar = np.stack([vv, vh], axis=-1).astype(np.float32)
        
        # Log transform
        if self.sar_log_transform:
            sar = np.log10(np.clip(sar, 1e-10, None))
        
        # Robust normalization using percentile clipping
        if self.sar_percentile_clip:
            p_low, p_high = self.sar_percentile_clip
            for c in range(sar.shape[-1]):
                channel = sar[..., c]
                vmin, vmax = np.percentile(channel, [p_low, p_high])
                sar[..., c] = np.clip(channel, vmin, vmax)
                # Normalize to [0, 1]
                if vmax > vmin:
                    sar[..., c] = (sar[..., c] - vmin) / (vmax - vmin)
        
        return sar
    
    def _preprocess_s2(self, s2):
        """Preprocess Sentinel-2 grayscale data"""
        s2 = s2.astype(np.float32)
        # Normalize to [0, 1]
        s2 = s2 / 255.0 if s2.max() > 1.0 else s2
        return s2
    
    def _random_crop(self, img, crop_size):
        """Random crop from image"""
        h, w = img.shape[:2]
        ch, cw = crop_size
        
        if h < ch or w < cw:
            # Pad if image is smaller than crop size
            pad_h = max(0, ch - h)
            pad_w = max(0, cw - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)) if len(img.shape) == 3 
                        else ((0, pad_h), (0, pad_w)), mode='reflect')
            h, w = img.shape[:2]
        
        top = np.random.randint(0, h - ch + 1)
        left = np.random.randint(0, w - cw + 1)
        
        img_crop = img[top:top+ch, left:left+cw]
        return img_crop, top, left
    
    def _generate_homography(self, image_shape, max_angle=30, max_scale=0.2, max_translate=0.2):
        """Generate random homography matrix for augmentation
        
        Returns:
            H: 3x3 homography matrix
        """
        h, w = image_shape[:2]
        center = np.array([w / 2, h / 2])
        
        # Random rotation
        angle = np.random.uniform(-max_angle, max_angle)
        angle_rad = np.deg2rad(angle)
        
        # Random scale
        scale = np.random.uniform(1 - max_scale, 1 + max_scale)
        
        # Random translation
        tx = np.random.uniform(-max_translate * w, max_translate * w)
        ty = np.random.uniform(-max_translate * h, max_translate * h)
        
        # Build transformation matrix
        # 1. Translate to origin
        T1 = np.array([[1, 0, -center[0]], 
                       [0, 1, -center[1]], 
                       [0, 0, 1]])
        
        # 2. Rotate and scale
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[scale * cos_a, -scale * sin_a, 0],
                      [scale * sin_a, scale * cos_a, 0],
                      [0, 0, 1]])
        
        # 3. Translate back and apply translation
        T2 = np.array([[1, 0, center[0] + tx],
                       [0, 1, center[1] + ty],
                       [0, 0, 1]])
        
        # Combine transformations
        H = T2 @ R @ T1
        
        return H.astype(np.float32)
    
    def _apply_homography(self, img, H, output_shape):
        """Apply homography to image"""
        return cv2.warpPerspective(img, H, (output_shape[1], output_shape[0]),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
    
    def __getitem__(self, idx):
        pair_info = self.pairs[idx]
        scene = self.scenes[pair_info['scene_idx']]
        pair_type = pair_info['type']
        
        # Load images
        vv = cv2.imread(str(scene['vv']), cv2.IMREAD_UNCHANGED)
        vh = cv2.imread(str(scene['vh']), cv2.IMREAD_UNCHANGED)
        s2 = cv2.imread(str(scene['s2']), cv2.IMREAD_GRAYSCALE)
        
        # Preprocess
        sar = self._preprocess_sar(vv, vh)  # Shape: (H, W, 2)
        s2 = self._preprocess_s2(s2)  # Shape: (H, W)
        s2 = np.expand_dims(s2, axis=-1)  # Shape: (H, W, 1)
        
        # Generate pairs based on type
        if pair_type == 'sar_sar':
            # Generate two views from same SAR patch with geometric augmentation
            sar_crop, _, _ = self._random_crop(sar, self.patch_size)
            
            if self.augment_sar:
                # First view (original)
                img0 = sar_crop
                
                # Second view (with homography)
                H = self._generate_homography(sar_crop.shape)
                img1 = self._apply_homography(sar_crop, H, self.patch_size)
                
                # Homography from img0 to img1
                T_0to1 = H
            else:
                # Without augmentation (identity transform)
                img0 = sar_crop
                img1 = sar_crop.copy()
                T_0to1 = np.eye(3, dtype=np.float32)
        
        elif pair_type == 'sar_s2':
            # Use aligned SAR and S2 patches
            sar_crop, top, left = self._random_crop(sar, self.patch_size)
            s2_crop = s2[top:top+self.patch_size[0], left:left+self.patch_size[1]]
            
            img0 = sar_crop
            img1 = s2_crop
            # Identity transform (already aligned)
            T_0to1 = np.eye(3, dtype=np.float32)
        
        elif pair_type == 's2_s2':
            # Two views from same S2 patch
            s2_crop, _, _ = self._random_crop(s2, self.patch_size)
            
            if self.augment_sar:  # Use same augmentation setting
                img0 = s2_crop
                H = self._generate_homography(s2_crop.shape)
                img1 = self._apply_homography(s2_crop, H, self.patch_size)
                T_0to1 = H
            else:
                img0 = s2_crop
                img1 = s2_crop.copy()
                T_0to1 = np.eye(3, dtype=np.float32)
        
        # Convert to torch tensors (C, H, W)
        img0 = torch.from_numpy(img0).permute(2, 0, 1).float()
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        T_0to1 = torch.from_numpy(T_0to1).float()
        
        return {
            'image0': img0,
            'image1': img1,
            'T_0to1': T_0to1,
            'pair_type': pair_type,
            'scene_idx': pair_info['scene_idx']
        }
