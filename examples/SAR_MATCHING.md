# SAR Image Matching Examples

This directory contains examples for using SuperGlue with Sentinel-1 SAR data.

## Example 1: Basic SAR Image Matching with 2-Channel Input

```python
import cv2
import numpy as np
import torch
from models.superpoint import SuperPoint
from models.superglue import SuperGlue

# Configuration for SAR (2-channel VV+VH)
sp_config = {
    'in_channels': 2,
    'nms_radius': 4,
    'keypoint_threshold': 0.005,
    'max_keypoints': 1024,
    'load_pretrained': True  # Load pretrained weights with channel adaptation
}

sg_config = {
    'weights': 'indoor',  # or 'outdoor'
    'sinkhorn_iterations': 100,
    'match_threshold': 0.2,
}

# Initialize models
superpoint = SuperPoint(sp_config).eval()
superglue = SuperGlue(sg_config).eval()

# Load and preprocess SAR images
def load_sar_image(vv_path, vh_path):
    """Load and preprocess Sentinel-1 VV and VH bands"""
    vv = cv2.imread(vv_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    vh = cv2.imread(vh_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    
    # Stack channels
    sar = np.stack([vv, vh], axis=-1)
    
    # Log transform
    sar = np.log10(np.clip(sar, 1e-10, None))
    
    # Robust normalization (percentile clipping)
    for c in range(2):
        channel = sar[..., c]
        vmin, vmax = np.percentile(channel, [2, 98])
        sar[..., c] = np.clip(channel, vmin, vmax)
        if vmax > vmin:
            sar[..., c] = (sar[..., c] - vmin) / (vmax - vmin)
    
    # Convert to tensor (C, H, W)
    sar_tensor = torch.from_numpy(sar).permute(2, 0, 1).float()
    return sar_tensor

# Load two SAR images
image0 = load_sar_image('image0_vv.tif', 'image0_vh.tif')
image1 = load_sar_image('image1_vv.tif', 'image1_vh.tif')

# Add batch dimension
image0 = image0.unsqueeze(0)
image1 = image1.unsqueeze(0)

# Extract keypoints and descriptors
with torch.no_grad():
    pred0 = superpoint({'image': image0})
    pred1 = superpoint({'image': image1})
    
    # Match with SuperGlue
    data = {
        'keypoints0': pred0['keypoints'],
        'keypoints1': pred1['keypoints'],
        'descriptors0': pred0['descriptors'],
        'descriptors1': pred1['descriptors'],
        'scores0': pred0['scores'],
        'scores1': pred1['scores'],
        'image0': image0,
        'image1': image1,
    }
    
    pred = superglue(data)

# Get matches
matches = pred['matches0'][0]
confidence = pred['matching_scores0'][0]

# Filter valid matches
valid = matches > -1
kpts0 = pred0['keypoints'][0][valid].cpu().numpy()
kpts1 = pred1['keypoints'][0][matches[valid]].cpu().numpy()
conf = confidence[valid].cpu().numpy()

print(f"Found {len(kpts0)} matches")
print(f"Average confidence: {conf.mean():.3f}")
```

## Example 2: Using Trained SAR Weights

```python
import torch
from models.superpoint import SuperPoint
from models.superglue import SuperGlue

# Load trained weights from checkpoint
checkpoint = torch.load('output/stage2_training/checkpoints/final_model.pth')

# Initialize models without pretrained weights
sp_config = {'in_channels': 2, 'load_pretrained': False}
sg_config = {'load_pretrained': False}

superpoint = SuperPoint(sp_config)
superglue = SuperGlue(sg_config)

# Load trained weights
superpoint.load_state_dict(checkpoint['superpoint_state_dict'])
superglue.load_state_dict(checkpoint['superglue_state_dict'])

superpoint.eval()
superglue.eval()

# Now use for inference as in Example 1
```

## Tips for SAR Matching

1. **Preprocessing is critical**: Always apply log transform and robust normalization
2. **Image quality**: Ensure SAR images are properly calibrated (sigma0 or gamma0)
3. **Co-registration**: For SAR↔optical, images must be co-registered
4. **Keypoint threshold**: Try different values (0.001-0.01) for your data
5. **Max keypoints**: Use 1024 for most cases, 2048 for complex scenes
6. **Training**: Fine-tune on your specific SAR data for best results

## See Also

- [TRAINING.md](../TRAINING.md) - Complete training guide
- [README.md](../README.md) - Main documentation
