# Training SuperGlue for SAR Image Matching

This guide explains how to train SuperGlue with Sentinel-1 SAR (VV+VH dual-channel) and Sentinel-2 optical data for improved SAR↔SAR matching performance.

## ⚠️ Important Note on Loss Implementation

**The current implementation uses a simplified loss computation** in the training loop. For production use, the loss function should be enhanced to:
1. Use the full assignment matrix from SuperGlue's log_optimal_transport
2. Compute cross-entropy loss on the assignment matrix with ground truth matches
3. Include both SuperGlue matching loss and optional SuperPoint detector/descriptor losses

The current implementation provides the infrastructure and can run training, but the loss computation in `train/train_utils.py` should be improved for optimal results. See the "Advanced Topics" section for details.

## Overview

The training implementation supports:
- **2-channel SuperPoint** for Sentinel-1 VV+VH input
- **Two-stage training**: 
  1. Stage 1: Train SuperGlue only (frozen SuperPoint)
  2. Stage 2: Joint fine-tuning (unfrozen SuperPoint layers + SuperGlue)
- **Three types of training pairs**: SAR↔SAR, SAR↔S2, S2↔S2
- **Geometric augmentation** for SAR↔SAR pairs with accurate homography
- **Pretrained weight adaptation** from 1/3-channel to 2-channel (non-random initialization)

## Data Organization

### Directory Structure

Organize your data in the following structure:

```
data/sentinel_pairs/
├── train_pairs.csv
├── val_pairs.csv (optional)
├── scene_0001/
│   ├── vv.tif
│   ├── vh.tif
│   └── s2_gray.tif
├── scene_0002/
│   ├── vv.tif
│   ├── vh.tif
│   └── s2_gray.tif
└── ...
```

### CSV File Format

Create a CSV file (`train_pairs.csv`) with the following columns:

```csv
vv_path,vh_path,s2_path
scene_0001/vv.tif,scene_0001/vh.tif,scene_0001/s2_gray.tif
scene_0002/vv.tif,scene_0002/vh.tif,scene_0002/s2_gray.tif
...
```

**Important notes:**
- Paths in CSV are relative to `data_root` specified in config
- VV, VH, and S2 images must be **strictly co-registered** (same pixel grid)
- Images should be GeoTIFF or any format supported by OpenCV
- S2 should be grayscale (single channel)
- Recommended image size: at least 512×512 pixels

### Data Requirements

1. **Sentinel-1 SAR Data**:
   - Two channels: VV and VH polarization
   - Format: GeoTIFF (16-bit or 32-bit float recommended)
   - Preprocessing: Basic calibration to sigma0 or gamma0

2. **Sentinel-2 Optical Data**:
   - Single channel grayscale
   - Can be derived from RGB by averaging or using specific band
   - Must be co-registered with S1 data

3. **Co-registration**:
   - S1 and S2 must be aligned to the same pixel grid
   - No geometric transformation needed (dataset assumes identity transform for SAR↔S2 pairs)
   - Use tools like SNAP, GDAL, or similar for co-registration

## Installation

Install additional dependencies for training:

```bash
pip install pyyaml tensorboard tqdm
```

## Training Workflow

### Stage 1: Train SuperGlue Only

In Stage 1, we freeze SuperPoint and train only SuperGlue to adapt to SAR data.

1. **Configure training** - Edit `configs/train_stage1.yaml`:
   ```yaml
   data_root: ./data/sentinel_pairs
   train_csv: train_pairs.csv
   batch_size: 4
   num_epochs: 30
   max_keypoints: 1024
   ```

2. **Start training**:
   ```bash
   python train/train_superglue_sar.py --config configs/train_stage1.yaml
   ```

3. **Monitor training** with TensorBoard:
   ```bash
   tensorboard --logdir output/stage1_training/logs
   ```

4. **Expected output**:
   - Checkpoints saved in `output/stage1_training/checkpoints/`
   - Training logs in `output/stage1_training/logs/`
   - Final model: `output/stage1_training/checkpoints/final_model.pth`

**Recommended settings for Stage 1:**
- Epochs: 20-40
- Learning rate: 1e-4
- Batch size: 4-8 (depending on GPU memory)
- Pair ratios: 60% SAR↔SAR, 40% SAR↔S2

### Stage 2: Joint Fine-tuning

In Stage 2, we unfreeze SuperPoint layers and fine-tune both networks together.

1. **Configure training** - Edit `configs/train_stage2.yaml`:
   ```yaml
   unfreeze_strategy: last_layers  # Options: all, descriptor, last_layers
   lr_superglue: 5e-5
   lr_superpoint: 1e-5  # Smaller LR for SuperPoint
   num_epochs: 20
   ```

2. **Start training** (resume from Stage 1):
   ```bash
   python train/train_superglue_sar.py \
       --config configs/train_stage2.yaml \
       --resume output/stage1_training/checkpoints/final_model.pth
   ```

3. **Monitor and evaluate** as in Stage 1.

**Recommended settings for Stage 2:**
- Epochs: 10-20
- Learning rate (SuperGlue): 5e-5
- Learning rate (SuperPoint): 1e-5 (10× smaller)
- Pair ratios: 70% SAR↔SAR, 30% SAR↔S2
- Unfreeze strategy: `last_layers` (conservative) or `all` (aggressive)

## Configuration Options

### Key Parameters

| Parameter | Description | Default | Recommendations |
|-----------|-------------|---------|-----------------|
| `training_stage` | 1 or 2 | 1 | Start with 1, then 2 |
| `in_channels` | Input channels | 2 | 2 for SAR VV+VH |
| `max_keypoints` | Max keypoints | 1024 | 1024 or 2048 |
| `patch_size` | Training patch size | [480, 480] | [480, 480] or [512, 512] |
| `pixel_threshold` | Match threshold (px) | 3.0 | 3-5 for SAR |
| `batch_size` | Batch size | 4 | Depends on GPU |
| `lr_superglue` | SuperGlue learning rate | 1e-4 | Stage 1: 1e-4, Stage 2: 5e-5 |
| `lr_superpoint` | SuperPoint learning rate | 1e-5 | Stage 2 only |

### Pair Type Ratios

Control the distribution of training pair types:

```yaml
pair_ratios:
  sar_sar: 0.6  # SAR↔SAR with geometric augmentation
  sar_s2: 0.4   # SAR↔S2 aligned pairs
  s2_s2: 0.0    # S2↔S2 pairs (optional)
```

**Recommendations:**
- For SAR↔SAR matching: Focus on `sar_sar` (60-70%)
- Include `sar_s2` (30-40%) to leverage S2 as regularization
- `s2_s2` typically not needed unless targeting cross-modal matching

### Unfreeze Strategies (Stage 2)

- **`all`**: Unfreeze all SuperPoint layers (most aggressive)
- **`last_layers`**: Unfreeze conv4 + descriptor + keypoint heads (balanced)
- **`descriptor`**: Unfreeze only descriptor head (most conservative)

## Using Trained Weights for Inference

After training, use the trained weights for SAR image matching:

### Option 1: Load trained checkpoint directly

```python
import torch
from models.superpoint import SuperPoint
from models.superglue import SuperGlue

# Load models with 2-channel support
sp_config = {
    'in_channels': 2,
    'load_pretrained': False,  # Don't load pretrained, we'll load trained weights
}
sg_config = {
    'load_pretrained': False,
}

superpoint = SuperPoint(sp_config)
superglue = SuperGlue(sg_config)

# Load trained checkpoint
checkpoint = torch.load('output/stage2_training/checkpoints/final_model.pth')
superpoint.load_state_dict(checkpoint['superpoint_state_dict'])
superglue.load_state_dict(checkpoint['superglue_state_dict'])

superpoint.eval()
superglue.eval()
```

### Option 2: Export to standalone weights

Save just the model weights for easier loading:

```python
# After training
import torch

checkpoint = torch.load('output/stage2_training/checkpoints/final_model.pth')

# Save SuperPoint weights
torch.save(checkpoint['superpoint_state_dict'], 'models/weights/superpoint_sar.pth')

# Save SuperGlue weights
torch.save(checkpoint['superglue_state_dict'], 'models/weights/superglue_sar.pth')
```

Then load in inference:

```python
from models.superpoint import SuperPoint
from models.superglue import SuperGlue

sp_config = {'in_channels': 2, 'load_pretrained': False}
superpoint = SuperPoint(sp_config)
superpoint.load_state_dict(torch.load('models/weights/superpoint_sar.pth'))

sg_config = {'load_pretrained': False}
superglue = SuperGlue(sg_config)
superglue.load_state_dict(torch.load('models/weights/superglue_sar.pth'))
```

### SAR Image Matching Example

```python
import cv2
import torch
import numpy as np

# Load SAR images (VV and VH)
vv = cv2.imread('image1_vv.tif', cv2.IMREAD_UNCHANGED)
vh = cv2.imread('image1_vh.tif', cv2.IMREAD_UNCHANGED)

# Stack and preprocess (log transform + normalize)
sar = np.stack([vv, vh], axis=-1).astype(np.float32)
sar = np.log10(np.clip(sar, 1e-10, None))

# Normalize each channel
for c in range(2):
    channel = sar[..., c]
    vmin, vmax = np.percentile(channel, [2, 98])
    sar[..., c] = np.clip(channel, vmin, vmax)
    if vmax > vmin:
        sar[..., c] = (sar[..., c] - vmin) / (vmax - vmin)

# Convert to tensor (C, H, W)
image_tensor = torch.from_numpy(sar).permute(2, 0, 1).float()[None]  # Add batch dim

# Extract features
with torch.no_grad():
    pred = superpoint({'image': image_tensor})
    keypoints = pred['keypoints'][0]
    descriptors = pred['descriptors'][0]

# Match with another image
# ... (similar preprocessing for image2)
```

## Testing Trained Models

After training, you can test and evaluate your models using the `train/test.py` script:

### Basic Testing

Test your trained model on SAR image pairs:

```bash
python train/test.py \
    --checkpoint output/stage2_training/checkpoints/final_model.pth \
    --image0_vv data/test/scene1/vv.tif \
    --image0_vh data/test/scene1/vh.tif \
    --image1_vv data/test/scene2/vv.tif \
    --image1_vh data/test/scene2/vh.tif
```

### Evaluation with Ground Truth

If you have ground truth homography for evaluation:

```bash
python train/test.py \
    --checkpoint model.pth \
    --eval \
    --gt_homography ground_truth/homography.npy \
    --image0_vv vv0.tif --image0_vh vh0.tif \
    --image1_vv vv1.tif --image1_vh vh1.tif \
    --threshold 3.0
```

The test script will output:
- Number of keypoints detected
- Number of matches found
- Average match confidence
- Matching accuracy (if ground truth provided)

See [train/README_TEST.md](train/README_TEST.md) for detailed testing documentation.

## Troubleshooting

### GPU Memory Issues

If you run out of GPU memory:
1. Reduce `batch_size` (try 2 or 1)
2. Reduce `max_keypoints` (try 512)
3. Reduce `patch_size` (try [384, 384])

### Training is slow

1. Increase `num_workers` for data loading
2. Use smaller `max_keypoints` if you don't need many matches
3. Reduce `num_pairs_per_scene`

### Poor matching performance

1. Ensure SAR and S2 data are properly co-registered
2. Check SAR preprocessing (log transform + normalization)
3. Increase training epochs
4. Adjust `pair_ratios` to focus more on SAR↔SAR
5. Try different `unfreeze_strategy` in Stage 2

### Data loading errors

1. Verify CSV file paths are correct
2. Ensure images exist and are readable
3. Check image format compatibility with OpenCV
4. Verify images have correct number of channels

## Advanced Topics

### Improving the Loss Function

**IMPORTANT**: The current loss implementation in `train/train_utils.py` is simplified and should be enhanced for production training.

To improve training effectiveness, modify the `SuperGlue.forward()` method to return the assignment matrix scores before thresholding, then use these in the loss computation:

```python
# In models/superglue.py, modify forward() to optionally return scores
def forward(self, data, return_scores=False):
    # ... existing code ...
    
    # Run the optimal transport
    scores = log_optimal_transport(
        scores, self.bin_score,
        iters=self.config['sinkhorn_iterations'])
    
    if return_scores:
        # Return scores for training
        return {
            'scores': scores,  # (B, M+1, N+1) assignment matrix
            # ... other outputs ...
        }
    
    # ... rest of existing code for inference ...
```

Then in training loop, compute proper loss:

```python
# In train_superglue_sar.py
pred = self.superglue(data, return_scores=True)
scores = pred['scores']  # (B, M+1, N+1)

# Compute cross-entropy loss
loss = compute_superglue_loss_from_scores(
    scores, gt_matches0_batch, gt_matches1_batch, self.device
)
```

The provided `compute_superglue_loss_from_scores` function in `train_utils.py` shows the structure for this computation.

### Custom Loss Functions

The current implementation uses a simplified loss. For better results, implement proper assignment matrix loss by modifying `train_utils.py`.

### Data Augmentation

Current augmentations include:
- Random crops
- Homography (rotation, scale, translation) for SAR↔SAR

You can extend by adding:
- Color jittering for S2
- Additional geometric transforms
- Speckle noise simulation for SAR

### Multi-GPU Training

To use multiple GPUs, modify the training script to use `torch.nn.DataParallel` or `DistributedDataParallel`.

## Citation

If you use this training code, please cite the original SuperGlue paper:

```bibtex
@inproceedings{sarlin20superglue,
  author    = {Paul-Edouard Sarlin and Daniel DeTone and 
               Tomasz Malisiewicz and Andrew Rabinovich},
  title     = {{SuperGlue}: Learning Feature Matching with Graph Neural Networks},
  booktitle = {CVPR},
  year      = {2020},
}
```

## Support

For issues related to:
- **Data preparation**: Check data format and co-registration
- **Training errors**: Review error messages and configuration
- **Performance**: Try different hyperparameters and training stages
- **Custom modifications**: Refer to code comments in source files
