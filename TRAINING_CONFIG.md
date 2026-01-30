# Configuration Examples for Scheme 3 Fine-tuning

This document provides configuration examples for fine-tuning SuperPoint on Sentinel-2 RGB PNG images.

## Basic Fine-tuning (Recommended)

Fine-tune with frozen backbone layers for faster training and better transfer:

```bash
python train_superpoint.py \
  --image_dir ./data/sentinel2_rgb \
  --batch_size 4 \
  --epochs 20 \
  --freeze_backbone \
  --learning_rate 0.001 \
  --output_dir ./output/finetune_frozen
```

**What this does:**
- Loads pretrained 1-channel weights and adapts to 3 channels (RGB)
- Freezes conv1-conv3 layers (early feature extractors)
- Uses reduced learning rate (0.0001) for fine-tuning
- Saves checkpoints every 5 epochs

## Training from Scratch

Train a new model without pretrained weights:

```bash
python train_superpoint.py \
  --image_dir ./data/sentinel2_rgb \
  --no_pretrained \
  --batch_size 8 \
  --epochs 100 \
  --learning_rate 0.001 \
  --output_dir ./output/from_scratch
```

**Note:** Training from scratch requires much more data and training time.

## Full Model Fine-tuning

Fine-tune all layers (no freezing):

```bash
python train_superpoint.py \
  --image_dir ./data/sentinel2_rgb \
  --batch_size 4 \
  --epochs 50 \
  --learning_rate 0.0001 \
  --output_dir ./output/finetune_full
```

**When to use:** When you have sufficient training data and want to adapt the entire model.

## Custom Image Patterns

If your images use different file extensions or naming:

```bash
python train_superpoint.py \
  --image_dir ./data/s2_images \
  --image_glob '*.png' '*.jpg' '*.tif' \
  --batch_size 4 \
  --epochs 20 \
  --freeze_backbone \
  --output_dir ./output/custom_patterns
```

## Working with Different Channel Counts

The same script automatically adapts to any channel count:

### Grayscale (1 channel)
```bash
python train_superpoint.py \
  --image_dir ./data/grayscale \
  --batch_size 8 \
  --epochs 20 \
  --output_dir ./output/grayscale
```

### 13-band SAR (original use case)
```bash
python train_superpoint.py \
  --image_dir ./data/sar_13band \
  --batch_size 2 \
  --epochs 20 \
  --output_dir ./output/sar_13band
```

**Note:** Channel count is automatically detected from the first image in the dataset.

## GPU/CPU Configuration

Force CPU usage (for testing):
```bash
python train_superpoint.py \
  --image_dir ./data/test \
  --force_cpu \
  --batch_size 1 \
  --epochs 1
```

Multi-worker data loading:
```bash
python train_superpoint.py \
  --image_dir ./data/sentinel2_rgb \
  --num_workers 8 \
  --batch_size 16 \
  --epochs 20
```

## Advanced: Custom SuperPoint Parameters

```bash
python train_superpoint.py \
  --image_dir ./data/sentinel2_rgb \
  --nms_radius 3 \
  --keypoint_threshold 0.01 \
  --max_keypoints 2048 \
  --batch_size 4 \
  --epochs 20 \
  --freeze_backbone \
  --output_dir ./output/custom_params
```

**Parameters explained:**
- `--nms_radius`: Non-maximum suppression radius (default: 4)
- `--keypoint_threshold`: Confidence threshold for keypoint detection (default: 0.005)
- `--max_keypoints`: Maximum keypoints per image (-1 for unlimited, default: -1)

## Checkpoint Management

Save checkpoints more frequently:
```bash
python train_superpoint.py \
  --image_dir ./data/sentinel2_rgb \
  --epochs 50 \
  --save_interval 2 \
  --output_dir ./output/frequent_checkpoints
```

## Important Notes

### Channel Detection
- The training script automatically detects the number of input channels from the first image
- No need to manually specify channel count
- Works with 1, 3, 13, or any other channel count

### Weight Adaptation
- When loading pretrained 1-channel weights for 3-channel input:
  - First conv layer weights are replicated across channels
  - Weights are scaled by 1/3 to preserve magnitude
  - All other layers remain unchanged
- Missing/unexpected keys are logged but don't stop training

### Memory Considerations
- RGB (3ch): ~1.5x memory vs grayscale (1ch)
- 13-band: ~6.5x memory vs grayscale
- Adjust `--batch_size` accordingly
- Recommended batch sizes:
  - 3ch RGB: 4-8 on 8GB GPU
  - 13ch: 2-4 on 8GB GPU

### Dataset Requirements
- Minimum: 100+ images for fine-tuning
- Recommended: 1000+ images for good results
- For training from scratch: 10,000+ images

## Python API Usage

You can also use the components programmatically:

```python
from data import S2RGBPNGDataset
from models.superpoint import SuperPoint
import torch

# Create dataset
dataset = S2RGBPNGDataset('./data/sentinel2_rgb')
print(f"Auto-detected {dataset.channels} channels")

# Infer channels and create model
sample = dataset[0]
in_channels = sample['image'].shape[0]

model = SuperPoint({
    'in_channels': in_channels,
    'load_pretrained': True,  # Auto-adapts weights
    'nms_radius': 4,
    'keypoint_threshold': 0.005,
})

# Use model
with torch.no_grad():
    output = model({'image': sample['image'].unsqueeze(0)})
    print(f"Detected {len(output['keypoints'][0])} keypoints")
```

## Troubleshooting

### Out of Memory
- Reduce `--batch_size`
- Use `--num_workers 0` to disable multiprocessing
- Try `--force_cpu` for testing

### No Images Found
- Check `--image_dir` path
- Verify image extensions match `--image_glob` patterns
- Ensure images exist and are readable

### Weight Adaptation Warnings
- Normal: "Adapting conv1a weights from 1 to 3 channels"
- Check: Missing/unexpected keys should only be for adapted layer
- If many missing keys: verify pretrained weights file exists

### Channel Mismatch
- The script auto-detects channels, no manual config needed
- If all images have different channel counts, first image determines the model
- Ensure all images in dataset have the same channel count
