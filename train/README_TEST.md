# Testing Trained SuperGlue SAR Models

The `test.py` script allows you to test and evaluate trained SuperGlue models on SAR image pairs.

## Usage

### Basic Testing

Test a trained model on a pair of SAR images:

```bash
python train/test.py \
    --checkpoint output/stage2_training/checkpoints/final_model.pth \
    --image0_vv data/scene1/vv.tif \
    --image0_vh data/scene1/vh.tif \
    --image1_vv data/scene2/vv.tif \
    --image1_vh data/scene2/vh.tif
```

### Evaluation with Ground Truth

If you have ground truth homography (for evaluation):

```bash
python train/test.py \
    --checkpoint model.pth \
    --eval \
    --gt_homography ground_truth/homography.npy \
    --image0_vv vv0.tif --image0_vh vh0.tif \
    --image1_vv vv1.tif --image1_vh vh1.tif \
    --threshold 3.0
```

### Using GPU

```bash
python train/test.py \
    --checkpoint model.pth \
    --device cuda \
    --image0_vv vv0.tif --image0_vh vh0.tif \
    --image1_vv vv1.tif --image1_vh vh1.tif
```

## Arguments

- `--checkpoint`: Path to trained model checkpoint (.pth file)
- `--device`: Device to run on (cpu or cuda, default: cpu)
- `--image0_vv`: Path to first image VV channel
- `--image0_vh`: Path to first image VH channel  
- `--image1_vv`: Path to second image VV channel
- `--image1_vh`: Path to second image VH channel
- `--eval`: Enable evaluation mode (requires ground truth)
- `--gt_homography`: Path to ground truth homography (.npy file)
- `--threshold`: Pixel threshold for correct matches (default: 3.0)
- `--no_log_transform`: Disable log transform for SAR data

## Output

The script outputs:
- Number of keypoints detected in each image
- Number of matches found
- Average match confidence
- (If --eval) Matching accuracy and number of correct matches

Example output:
```
Using device: cpu
Loading checkpoint from output/stage2_training/checkpoints/final_model.pth
Models loaded successfully
Loading images...
Image 0 shape: torch.Size([2, 512, 512])
Image 1 shape: torch.Size([2, 512, 512])

Running matching...

============================================================
MATCHING RESULTS
============================================================
Keypoints in image 0: 1024
Keypoints in image 1: 987
Number of matches: 543
Average match confidence: 0.823

Evaluation (threshold=3.0px):
Correct matches: 489/543
Accuracy: 90.05%
============================================================
```

## Ground Truth Format

The ground truth homography should be a 3x3 numpy array saved as .npy:

```python
import numpy as np

# Your 3x3 homography matrix
H = np.array([[...], [...], [...]])

# Save it
np.save('homography.npy', H)
```

## See Also

- [TRAINING.md](../TRAINING.md) - Training guide
- [train_superglue_sar.py](train_superglue_sar.py) - Training script
