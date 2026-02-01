#!/usr/bin/env python3
"""
Test/Evaluation script for trained SuperGlue SAR models

This script allows you to test trained models on SAR image pairs and evaluate
their matching performance.

Usage:
    # Test with trained checkpoint
    python train/test.py --checkpoint output/stage2_training/checkpoints/final_model.pth \
                         --image0_vv path/to/vv0.tif --image0_vh path/to/vh0.tif \
                         --image1_vv path/to/vv1.tif --image1_vh path/to/vh1.tif
    
    # Test with evaluation metrics (if ground truth homography available)
    python train/test.py --checkpoint model.pth --eval --gt_homography H.npy \
                         --image0_vv vv0.tif --image0_vh vh0.tif \
                         --image1_vv vv1.tif --image1_vh vh1.tif
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.superpoint import SuperPoint
from models.superglue import SuperGlue


def load_sar_image(vv_path, vh_path, log_transform=True, percentile_clip=(2, 98)):
    """Load and preprocess Sentinel-1 VV and VH bands"""
    vv = cv2.imread(str(vv_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    vh = cv2.imread(str(vh_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    
    if vv is None or vh is None:
        raise ValueError(f"Failed to load images: VV={vv_path}, VH={vh_path}")
    
    # Stack channels
    sar = np.stack([vv, vh], axis=-1)
    
    # Log transform
    if log_transform:
        sar = np.log10(np.clip(sar, 1e-10, None))
    
    # Robust normalization (percentile clipping)
    for c in range(2):
        channel = sar[..., c]
        vmin, vmax = np.percentile(channel, percentile_clip)
        sar[..., c] = np.clip(channel, vmin, vmax)
        if vmax > vmin:
            sar[..., c] = (sar[..., c] - vmin) / (vmax - vmin)
    
    # Convert to tensor (C, H, W)
    sar_tensor = torch.from_numpy(sar).permute(2, 0, 1).float()
    return sar_tensor


def load_checkpoint(checkpoint_path, device='cpu'):
    """Load trained SuperPoint and SuperGlue models from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize models
    sp_config = {
        'in_channels': 2,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024,
        'load_pretrained': False
    }
    
    sg_config = {
        'descriptor_dim': 256,
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
        'load_pretrained': False
    }
    
    superpoint = SuperPoint(sp_config).to(device)
    superglue = SuperGlue(sg_config).to(device)
    
    # Load weights
    superpoint.load_state_dict(checkpoint['superpoint_state_dict'])
    superglue.load_state_dict(checkpoint['superglue_state_dict'])
    
    superpoint.eval()
    superglue.eval()
    
    return superpoint, superglue


def compute_matching_accuracy(kpts0, kpts1, matches, homography, threshold=3.0):
    """Compute matching accuracy given ground truth homography"""
    if matches is None or len(kpts0) == 0:
        return 0.0, 0
    
    valid = matches > -1
    if valid.sum() == 0:
        return 0.0, 0
    
    # Get matched keypoints
    matched_kpts0 = kpts0[valid]
    matched_kpts1 = kpts1[matches[valid]]
    
    # Warp kpts0 using homography
    kpts0_h = np.concatenate([matched_kpts0, np.ones((len(matched_kpts0), 1))], axis=1)
    kpts0_warped_h = (homography @ kpts0_h.T).T
    kpts0_warped = kpts0_warped_h[:, :2] / kpts0_warped_h[:, 2:3]
    
    # Compute distances
    distances = np.linalg.norm(kpts0_warped - matched_kpts1, axis=1)
    correct = distances < threshold
    
    accuracy = correct.sum() / len(correct)
    num_correct = correct.sum()
    
    return accuracy, num_correct


def test_image_pair(superpoint, superglue, image0, image1, device='cpu', 
                   gt_homography=None, visualize=False):
    """Test matching on a single image pair"""
    # Add batch dimension
    image0 = image0.unsqueeze(0).to(device)
    image1 = image1.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Extract features
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
    
    # Get results
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()
    kpts0 = pred0['keypoints'][0].cpu().numpy()
    kpts1 = pred1['keypoints'][0].cpu().numpy()
    
    # Count valid matches
    valid = matches > -1
    num_matches = valid.sum()
    
    results = {
        'num_keypoints0': len(kpts0),
        'num_keypoints1': len(kpts1),
        'num_matches': int(num_matches),
        'matches': matches,
        'confidence': confidence,
        'keypoints0': kpts0,
        'keypoints1': kpts1
    }
    
    # Compute accuracy if ground truth available
    if gt_homography is not None:
        accuracy, num_correct = compute_matching_accuracy(
            kpts0, kpts1, matches, gt_homography
        )
        results['accuracy'] = accuracy
        results['num_correct'] = int(num_correct)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Test trained SuperGlue SAR models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run on (cpu or cuda)')
    
    # Input images
    parser.add_argument('--image0_vv', type=str, required=True,
                       help='Path to first image VV channel')
    parser.add_argument('--image0_vh', type=str, required=True,
                       help='Path to first image VH channel')
    parser.add_argument('--image1_vv', type=str, required=True,
                       help='Path to second image VV channel')
    parser.add_argument('--image1_vh', type=str, required=True,
                       help='Path to second image VH channel')
    
    # Evaluation options
    parser.add_argument('--eval', action='store_true',
                       help='Evaluate with ground truth homography')
    parser.add_argument('--gt_homography', type=str, default=None,
                       help='Path to ground truth homography (.npy file)')
    parser.add_argument('--threshold', type=float, default=3.0,
                       help='Pixel threshold for correct matches')
    
    # Preprocessing options
    parser.add_argument('--no_log_transform', action='store_true',
                       help='Disable log transform for SAR data')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    superpoint, superglue = load_checkpoint(args.checkpoint, device)
    print("Models loaded successfully")
    
    # Load images
    print("Loading images...")
    image0 = load_sar_image(
        args.image0_vv, args.image0_vh,
        log_transform=not args.no_log_transform
    )
    image1 = load_sar_image(
        args.image1_vv, args.image1_vh,
        log_transform=not args.no_log_transform
    )
    print(f"Image 0 shape: {image0.shape}")
    print(f"Image 1 shape: {image1.shape}")
    
    # Load ground truth if provided
    gt_homography = None
    if args.eval:
        if args.gt_homography is None:
            print("Warning: --eval specified but no --gt_homography provided")
        else:
            gt_homography = np.load(args.gt_homography)
            print(f"Loaded ground truth homography: {gt_homography.shape}")
    
    # Test matching
    print("\nRunning matching...")
    results = test_image_pair(
        superpoint, superglue,
        image0, image1,
        device=device,
        gt_homography=gt_homography
    )
    
    # Print results
    print("\n" + "="*60)
    print("MATCHING RESULTS")
    print("="*60)
    print(f"Keypoints in image 0: {results['num_keypoints0']}")
    print(f"Keypoints in image 1: {results['num_keypoints1']}")
    print(f"Number of matches: {results['num_matches']}")
    
    if results['num_matches'] > 0:
        avg_confidence = results['confidence'][results['matches'] > -1].mean()
        print(f"Average match confidence: {avg_confidence:.3f}")
    
    if 'accuracy' in results:
        print(f"\nEvaluation (threshold={args.threshold}px):")
        print(f"Correct matches: {results['num_correct']}/{results['num_matches']}")
        print(f"Accuracy: {results['accuracy']:.2%}")
    
    print("="*60)
    
    return results


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
