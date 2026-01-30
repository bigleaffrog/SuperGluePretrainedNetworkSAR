#!/usr/bin/env python3
"""
Fine-tuning script for SuperPoint with Sentinel-2 RGB PNG data (Scheme 3).

This script demonstrates fine-tuning SuperPoint on RGB PNG images
(Sentinel-2 B4/B3/B2 bands) with automatic channel detection and
weight adaptation from pretrained 1-channel grayscale model.
"""

import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from models.superpoint import SuperPoint
from data import S2RGBPNGDataset


def get_channels_from_dataset(dataset):
    """Probe dataset to determine number of input channels."""
    sample = dataset[0]
    if 'image' in sample:
        img = sample['image']
        if isinstance(img, torch.Tensor):
            channels = img.shape[0]  # CHW format
        elif isinstance(img, np.ndarray):
            if len(img.shape) == 3:
                channels = img.shape[0]  # CHW format
            else:
                channels = 1
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
    else:
        raise ValueError("Sample does not contain 'image' key")
    
    print(f"Detected {channels} input channels from dataset")
    return channels


def create_superpoint_model(in_channels, config):
    """Create SuperPoint model with specified input channels.
    
    Args:
        in_channels: Number of input channels
        config: Configuration dict for SuperPoint
        
    Returns:
        SuperPoint model instance
    """
    model_config = {
        'in_channels': in_channels,
        'load_pretrained': config.get('load_pretrained', True),
        'nms_radius': config.get('nms_radius', 4),
        'keypoint_threshold': config.get('keypoint_threshold', 0.005),
        'max_keypoints': config.get('max_keypoints', -1),
        'descriptor_dim': config.get('descriptor_dim', 256),
    }
    
    model = SuperPoint(model_config)
    return model


def freeze_backbone(model, freeze_layers=None):
    """Freeze specified layers of the model.
    
    Args:
        model: SuperPoint model
        freeze_layers: List of layer name prefixes to freeze, or None to freeze all conv layers
    """
    if freeze_layers is None:
        # Default: freeze early conv layers (conv1, conv2, conv3)
        freeze_layers = ['conv1a', 'conv1b', 'conv2a', 'conv2b', 'conv3a', 'conv3b']
    
    for name, param in model.named_parameters():
        should_freeze = any(name.startswith(layer) for layer in freeze_layers)
        if should_freeze:
            param.requires_grad = False
            print(f"Frozen layer: {name}")


def dummy_loss(pred, batch):
    """Dummy loss function for demonstration.
    
    In a real training setup, you would implement proper losses for:
    - Keypoint detection (e.g., heatmap loss)
    - Descriptor learning (e.g., triplet loss, contrastive loss)
    
    This is just a placeholder to show the training loop structure.
    """
    # For now, return a dummy loss based on number of detected keypoints
    # This is NOT a real training loss!
    num_keypoints = sum(len(kpts) for kpts in pred['keypoints'])
    # Encourage detecting some keypoints (target: ~1000 per image)
    target_kpts = 1000.0
    loss = torch.abs(torch.tensor(num_keypoints, dtype=torch.float32) - target_kpts) / target_kpts
    return loss


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        images = batch['image'].to(device)
        
        # Prepare input for SuperPoint
        data = {'image': images}
        
        # Forward pass
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            pred = model(data)
            
            # Compute loss (dummy for now - replace with real loss)
            loss = dummy_loss(pred, batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print(f"\nLoading dataset from: {args.image_dir}")
    dataset = S2RGBPNGDataset(
        image_dir=args.image_dir,
        image_glob=args.image_glob,
    )
    
    # Infer number of channels from dataset
    in_channels = get_channels_from_dataset(dataset)
    print(f"Dataset: {len(dataset)} images, {in_channels} channels, band_mapping: {dataset.band_mapping}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    # Create model with inferred channel count
    print(f"\nCreating SuperPoint model with {in_channels} input channels...")
    model_config = {
        'load_pretrained': args.load_pretrained,
        'nms_radius': args.nms_radius,
        'keypoint_threshold': args.keypoint_threshold,
        'max_keypoints': args.max_keypoints,
    }
    model = create_superpoint_model(in_channels, model_config)
    model = model.to(device)
    
    # Optionally freeze backbone layers
    if args.freeze_backbone:
        print("\nFreezing backbone layers...")
        freeze_backbone(model)
    
    # Setup optimizer with potentially different learning rates
    if args.freeze_backbone:
        # Use smaller learning rate for fine-tuning
        lr = args.learning_rate * 0.1
        print(f"Using reduced learning rate for fine-tuning: {lr}")
    else:
        lr = args.learning_rate
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=args.weight_decay,
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, device, epoch + 1)
        print(f"Epoch {epoch + 1}/{args.epochs} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = Path(args.output_dir) / f"superpoint_epoch_{epoch+1}.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'in_channels': in_channels,
                'band_mapping': dataset.band_mapping,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = Path(args.output_dir) / "superpoint_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'in_channels': in_channels,
        'band_mapping': dataset.band_mapping,
    }, final_path)
    print(f"\nTraining completed. Final model saved to: {final_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fine-tune SuperPoint on Sentinel-2 RGB PNG images (Scheme 3)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Dataset arguments
    parser.add_argument(
        '--image_dir', type=str, required=True,
        help='Directory containing PNG images')
    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png'],
        help='Glob patterns for image files')
    
    # Model arguments
    parser.add_argument(
        '--load_pretrained', action='store_true', default=True,
        help='Load pretrained SuperPoint weights (will adapt to input channels)')
    parser.add_argument(
        '--no_pretrained', dest='load_pretrained', action='store_false',
        help='Do not load pretrained weights (train from scratch)')
    parser.add_argument(
        '--freeze_backbone', action='store_true',
        help='Freeze early convolutional layers for fine-tuning')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint NMS radius')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint threshold')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum keypoints to detect (-1 for no limit)')
    
    # Training arguments
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='Batch size for training')
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='Number of training epochs')
    parser.add_argument(
        '--learning_rate', type=float, default=0.001,
        help='Learning rate')
    parser.add_argument(
        '--weight_decay', type=float, default=0.0001,
        help='Weight decay for optimizer')
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers')
    
    # Output arguments
    parser.add_argument(
        '--output_dir', type=str, default='./output',
        help='Directory to save checkpoints and final model')
    parser.add_argument(
        '--save_interval', type=int, default=5,
        help='Save checkpoint every N epochs')
    
    # Device arguments
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force using CPU even if GPU is available')
    
    args = parser.parse_args()
    main(args)
