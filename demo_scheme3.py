#!/usr/bin/env python3
"""
Minimal demonstration of Scheme 3 fine-tuning workflow.

This script demonstrates the complete workflow:
1. Create sample RGB PNG images
2. Load dataset with automatic channel detection
3. Create SuperPoint model with channel adaptation
4. Run a minimal training iteration
"""

import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.superpoint import SuperPoint
from data import S2RGBPNGDataset


def create_sample_rgb_images(output_dir, num_images=10):
    """Create sample RGB PNG images for demonstration."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_images} sample RGB images...")
    for i in range(num_images):
        # Create a random RGB image (simulating S2 B4/B3/B2)
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # Add some structure to make it more realistic
        # Simulate some features/edges
        h, w = img.shape[:2]
        cv2.rectangle(img, (w//4, h//4), (3*w//4, 3*h//4), (255, 255, 255), 2)
        cv2.circle(img, (w//2, h//2), 50, (128, 128, 128), -1)
        
        # Save as PNG
        img_path = output_dir / f"s2_image_{i:03d}.png"
        cv2.imwrite(str(img_path), img)
    
    print(f"✓ Created {num_images} images in {output_dir}")
    return output_dir


def demonstrate_workflow():
    """Demonstrate the complete Scheme 3 workflow."""
    
    print("="*70)
    print("Scheme 3 Fine-tuning Demonstration")
    print("="*70)
    
    # Create temporary directory for sample data
    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Create sample RGB images
        print("\n[Step 1] Creating sample RGB PNG images...")
        image_dir = create_sample_rgb_images(tmpdir, num_images=10)
        
        # Step 2: Load dataset with automatic channel detection
        print("\n[Step 2] Loading dataset with automatic channel detection...")
        dataset = S2RGBPNGDataset(image_dir)
        
        print(f"  ✓ Dataset loaded: {len(dataset)} images")
        print(f"  ✓ Channels detected: {dataset.channels}")
        print(f"  ✓ Band mapping: {dataset.band_mapping}")
        
        # Examine a sample
        sample = dataset[0]
        print(f"  ✓ Sample image shape: {sample['image'].shape} (CHW)")
        print(f"  ✓ Sample dtype: {sample['image'].dtype}")
        print(f"  ✓ Sample range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
        
        # Step 3: Infer channels and create model
        print("\n[Step 3] Creating SuperPoint model with automatic channel adaptation...")
        in_channels = sample['image'].shape[0]
        print(f"  → Inferred {in_channels} input channels from dataset")
        
        model = SuperPoint({
            'in_channels': in_channels,
            'load_pretrained': True,  # Will adapt 1ch→3ch automatically
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024,
        })
        
        print(f"  ✓ Model created with {in_channels} input channels")
        print(f"  ✓ First conv layer: {model.conv1a}")
        
        # Step 4: Run inference to verify model works
        print("\n[Step 4] Running inference to verify model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            # Prepare batch
            images = torch.stack([dataset[i]['image'] for i in range(min(4, len(dataset)))])
            images = images.to(device)
            
            # Run model
            output = model({'image': images})
            
            print(f"  ✓ Inference successful on batch of {images.shape[0]} images")
            print(f"  ✓ Detected keypoints per image: {[len(kp) for kp in output['keypoints']]}")
            print(f"  ✓ Descriptor shape per image: {[desc.shape for desc in output['descriptors']]}")
        
        # Step 5: Demonstrate training iteration
        print("\n[Step 5] Demonstrating a training iteration...")
        model.train()
        
        # Create dummy optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training iteration
        optimizer.zero_grad()
        
        # Forward pass with gradient enabled
        output = model({'image': images})
        
        # Dummy loss using descriptor magnitudes (demonstrates training flow)
        # In real training, you would use proper losses like:
        # - Heatmap loss for keypoint detection
        # - Triplet/contrastive loss for descriptors
        # Concatenate all descriptors into one tensor for loss computation
        all_descriptors = torch.cat(output['descriptors'], dim=1)  # Concatenate along keypoint dimension
        loss = all_descriptors.abs().mean()  # Simple dummy loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"  ✓ Training iteration completed")
        print(f"  ✓ Loss: {loss.item():.4f}")
        
        # Summary
        print("\n" + "="*70)
        print("Workflow Summary")
        print("="*70)
        print("✓ RGB PNG images created and loaded")
        print(f"✓ Channels automatically detected: {in_channels}")
        print(f"✓ Pretrained weights adapted: 1ch → {in_channels}ch")
        print("✓ Model inference verified")
        print("✓ Training iteration successful")
        print("\nThe model is ready for fine-tuning!")
        print("="*70)


if __name__ == '__main__':
    demonstrate_workflow()
