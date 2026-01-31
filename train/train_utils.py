"""
Training utilities for SuperGlue with SAR data
"""

import torch
import torch.nn.functional as F
import numpy as np


def generate_ground_truth_matches(kpts0, kpts1, T_0to1, pixel_threshold=3.0):
    """Generate ground truth matches based on geometric transformation
    
    Args:
        kpts0: Keypoints from image0, shape (N0, 2) in (x, y) format
        kpts1: Keypoints from image1, shape (N1, 2) in (x, y) format
        T_0to1: Homography matrix from image0 to image1, shape (3, 3)
        pixel_threshold: Maximum pixel distance for a valid match
    
    Returns:
        gt_matches0: Ground truth matches for kpts0, shape (N0,)
                    -1 indicates unmatched (dustbin)
        gt_matches1: Ground truth matches for kpts1, shape (N1,)
    """
    if kpts0.shape[0] == 0 or kpts1.shape[0] == 0:
        return (torch.full((kpts0.shape[0],), -1, dtype=torch.long, device=kpts0.device),
                torch.full((kpts1.shape[0],), -1, dtype=torch.long, device=kpts1.device))
    
    # Convert keypoints to homogeneous coordinates
    kpts0_h = torch.cat([kpts0, torch.ones((kpts0.shape[0], 1), device=kpts0.device)], dim=1)
    
    # Warp kpts0 to image1 coordinates
    kpts0_warped_h = torch.matmul(T_0to1, kpts0_h.T).T  # (N0, 3)
    kpts0_warped = kpts0_warped_h[:, :2] / (kpts0_warped_h[:, 2:3] + 1e-8)
    
    # Compute pairwise distances
    dist_matrix = torch.cdist(kpts0_warped, kpts1)  # (N0, N1)
    
    # Find nearest neighbors
    min_dist0, nn_idx0 = dist_matrix.min(dim=1)  # For each kpt in img0
    min_dist1, nn_idx1 = dist_matrix.min(dim=0)  # For each kpt in img1
    
    # Mutual nearest neighbor check
    mutual_0to1 = (nn_idx1[nn_idx0] == torch.arange(kpts0.shape[0], device=kpts0.device))
    mutual_1to0 = (nn_idx0[nn_idx1] == torch.arange(kpts1.shape[0], device=kpts1.device))
    
    # Apply distance threshold
    valid_0 = (min_dist0 < pixel_threshold) & mutual_0to1
    valid_1 = (min_dist1 < pixel_threshold) & mutual_1to0
    
    # Create ground truth matches (-1 for dustbin/unmatched)
    gt_matches0 = torch.full((kpts0.shape[0],), -1, dtype=torch.long, device=kpts0.device)
    gt_matches1 = torch.full((kpts1.shape[0],), -1, dtype=torch.long, device=kpts1.device)
    
    gt_matches0[valid_0] = nn_idx0[valid_0]
    gt_matches1[valid_1] = nn_idx1[valid_1]
    
    return gt_matches0, gt_matches1


def compute_superglue_loss(pred, kpts0, kpts1, T_0to1, pixel_threshold=3.0):
    """Compute SuperGlue loss based on geometric consistency
    
    Args:
        pred: Predictions from SuperGlue model containing 'scores' or matching matrices
        kpts0: Keypoints from image0, list or tensor
        kpts1: Keypoints from image1, list or tensor
        T_0to1: Homography matrix, tensor of shape (B, 3, 3)
        pixel_threshold: Pixel threshold for valid matches
    
    Returns:
        loss: Scalar loss value
        stats: Dictionary with loss statistics
    """
    batch_size = T_0to1.shape[0]
    total_loss = 0.0
    num_correct = 0
    num_total = 0
    
    for b in range(batch_size):
        # Get keypoints for this batch item
        if isinstance(kpts0, list):
            kp0 = kpts0[b]
            kp1 = kpts1[b]
        else:
            kp0 = kpts0[b]
            kp1 = kpts1[b]
        
        # Generate ground truth matches
        gt_matches0, gt_matches1 = generate_ground_truth_matches(
            kp0, kp1, T_0to1[b], pixel_threshold)
        
        if kp0.shape[0] == 0 or kp1.shape[0] == 0:
            continue
        
        # Get predicted scores from log_optimal_transport output
        # pred contains the full assignment matrix including dustbin
        # We need to extract it from the model's forward pass
        # For now, use a simplified approach based on matching_scores
        
        # Since we're in training mode, we need to compute loss on the assignment matrix
        # This requires access to the scores before thresholding
        # We'll need to modify this based on the actual model output
        
        # For simplicity, we'll compute cross-entropy loss on the assignment
        # This is a placeholder that should be refined
        matches0_pred = pred['matches0'][b] if isinstance(pred['matches0'], list) else pred['matches0'][b]
        
        # Compute accuracy
        valid_mask = gt_matches0 >= 0
        if valid_mask.sum() > 0:
            correct = (matches0_pred[valid_mask] == gt_matches0[valid_mask]).float().sum()
            num_correct += correct.item()
            num_total += valid_mask.sum().item()
    
    # Placeholder loss - this should be replaced with proper assignment matrix loss
    loss = torch.tensor(0.0, requires_grad=True)
    
    stats = {
        'loss': loss.item(),
        'accuracy': num_correct / max(num_total, 1),
        'num_matches': num_total
    }
    
    return loss, stats


def compute_superglue_loss_from_scores(scores, gt_matches0, gt_matches1, device):
    """Compute cross-entropy loss from assignment matrix and ground truth
    
    Args:
        scores: Log assignment matrix from log_optimal_transport, shape (B, M+1, N+1)
        gt_matches0: Ground truth matches for image0, shape (B, M)
        gt_matches1: Ground truth matches for image1, shape (B, N)
        device: torch device
    
    Returns:
        loss: Scalar loss value
    """
    batch_size, m_plus_1, n_plus_1 = scores.shape
    m, n = m_plus_1 - 1, n_plus_1 - 1
    
    total_loss = 0.0
    num_items = 0
    
    for b in range(batch_size):
        # Get ground truth for this batch
        gt_m0 = gt_matches0[b]  # (M,)
        
        # Create target indices (M,) where each element is the match index or dustbin
        targets = gt_m0.clone()
        targets[gt_m0 == -1] = n  # Dustbin is at index N
        targets = targets.clamp(0, n)  # Ensure valid range
        
        # Extract scores for this batch (M+1, N+1)
        batch_scores = scores[b]  # (M+1, N+1)
        
        # Use only the first M rows (keypoints, not dustbin row)
        kpt_scores = batch_scores[:m, :]  # (M, N+1)
        
        # Compute cross-entropy loss
        loss_b = F.cross_entropy(kpt_scores, targets, reduction='mean')
        total_loss += loss_b
        num_items += 1
    
    if num_items > 0:
        loss = total_loss / num_items
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    return loss


def freeze_layers(model, layer_names):
    """Freeze specific layers in a model
    
    Args:
        model: PyTorch model
        layer_names: List of layer name patterns to freeze
    """
    for name, param in model.named_parameters():
        for layer_pattern in layer_names:
            if layer_pattern in name:
                param.requires_grad = False
                break


def unfreeze_layers(model, layer_names):
    """Unfreeze specific layers in a model
    
    Args:
        model: PyTorch model
        layer_names: List of layer name patterns to unfreeze
    """
    for name, param in model.named_parameters():
        for layer_pattern in layer_names:
            if layer_pattern in name:
                param.requires_grad = True
                break


def freeze_model(model):
    """Freeze all parameters in a model"""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    """Unfreeze all parameters in a model"""
    for param in model.parameters():
        param.requires_grad = True


def get_trainable_params(model):
    """Get count of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(epoch, model_sp, model_sg, optimizer_sp, optimizer_sg, 
                    loss, path):
    """Save training checkpoint
    
    Args:
        epoch: Current epoch number
        model_sp: SuperPoint model
        model_sg: SuperGlue model
        optimizer_sp: SuperPoint optimizer (can be None)
        optimizer_sg: SuperGlue optimizer
        loss: Current loss value
        path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'superpoint_state_dict': model_sp.state_dict(),
        'superglue_state_dict': model_sg.state_dict(),
        'optimizer_sg_state_dict': optimizer_sg.state_dict(),
        'loss': loss,
    }
    
    if optimizer_sp is not None:
        checkpoint['optimizer_sp_state_dict'] = optimizer_sp.state_dict()
    
    torch.save(checkpoint, path)


def load_checkpoint(path, model_sp, model_sg, optimizer_sp=None, optimizer_sg=None):
    """Load training checkpoint
    
    Args:
        path: Path to checkpoint file
        model_sp: SuperPoint model
        model_sg: SuperGlue model
        optimizer_sp: SuperPoint optimizer (optional)
        optimizer_sg: SuperGlue optimizer (optional)
    
    Returns:
        start_epoch: Epoch to resume from
        loss: Loss value from checkpoint
    """
    checkpoint = torch.load(path)
    
    model_sp.load_state_dict(checkpoint['superpoint_state_dict'])
    model_sg.load_state_dict(checkpoint['superglue_state_dict'])
    
    if optimizer_sg is not None and 'optimizer_sg_state_dict' in checkpoint:
        optimizer_sg.load_state_dict(checkpoint['optimizer_sg_state_dict'])
    
    if optimizer_sp is not None and 'optimizer_sp_state_dict' in checkpoint:
        optimizer_sp.load_state_dict(checkpoint['optimizer_sp_state_dict'])
    
    return checkpoint['epoch'], checkpoint.get('loss', 0.0)
