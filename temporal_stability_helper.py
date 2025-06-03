#
# Copyright (C) 2025, Felix Hirt
# All rights reserved.
#

import torch

def temporal_smoothness_loss(rotation_current, scale_current, opacity_current,features_dc_current,features_rest_current, rotation_prev, scale_prev, opacity_prev,features_dc_prev,features_rest_prev, mask):
    '''computes temporal stability loss with focus on smoothing the properties'''
    #parameters: position, opacity, scale, rotation
    #pos_diff = (position_current - position_prev)**2
    #pos_loss = torch.mean(pos_diff[~mask])
    opacity_diff = (opacity_current - opacity_prev)**2
    opacity_loss = torch.mean(opacity_diff[~mask])
    scale_diff = (scale_current - scale_prev)**2
    scale_loss = torch.mean(scale_diff[~mask])
    features_dc_diff = (features_dc_current - features_dc_prev)**2
    features_dc_loss = torch.mean(features_dc_diff[~mask])
    features_rest_diff = (features_rest_current - features_rest_prev)**2
    features_rest_loss = torch.mean(features_rest_diff[~mask])
    
    # For quaternion rotations, use angular difference
    rotation_diff = 1 - torch.abs(torch.sum(rotation_current * rotation_prev, dim=-1))
    rotation_loss = torch.mean(rotation_diff[~mask]**2)
    
    return  opacity_loss + scale_loss + rotation_loss + features_dc_loss + features_rest_loss


def spatial_coherence_loss(position_current, position_prev, k=5, mask=None):
    '''computes loss temporal stability with focus on coherent motion of gaussian positions'''
    positions = position_current
    motion = positions - position_prev

    #find k-nearest neighbors
    indices = knn_search(positions, k=k)

    #mask for valid neighbors (non-relocated)
    if mask is not None:
        neighbor_mask = mask[indices]  #true for relocated neighbors
        valid_neighbor_counts = (~neighbor_mask).sum(dim=1)

    #compute neighbor mean motion, excluding relocated neighbors
    neighbor_motions = motion[indices] 
    if mask is not None:
        neighbor_motions[neighbor_mask] = 0  #zero out relocated neighbors

    neighbor_mean_motion = neighbor_motions.sum(dim=1) / valid_neighbor_counts.clamp(min=1).unsqueeze(-1)
    
    #compute motion loss only for non-relocated points
    if mask is not None:
        motion = motion[~mask]
        neighbor_mean_motion = neighbor_mean_motion[~mask]
    
    motion_loss = torch.mean(torch.norm(motion - neighbor_mean_motion, dim=-1)**2)
    return motion_loss

def knn_search(points, k):
    """
    Perform KNN search using PyTorch
    """
    pairwise_distances = torch.cdist(points, points)  # Compute pairwise distances
    _, indices = torch.topk(pairwise_distances, k=k, largest=False)  # Find k smallest distances
    return indices

def get_relocation_mask(position_current, position_prev, thresh=0.1):
    displacement = torch.norm(position_current - position_prev, dim=-1)
    mask = displacement > thresh  # True for relocated points
    return mask

