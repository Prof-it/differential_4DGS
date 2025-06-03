#
# Copyright (C) 2025, Felix Hirt
# All rights reserved.
#

import torch

class GaussianFrame:
    def __init__(self, new_xyz, new_features_dc, new_features_rest, new_scaling, new_rotation, new_opacity, new_max_radii=None, totalFrames=None, new_reference_counts = None):
        self.total_frames = totalFrames
        self._xyz : torch.Tensor = new_xyz
        self._features_dc : torch.Tensor = new_features_dc
        self._features_rest : torch.Tensor= new_features_rest
        self._scaling : torch.Tensor = new_scaling
        self._rotation : torch.Tensor = new_rotation
        self._opacity : torch.Tensor = new_opacity
        self.max_radii2D : torch.Tensor = new_max_radii
        self.reference_counts = new_reference_counts
    
        