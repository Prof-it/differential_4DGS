#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# This code was modified by Felix Hirt to adapt for a multi optimizer differential 4D Gaussian Splatting method


import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, frame, raw_render = False, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    full_xyz = pc.get_xyz(frame, raw_render)
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(full_xyz, dtype=pc._xyz[frame].dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
      
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = full_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity(frame, raw_render)
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(frame, scaling_modifier)
    else:
        scales = pc.get_scaling(frame, raw_render)
        rotations = pc.get_rotation(frame, raw_render)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features(frame, raw_render).transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features(frame, raw_render).shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc(frame, raw_render), pc.get_features_rest(frame, raw_render)
            else:
                shs = pc.get_features(frame, raw_render)
    else:
        colors_precomp = override_color

   # print("renderinfo")
  #  print(means3D.shape)
   # print(means2D.shape)
    #print(dc.shape)
  #  print(shs.shape)
    #print(colors_precomp.shape)
    #print(opacity.shape)
   # print(scales)
    #print(rotations.shape)
    #print(cov3D_precomp.shape)
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out



def render_interpolated(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, frame, future_frame, raw_render = False, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    start_frame = math.floor(frame)
    end_frame = future_frame
    interpolation_position = frame - start_frame 

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(interpolate_tensors(pc.get_xyz(start_frame, raw_render),pc.get_xyz(end_frame, raw_render),interpolation_position), dtype=pc._xyz[start_frame].dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
      
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = interpolate_tensors(pc.get_xyz(start_frame, raw_render),pc.get_xyz(end_frame, raw_render),interpolation_position)
    means2D = screenspace_points
    opacity = interpolate_tensors(pc.get_opacity(start_frame, raw_render), pc.get_opacity(end_frame, raw_render), interpolation_position)


    if(interpolation_position > 0):
        #hide long distance points
        # Example tensors containing 3D points
        points1 = pc.get_xyz(start_frame, raw_render)
        points2 = pc.get_xyz(end_frame, raw_render)

        diff = points1 - points2  # This assumes both tensors have the same shape
        dists = torch.norm(diff, dim=1)  

        # Define distance threshold
        threshold = 0.5

        # Get indices where distance exceeds threshold
        indices = torch.nonzero(dists > threshold, as_tuple=True)

        opacity[indices] = 0.001

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(start_frame, scaling_modifier)
    else:
        scales = interpolate_tensors(pc.get_scaling(start_frame, raw_render), pc.get_scaling(end_frame, raw_render), interpolation_position)
        rotations = pc.rotation_activation(interpolate_tensors(pc.get_rotation_unnormalized(start_frame, raw_render), pc.get_rotation_unnormalized(end_frame, raw_render), interpolation_position))

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = interpolate_tensors(pc.get_features(start_frame, raw_render), pc.get_features(end_frame, raw_render), interpolation_position).transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(interpolate_tensors(pc.get_features(start_frame, raw_render), pc.get_features(end_frame, raw_render), interpolation_position).shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = interpolate_tensors(pc.get_features_dc(start_frame, raw_render), pc.get_features_dc(end_frame, raw_render), interpolation_position), interpolate_tensors(pc.get_features_rest(start_frame, raw_render), pc.get_features_rest(end_frame, raw_render), interpolation_position)
            else:
                shs = interpolate_tensors(pc.get_features(start_frame, raw_render), pc.get_features(end_frame, raw_render), interpolation_position)
    else:
        colors_precomp = override_color

   # print("renderinfo")
  #  print(means3D.shape)
   # print(means2D.shape)
    #print(dc.shape)
  #  print(shs.shape)
    #print(colors_precomp.shape)
    #print(opacity.shape)
   # print(scales)
    #print(rotations.shape)
    #print(cov3D_precomp.shape)
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 

    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out


def interpolate_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, position: float) -> torch.Tensor:
    return tensor1
    #return slerp_tensors(tensor1, tensor2, position)

    """
    Linearly interpolates between two tensors.
    
    Args:
        tensor1 (torch.Tensor): The starting tensor.
        tensor2 (torch.Tensor): The ending tensor.
        position (float): A value between 0 and 1 indicating interpolation position.
            - 0 returns tensor1
            - 1 returns tensor2
            - Values in between return interpolated tensors
    
    Returns:
        torch.Tensor: The interpolated tensor.
    """
    if(tensor1.shape[0] != tensor2.shape[0]):
        return tensor1
    if not (0.0 <= position <= 1.0):
        raise ValueError("Position must be between 0 and 1.")
    
    new_tensor = (1 - position) * tensor1 + position * tensor2
    
    return new_tensor

def slerp_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, position: float) -> torch.Tensor:
    """
    Spherical linear interpolation (SLERP) between two tensors.
    
    Args:
        tensor1 (torch.Tensor): The starting tensor.
        tensor2 (torch.Tensor): The ending tensor.
        position (float): A value between 0 and 1 indicating interpolation position.
            - 0 returns tensor1
            - 1 returns tensor2
            - Values in between return interpolated tensors
    
    Returns:
        torch.Tensor: The interpolated tensor.
    """
    if not (0.0 <= position <= 1.0):
        raise ValueError("Position must be between 0 and 1.")
    
    tensor1, tensor2 = tensor1.to(torch.float32), tensor2.to(torch.float32)
    
    dot_product = (tensor1 * tensor2).sum()
    omega = torch.acos(dot_product / (tensor1.norm() * tensor2.norm()))
    
    if torch.isclose(omega, torch.tensor(0.0)):
        return (1 - position) * tensor1 + position * tensor2
    
    sin_omega = torch.sin(omega)
    interp_tensor = (torch.sin((1 - position) * omega) / sin_omega) * tensor1 + (torch.sin(position * omega) / sin_omega) * tensor2
    
    return interp_tensor