#
# Copyright (C) 2025, Felix Hirt
# All rights reserved.
#

import torch
import PIL as plt
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image
import cv2
import scipy
from typing import List
oneShow = 0

def dilate_mask(mask: torch.Tensor, iterations: int) -> torch.Tensor:
    """Dilates binary mask """
    kernel = torch.tensor([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)

    #ensure input is float and in shape (1, 1, H, W)
    dilated = mask.float().unsqueeze(0).unsqueeze(0)
    
    for _ in range(iterations):
        dilated = F.conv2d(dilated, kernel, padding=1)
        dilated = (dilated > 0).float()
    
    #return back to shape (H, W)
    return dilated.squeeze(0).squeeze(0)

def binary_difference_matte(img1, img2, threshold: float = 0.04, dilation: int = 0):
    """
    Creates a binary mask highlighting differences between two images.
    """
    #compute difference between images
    diff = torch.abs(img1 - img2)
    
    #sum across the color channels
    diff_sum = torch.sum(diff, dim=-3)
    
    #create binary mask using threshold
    binary_diff_mask = (diff_sum > threshold).float()

    #dilate mask to capture more Gaussian points
    binary_diff_mask = dilate_mask(binary_diff_mask, dilation)

    return binary_diff_mask

def visualize_binary_matte(binary_matte: torch.Tensor):
    """
    Visualizes a binary difference matte using PIL
    """
    #convert to numpy and scale to 255
    matte_np = (binary_matte.cpu().numpy() * 255).astype(np.uint8)
    
    #convert to PIL image
    matte_img = Image.fromarray(matte_np, mode="L")  # "L" mode for grayscale
    
    #show the image
    matte_img.show()

def compute_frame_references(
    frames, 
    threshold,
    dilation
):
    """
    computes per-frame reference counts indicating how many frames back can be reused.
    """
    if not frames:
        return []
    
    no_change_masks = []
    for i in range(len(frames)-1):
        diff_mask = binary_difference_matte(frames[i], frames[i+1], threshold, dilation=dilation)
        no_change_masks.append(1 - diff_mask)  # Invert 
    
    
    device = frames[0].device
    shape = frames[0].shape[:-3] + frames[0].shape[-2:]  # Preserve batch dimensions
    reference_counts = [torch.zeros(shape, device=device, dtype=torch.float32)]
    
    #compute reference counts
    for i in range(1, len(frames)):
        if i-1 < len(no_change_masks):
            current_mask = no_change_masks[i-1]
        else:
            current_mask = torch.zeros_like(reference_counts[-1])
            
        new_count = current_mask * (reference_counts[i-1] + 1)
        reference_counts.append(new_count)
 

    global oneShow
    if(oneShow<1):
        for i in range(1,2):
            #convert binary_diff_mask tensor to a NumPy array
            binary_diff_mask_np = reference_counts[i].squeeze().cpu().numpy()  # Remove batch and channel dimensions, and convert to NumPy

            #convert NumPy array to a PIL image
            binary_diff_mask_pil = Image.fromarray((binary_diff_mask_np * 255).astype('uint8'))
            #display PIL image
            binary_diff_mask_pil.show()
    oneShow += 1
    return reference_counts

def filter_by_reference(input_tensor, reference_counts, frame):
    """
    extracts elements from input_tensor where reference_counts are 0.
    """
    mask = reference_counts[frame] == 0  # Create a boolean mask where reference_counts is 0
    
    return input_tensor[mask]

def select_masked_elements_in_tensorUNOPTIMIZED(current_frame_idx, reference_counts, tensor_list):
    device = tensor_list[0].device
    property_dims = tensor_list[0].shape[1:]
    new_tensor = torch.zeros((reference_counts[0].shape[0], *property_dims), dtype=tensor_list[0].dtype, device=device)
    for i in range(current_frame_idx+1):
        mask = (reference_counts[current_frame_idx] == (current_frame_idx - i)) & (reference_counts[i] == 0)
        scaled_mask = mask[reference_counts[i] == 0].nonzero(as_tuple=True)[0]
        new_tensor[mask.nonzero(as_tuple=True)[0]] = tensor_list[i][scaled_mask]
    return new_tensor

def select_masked_elements_in_tensor(current_frame_idx, reference_counts, tensor_list):
    """
    reconstructs property values from previous frames based on reference_counts
    """

    full_tensors = []
    total_num_points = reference_counts[0].shape[0]
    #reconstruct full tensors from the filtered ones for frames 0..current_frame_idx.
    for f in range(current_frame_idx + 1):
        # Create the mask that was used during filtering.
        mask = (reference_counts[f] == 0)  # shape: [total_num_points]
        filtered_tensor = tensor_list[f]
        
        #some properties have different dimensions(rotation, position, features).
        property_dims = filtered_tensor.shape[1:]  #could be (K,) or more dimensions
        full_tensor = torch.zeros((total_num_points, *property_dims),
                                    dtype=filtered_tensor.dtype,
                                    device=filtered_tensor.device)
        
        #place the filtered values back into their original positions.
        full_tensor[mask] = filtered_tensor
        full_tensors.append(full_tensor)
    
    tensor_stacked = torch.stack(full_tensors, dim=0)
    
    #for each point in the current frame, determine from which previous frame to pick the property value.
    current_ref = reference_counts[current_frame_idx]
    
    source_frames = current_frame_idx - current_ref.long()
    source_frames = torch.clamp(source_frames, min=0, max=current_frame_idx)  #avoid negative indices
    
    #create points indices
    points_idx = torch.arange(total_num_points, device=tensor_stacked.device)
    
    #select elements
    if(tensor_stacked.dim() > 3):
        selected_elements = tensor_stacked[source_frames, points_idx, :, :]
    else:
        selected_elements = tensor_stacked[source_frames, points_idx]
    
    return selected_elements

def calculate_frame_variations(viewpoint_stack, threshold, dilation=0):
    reference_counts =[]

    for i in range(len(viewpoint_stack[0])):
        list_of_frames = []
        for j in range(len(viewpoint_stack)):
            list_of_frames.append(viewpoint_stack[j][i].original_image.cuda())

        reference_counts.append(compute_frame_references(list_of_frames, threshold, dilation))

    return reference_counts

def generate_frame_references(cameras_list,frame_variations,points3D, radius):
    """
    Compute frame reference for each 3D point per frame
    """
    num_frames = len(cameras_list)
    num_points = points3D.shape[0]
    device = points3D.device
    output = []

    for frame_idx in range(num_frames):
        frame_cameras = cameras_list[frame_idx]
        frame_output = torch.full((num_points,), float('inf'), device=device)

        #count frame references for all points
        max_ref = max([fv.max().item() for fv_list in frame_variations for fv in fv_list]) + 1
        ref_counts = torch.zeros((num_points, int(max_ref)), device=device, dtype=torch.long)


        for vp_idx, camera in enumerate(frame_cameras):
            #frame variations tensor for this viewpoint and frame
            fv_tensor = frame_variations[vp_idx][frame_idx]

            corrected_world_view = torch.inverse(camera.world_view_transform)
            corrected_world_view = offset_world_view_transform(corrected_world_view, camera.camera_center)

            #transform 3D points from world space to camera space
            points_homog = torch.cat([points3D, torch.ones_like(points3D[:, :1])], dim=1)
            camera_space = torch.matmul(points_homog, corrected_world_view.T)

            projection_matrix = compute_projection_matrix(camera.FoVx, camera.FoVy, 0, 10000000)

            #project points from camera space to clip space
            clip_space = torch.matmul(camera_space,  projection_matrix.T)
            w = clip_space[:, 3]
            valid_w = w != 0
            ndc = clip_space[:, :3] / w.unsqueeze(1)

            #check if points are within the view
            valid_x = (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0)
            valid_y = (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
            valid_z = (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
            valid_ndc = valid_x & valid_y & valid_z & valid_w

            #convert clip space to pixel coordinates
            width, height = camera.image_width, camera.image_height
            u = (1.0-ndc[:, 0]) * (width - 1) / 2.0
            v = (1.0 - ndc[:, 1]) * (height - 1) / 2.0

            u_idx = torch.round(u).long().clamp(0, width - 1)
            v_idx = torch.round(v).long().clamp(0, height - 1)

            valid_uv = (u_idx >= 0) & (u_idx < width) & (v_idx >= 0) & (v_idx < height)
            valid = valid_ndc & valid_uv

            #get frame references for valid points
            valid_indices = valid.nonzero(as_tuple=False).squeeze(1)
            refs = fv_tensor[v_idx[valid_indices], u_idx[valid_indices]].long()

            ref_counts[valid_indices, refs] += 1

        #use the most frequent frame reference for each point
        frame_output = ref_counts.argmax(dim=1)

        #use current frame if no valid viewpoints
        frame_output[ref_counts.sum(dim=1) == 0] = frame_idx

        #assign additional points if they are within a radius to a valid point
        frame_output = assign_nearby_points(frame_output, points3D, radius=radius)  # Adjust radius as needed

        output.append(frame_output)
        
        #visualize_frame_variation(fv_tensor)
        #visualize_points_projection(points3D, camera)

        #visualize_points_projection(frame_output, camera)

    return output

def assign_nearby_points(frame_output, points3D, radius):
    """
    assigns reference to unchosen points if they are within a certain radius
    of a chosen point.
    """
    chosen_mask = frame_output == 0#float('inf')  #chosen points
    unchosen_mask = ~chosen_mask  #unchosen points
    
    if unchosen_mask.sum() == 0 or chosen_mask.sum() == 0:
        return frame_output  #no updates needed if all are chosen or none are chosen
    
    chosen_points = points3D[chosen_mask]
    unchosen_points = points3D[unchosen_mask]

    #compute pairwise distances between unchosen and chosen points
    #dist_matrix = torch.cdist(unchosen_points, chosen_points) 
    min_dist, nearest_idx = batched_cdist(unchosen_points, chosen_points)
    
    #find the nearest chosen point for each unchosen point
    #min_dist, nearest_idx = dist_matrix.min(dim=1)
    #assign references for unchosen points within the radius
    within_radius_mask = min_dist <= radius
    if within_radius_mask.any():
        unchosen_indices = unchosen_mask.nonzero(as_tuple=True)[0]  #get valid indices
        frame_output[unchosen_indices[within_radius_mask]] = 0
            #frame_output[chosen_mask.nonzero(as_tuple=True)[0][nearest_idx[within_radius_mask]]]

    return frame_output

def batched_cdist(unchosen_points, chosen_points, batch_size=1024):
    min_dists = []
    nearest_indices = []

    for i in range(0, unchosen_points.shape[0], batch_size):
        batch = unchosen_points[i:i+batch_size]
        dists = torch.cdist(batch, chosen_points)
        min_dist, idx = dists.min(dim=1)
        min_dists.append(min_dist)
        nearest_indices.append(idx)

    return torch.cat(min_dists), torch.cat(nearest_indices)


def points_changed_frame_to_frame(viewpoint_list_1, viewpoint_list_2,points3D):
    """
    compute the minimum frame reference for each 3D point from one frame to antoher
    """
    num_points = points3D.shape[0]
    frame_variations = calculate_frame_variations([viewpoint_list_1, viewpoint_list_2], 0.05)
    device = points3D.device
    output = []

    frame_cameras = viewpoint_list_2
    frame_output = torch.full((num_points,), float('inf'), device=device)

    
    #initialize a tensor to count frame references for all points
    max_ref = max([fv.max().item() for fv_list in frame_variations for fv in fv_list]) + 1
    ref_counts = torch.zeros((num_points, int(max_ref)), device=device, dtype=torch.long)


    for vp_idx, camera in enumerate(frame_cameras):
        #get the frame variation tensor for this viewpoint and frame
        fv_tensor = frame_variations[vp_idx][1]

        corrected_world_view = torch.inverse(camera.world_view_transform)
        #correct the world_view_transform if mirrored on the x-axis
        #corrected_world_view = fix_mirrored_world_view_transform(corrected_world_view)

        corrected_world_view = offset_world_view_transform(corrected_world_view, camera.camera_center)

        #transform 3D points from world space to camera space using world_view_transform
        points_homog = torch.cat([points3D, torch.ones_like(points3D[:, :1])], dim=1)
        camera_space = torch.matmul(points_homog, corrected_world_view.T)

        projection_matrix = compute_projection_matrix(camera.FoVx, camera.FoVy, 0, 10000000)

        #project points from camera space to clip space using full_proj_transform
        clip_space = torch.matmul(camera_space,  projection_matrix.T)#camera.full_proj_transform.T)
        w = clip_space[:, 3]
        valid_w = w != 0
        ndc = clip_space[:, :3] / w.unsqueeze(1)

        #check if points are within the view frustum
        valid_x = (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0)
        valid_y = (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
        valid_z = (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
        valid_ndc = valid_x & valid_y & valid_z & valid_w

        #convert NDC to pixel coordinates
        width, height = camera.image_width, camera.image_height
        u = (1.0-ndc[:, 0]) * (width - 1) / 2.0
        v = (1.0 - ndc[:, 1]) * (height - 1) / 2.0

        u_idx = torch.round(u).long().clamp(0, width - 1)
        v_idx = torch.round(v).long().clamp(0, height - 1)

        valid_uv = (u_idx >= 0) & (u_idx < width) & (v_idx >= 0) & (v_idx < height)
        valid = valid_ndc & valid_uv

        #get frame references for valid points
        valid_indices = valid.nonzero(as_tuple=False).squeeze(1)
        refs = fv_tensor[v_idx[valid_indices], u_idx[valid_indices]].long()

        #update reference counts for valid points
        ref_counts[valid_indices, refs] += 1

    #determine the most frequent frame reference for each point
    frame_output = ref_counts.argmax(dim=1)

    #default to 1 (prev frame) if no valid viewpoints
    frame_output[ref_counts.sum(dim=1) == 0] = 1
    #visualize_frame_variation(fv_tensor)
    #visualize_points_projection(points3D, camera)

    #visualize_points_projection(points3D[frame_output == 0], camera).show()

    return frame_output


    #default to first frame if no valid viewpoints
    #frame_output[frame_output == float('inf')] = frame_idx
    output.append(frame_output)
    
    #visualize_frame_variation(fv_tensor)
    #visualize_points_projection(points3D, camera)

    #visualize_points_projection(frame_output, camera)

    return output


def visualize_points_projection(points3D, camera):
    """
    visualize the projection of 3D points into the camera's 2D space.
    """
    corrected_world_view = torch.inverse(camera.world_view_transform)
    #correct the world_view_transform if mirrored on the x-axis
    #corrected_world_view = fix_mirrored_world_view_transform(corrected_world_view)

    corrected_world_view = offset_world_view_transform(corrected_world_view, camera.camera_center)

    #transform 3D points from world space to camera space using world_view_transform
    points_homog = torch.cat([points3D, torch.ones_like(points3D[:, :1])], dim=1)
    camera_space = torch.matmul(points_homog, corrected_world_view.T)

    #correct z-axis mirroring if necessary
    #if torch.det(camera.world_view_transform[:3, :3]) < 0:
    #camera_space[:, 2] *= -1

    projection_matrix = compute_projection_matrix(camera.FoVx, camera.FoVy, 0, 10000000)
    #projection_matrix = offset_projection_matrix(projection_matrix, camera.camera_center)

    #projection_matrix = adjust_projection_matrix(projection_matrix, camera.camera_center)

    #print(test_projection_alignment(projection_matrix, camera.camera_center))

    #project points from camera space to clip space using full_proj_transform
    clip_space = torch.matmul(camera_space,  projection_matrix.T)#camera.full_proj_transform.T)
    w = clip_space[:, 3]
    valid_w = w != 0
    ndc = clip_space[:, :3] / w.unsqueeze(1)

    valid_x = (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0)
    valid_y = (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
    valid_z = (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
    valid = valid_x & valid_y & valid_z & valid_w

    width = int(camera.image_width)
    height = int(camera.image_height)

    u = (1.0 - ndc[:, 0]) * (width - 1) / 2.0
    v = (1.0 - ndc[:, 1]) * (height - 1) / 2.0

    u_idx = torch.round(u).long()
    v_idx = torch.round(v).long()

    u_idx = torch.clamp(u_idx, 0, width - 1)
    v_idx = torch.clamp(v_idx, 0, height - 1)

    image = Image.new('L', (width, height), 0)
    pixels = image.load()

    valid_indices = valid.nonzero(as_tuple=True)[0].cpu().numpy()
    u_valid = u_idx[valid].cpu().numpy()
    v_valid = v_idx[valid].cpu().numpy()

    for u, v in zip(u_valid, v_valid):
        if 0 <= u < width and 0 <= v < height:
            pixels[u, v] = 255

    #image.show()
    return image

def visualize_frame_variation(frame_variation_tensor):
    """
    visualize a frame variation tensor using a colormap.
    """
    if frame_variation_tensor.device != 'cpu':
        fv_np = frame_variation_tensor.cpu().numpy()
    else:
        fv_np = frame_variation_tensor.numpy()

    if np.ptp(fv_np) == 0:
        normalized = np.zeros_like(fv_np, dtype=np.uint8)
    else:
        normalized = ((fv_np - np.min(fv_np)) / np.ptp(fv_np) * 255).astype(np.uint8)

    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(colored_rgb)
    image.show()


def compute_projection_matrix(FoVx, FoVy, n, f):
    """
    compute the projection matrix using FoVx, FoVy, near, and far planes.
    """
    #compute focal lengths from FoV
    fx = 1 / math.tan(FoVx / 2)
    fy = 1 / math.tan(FoVy / 2)
    
    #standard perspective projection matrix
    projection_matrix = torch.tensor([
        [fx, 0,  0, 0],
        [0, fy, 0, 0],
        [0, 0, (f+n)/(n-f), (2*f*n)/(n-f)],
        [0, 0, -1, 0]
    ], dtype=torch.float32, device="cuda")
        
    
    return projection_matrix

def fix_mirrored_world_view_transform(world_view_transform):
    """
    corrects the world_view_transform if it is mirrored on the x-axis.
    """
    #flip the sign of the x-axis components (first column)
    corrected_transform = world_view_transform.clone()
    corrected_transform[:, 2] *= -1
    return corrected_transform

def offset_world_view_transform(world_view_transform, camera_center):
    """
    adjusts the world_view_transform so that its camera center matches the provided camera_center
    """
    #extract the rotation component from the world_view_transform
    rotation = world_view_transform[:3, :3]

    #compute the new translation component
    translation = -torch.matmul(rotation, camera_center)

    #construct the corrected world_view_transform
    corrected_transform = torch.eye(4, device=world_view_transform.device)
    corrected_transform[:3, :3] = rotation
    corrected_transform[:3, 3] = translation

    return corrected_transform

def zero_elements_with_reference(current_frame_idx, reference_counts, tensor_list):
    """
    zeros out elements in the current frame that have a reference_count > 0
    """

    new_tensor_list = [tensor.clone() for tensor in tensor_list]
    
    #get the reference counts for the current frame
    current_ref_counts = reference_counts[current_frame_idx]  # [N]
    
    #find indices where reference count is greater than 0
    mask = current_ref_counts > 0
    if not mask.any():
        return new_tensor_list  # No elements to zero out
    
    #zero out the elements in the current frame where reference_count > 0
    new_tensor_list[current_frame_idx].data[mask] = 0
    
    return new_tensor_list
    
def get_shared_masks(reference_counts, frame_a, frame_b):
    """
    using reference_counts for each frame this returns two boolean masks that indicate which elements in the filtered tensors for frame a and frame b
    are shared between the two frames.
    """
    #global masks, True for points used in that frame.
    global_mask_a = (reference_counts[frame_a] == 0)
    global_mask_b = (reference_counts[frame_b] == 0)
    
    #the global indices (in sorted order) corresponding to each filtered tensor
    idx_a = torch.nonzero(global_mask_a, as_tuple=True)[0]
    idx_b = torch.nonzero(global_mask_b, as_tuple=True)[0]
    
    #compute the set of global indices that are shared between the two frames
    set_a = set(idx_a.tolist())
    set_b = set(idx_b.tolist())
    common_global = sorted(set_a.intersection(set_b))
    
    #convert common_global back to a tensor.
    if len(common_global) > 0:
        common_global_tensor = torch.tensor(common_global, dtype=idx_a.dtype, device=idx_a.device)
    else:
        #if there is no intersection, return empty boolean masks.
        return torch.zeros_like(idx_a, dtype=torch.bool), torch.zeros_like(idx_b, dtype=torch.bool)
    
    #create masks
    try:
        # If torch.isin is available, use it.
        mask_a = torch.isin(idx_a, common_global_tensor)
        mask_b = torch.isin(idx_b, common_global_tensor)
    except AttributeError:
        # Fallback if torch.isin is not available.
        mask_a = torch.tensor([val in common_global for val in idx_a.tolist()], device=idx_a.device)
        mask_b = torch.tensor([val in common_global for val in idx_b.tolist()], device=idx_b.device)
    
    return mask_a, mask_b

from scipy.optimize import linear_sum_assignment

def reassign_selected_points(xyz, selected_points, pruned_points, pruned_points_subset, threshold = None):
    """
    reassigns selected points to pruned points
    """
    device = xyz.device
    #if there are fewer pruned points than selected points shorten the selection
    if pruned_points.size(0) == selected_points.size(0):
        return pruned_points
    
    n_selected = selected_points.size(0)
    if pruned_points_subset.dim() == 0:
        pruned_points_subset = pruned_points_subset.unsqueeze(0)
    n_subset = pruned_points_subset.size(0)
    
    new_selected = torch.empty_like(selected_points)
    
    #if the subset is large enough to cover all selected points it can be used.
    if n_subset >= n_selected:
        new_selected = pruned_points_subset[:n_selected]
        return new_selected
    
    assigned_list = []
    
    #assign the entries from the pruned_points_subset.
    for idx in range(min(n_subset, n_selected)):
        assigned_list.append(pruned_points_subset[idx].item())
    
    #if subset covers selected points it can be used.
    if n_subset >= n_selected:
        return torch.tensor(assigned_list, device=device, dtype=pruned_points.dtype)
    
    #remaining selected points
    remaining_selected = selected_points[n_subset:]
    remaining_coords = xyz[remaining_selected]  # shape: (n_remaining, 3)
    
    #find available pruned points not in subset
    mask = ~torch.isin(pruned_points, pruned_points_subset)
    available_pruned = pruned_points[mask]
    available_coords = xyz[available_pruned]  # shape: (n_available, 3)
    
    #compute cost matrix.
    cost_matrix = torch.cdist(remaining_coords, available_coords)
    cost_matrix_original = cost_matrix.clone()
    
    #threshold
    if threshold is not None:
        cost_matrix_modified = cost_matrix.clone()
        cost_matrix_modified[cost_matrix_modified > threshold] = 1e10  # a large number to avoid assignment
    else:
        cost_matrix_modified = cost_matrix
    
    #convert cost matrix to a NumPy for scipy.
    cost_np = cost_matrix_modified.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)
    
    #only include assignments if the original cost is within the threshold
    for i, j in zip(row_ind, col_ind):
        if threshold is not None and cost_matrix_original[i, j] > threshold:
            continue  # do not add an assignment for this selected point
        else:
            assigned_list.append(available_pruned[j].item())
    
    return torch.tensor(assigned_list, device=device, dtype=pruned_points.dtype)

def compute_new_values(refs, frame_idx, mask):
    """
    computes new reference values for the specified frame and mask by finding the latest
    previous frame where the reference was 0 for each masked position.
    """
    new_values = torch.zeros_like(refs[frame_idx])
    mask_indices = torch.nonzero(mask, as_tuple=True)[0]  #True indices
    
    for j in mask_indices:
        j = j.item()  #integer
        #search from frame_idx-1 down to 0
        for k in range(frame_idx - 1, -1, -1):
            if refs[k][j] == 0:
                new_values[j] = frame_idx - k
                break
        #if no previous frame with 0 is found, make it 0 (current frame)
        else:
            new_values[j] = 0
    return new_values

def swap_random_points(tensor1, tensor2, row_ind, col_ind):
    tensor1[row_ind] = tensor2[col_ind].clone()
    #temporarily disabled
    return tensor1
    #get the number of points in each tensor
    n = tensor1.size(0)
    m = tensor2.size(0)
    k = min(n, m)
    
    if m >= n:
        #tensor2 is longer
        indices = torch.randperm(m)[:k] if new_indices is None else new_indices
        selected = tensor2[indices]
        tensor1 = selected.clone()
    else:
        #tensor1 is longer
        indices = torch.randperm(n)[:k] if new_indices is None else new_indices
        #replace those indices in tensor1 with all points from tensor2
        tensor1[indices] = tensor2.clone()
    
    return tensor1, indices

def swap_closest_points(tensor1, tensor2):
    device = tensor1.device 
    tensor2 = tensor2.to(device)

    n, m = tensor1.size(0), tensor2.size(0)
    k = min(n, m) 

    #compute pairwise distances
    dists = torch.cdist(tensor1, tensor2)  # Shape: (n, m)

    #optimal assignment
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(dists.cpu().numpy())

    #ensure indices are within bounds
    row_ind, col_ind = torch.tensor(row_ind[:k], device=device), torch.tensor(col_ind[:k], device=device)

    tensor1[row_ind] = tensor2[col_ind].clone()

    return tensor1, row_ind, col_ind