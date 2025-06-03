#
# Copyright (C) 2025, Felix Hirt
# All rights reserved.
#

from typing import List, NamedTuple
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
import os
import torch

class BasicPointCloud(NamedTuple):
    points: np.ndarray 
    colors: np.ndarray 
    normals: np.ndarray  

def merge_pointclouds(pointclouds, model_path, merge_radius = 0.1, prune_radius = 0.04):
    #return first pointcloud for now
    #return pointclouds[0]
    """
    Merge multiple point clouds, removing redundant points and pruning isolated points.

    """
    #cmbine points from all point clouds
    all_points = np.vstack([pc.points for pc in pointclouds])
    all_colors = np.vstack([pc.colors for pc in pointclouds])
    all_normals = np.vstack([pc.normals for pc in pointclouds])
    
    #use KD-Tree for efficient nearest neighbor search
    tree = cKDTree(all_points)
    
    #find indices of points to keep and their colors
    unique_indices = []
    unique_colors = []
    used_indices = set()
    
    for i in range(len(all_points)):
        if i in used_indices:
            continue
        
        # find points within merge radius
        nearby_indices = tree.query_ball_point(all_points[i], merge_radius)
        
        #average colors for points in the cluster
        cluster_colors = all_colors[nearby_indices]
        avg_color = np.mean(cluster_colors, axis=0)
        
        #keep the first point in the cluster
        unique_indices.append(i)
        unique_colors.append(avg_color)
        
        #mark all other points in this cluster as used
        for idx in nearby_indices:
            used_indices.add(idx)
    
    #create point cloud with merged points
    merged_points = all_points[unique_indices]
    merged_colors = np.array(unique_colors)
    merged_normals = all_normals[unique_indices]
    
    #prune isolated points
    tree_merged = cKDTree(merged_points)
    isolated_indices = []
    
    for i in range(len(merged_points)):
        #check if point has any neighbors within prune radius
        nearby_points = tree_merged.query_ball_point(merged_points[i], prune_radius)
        if len(nearby_points) <= 1:  # Point itself is in the list
            isolated_indices.append(i)
    
    #remove isolated points
    final_points = np.delete(merged_points, isolated_indices, axis=0)
    final_colors = np.delete(merged_colors, isolated_indices, axis=0)
    final_normals = np.delete(merged_normals, isolated_indices, axis=0)
    newBPC = BasicPointCloud(
        points=final_points,
        colors=final_colors,
        normals=final_normals
    )
    print(str(newBPC.points.shape[0]) + " Points")
    
    save_path = os.path.join(model_path, "input.ply")
    if(os.path.isfile(save_path)):
        os.remove(save_path)
    save_pointcloud_to_ply(newBPC, os.path.join(model_path, "input.ply"))
    #create merged point cloud with unique points
    return newBPC

def save_pointcloud_to_ply(pointcloud, filename):
    """
    Saves a BasicPointCloud object to a PLY file.
    """
    #ensure colors are in the range [0, 255] and uint8
    colors = np.clip(pointcloud.colors, 0, 255).astype(np.uint8)

    points_with_data = np.hstack((
        pointcloud.points,
        colors,
        pointcloud.normals
    ))

    vertex = np.array(
        [tuple(row) for row in points_with_data],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
               ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
    )

    ply_element = PlyElement.describe(vertex, 'vertex')

   
    PlyData([ply_element]).write(filename)