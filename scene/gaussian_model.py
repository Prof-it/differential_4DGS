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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import math
import difference_matte_helper
import file_helper

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, totalFrames, training_batch, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree 
        self.total_frames = totalFrames
        self.training_batch = training_batch
        self.current_reference_counts = None
        self._xyz = []
        for i in range(round(totalFrames)):#totalFrames)): 
            self._xyz.append(torch.empty(0))
        self._features_dc = []
        for i in range(round(totalFrames)):
            self._features_dc.append(torch.empty(0))
        self._features_rest = []
        for i in range(round(totalFrames)):
            self._features_rest.append(torch.empty(0))
        self._scaling = []
        for i in range(round(totalFrames)):
            self._scaling.append(torch.empty(0))
        self._rotation = []
        for i in range(round(totalFrames)):
            self._rotation.append(torch.empty(0))
        self._opacity = []
        for i in range(totalFrames):
            self._opacity.append(torch.empty(0))
        self.max_radii2D = []
        for i in range(totalFrames):
            self.max_radii2D.append(torch.empty(0))
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = []
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.current_initial_pointcloud = None
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    
    def get_scaling(self, frame, raw_render = False):
        if(raw_render):
            return self.get_scaling_raw(frame)
        multiplier = len(self._scaling)/self.total_frames
        frame = math.floor(frame*multiplier)
        #return self.scaling_activation(self._scaling[frame])
        new_scaling = []
        for i in range(len(self._scaling)):
            new_scaling.append(self.scaling_activation(self._scaling[i]))
        return difference_matte_helper.select_masked_elements_in_tensor(frame, self.current_reference_counts, new_scaling)
    
    def get_rotation(self, frame, raw_render=False):
        if(raw_render):
            return self.get_rotation_raw(frame)
        multiplier = len(self._rotation)/self.total_frames
        frame = math.floor(frame*multiplier)
        new_rotation = []
        for i in range(len(self._rotation)):
            new_rotation.append(self.rotation_activation(self._rotation[i]))
        return difference_matte_helper.select_masked_elements_in_tensor(frame, self.current_reference_counts, new_rotation)
    
    def get_rotation_unnormalized(self, frame, raw_render = False):
        if(raw_render):
            return self._rotation[frame]
        multiplier = len(self._rotation)/self.total_frames
        frame = math.floor(frame*multiplier)
        return difference_matte_helper.select_masked_elements_in_tensor(frame, self.current_reference_counts, self._rotation)
    
    def get_xyz(self, frame, raw_render = False):
        if(raw_render):
            return self.get_xyz_raw(frame)
        multiplier = len(self._xyz)/self.total_frames
        frame = math.floor(frame*multiplier)
        return difference_matte_helper.select_masked_elements_in_tensor(frame,self.current_reference_counts, self._xyz)
    
    def get_features(self, frame, raw_render = False):
        if(raw_render):
            return self.get_features_raw(frame)
        features_dc = self.get_features_dc(frame)
        features_rest = self.get_features_rest(frame)
        return torch.cat((features_dc, features_rest), dim=1)
    
    def get_features_dc(self, frame, raw_render = False):
        if(raw_render):
            return self.get_features_dc_raw(frame)
        multiplier = len(self._features_dc)/self.total_frames
        frame = math.floor(frame*multiplier)
        return difference_matte_helper.select_masked_elements_in_tensor(frame,self.current_reference_counts, self._features_dc)
    
    def get_features_rest(self, frame, raw_render = False):
        if(raw_render):
            return self.get_features_rest_raw(frame)
        multiplier = len(self._features_rest)/self.total_frames
        frame = math.floor(frame*multiplier)
        return difference_matte_helper.select_masked_elements_in_tensor(frame, self.current_reference_counts, self._features_rest)
    
    def get_opacity(self, frame, raw_render = False):
        if(raw_render):
            return self.get_opacity_raw(frame)
        multiplier = len(self._opacity)/self.total_frames
        frame = math.floor(frame*multiplier)
        new_opacity = []
        for i in range(len(self._opacity)):
            new_opacity.append(self.opacity_activation(self._opacity[i]))
        return difference_matte_helper.select_masked_elements_in_tensor(frame, self.current_reference_counts, new_opacity)
    
    #Getter Functions without Background
    def get_scaling_raw(self, frame):
        multiplier = len(self._scaling)/self.total_frames
        frame = math.floor(frame*multiplier)
        #return self.scaling_activation(self._scaling[frame])
        return self.scaling_activation(self._scaling[frame])
    

    def get_rotation_raw(self, frame):
        multiplier = len(self._rotation)/self.total_frames
        frame = math.floor(frame*multiplier)
        #return self.rotation_activation(self._rotation[frame])
        return self.rotation_activation(self._rotation[frame])
    
    
    def get_xyz_raw(self, frame):
        multiplier = len(self._xyz)/self.total_frames
        frame = math.floor(frame*multiplier)
        return self._xyz[frame]
    
    def get_features_raw(self, frame):
        features_dc = self.get_features_dc(frame, True)
        features_rest = self.get_features(frame,True)
        return torch.cat((features_dc, features_rest), dim=1)
    
    def get_features_dc_raw(self, frame):
        multiplier = len(self._features_dc)/self.total_frames
        frame = math.floor(frame*multiplier)
        #return self._features_dc[frame]
        return self._features_dc[frame]
    
    def get_features_rest_raw(self, frame):
        multiplier = len(self._features_rest)/self.total_frames
        frame = math.floor(frame*multiplier)
        #return self._features_rest[frame]
        return self._features_rest[frame]
    
    def get_opacity_raw(self, frame):
        multiplier = len(self._opacity)/self.total_frames
        frame = math.floor(frame*multiplier)
        #return self.opacity_activation(self._opacity[frame])
        return self.opacity_activation(self._opacity[frame])
    


    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, frame, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling(frame), scaling_modifier, self.get_rotation_unnormalized(frame))

    def oneupSHdegree(self):
        
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            print("raising SH degree to " + str(self.active_sh_degree))

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float, frame : int):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        self.current_initial_pointcloud = pcd
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        #xyz    
        self._xyz[0] = fused_point_cloud
        #features_dc
        self._features_dc[0] = features[:,:,0:1].transpose(1, 2).contiguous()
        #features_rest
        self._features_rest[0] =features[:,:,1:].transpose(1, 2).contiguous()
        #scaling
        self._scaling[0] = scales
        #rotation
        self._rotation[0] = rots
        #opacity
        self._opacity[0] = opacities
        self.max_radii2D[0] = torch.zeros((self._xyz[0].shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def create_optimizer(self, training_args, frame):
        l = []
        l.append({'params': [self._xyz[frame]], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": str(frame)+"_"+"xyz"})
        l.append({'params': [self._features_dc[frame]], 'lr': training_args.feature_lr, "name": str(frame)+"_"+"f_dc"})
        l.append({'params': [self._features_rest[frame]], 'lr': training_args.feature_lr / 20.0, "name": str(frame)+"_"+"f_rest"})
        l.append({'params': [self._opacity[frame]], 'lr': training_args.opacity_lr, "name": str(frame)+"_"+"opacity"})
        l.append({'params': [self._scaling[frame]], 'lr': training_args.scaling_lr, "name": str(frame)+"_"+"scaling"})
        l.append({'params': [self._rotation[frame]], 'lr': training_args.rotation_lr, "name": str(frame)+"_"+"rotation"})
        if self.optimizer_type == "default":
            return torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                return SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                print("sparse didnt work")
                return torch.optim.Adam(l, lr=0.0, eps=1e-15)
        return None
    
    def create_optimizerSINGLE(self, training_args):
        l = []
        for frame in range(self.total_frames):
            l.append({'params': [self._xyz[frame]], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": str(frame)+"_"+"xyz"})
            l.append({'params': [self._features_dc[frame]], 'lr': training_args.feature_lr, "name": str(frame)+"_"+"f_dc"})
            l.append({'params': [self._features_rest[frame]], 'lr': training_args.feature_lr / 20.0, "name": str(frame)+"_"+"f_rest"})
            l.append({'params': [self._opacity[frame]], 'lr': training_args.opacity_lr, "name": str(frame)+"_"+"opacity"})
            l.append({'params': [self._scaling[frame]], 'lr': training_args.scaling_lr, "name": str(frame)+"_"+"scaling"})
            l.append({'params': [self._rotation[frame]], 'lr': training_args.rotation_lr, "name": str(frame)+"_"+"rotation"})
        if self.optimizer_type == "default":
            return torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                return SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                return torch.optim.Adam(l, lr=0.0, eps=1e-15)
        return None

    def training_setup(self, training_args, frame_variations, viewpoint_stack, batch = 0, move_pointcloud = False):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self._xyz[0].shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz[0].shape[0], 1), device="cuda")
        base_tensor_xyz = self._xyz[0].clone()
        base_tensor_opacity = self._opacity[0].clone()
        base_tensor_rotation = self._rotation[0].clone()
        base_tensor_scaling = self._scaling[0].clone()
        base_tensor_features_dc =  self._features_dc[0].clone()
        base_tensor_features_rest = self._features_rest[0].clone()
        if(move_pointcloud):
            fused_point_cloud = torch.tensor(np.asarray(self.current_initial_pointcloud.points)).float().cuda()
            pointcloud_references = difference_matte_helper.generate_frame_references(viewpoint_stack, frame_variations, fused_point_cloud, training_args.difference_radius)
            fused_color = RGB2SH(torch.tensor(np.asarray(self.current_initial_pointcloud.colors)).float().cuda())
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0
            dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
            scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, 0] = 1
            opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        for i in range(len(self._xyz)):
            with torch.no_grad():  
                #bring loaded pointcloud info to other parameters

                new_xyz = difference_matte_helper.filter_by_reference(base_tensor_xyz, self.current_reference_counts, i)#.clone()
                new_features_dc = difference_matte_helper.filter_by_reference(base_tensor_features_dc, self.current_reference_counts, i)#.clone()
                new_features_rest = difference_matte_helper.filter_by_reference(base_tensor_features_rest, self.current_reference_counts, i)#.clone()
                new_scaling = difference_matte_helper.filter_by_reference(base_tensor_scaling, self.current_reference_counts, i)#.clone()
                new_rotation = difference_matte_helper.filter_by_reference(base_tensor_rotation, self.current_reference_counts, i)#.clone()
                new_opacity = difference_matte_helper.filter_by_reference(base_tensor_opacity, self.current_reference_counts, i)#.clone()
                if(move_pointcloud and batch>0 and i > 0):
                    #calculate merged pointcloud references
                    selected_pointcloud_points = fused_point_cloud[pointcloud_references[i] == 0]
                    #move random points from new xyz to pointcloud points
                    #eventually add distance checking here
                    new_xyz, row_ind, col_ind = difference_matte_helper.swap_closest_points(new_xyz, selected_pointcloud_points)
                    #features_dc
                    new_features_dc = difference_matte_helper.swap_random_points(new_features_dc, features[:,:,0:1].transpose(1, 2).contiguous()[pointcloud_references[i] == 0],row_ind, col_ind)
                    #features_rest
                    new_features_rest = difference_matte_helper.swap_random_points(new_features_rest, features[:,:,1:].transpose(1, 2).contiguous()[pointcloud_references[i] == 0],row_ind, col_ind)
                    #scaling
                    new_scaling = difference_matte_helper.swap_random_points(new_scaling, scales[pointcloud_references[i]==0],row_ind, col_ind)
                    #rotation
                    new_rotation = difference_matte_helper.swap_random_points(new_rotation, rots[pointcloud_references[i]==0],row_ind, col_ind)
                    #opacity
                    new_opacity = difference_matte_helper.swap_random_points(new_opacity, opacities[pointcloud_references[i]==0],row_ind, col_ind)
                self._xyz[i] =  nn.Parameter(new_xyz.clone().requires_grad_(True))
                self._opacity[i] =  nn.Parameter(new_opacity.clone().requires_grad_(True))
                self._rotation[i] =  nn.Parameter(new_rotation.clone().requires_grad_(True))
                self._scaling[i] =  nn.Parameter(new_scaling.clone().requires_grad_(True))
                self._features_dc[i] =  nn.Parameter(new_features_dc.clone().requires_grad_(True))
                self._features_rest[i] =  nn.Parameter(new_features_rest.clone().requires_grad_(True))
                self.max_radii2D[i] = torch.zeros((base_tensor_xyz.shape[0]), device="cuda")



                self.optimizer.append(self.create_optimizer(training_args, i))

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for i in range(len(self.optimizer)):
            for param_group in self.optimizer[i].param_groups:
                if param_group["name"].endswith("xyz"):
                    lr = self.xyz_scheduler_args(iteration)
                    param_group['lr'] = lr
                    return lr

    def construct_list_of_attributes(self, frame):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc[frame].shape[1]*self._features_dc[frame].shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest[frame].shape[1]*self._features_rest[frame].shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling[frame].shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation[frame].shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def construct_list_of_attributes_arrayversion(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz'] #need to add these
        for j in range(len(self._features_dc)):
            # All channels except the 3 DC
            for i in range(self._features_dc[j].shape[1]*self._features_dc[0].shape[2]):
                l.append('{}_f_dc_{}'.format(j,i))
        for j in range(len(self._features_rest)):
            for i in range(self._features_rest[j].shape[1]*self._features_rest[0].shape[2]):
                l.append('{}_f_rest_{}'.format(j,i))
        for j in range(len(self._opacity)):
            l.append('{}_opacity'.format(j))
        for j in range(len(self._scaling)):
            for i in range(self._scaling[j].shape[1]):
                l.append('{}_scale_{}'.format(j,i))
        for j in range(len(self._rotation)):
            for i in range(self._rotation[j].shape[1]):
                l.append('{}_rot_{}'.format(j,i))
        return l
    
    def quantize_array(self, arr, q):
        return np.clip(np.round(arr * q), -32768, 32767).astype(np.int16)

    def save_ply(self, path, quantize=True):
        mkdir_p(path)
        for frame in range(1 if self.training_batch>0 else 0, self.total_frames):

            ply_save_path = os.path.join(path, "frame" + str(frame + (self.training_batch * (self.total_frames-1))) + ".ply")
            reference_save_path = os.path.join(path, "frame" + str(frame + (self.training_batch * (self.total_frames-1))) + "_references.move")

            xyz = self._xyz[frame].detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc[frame].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest[frame].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities =  self._opacity[frame].detach().cpu().numpy()
            
            scale = self._scaling[frame].detach().cpu().numpy()
            rotation = self._rotation[frame].detach().cpu().numpy()

            dtype_code = 'f4'
            #quantize if desired
            if quantize:
                #precision
                quant_scales = {
                    'xyz': 1000,
                    'f_dc': 1000,
                    'f_rest': 1000,
                    'opacities': 10000,
                    'scale': 1000,
                    'rotation': 10000
                    }
                
                for name, arr in [('xyz', xyz), ('f_dc', f_dc), ('f_rest', f_rest),
                  ('opacities', opacities), ('scale', scale), ('rotation', rotation)]:
                    q = quant_scales[name]
                    if q is not None:
                        arr[:] = self.quantize_array(arr, q)

                dtype_code = 'i2'

            dtype_full = [(attribute, dtype_code) for attribute in self.construct_list_of_attributes(frame)]


            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(ply_save_path)
            #torch.save(self.current_reference_counts[frame], reference_save_path)
            file_helper.save_tensor_to_file(self.current_reference_counts[frame], reference_save_path)

    def save_plyNEWOLD(self, path):
        mkdir_p(os.path.dirname(path))

        #  xyc = []
        # normals = []
        # f_dc = []

        # for i in range(len(self._xyz):

        xyz = self._xyz[0].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = []
        for i in range(len(self._features_dc)):
            f_dc.append(difference_matte_helper.zero_elements_with_reference(i, self.current_reference_counts, self._features_dc)[i].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy())
        f_dc = np.concatenate(f_dc, axis=1)
        f_rest = []
        for i in range(len(self._features_rest)):
            f_rest.append(difference_matte_helper.zero_elements_with_reference(i, self.current_reference_counts, self._features_rest)[i].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy())
        f_rest = np.concatenate(f_rest, axis=1)

        opacities = []
        for i in range(len(self._opacity)):
            opacities.append(difference_matte_helper.zero_elements_with_reference(i, self.current_reference_counts, self._opacity)[i].detach().cpu().numpy())
        opacities = np.concatenate(opacities, axis=1)

        scale = []
        for i in range(len(self._scaling)):
            scale.append(difference_matte_helper.zero_elements_with_reference(i, self.current_reference_counts, self._scaling)[i].detach().cpu().numpy())
        scale = np.concatenate(scale, axis=1)

        rotation = []
        for i in range(len(self._rotation)):
            rotation.append(difference_matte_helper.zero_elements_with_reference(i, self.current_reference_counts, self._rotation)[i].detach().cpu().numpy())
        rotation = np.concatenate(rotation, axis=1)   

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_arrayversion()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        #opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity(frame), torch.ones_like(self.get_opacity(frame))*0.01))
        #optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, str(frame)+"_"+"opacity")
        #self._opacity[frame] = optimizable_tensors[str(frame)+"_"+"opacity"]
        #return
        for i in range(len(self._opacity)):
            opacities_new = torch.min(self._opacity[i], torch.ones_like(self._opacity[i])*0.01)
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, str(i)+"_"+"opacity", i)
            self._opacity[i] = optimizable_tensors[str(i)+"_"+"opacity"]

    def reset_opacity_batch(self):
        for i in range(1, len(self._opacity)):
            opacities_new = torch.min(self._opacity[i], torch.ones_like(self._opacity[i])*0.01)
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, str(i)+"_"+"opacity", i)
            self._opacity[i] = optimizable_tensors[str(i)+"_"+"opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        #opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        for i in range(len(self._xyz)):
            self._xyz[i] = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        for i in range(len(self._features_dc)):
            self._features_dc[i] = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        for i in range(len(self._features_rest)):
            self._features_rest[i] = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        for i in range(len(self._opacity)):
            self._opacity[i] = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        for i in range(len(self._scaling)):
            self._scaling[i] = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        for i in range(len(self._rotation)):
            self._rotation[i] = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name, frame):
        optimizable_tensors = {}
        for group in self.optimizer[frame].param_groups:
            if group["name"] == name:
                stored_state = self.optimizer[frame].state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer[frame].state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer[frame].state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask, frame):
        optimizable_tensors = {}
        for group in self.optimizer[frame].param_groups:
            stored_state = self.optimizer[frame].state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer[frame].state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer[frame].state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, frame):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask, frame)
           
        self._xyz[frame] = optimizable_tensors[str(frame)+"_"+"xyz"]     
        self._features_dc[frame] = optimizable_tensors[str(frame)+"_"+"f_dc"]
        self._features_rest[frame] = optimizable_tensors[str(frame)+"_"+"f_rest"]
        self._opacity[frame] = optimizable_tensors[str(frame)+"_"+"opacity"]
        self._scaling[frame] = optimizable_tensors[str(frame)+"_"+"scaling"]
        self._rotation[frame] = optimizable_tensors[str(frame)+"_"+"rotation"]

        if(frame == 0):
            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
            self.denom = self.denom[valid_points_mask]
            #self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict, frame):
        optimizable_tensors = {}
        
        for group in self.optimizer[frame].param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer[frame].state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer[frame].state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer[frame].state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def relocate_tensors_to_optimizer(self, tensors_dict, frame):
        optimizable_tensors = {}
        
        for group in self.optimizer[frame].param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer[frame].state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer[frame].state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer[frame].state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    
    def fix_references_length(self, reference_tensor = None):
        if(reference_tensor is None):
            reference_tensor = self._xyz[0]
        diff = reference_tensor.shape[0] - self.current_reference_counts[0].shape[0]
        for i in range(len(self.current_reference_counts)):
            if diff < 0:
                # Truncate if too large
                self.current_reference_counts[i] = self.current_reference_counts[i][:reference_tensor.shape[0]]
            elif diff > 0:
                # Pad if too small
                pad = torch.zeros(diff, dtype=self.current_reference_counts[i].dtype, device=self.current_reference_counts[i].device)  # Create padding tensor
                self.current_reference_counts[i] = torch.cat([self.current_reference_counts[i], pad], dim=0)  
    
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_reference_counts, frame):
        d = {}
        d[str(frame)+"_"+"f_dc"] = new_features_dc
        
        d[str(frame)+"_"+"f_rest"] = new_features_rest
    
        d[str(frame)+"_"+"xyz"] = new_xyz
    
        d[str(frame)+"_"+"opacity"] = new_opacities
    
        d[str(frame)+"_"+"scaling"] = new_scaling
    
        d[str(frame)+"_"+"rotation"] = new_rotation

        optimizable_tensors = self.cat_tensors_to_optimizer(d, frame) 

        self._xyz[frame] = optimizable_tensors[str(frame)+"_"+"xyz"]   
        self._features_dc[frame] = optimizable_tensors[str(frame)+"_"+"f_dc"]      
        self._features_rest[frame] = optimizable_tensors[str(frame)+"_"+"f_rest"]      
        self._opacity[frame] = optimizable_tensors[str(frame)+"_"+"opacity"]      
        self._scaling[frame] = optimizable_tensors[str(frame)+"_"+"scaling"]     
        self._rotation[frame] = optimizable_tensors[str(frame)+"_"+"rotation"]

        self.current_reference_counts[frame] = torch.cat((self.current_reference_counts[frame], new_reference_counts))

        self.max_radii2D[frame] = torch.zeros((self.current_reference_counts[frame].shape[0]), device="cuda")

        if(frame==0):
            #self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
            self.xyz_gradient_accum = torch.zeros((self._xyz[0].shape[0], 1), device="cuda")
            self.denom = torch.zeros((self._xyz[0].shape[0], 1), device="cuda")

    def relocation_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_reference_counts, frame):
        d = {}
        d[str(frame)+"_"+"f_dc"] = new_features_dc
        
        d[str(frame)+"_"+"f_rest"] = new_features_rest
    
        d[str(frame)+"_"+"xyz"] = new_xyz
    
        d[str(frame)+"_"+"opacity"] = new_opacities
    
        d[str(frame)+"_"+"scaling"] = new_scaling
    
        d[str(frame)+"_"+"rotation"] = new_rotation

        optimizable_tensors = self.relocate_tensors_to_optimizer(d, frame) 

        self._xyz[frame] = optimizable_tensors[str(frame)+"_"+"xyz"]   
        self._features_dc[frame] = optimizable_tensors[str(frame)+"_"+"f_dc"]      
        self._features_rest[frame] = optimizable_tensors[str(frame)+"_"+"f_rest"]      
        self._opacity[frame] = optimizable_tensors[str(frame)+"_"+"opacity"]      
        self._scaling[frame] = optimizable_tensors[str(frame)+"_"+"scaling"]     
        self._rotation[frame] = optimizable_tensors[str(frame)+"_"+"rotation"]
    

        self.max_radii2D[frame] = torch.zeros((self.current_reference_counts[frame].shape[0]), device="cuda")

        self.current_reference_counts[frame] = new_reference_counts

       
        #self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self._xyz[0].shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz[0].shape[0], 1), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self._xyz[0].shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        scaling_tensor_sum = []
        for i in range(0, len(self._scaling)):
            scaling_tensor_sum.append(self.get_scaling(i))
        scaling_tensor_sum = torch.stack(scaling_tensor_sum).max(dim=0).values
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(scaling_tensor_sum, dim=1).values > self.percent_dense*scene_extent)
        filter_for_reference_counts = None

        for i in range(self.total_frames):
            #!!!!!!!!!!!!!select points mask for this frame!!
            new_selected_pts_mask = difference_matte_helper.filter_by_reference(selected_pts_mask, self.current_reference_counts, i)
            stds = self.scaling_activation(self._scaling[i])[new_selected_pts_mask].repeat(N,1)
            means =torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[i][new_selected_pts_mask]).repeat(N,1,1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[i][new_selected_pts_mask].repeat(N, 1)
            new_scaling=self.scaling_inverse_activation(self.scaling_activation(self._scaling[i][new_selected_pts_mask]).repeat(N,1) / (0.8*N))
            new_rotation=self._rotation[i][new_selected_pts_mask].repeat(N,1)
            new_features_dc=self._features_dc[i][new_selected_pts_mask].repeat(N,1,1)
            new_features_rest=self._features_rest[i][new_selected_pts_mask].repeat(N,1,1)
            new_opacity=self._opacity[i][new_selected_pts_mask].repeat(N,1)

            new_reference_counts = self.current_reference_counts[i][selected_pts_mask].repeat(N) #use original pts mask
            
            #if(i == 0):
            new_tmp_radii = 0 # self.tmp_radii[selected_pts_mask].repeat(N)

            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii, new_reference_counts, i)
            prune_filter = torch.cat((new_selected_pts_mask, torch.zeros(N * new_selected_pts_mask.sum(), device="cuda", dtype=bool)))
            if(i == 0):
                filter_for_reference_counts = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
            self.prune_points(prune_filter, i)
            #prune reference counts
            self.current_reference_counts[i] = self.current_reference_counts[i][~filter_for_reference_counts]
            self.max_radii2D[i] = self.max_radii2D[i][~filter_for_reference_counts]


    def densify_and_clone(self, grads, grad_threshold, scene_extent):
            # Extract points that satisfy the gradient condition
            selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
            scaling_tensor_sum = []
            for i in range(0, len(self._scaling)):
                scaling_tensor_sum.append(self.get_scaling(i))
            scaling_tensor_sum = torch.stack(scaling_tensor_sum).max(dim=0).values
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(scaling_tensor_sum, dim=1).values <= self.percent_dense*scene_extent)

            new_tmp_radii = 0# self.tmp_radii[selected_pts_mask]


            new_reference_counts = []
                        #fix referecen counts
            for i in range(self.total_frames):
                new_reference_counts.append(self.current_reference_counts[i][selected_pts_mask])
            
            for i in range(self.total_frames):
                #!!!!!!!!!! sekec pts mask for this frame
                new_selected_pts_mask = difference_matte_helper.filter_by_reference(selected_pts_mask, self.current_reference_counts, i)
            
                new_xyz=self._xyz[i][new_selected_pts_mask]
                new_features_dc=self._features_dc[i][new_selected_pts_mask]
                new_features_rest=self._features_rest[i][new_selected_pts_mask]         
                new_opacities=self._opacity[i][new_selected_pts_mask]          
                new_scaling=self._scaling[i][new_selected_pts_mask]        
                new_rotation=self._rotation[i][new_selected_pts_mask]

                new_reference_counts = self.current_reference_counts[i][selected_pts_mask] #Use Original pts mask

                self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_reference_counts, i)

    def densify_and_split_relocate(self, grads, grad_threshold, scene_extent, min_opacity, max_screen_size, extent, viewpoint_stack, previous_viewpoint_stack, main_reference_viewpoints, N=2):
        dens_log_string = "Relocating and Splitting: "
        n_init_points = self._xyz[0].shape[0]
       
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        moved_points = []

        for i in range(1, self.total_frames):
           # moved_points.append(torch.zeros(self._xyz[0].shape[0], dtype=torch.bool, device=self._xyz[0].device))

                # Extract points that satisfy the gradient condition
            scaling_tensor = self.get_scaling(i)
            filtered_selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(scaling_tensor, dim=1).values > self.percent_dense*scene_extent)

            prune_mask = (self.get_opacity(i) < min_opacity).squeeze()
            if max_screen_size:
                max_radii2D_tensor_sum = torch.stack(self.max_radii2D).max(dim=0).values
                big_points_vs = max_radii2D_tensor_sum > max_screen_size
                big_points_ws = scaling_tensor.max(dim=1).values > 0.1 * extent
                prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

            if(not prune_mask.any()):
                continue
            
            #get available points
            scaled_prune_mask = difference_matte_helper.filter_by_reference(prune_mask, self.current_reference_counts, i)
            scaled_prune_mask = torch.nonzero(scaled_prune_mask).squeeze()

            #get available points
            #available_points = difference_matte_helper.points_changed_frame_to_frame(main_reference_viewpoints, viewpoint_stack, self.get_xyz(i))
            #available_points = torch.nonzero(available_points).squeeze()
            #print("available points without prunes: " + str(available_points.shape))
            #available_points = torch.unique(torch.cat((available_points, scaled_prune_mask), dim=0))
            #print("available points " + str(available_points.shape)) 
            #available_points = scaled_prune_mask

            #only points in this frame
            filtered_selected_pts_mask = difference_matte_helper.filter_by_reference(selected_pts_mask, self.current_reference_counts, i)

            if(not filtered_selected_pts_mask.any()):
                continue

            selected_points = torch.nonzero(filtered_selected_pts_mask).squeeze()

            if(selected_points.dim() == 0 or selected_points.shape[0] == 0):
                continue

            # Remove duplicates from available points
            #new_available_points = available_points[~torch.isin(available_points, selected_points)]
            #print("Available points " + str(new_available_points.shape[0]))
            prune_points = torch.nonzero(prune_mask).squeeze()
            if(prune_points.dim() == 0 or prune_points.shape[0] == 0):   
                continue

                        #points that changed, but aren't part of this farme anymore should be able to come back
            #changed_points = difference_matte_helper.points_changed_frame_to_frame(viewpoint_stack, main_reference_viewpoints, self.get_xyz(i))

            #changed_points = changed_points[(changed_points == 0) & (self.current_reference_counts[i] != 0)]

            #prune_points = torch.unique(torch.cat((changed_points, prune_points), dim=0))

            #shuffle selected points    
            shuffled_indices = torch.randperm(selected_points.size(0))

            # Apply the shuffled indices to the tensor
            selected_points = selected_points[shuffled_indices]

            #if more selected points than prunes
            if prune_points.size(0) < selected_points.size(0):
                 selected_points = selected_points[:prune_points.size(0)]



            new_available_points = difference_matte_helper.reassign_selected_points(self.get_xyz(i), selected_points, prune_points, scaled_prune_mask, 0.2)
            if(new_available_points.nelement() == 0):
                continue
            # Compute how many points we need to fill
            min_length = min(new_available_points.shape[0], selected_points.shape[0])

            # Trim both tensors to the same length
            #new_available_points = new_available_points[:min_length]
            selected_points = selected_points[:min_length]

            #bring new points into this frame
            new_reference_counts = self.current_reference_counts[i].clone()
            new_reference_counts[new_available_points] = 0

            #create new points mask
            new_points = torch.zeros(self._xyz[0].shape[0], dtype=torch.bool, device=self._xyz[0].device)
            new_points[new_available_points] = True
            #new_points = torch.nonzero(new_points).squeeze()
            new_points = new_points[new_reference_counts == 0]

            # Assign values from selected points to available points
            stds = self.scaling_activation(self._scaling[i])[selected_points]
            means =torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[i][selected_points])
            new_xyz = self.get_xyz(i)[new_reference_counts == 0]
            new_xyz[new_points] = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[i][selected_points]

            new_features_dc = self.get_features_dc(i)[new_reference_counts == 0]
            new_features_dc[new_points] = self._features_dc[i][selected_points]

            new_features_rest = self.get_features_rest(i)[new_reference_counts == 0]
            new_features_rest[new_points] = self._features_rest[i][selected_points]  

            new_opacities = self.inverse_opacity_activation(self.get_opacity(i))[new_reference_counts==0]
            new_opacities[new_points] =  self._opacity[i][selected_points]   

            new_rotation = self.get_rotation_unnormalized(i)[new_reference_counts == 0]
            new_rotation[new_points] = self._rotation[i][selected_points]

            new_scaling = self.scaling_inverse_activation(self.get_scaling(i))[new_reference_counts == 0]
            new_scaling[selected_points] = self.scaling_inverse_activation(self.scaling_activation(self._scaling[i][selected_points]) / (0.8*N))
            new_scaling[new_points] = self.scaling_inverse_activation(self.scaling_activation(self._scaling[i][selected_points]) / (0.8*N))

            self.relocation_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, 0, new_reference_counts, i)

            dens_log_string += str(new_available_points.shape[0]) + " relocated in Frame "+ str(i) + ", "
           # moved_points[i]= new_available_points
        return dens_log_string, moved_points

    def densify_and_clone_relocate(self, grads, grad_threshold, scene_extent, min_opacity, max_screen_size, extent, viewpoint_stack, previous_viewpoint_stack, main_reference_viewpoints):

        dens_log_string = "Relocating and Cloning: "
        moved_points = []
        for i in range(1, self.total_frames):
            #moved_points.append(torch.zeros(self._xyz[0].shape[0], dtype=torch.bool, device=self._xyz[0].device))
            # Extract points that satisfy the gradient condition
            selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
            scaling_tensor = self.get_scaling(i)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(scaling_tensor, dim=1).values <= self.percent_dense*scene_extent)
            
            prune_mask = (self.get_opacity(i) < min_opacity).squeeze()
            if max_screen_size:
                max_radii2D_tensor_sum = torch.stack(self.max_radii2D).max(dim=0).values
                big_points_vs = max_radii2D_tensor_sum > max_screen_size
                big_points_ws = scaling_tensor.max(dim=1).values > 0.1 * extent
                prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

            if(not prune_mask.any()):
                continue

            #get available points
            scaled_prune_mask = difference_matte_helper.filter_by_reference(prune_mask, self.current_reference_counts, i)
            scaled_prune_mask = torch.nonzero(scaled_prune_mask).squeeze()

            #get available points
            #available_points = difference_matte_helper.points_changed_frame_to_frame(main_reference_viewpoints, viewpoint_stack, self.get_xyz(i))
            #available_points = torch.nonzero(available_points).squeeze()
            #print("available points without prunes: " + str(available_points.shape))
            #available_points = torch.unique(torch.cat((available_points, scaled_prune_mask), dim=0))
            #print("available points " + str(available_points.shape)) 
            #available_points = scaled_prune_mask

            #only points in this frame
            filtered_selected_pts_mask = difference_matte_helper.filter_by_reference(selected_pts_mask, self.current_reference_counts, i)

            if(not filtered_selected_pts_mask.any()):
                continue

            selected_points = torch.nonzero(filtered_selected_pts_mask).squeeze()

            if(selected_points.dim() == 0 or selected_points.shape[0] == 0):
                continue

            # Remove duplicates from available points
            #new_available_points = available_points[~torch.isin(available_points, selected_points)]
            #print("Available points " + str(new_available_points.shape[0]))
            prune_points = torch.nonzero(prune_mask).squeeze()
            if(prune_points.dim() == 0 or prune_points.shape[0] == 0):    
                continue

                        #points that changed, but aren't part of this farme anymore should be able to come back
            #changed_points = difference_matte_helper.points_changed_frame_to_frame(viewpoint_stack, main_reference_viewpoints, self.get_xyz(i))

            #changed_points = changed_points[(changed_points == 0) & (self.current_reference_counts[i] != 0)]

            #prune_points = torch.unique(torch.cat((changed_points, prune_points), dim=0))

            #shuffle selected points    
            shuffled_indices = torch.randperm(selected_points.size(0))

            # Apply the shuffled indices to the tensor
            selected_points = selected_points[shuffled_indices]

            #if more selected points than prunes
            if prune_points.size(0) < selected_points.size(0):
                 selected_points = selected_points[:prune_points.size(0)]

            new_available_points = difference_matte_helper.reassign_selected_points(self.get_xyz(i), selected_points, prune_points, scaled_prune_mask, 0.2)
            if(new_available_points.nelement() == 0):
                continue

            # Compute how many points we need to fill
            min_length = min(new_available_points.shape[0], selected_points.shape[0])

            # Trim both tensors to the same length
            #new_available_points = new_available_points[:min_length]
            selected_points = selected_points[:min_length]

            #bring new points into this frame
            new_reference_counts = self.current_reference_counts[i].clone()
            new_reference_counts[new_available_points] = 0

            #create new points mask
            new_points = torch.zeros(self._xyz[0].shape[0], dtype=torch.bool, device=self._xyz[0].device)
            new_points[new_available_points] = True
            #new_points = torch.nonzero(new_points).squeeze()
            new_points = new_points[new_reference_counts == 0]

                    # Assign values from selected points to available points
            new_xyz = self.get_xyz(i)[new_reference_counts == 0]
            new_xyz[new_points] = self._xyz[i][selected_points]

            new_features_dc = self.get_features_dc(i)[new_reference_counts == 0]
            new_features_dc[new_points] = self._features_dc[i][selected_points]

            new_features_rest = self.get_features_rest(i)[new_reference_counts == 0]
            new_features_rest[new_points] = self._features_rest[i][selected_points]  

            new_opacities = self.inverse_opacity_activation(self.get_opacity(i))[new_reference_counts==0]
            new_opacities[new_points] =  self._opacity[i][selected_points]   

            new_rotation = self.get_rotation_unnormalized(i)[new_reference_counts == 0]
            new_rotation[new_points] = self._rotation[i][selected_points]

            new_scaling = self.scaling_inverse_activation(self.get_scaling(i))[new_reference_counts == 0]
            new_scaling[new_points] = self._scaling[i][selected_points]

            self.relocation_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, 0, new_reference_counts, i)

            dens_log_string += (str(new_available_points.shape[0]) + " relocated in Frame " +str(i) + ", ")
            #moved_points[i]= new_available_points
        return dens_log_string, moved_points


    def densify_and_prune_relocate(self, max_grad, min_opacity, extent, max_screen_size, radii, viewpoint_stack, previous_viewpoint_stack, frame_variations, frame, main_reference_viewpoints):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        log_string, moved_points = self.densify_and_clone_relocate(grads, max_grad, extent, min_opacity, max_screen_size, extent, viewpoint_stack, previous_viewpoint_stack, main_reference_viewpoints)
        print(log_string)
        log_string, moved_points_split = self.densify_and_split_relocate(grads, max_grad, extent, min_opacity, max_screen_size, extent, viewpoint_stack, previous_viewpoint_stack, main_reference_viewpoints)

        merged_moved_points = [torch.unique(torch.cat((t1, t2), dim=0)) for t1, t2 in zip(moved_points, moved_points_split)]
        prune_log_string = "Unlinked "
        for i in range(1, self.total_frames): #exclude first frame as it's the reference frame
                    #instead of pruning make them invisible and available for the future
            prune_mask = (self.get_opacity(i) < min_opacity).squeeze()
            if max_screen_size:
                max_radii2D_tensor_sum = torch.stack(self.max_radii2D).max(dim=0).values
                big_points_vs = max_radii2D_tensor_sum > max_screen_size
                big_points_ws = self.get_scaling(i).max(dim=1).values > 0.1 * extent
                prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            if(not prune_mask.any()):
                continue
            if(prune_mask.dim() == 0):
                continue
            scaled_prune_mask = difference_matte_helper.filter_by_reference(prune_mask, self.current_reference_counts, i)
            if(not scaled_prune_mask.any()):
                continue
            if(scaled_prune_mask.dim() == 0):
                continue
            #self.prune_points(scaled_prune_mask, i)
            prune_mask[self.current_reference_counts[i] != 0] = False
            #self.current_reference_counts = difference_matte_helper.update_frame_reference(self.current_reference_counts, i, prune_mask)
            prune_log_string += str(torch.sum(scaled_prune_mask).item()) + " from Frame " + str(i) + ", "
        print(prune_log_string)
            #self.current_reference_counts[i][prune_mask] = self.total_frames-i-1
            #new_opacity = self._opacity[i].clone()
            #new_opacity[scaled_prune_mask] = self.inverse_opacity_activation(torch.ones_like(new_opacity[scaled_prune_mask]*0.004))
           # self._opacity[i] = self.replace_tensor_to_optimizer(new_opacity, str(i)+"_opacity", i)[str(i)+"_opacity"]
            #new_scaling = self._scaling[i].clone()
            #new_scaling[scaled_prune_mask] = self.scaling_inverse_activation(torch.ones_like(new_scaling[scaled_prune_mask]*0.004))
            #self._scaling[i] = self.replace_tensor_to_optimizer(new_scaling, str(i)+"_scaling", i)[str(i)+"_scaling"]

        #tmp_radii = self.tmp_radii
        #self.tmp_radii = None

                #remap new points to 2d space 
        #self.current_reference_counts = difference_matte_helper.generate_frame_references(viewpoint_stack, frame_variations, self._xyz[0])
        return merged_moved_points


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii, viewpoint_stack, frame_variations):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        #self.tmp_radii = radii

        
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        opacity_tensor_sum = []
        for i in range(0, len(self._scaling)):
            opacity_tensor_sum.append(self.get_opacity(i))
        opacity_tensor_sum = torch.stack(opacity_tensor_sum).min(dim=0).values
        prune_mask = (opacity_tensor_sum < min_opacity).squeeze()
        if max_screen_size:
            max_radii2D_tensor_sum = torch.stack(self.max_radii2D).max(dim=0).values
            big_points_vs = max_radii2D_tensor_sum > max_screen_size
            scaling_tensor_sum = []
            for i in range(0, len(self._scaling)):
                scaling_tensor_sum.append(self.get_scaling(i))
            scaling_tensor_sum = torch.stack(scaling_tensor_sum).max(dim=0).values
            big_points_ws = scaling_tensor_sum.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        filter_for_reference_counts = prune_mask
        for i in range(self.total_frames):
            if(prune_mask.dim() == 0):
                continue
            scaled_prune_mask = difference_matte_helper.filter_by_reference(prune_mask, self.current_reference_counts, i)
            self.prune_points(scaled_prune_mask, i)
            self.current_reference_counts[i] = self.current_reference_counts[i][~filter_for_reference_counts]
            self.max_radii2D[i] = self.max_radii2D[i][~filter_for_reference_counts]
        #tmp_radii = self.tmp_radii
        #self.tmp_radii = None

        
                #remap new points to 2d space 
        #self.current_reference_counts = difference_matte_helper.generate_frame_references(viewpoint_stack, frame_variations, self._xyz[0])
        #for i in range(self.total_frames):
        #    self.xyz[i] = difference_matte_helper.filter_by_reference(self.xyz)

        #difference_matte_helper.propagate_current_frame(i, self.current_reference_counts, self._xyz)
                #propagate
        #for i in range(self.total_frames):
            
        #    difference_matte_helper.propagate_current_frame(i, self.current_reference_counts, self._opacity)                
        #    difference_matte_helper.propagate_current_frame(i, self.current_reference_counts, self._scaling)
        #    difference_matte_helper.propagate_current_frame(i, self.current_reference_counts, self._rotation)
        #    difference_matte_helper.propagate_current_frame(i, self.current_reference_counts, self._features_dc)
        #    difference_matte_helper.propagate_current_frame(i, self.current_reference_counts, self._features_rest)
        


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


