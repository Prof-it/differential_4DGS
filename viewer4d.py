#
# Copyright (C) 2025, Felix Hirt
# All rights reserved.
#

import torch
import numpy as np
import os
import json
from plyfile import PlyData, PlyElement
import math
import file_helper
import threading
import traceback

import os
import torch
from gaussian_renderer import render, network_gui, render_interpolated
import sys
from scene import Scene, GaussianModel
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import gaussian_frame
import time
import difference_matte_helper
from PIL import Image
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False



class viewer4d:
    def __init__(self, new_total_frames, new_gaussians, new_source_path, new_frames_per_batch, buffer_size, new_active_sh):
        self.total_frames = new_total_frames
        self.play_gaussians :GaussianModel
        self.source_path = new_source_path
        self.max_sh_degree = new_active_sh
        self.gaussians = new_gaussians
        self.frames : list[gaussian_frame.GaussianFrame]= []
        self.frames_per_batch = new_frames_per_batch
        self.cuda_stream = torch.cuda.Stream()
        self.buffer_size = buffer_size
        self.main_base_frame = None
    

    def start_viewer(self, args, pipe):
        
        self.load_data_to_gaussians() #fills self.frames
        self.play_gaussians = self.fill_gaussian_obj(self.frames)
        self.gaussians.current_reference_counts = [None]*args.buffer_size

            # Preload initial buffer
        for i in range(min(args.buffer_size, self.total_frames)):
            self.move_to_cuda(i, self.gaussians, i, self.play_gaussians)

            # Setup prefetching
        self.prefetch_queue = []
        self.lock = threading.Lock()
        self.stop_prefetch = False
        self.prefetch_thread = threading.Thread(target=self.prefetch_worker)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()
        
        view_frame = 0
        future_frame = 1 * args.skip_frame
        start_time = time.time()
        viewer_available = True

        del self.frames

        current_batch = 0
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        print("Starting viewer")
        print("Frames "+ str(len(self.play_gaussians._xyz)))
        print("Listeing for SIBR Viewer at host: " + str(args.ip) + " port: " + str(args.port))
        print("--> SIBR_remoteGaussian_app.exe --ip " + str(args.ip) + " --port " + str(args.port))
        last_logged = 0

        # Use high-resolution timer
        start_time = time.perf_counter()

        torch.cuda.empty_cache()
        while viewer_available:
            if network_gui.conn is None:
                network_gui.try_connect()

            while network_gui.conn is not None:
                try:
                    # Receive commands
                    custom_cam, do_training, do_scale_points, play_pause, keep_alive, scaling_modifer = network_gui.receive()
                    scaling_modifer = scaling_modifer or 1

                    # Update frame rate or skip value depending on mode
                    if do_training:
                        args.frame_rate = scaling_modifer * args.frame_rate + (1 - scaling_modifer) * 0.1
                    else:
                        args.skip_frame = max(1, math.floor(scaling_modifer * 1 + (1 - scaling_modifer) * 9))

                    # Playback mode
                    if play_pause:
                        if not do_scale_points:
                            interpolation_index = scaling_modifer  # static index
                    else:
                        current_time = time.perf_counter()
                        elapsed = current_time - start_time
                        interpolation_index = elapsed / ((1 / args.frame_rate) * args.skip_frame)

                        if interpolation_index >= 1:
                            # Schedule prefetch for next frame after this advancement
                            next_global = (current_batch * self.buffer_size + future_frame + args.skip_frame) % self.total_frames
                            self.schedule_prefetch(next_global)
                            
                            # Update frame pointers
                            interpolation_index = 0
                            old_view_frame = view_frame
                            view_frame = future_frame
                            real_frame = current_batch * self.buffer_size + view_frame

                            # Set future frame index
                            future_frame += args.skip_frame
                            real__future_frame = current_batch * self.buffer_size + future_frame

                            # Handle batch wrapping
                            if future_frame >= self.buffer_size or real__future_frame >= self.total_frames:
                                if real__future_frame >= self.total_frames:
                                    view_frame = 0
                                    future_frame = 1 + args.skip_frame
                                    additional_batch = future_frame // args.buffer_size
                                    future_frame %= args.buffer_size
                                    current_batch = additional_batch
                                    # Preload first frame of new batch
                                    self.schedule_prefetch(current_batch * self.buffer_size)
                                else:
                                    additional_batch = future_frame // args.buffer_size
                                    future_frame = 1 + future_frame % args.buffer_size
                                    base_frame = old_view_frame if args.skip_frame > 1 else len(self.gaussians._xyz) - 1
                                    current_batch = real__future_frame // self.buffer_size - 1 + additional_batch
                                    # Schedule prefetch for next frame in new batch
                                    next_in_batch = (current_batch * self.buffer_size + future_frame) % self.total_frames
                                    self.schedule_prefetch(next_in_batch)

                            start_time = time.perf_counter()
                            #torch.cuda.empty_cache()

                    if not do_scale_points:
                        scaling_modifer = 1

                    # Render frame if camera exists
                    if custom_cam is not None:
                        if not args.do_interpolate:
                            interpolation_index = 0

                        net_image = render_interpolated(
                            custom_cam,
                            self.gaussians,
                            pipe,
                            background,
                            view_frame + interpolation_index,
                            future_frame,
                            raw_render=True,
                            scaling_modifier=scaling_modifer,
                            use_trained_exp=False,
                            separate_sh=SPARSE_ADAM_AVAILABLE
                        )["render"]

                        net_image_bytes = memoryview(
                            (torch.clamp(net_image, 0, 1.0) * 255)
                            .byte()
                            .permute(1, 2, 0)
                            .contiguous()
                            .cpu()
                            .numpy()
                        ) if net_image is not None else None
                    else:
                        net_image_bytes = None

                    # Send rendered image
                    network_gui.send(
                        net_image_bytes,
                        os.path.join(args.source_path, "..", "..", "..", "point", "colmap_0")
                    )

                    # Efficient logging
                    if time.perf_counter() - last_logged > 1.0:
                        print(f"[INFO] Skip frames: {args.skip_frame}, Frame rate: {args.frame_rate:.2f}")
                        last_logged = time.perf_counter()

                except Exception as e:
                    print(f"[ERROR] {e}")
  
                    print(traceback.format_exc())
                    network_gui.conn = None

            
        # Cleanup when viewer exits
        self.stop_prefetch = True
        self.prefetch_thread.join()

    def schedule_prefetch(self, global_frame):
        """Schedule a frame to be prefetched into the buffer"""
        with self.lock:
            if global_frame not in self.prefetch_queue:
                self.prefetch_queue.append(global_frame)
                # Keep only the most recent requests to avoid backlog
                if len(self.prefetch_queue) > self.buffer_size * 2:
                    self.prefetch_queue = self.prefetch_queue[-self.buffer_size * 2:]

    def prefetch_worker(self):
        """Background thread that prefetches frames into GPU memory"""
        while not self.stop_prefetch:
            if self.prefetch_queue:
                with self.lock:
                    global_frame = self.prefetch_queue.pop(0)
                
                # Calculate buffer slot using modulo arithmetic
                target_slot = global_frame % self.buffer_size
                self.move_to_cuda(target_slot, self.gaussians, global_frame, self.play_gaussians)
            else:
                time.sleep(0.001)  # Short sleep to prevent busy-waiting

    def move_to_cuda(self, target_frame, target, source_frame, source):
        device = torch.device("cuda")
        # Asynchronous transfers with non_blocking=True
        target._xyz[target_frame] = source._xyz[source_frame].to(device, non_blocking=True)
        target._opacity[target_frame] = source._opacity[source_frame].to(device, non_blocking=True)
        target._rotation[target_frame] = source._rotation[source_frame].to(device, non_blocking=True)
        target._features_dc[target_frame] = source._features_dc[source_frame].to(device, non_blocking=True)
        target._features_rest[target_frame] = source._features_rest[source_frame].to(device, non_blocking=True)
        target._scaling[target_frame] = source._scaling[source_frame].to(device, non_blocking=True)
        target.current_reference_counts[target_frame] = source.current_reference_counts[source_frame].to(device, non_blocking=True)

    def reconstruct_source_into_gaussian_obj(self, source_frame, target_frame, source_g:GaussianModel, target_g:GaussianModel):
        target_g.current_reference_counts[target_frame] = source_g.current_reference_counts[source_frame].clone().cpu().pin_memory()
        target_g._xyz[target_frame] = source_g.get_xyz(source_frame).cpu().pin_memory()
        target_g._opacity[target_frame] = source_g.inverse_opacity_activation(source_g.get_opacity(source_frame)).cpu().pin_memory()
        target_g._rotation[target_frame] = source_g.get_rotation_unnormalized(source_frame).cpu().pin_memory()
        target_g._features_dc[target_frame] = source_g.get_features_dc(source_frame).cpu().pin_memory()
        target_g._features_rest[target_frame] = source_g.get_features_rest(source_frame).cpu().pin_memory()
        target_g._scaling[target_frame] = source_g.scaling_inverse_activation(source_g.get_scaling(source_frame)).cpu().pin_memory()
        #target_g._xyz[target_frame] = difference_matte_helper.select_masked_elements_in_tensor(source_frame, source_g.current_reference_counts, source_g._xyz).clone().to("cpu", non_blocking=True).pin_memory()
        #target_g._opacity[target_frame] = difference_matte_helper.select_masked_elements_in_tensor(source_frame, source_g.current_reference_counts, source_g._opacity).clone().to("cpu", non_blocking=True).pin_memory()
        #target_g._rotation[target_frame] = difference_matte_helper.select_masked_elements_in_tensor(source_frame, source_g.current_reference_counts, source_g._rotation).clone().to("cpu", non_blocking=True).pin_memory()
        #target_g._features_dc[target_frame] = difference_matte_helper.select_masked_elements_in_tensor(source_frame, source_g.current_reference_counts, source_g._features_dc).clone().to("cpu", non_blocking=True).pin_memory()
        #target_g._features_rest[target_frame] = difference_matte_helper.select_masked_elements_in_tensor(source_frame, source_g.current_reference_counts, source_g._features_rest).clone().to("cpu", non_blocking=True).pin_memory()
        #target_g._scaling[target_frame] = difference_matte_helper.select_masked_elements_in_tensor(source_frame, source_g.current_reference_counts, source_g._scaling).clone().to("cpu", non_blocking=True).pin_memory()

    def fill_frame_into_gaussian_obj(self, viewframe, gaussian_obj, gaussian_frame, reconstruct = False):
        if(reconstruct):
            gaussian_obj._xyz[viewframe] = difference_matte_helper.add_masked_elements_to_tensor(viewframe, gaussian_frame.reference_counts, gaussian_obj._xyz, gaussian_frame._xyz.clone()).pin_memory()
            gaussian_obj._opacity[viewframe] = difference_matte_helper.add_masked_elements_to_tensor(viewframe, gaussian_frame.reference_counts, gaussian_obj._opacity, gaussian_frame._opacity.clone()).pin_memory()
            gaussian_obj._rotation[viewframe] = difference_matte_helper.add_masked_elements_to_tensor(viewframe, gaussian_frame.reference_counts, gaussian_obj._rotation, gaussian_frame._rotation.clone()).pin_memory()
            gaussian_obj._features_dc[viewframe] = difference_matte_helper.add_masked_elements_to_tensor(viewframe, gaussian_frame.reference_counts, gaussian_obj._features_dc, gaussian_frame._features_dc.clone()).pin_memory()
            gaussian_obj._features_rest[viewframe] = difference_matte_helper.add_masked_elements_to_tensor(viewframe, gaussian_frame.reference_counts, gaussian_obj._features_rest, gaussian_frame._features_rest.clone()).pin_memory()
            gaussian_obj._scaling[viewframe] = difference_matte_helper.add_masked_elements_to_tensor(viewframe, gaussian_frame.reference_counts, gaussian_obj._scaling, gaussian_frame._scaling.clone()).pin_memory()
            gaussian_obj.current_reference_counts[viewframe] = gaussian_frame.reference_counts.clone().pin_memory()
            #gaussian_obj._xyz[viewframe] = difference_matte_helper.select_masked_elements_in_tensor(viewframe, gaussian_obj.current_reference_counts, gaussian_obj._xyz).pin_memory()
            #gaussian_obj._opacity[viewframe] = difference_matte_helper.select_masked_elements_in_tensor(viewframe, gaussian_obj.current_reference_counts, gaussian_obj._opacity).pin_memory()
            #gaussian_obj._rotation[viewframe] = difference_matte_helper.select_masked_elements_in_tensor(viewframe, gaussian_obj.current_reference_counts, gaussian_obj._rotation).pin_memory()
            #gaussian_obj._features_dc[viewframe] = difference_matte_helper.select_masked_elements_in_tensor(viewframe, gaussian_obj.current_reference_counts, gaussian_obj._features_dc).pin_memory()
            #gaussian_obj._features_rest[viewframe] = difference_matte_helper.select_masked_elements_in_tensor(viewframe, gaussian_obj.current_reference_counts, gaussian_obj._features_rest).pin_memory()
            #gaussian_obj._scaling[viewframe] = difference_matte_helper.select_masked_elements_in_tensor(viewframe, gaussian_obj.current_reference_counts, gaussian_obj._scaling).pin_memory()
        else:
            gaussian_obj._xyz[viewframe] = gaussian_frame._xyz.clone().cuda()
            gaussian_obj._opacity[viewframe] = gaussian_frame._opacity.clone().cuda()
            gaussian_obj._rotation[viewframe] =gaussian_frame._rotation.clone().cuda()
            gaussian_obj._features_dc[viewframe] = gaussian_frame._features_dc.clone().cuda()
            gaussian_obj._features_rest[viewframe] = gaussian_frame._features_rest.clone().cuda()
            gaussian_obj._scaling[viewframe] = gaussian_frame._scaling.clone().cuda()
            gaussian_obj.current_reference_counts[viewframe] = gaussian_frame.reference_counts.clone().cuda()

    def fill_gaussian_objOLD(self, frames_array):
        new_gaussian_obj = GaussianModel(3, len(frames_array), 0)
        new_gaussian_obj.current_reference_counts = [None]*len(frames_array)
        for i in range(len(frames_array)):
            if(i==0):
                self.fill_frame_into_gaussian_obj(i, new_gaussian_obj, frames_array[i], False)
            else:
                self.fill_frame_into_gaussian_obj(i, new_gaussian_obj, frames_array[i], True)
        return new_gaussian_obj
    
    def fill_gaussian_obj(self, frames_array):
        new_gaussian_obj = GaussianModel(3, len(frames_array), 0)
        new_gaussian_obj.current_reference_counts = [None]*len(frames_array)
        
        frames_in_batch = 10
        tmp_gaussian_obj = GaussianModel(3, frames_in_batch, 0)
        tmp_gaussian_obj.current_reference_counts = [None]*frames_in_batch
        batch = 0
        frame_counter = 0
        for i in range(len(frames_array)):
            self.fill_frame_into_gaussian_obj(frame_counter, tmp_gaussian_obj, frames_array[i], False)

            if(frame_counter >= frames_in_batch-1 or i == len(frames_array)-1):
                for j in range(1 if batch > 0 else 0,len(tmp_gaussian_obj._xyz)):
                    target_frame = j-(1 if batch>0 else 0)+batch*(frames_in_batch - (1 if batch>0 else 0))
                    if(target_frame >= len(frames_array)-1):
                        break
                    self.reconstruct_source_into_gaussian_obj(j, target_frame, tmp_gaussian_obj, new_gaussian_obj)
                frame_counter=1
                last_frame_index = len(tmp_gaussian_obj._xyz)-1
                self.fill_frame_into_gaussian_obj(0,
                                                   tmp_gaussian_obj,
                                                    gaussian_frame.GaussianFrame(tmp_gaussian_obj.get_xyz(last_frame_index).clone(),
                                                                                tmp_gaussian_obj.get_features_dc(last_frame_index).clone(),
                                                                                tmp_gaussian_obj.get_features_rest(last_frame_index).clone(), 
                                                                                tmp_gaussian_obj.scaling_inverse_activation(tmp_gaussian_obj.get_scaling(last_frame_index)).clone(), 
                                                                                tmp_gaussian_obj.get_rotation_unnormalized(last_frame_index).clone(), 
                                                                                tmp_gaussian_obj.inverse_opacity_activation(tmp_gaussian_obj.get_opacity(last_frame_index)).clone(), 
                                                                                new_reference_counts=frames_array[0].reference_counts.clone())) 
                batch += 1
            else:
                frame_counter +=1
            torch.cuda.empty_cache()

            print(f"\rPreparing Frames: {int((i/len(frames_array))*100)}%", end="")
        
        del tmp_gaussian_obj
        torch.cuda.empty_cache()

        return new_gaussian_obj
    

    def move_to_cudaSTREAM(self, target_frame, target, source_frame, source):
        stream = self.cuda_stream

        with torch.cuda.stream(stream):
            target._xyz[target_frame] = source._xyz[source_frame].clone().to("cuda", non_blocking=True)
            target._opacity[target_frame] = source._opacity[source_frame].clone().to("cuda", non_blocking=True)
            target._rotation[target_frame] = source._rotation[source_frame].clone().to("cuda", non_blocking=True)
            target._features_dc[target_frame] = source._features_dc[source_frame].clone().to("cuda", non_blocking=True)
            target._features_rest[target_frame] = source._features_rest[source_frame].clone().to("cuda", non_blocking=True)
            target._scaling[target_frame] = source._scaling[source_frame].clone().to("cuda", non_blocking=True)
            target.current_reference_counts[target_frame] = source.current_reference_counts[source_frame].clone().to("cuda", non_blocking=True)

        # Optional: wait for stream to finish if dependent work needs these immediately
        torch.cuda.current_stream().wait_stream(stream)

    def move_to_cudaOLD(self, target_frame, target, source_frame, source):
        device = torch.device("cuda")
        target._xyz[target_frame] = source._xyz[source_frame].to(device, non_blocking=True)
        target._opacity[target_frame] = source._opacity[source_frame].to(device, non_blocking=True)
        target._rotation[target_frame] = source._rotation[source_frame].to(device, non_blocking=True)
        target._features_dc[target_frame] = source._features_dc[source_frame].to(device, non_blocking=True)
        target._features_rest[target_frame] = source._features_rest[source_frame].to(device, non_blocking=True)
        target._scaling[target_frame] = source._scaling[source_frame].to(device, non_blocking=True)
        target.current_reference_counts[target_frame] = source.current_reference_counts[source_frame].to(device, non_blocking=True)

    def fill_buffer(self, viewframe, gaussian_frame, reconstruct = True):
        if(reconstruct):
            self.gaussians._xyz[viewframe] = difference_matte_helper.add_masked_elements_to_tensor(viewframe, gaussian_frame.reference_counts, self.gaussians._xyz, gaussian_frame._xyz.clone().cuda())
            self.gaussians._opacity[viewframe] = difference_matte_helper.add_masked_elements_to_tensor(viewframe, gaussian_frame.reference_counts, self.gaussians._opacity, gaussian_frame._opacity.clone().cuda())
            self.gaussians._rotation[viewframe] = difference_matte_helper.add_masked_elements_to_tensor(viewframe, gaussian_frame.reference_counts, self.gaussians._rotation, gaussian_frame._rotation.clone().cuda())
            self.gaussians._features_dc[viewframe] = difference_matte_helper.add_masked_elements_to_tensor(viewframe, gaussian_frame.reference_counts, self.gaussians._features_dc, gaussian_frame._features_dc.clone().cuda())
            self.gaussians._features_rest[viewframe] = difference_matte_helper.add_masked_elements_to_tensor(viewframe, gaussian_frame.reference_counts, self.gaussians._features_rest, gaussian_frame._features_rest.clone().cuda())
            self.gaussians._scaling[viewframe] = difference_matte_helper.add_masked_elements_to_tensor(viewframe, gaussian_frame.reference_counts, self.gaussians._scaling, gaussian_frame._scaling.clone().cuda())
            self.gaussians.current_reference_counts[viewframe] = gaussian_frame.reference_counts.clone().cuda()
        else:
            self.gaussians._xyz[viewframe] = gaussian_frame._xyz.clone().cuda()
            self.gaussians._opacity[viewframe] = gaussian_frame._opacity.clone().cuda()
            self.gaussians._rotation[viewframe] = gaussian_frame._rotation.clone().cuda()
            self.gaussians._features_dc[viewframe] = gaussian_frame._features_dc.clone().cuda()
            self.gaussians._features_rest[viewframe] = gaussian_frame._features_rest.clone().cuda()
            self.gaussians._scaling[viewframe] = gaussian_frame._scaling.clone().cuda()
            self.gaussians.current_reference_counts[viewframe] = gaussian_frame.reference_counts.clone().cuda()

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

        #dequantization
        if(args.quantized):
            quant_scales = {
            'xyz': 1000,
            'f_dc': 1000,
            'f_rest': 1000,
            'opacities': 10000,
            'scale': 1000,
            'rotation': 10000
            }
            for name, arr in [('xyz', xyz), ('f_dc', features_dc), ('f_rest', features_extra),
                    ('opacities', opacities), ('scale', scales), ('rotation', rots)]:
                scale = quant_scales[name]
                if scale:
                    arr = arr.astype(np.float32) / scale

                if name == 'scale':
                    arr = np.clip(arr, -5.0, 1.5) 

                #store back to correct variable
                if name == 'xyz': xyz = arr
                elif name == 'f_dc': features_dc = arr
                elif name == 'f_rest': features_extra = arr
                elif name == 'opacities': opacities = arr
                elif name == 'scale': scales = arr
                elif name == 'rotation': rots = arr

        new_xyz = torch.tensor(xyz, dtype=torch.float, device="cpu")
        new_features_dc = torch.tensor(features_dc, dtype=torch.float, device="cpu").transpose(1, 2).contiguous()
        new_features_rest= torch.tensor(features_extra, dtype=torch.float, device="cpu").transpose(1, 2).contiguous()
        new_opacity=torch.tensor(opacities, dtype=torch.float, device="cpu")
        new_scaling = torch.tensor(scales, dtype=torch.float, device="cpu")
        new_rotation = torch.tensor(rots, dtype=torch.float, device="cpu")

        active_sh_degree = self.max_sh_degree

        return new_xyz, new_opacity, new_scaling, new_rotation, new_features_dc, new_features_rest, active_sh_degree 
    
    def load_data_to_gaussiansOLD(self, frames_per_batch):
        #find how many frames there are
        self.gaussians.current_reference_counts = []

        for i in range(self.total_frames):
            print(f"\rLoading Frame: {i}", end="")
            frame_path = os.path.join(self.source_path, "frame" + str(i) + ".ply")
            frame_references_path = os.path.join(self.source_path, "frame" + str(i) + "_references.move")
            import_pkg = self.load_ply(frame_path)
            self.gaussians._xyz[i] = import_pkg[0].cuda()
            self.gaussians._opacity[i] = import_pkg[1].cuda()
            self.gaussians._scaling[i] = import_pkg[2].cuda()
            self.gaussians._rotation[i] = import_pkg[3].cuda()
            self.gaussians._features_dc[i] = import_pkg[4].cuda()
            self.gaussians._features_rest[i] = import_pkg[5].cuda()
            self.gaussians.active_sh_degree = import_pkg[6]
            new_reference_counts = file_helper.load_tensor_from_file(frame_references_path)
            addition_framecount = math.floor(i / frames_per_batch) * (frames_per_batch - 1)
            if(addition_framecount > 0):
                print("addition framecount " + str(addition_framecount))
                base_frame_index = i - addition_framecount
                print("base frame index " + str(base_frame_index))
                new_reference_counts[new_reference_counts == base_frame_index] += addition_framecount 
                #new_reference_counts = new_reference_counts[:self.gaussians.current_reference_counts[0].shape[0]]
            self.gaussians.current_reference_counts.append(new_reference_counts)
        
        
            #print(self.gaussians._xyz[i].shape)
            #print(self.gaussians.current_reference_counts[i].shape)

    def load_data_to_gaussians(self):
        #find how many frames there are
        i = 0
        while(True):
            print(f"\rLoading Frame: {i}", end="")
            frame_path = os.path.join(self.source_path, "frame" + str(i) + ".ply")
            frame_references_path = os.path.join(self.source_path, "frame" + str(i) + "_references.move")
            try:
                import_pkg = self.load_ply(frame_path)
                new_reference_counts = file_helper.load_tensor_from_file(frame_references_path)
            except Exception as e: 
                #print(repr(e))
                #print("not found")
                #print("SH degree set correctly?")
                self.total_frames = i-1
                break
            new_xyz = import_pkg[0]
            new_opacity = import_pkg[1]
            new_scaling = import_pkg[2]
            new_rotation = import_pkg[3]
            new_features_dc = import_pkg[4]
            new_features_rest = import_pkg[5]
            self.gaussians.active_sh_degree = import_pkg[6]
            self.frames.append(gaussian_frame.GaussianFrame(new_xyz, new_features_dc, new_features_rest, new_scaling, new_rotation, new_opacity, new_reference_counts=new_reference_counts))
            i+=1
    



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--source_path', type=str, required=True)

    args = parser.parse_args(sys.argv[1:])
    
    args.total_frames = 300
    args.max_sh_degree = 3
    args.frames_per_batch = 10
    args.start_frame = 0
    args.end_frame = 300
    args.buffer_size = 10
    args.frame_rate = 30
    args.skip_frame = 1
    args.do_interpolate =  False
    args.quantized = True

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)

    #create viewer

    player = viewer4d(args.total_frames, GaussianModel(args.max_sh_degree, args.buffer_size, 0), args.source_path, args.frames_per_batch, args.buffer_size, args.max_sh_degree)

    player.start_viewer(args, pp.extract(args))
    