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

import os
import torch
import yaml
import torchvision.transforms as T
from torch import nn
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
import pprint
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import difference_matte_helper
import temporal_stability_helper
import gaussian_frame
from utils.general_utils import PILtoTorch
import gc
import time
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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):


    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    #class variable main reference frame include tensors and image
    #class variable last frame of previous batch include tensors and image

    batch = 0
    main_reference_viewpoints = None
    last_previous_gaussian_frame : gaussian_frame.GaussianFrame = None
    spare_gaussians = torch.empty(args.spare_gaussians)

    while(True):
        dataset.start_frame = args.start_frame + (batch * (args.total_frames - 1))

        if(dataset.start_frame>=args.end_frame):
            break
        if(args.end_frame - dataset.start_frame < args.total_frames-1):
            args.total_frames = args.end_frame - dataset.start_frame 

        first_iter = 0
        tb_writer = prepare_output_and_logger(dataset, batch)
        gaussians = GaussianModel(dataset.sh_degree, dataset.total_frames,batch, opt.optimizer_type)
        scene = Scene(dataset, gaussians)

        use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
        depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

        viewpoint_stack = []
        viewpoint_indices = []
        for i in range(args.total_frames):
            viewpoint_stack.append(scene.getTrainCameras(i).copy())
            viewpoint_indices.append(list(range(len(viewpoint_stack[i]))))

        if(batch == 0):
            main_reference_viewpoints = scene.getTrainCameras(0).copy()

        if(batch > 0):
        #copy previous last frame to this current first one
            #gaussians._xyz[0] = nn.Parameter(last_previous_gaussian_frame._xyz.requires_grad_(True))
            #gaussians._features_dc[0] = nn.Parameter(last_previous_gaussian_frame._features_dc.requires_grad_(True))
            #gaussians._features_rest[0] = nn.Parameter(last_previous_gaussian_frame._features_rest.requires_grad_(True))
            #gaussians._scaling[0] = nn.Parameter(last_previous_gaussian_frame._scaling.requires_grad_(True))
            #gaussians._rotation[0] = nn.Parameter(last_previous_gaussian_frame._rotation.requires_grad_(True))
            #gaussians._opacity[0] = nn.Parameter(last_previous_gaussian_frame._opacity.requires_grad_(True))
            #gaussians.max_radii2D[0] = torch.zeros((gaussians._xyz[0].shape[0]), device="cuda")
            gaussians._xyz[0] = last_previous_gaussian_frame._xyz.clone().requires_grad_(True)
            gaussians._features_dc[0] = last_previous_gaussian_frame._features_dc.clone().requires_grad_(True)
            gaussians._features_rest[0] = last_previous_gaussian_frame._features_rest.clone().requires_grad_(True)
            gaussians._scaling[0] = last_previous_gaussian_frame._scaling.clone().requires_grad_(True)
            gaussians._rotation[0] = last_previous_gaussian_frame._rotation.clone().requires_grad_(True)
            gaussians._opacity[0] = last_previous_gaussian_frame._opacity.clone().requires_grad_(True)
            gaussians.max_radii2D[0] = torch.zeros((gaussians._xyz[0].shape[0]), device="cuda")
            gaussians.active_sh_degree = dataset.sh_degree
        

            last_previous_gaussian_frame = None

        #reference counts. possibly move this to scene class
        frame_variations = difference_matte_helper.calculate_frame_variations(viewpoint_stack, args.difference_threshold, args.difference_dilation)
        #map 3d points to 2d space for the first time, later do it right after densification
        if(gaussians.current_reference_counts is None):
            gaussians.current_reference_counts = difference_matte_helper.generate_frame_references(scene.train_cameras.copy(), frame_variations, gaussians._xyz[0], args.difference_radius)
        print("reference Counts in frames:")
        for i in range(len(gaussians.current_reference_counts)):
            print(gaussians.current_reference_counts[i][gaussians.current_reference_counts[i] == 0].shape)
        
        gaussians.training_setup(opt, frame_variations, viewpoint_stack, batch=batch, move_pointcloud=args.move_pointcloud)

        print("points in frames:")
        for i in range(len(gaussians._xyz)):
            print(gaussians._xyz[i].shape)

        #if(batch>0):
            #gaussians.xyz_gradient_accum = last_previous_gaussian_frame.xyz_grad_accum
            #gaussians.denom = last_previous_gaussian_frame.denom
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)

        print("Starting to process Batch "+str(batch))

        ema_loss_for_log = 0.0
        ema_Ll1depth_for_log = 0.0

        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", disable=False)
        first_iter += 1

        frame_counter = 0
        view_frame = 0
        frame = 1 if batch>0 else 0
        start_time = time.time()
        used_cam = None
        moved_points = None

        gc.collect()
        torch.cuda.empty_cache()
        
        for iteration in range(first_iter, opt.iterations + 1):
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, do_scale_points, play_pause, keep_alive, scaling_modifer = network_gui.receive()
                    
                    if(scaling_modifer is None):
                        scaling_modifer = 1

                    if(play_pause):
                        if(not do_scale_points):
                            view_frame = round(scaling_modifer * args.total_frames) - 1
                        if(view_frame < 0):
                            view_frame = 0
                    else:
                        if(time.time() - start_time) > (1/30):
                            view_frame +=1
                            if(view_frame >= args.total_frames):
                                view_frame = 0
                            start_time = time.time()
                    if(not do_scale_points):
                        scaling_modifer = 1
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, view_frame, raw_render=(not keep_alive), scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, os.path.join(dataset.source_path, args.frame_folder_name+str((args.start_frame))))
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    print(e)
                    network_gui.conn = None

            iter_start.record()

            gaussians.update_learning_rate(iteration)
            
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000*(args.total_frames/2) == 0:
                gaussians.oneupSHdegree()

            #go through all frames within 1 densification interval
            limit = 1#args.densification_interval/args.total_frames
            if(frame_counter>= limit):
                frame += 1
                if(frame > args.total_frames-1):
                    if(batch>0): ##only first batch trains the first frame (main reference frame)
                        frame = 1
                    else:
                        frame=0
                frame_counter = 0
            frame_counter += 1

            # Pick a random Camera
            if not viewpoint_stack[frame]:
                viewpoint_stack[frame] = scene.getTrainCameras(frame).copy()
                viewpoint_indices[frame] = list(range(len(viewpoint_stack[frame])))
            rand_idx = randint(0, len(viewpoint_indices[frame]) - 1)
            viewpoint_cam = viewpoint_stack[frame].pop(rand_idx)
            vind = viewpoint_indices[frame].pop(rand_idx)

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, frame, raw_render=False, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                image *= alpha_mask


            # Loss
            gt_image = viewpoint_cam.original_image.cuda()

            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                print("no FUSEDSSIM!!")
                ssim_value = ssim(image, gt_image)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            # Depth regularization
            Ll1depth_pure = 0.0
            if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                invDepth = render_pkg["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()

                Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
                loss += Ll1depth
                Ll1depth = Ll1depth.item()
            else:
                Ll1depth = 0


            #temporal stability loss
            window_size = args.temporal_frame_window  #number of frames to consider for loss      
            temporal_loss = 0.0  
            spatial_loss = 0.0  
            if frame >= window_size:
                for t in range(1, window_size):
                   
                    mask1, mask2 = difference_matte_helper.get_shared_masks(gaussians.current_reference_counts, frame, frame-t)
                    xyz_f, xyz_t = gaussians._xyz[frame][mask1], gaussians._xyz[frame-t][mask2]
                    relocation_mask = temporal_stability_helper.get_relocation_mask(xyz_f, xyz_t, 1.0)

                    temporal_loss += temporal_stability_helper.temporal_smoothness_loss(gaussians.rotation_activation(gaussians._rotation[frame][mask1]),gaussians.scaling_activation(gaussians._scaling[frame][mask1]), gaussians.opacity_activation(gaussians._opacity[frame][mask1]), gaussians._features_dc[frame][mask1], gaussians._features_rest[frame][mask1],
                                                                                        gaussians.rotation_activation(gaussians._rotation[frame-t][mask2]),gaussians.scaling_activation(gaussians._scaling[frame-t][mask2]), gaussians.opacity_activation(gaussians._opacity[frame-t][mask2]), gaussians._features_dc[frame-t][mask2], gaussians._features_rest[frame-t][mask2], relocation_mask)
                    spatial_loss += temporal_stability_helper.spatial_coherence_loss(xyz_f, xyz_t, mask=relocation_mask)
                temporal_loss /= (window_size-1)
                spatial_loss /= (window_size-1)
                #total Loss
                loss = torch.add(loss, args.lambda_temporal_smoothing * temporal_loss)
                loss = torch.add(loss, args.lambda_pos * spatial_loss)

            loss.backward()

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, spatial_loss, temporal_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, frame, False, 1.0, SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, frame)
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
                    #continue
                
                reference_counts_copy = None
                # Densification
                if iteration < opt.densify_until_iter and iteration not in saving_iterations:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[frame][visibility_filter] = torch.max(gaussians.max_radii2D[frame][visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        #if(batch > 0):
                        #    difference = difference_matte_helper.binary_difference_matte(gt_image, image, 0.1)
                        #    difference_matte_helper.visualize_binary_matte(difference)
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        size_threshold = None
                        reference_counts_copy = gaussians.current_reference_counts[frame].clone()
                        if(batch>0): #only first batch densifies and prunes, after points only get relocated
                            new_moved_points = gaussians.densify_and_prune_relocate(opt.densify_grad_threshold, opt.min_opacity, scene.cameras_extent, size_threshold, radii, scene.getTrainCameras(frame).copy(),scene.getTrainCameras(frame-1).copy(), frame_variations, frame, main_reference_viewpoints)
                            #if(moved_points is None):
                             #   moved_points =  new_moved_points
                           # else:
                             #   moved_points = [torch.unique(torch.cat((t1, t2), dim=0)) for t1, t2 in zip(moved_points, new_moved_points)]
                        else:
                            gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, scene.cameras_extent, size_threshold, radii, scene.train_cameras, frame_variations)
                        print("\n" + str(gaussians._xyz[0].shape[0]) + " Gaussians")
                        l = []
                        l.append(gaussians._xyz[frame])
                        l.append(gaussians._features_dc[frame])
                        l.append(gaussians._features_rest[frame])
                        l.append(gaussians._opacity[frame])
                        l.append(gaussians._scaling[frame])
                        l.append(gaussians._rotation[frame])
                        param_counter = 0
                        for param in l:
                            if param.grad is not None:
                                if(param.grad.norm() > 1):
                                    print("param counter" + str(param_counter))
                                    print(param.grad.norm())  # Ensure values are reasonable
                            param_counter +=1
                        torch.nn.utils.clip_grad_norm_(gaussians._features_dc[frame], max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(gaussians._opacity[frame], max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(gaussians._scaling[frame], max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(gaussians._features_rest[frame], max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(gaussians._xyz[frame], max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(gaussians._rotation[frame], max_norm=1.0)

                        #gc.collect()
                        #torch.cuda.empty_cache()



                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        if(batch>0):
                            #print("no opacity reset")
                            gaussians.reset_opacity_batch()
                        else:
                            #print("no opacity reset")
                            gaussians.reset_opacity()
                    

                # Optimizer step
                if iteration < opt.iterations:
                    #gaussians.exposure_optimizer.step()
                    #gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                    if use_sparse_adam:
                        # adjust radii to frame mask. Use old reference counts when doing densification
                        radii_mask = (gaussians.current_reference_counts[frame] if reference_counts_copy is None else reference_counts_copy) == 0
                        radii = radii[radii_mask]
                        visible = radii > 0
                        gaussians.optimizer[frame].step(visible, radii.shape[0])
                        gaussians.optimizer[frame].zero_grad(set_to_none = True)
                    else:
                        gaussians.optimizer[frame].step()
                        gaussians.optimizer[frame].zero_grad(set_to_none = True)



                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

                #del gt_image
                #del render_pkg
                #del image
                #del viewspace_point_tensor 
                #del visibility_filter
                
        del last_previous_gaussian_frame
        last_previous_gaussian_frame = gaussian_frame.GaussianFrame(gaussians.get_xyz(args.total_frames-1).detach().clone(),gaussians.get_features_dc(args.total_frames-1).detach().clone(),gaussians.get_features_rest(args.total_frames-1).detach().clone(),gaussians.scaling_inverse_activation(gaussians.get_scaling(args.total_frames-1)).detach().clone(),gaussians.get_rotation_unnormalized(args.total_frames-1).detach().clone(),gaussians.inverse_opacity_activation(gaussians.get_opacity(args.total_frames-1)).detach().clone(), gaussians.max_radii2D[args.total_frames-1].detach().clone(), args.total_frames)
        for optimizer in gaussians.optimizer:
            del optimizer
        del gaussians.exposure_optimizer
        del gaussians
        del scene
        del loss
        del iter_start
        del iter_end
        gc.collect()
        torch.cuda.empty_cache()
        batch += 1



def prepare_output_and_logger(args, current_batch):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_log_path = os.path.join(args.model_path, "tb_logs")
        os.makedirs(tb_log_path, exist_ok = True)
        amount_of_runs = sum(os.path.isdir(os.path.join(tb_log_path, entry)) for entry in os.listdir(tb_log_path))
        tb_log_path = os.path.join(tb_log_path, "run_"+str(amount_of_runs+1)+"_batch_"+str(current_batch))
        os.makedirs(tb_log_path, exist_ok = True)
        tb_writer = SummaryWriter(tb_log_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, spatial_loss, temporal_smoothing_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, frame):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/spatial_loss', spatial_loss, iteration)
        tb_writer.add_scalar('train_loss_patches/temporal_smoothing_loss', temporal_smoothing_loss, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTrainCameras(0)}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras(0)[idx % len(scene.getTrainCameras(0))] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity(frame), iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz(frame).shape[0], iteration)
        torch.cuda.empty_cache()

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        print("No 4D config file found. Using Default settings.")
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "default_config.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[200, 2000, 15000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    
    config_path = os.path.join(args.source_path, "config.yaml")
    #load config file
    config = load_config(config_path)

    #add additional cofig parameters
    for key, value in config.items():
        setattr(args, key, value)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    #torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
