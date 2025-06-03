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

import os
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import pointcloud_helper

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = []
        self.test_cameras = []
        self.reference_counts = []
        scene_info = []
        frame_input_paths = []
        for i in range(args.total_frames):
            frame_input_paths.append(os.path.join(args.source_path, args.frame_folder_name+str(args.start_frame + i)))
            

            if os.path.exists(os.path.join(frame_input_paths[i], "sparse")):
                scene_info.append(sceneLoadTypeCallbacks["Colmap"](frame_input_paths[i], args.images, args.depths, args.eval, args.train_test_exp))
            elif os.path.exists(os.path.join(frame_input_paths[i], "transforms_train.json")):
                print("Found transforms_train.json file, assuming Blender data set!")
                #convert this in future
                scene_info = sceneLoadTypeCallbacks["Blender"](frame_input_paths[i], args.white_background, args.depths, args.eval)
            else:
                assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info[0].ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            for i in range(args.total_frames):
                if scene_info[i].test_cameras:
                    camlist.extend(scene_info[i].test_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        #if shuffle:
           #for i in range(args.total_frames):
             #   random.shuffle(scene_info[i].train_cameras)  # Multi-res consistent random shuffling
             #   random.shuffle(scene_info[i].test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info[0].nerf_normalization["radius"]

        for i in range(args.total_frames):        
            print("Loading Training Cameras of Frame " + str(i))
            self.train_cameras.append(cameraList_from_camInfos(scene_info[i].train_cameras, 1.0, args, scene_info[i].is_nerf_synthetic, False))
            #print("Loading Test Cameras of Frame " + str(i))
            
            #self.test_cameras.append(cameraList_from_camInfos(scene_info[i].test_cameras, 1.0, args, scene_info[i].is_nerf_synthetic, True))
        


        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "point_cloud.ply"), args.train_test_exp)
        else:
            #make a list of all the pointcloud objects
            all_frames_pointclouds = []
            for i in range(args.total_frames):
                all_frames_pointclouds.append(scene_info[i].point_cloud)
            print(str(len(all_frames_pointclouds)) + " pointclouds" )
            merged_pointcloud = pointcloud_helper.merge_pointclouds(all_frames_pointclouds, args.model_path, 0.05, 0.08)#0.19, 0.3)
            self.gaussians.create_from_pcd(merged_pointcloud, scene_info[0].train_cameras, self.cameras_extent, 0)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(point_cloud_path)
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, frame, scale=1.0):
        return self.train_cameras[frame]

    def getTestCameras(self, frame, scale=1.0):
        return self.test_cameras[frame]