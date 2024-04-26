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

import torch
import torchvision
import cv2
import os
from os import makedirs
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from scene import Scene
from gaussian_renderer import render
from utils.general_utils import safe_state, vis_depth
from utils.graphics_utils import getWorld2View2
from utils.pose_utils import generate_ellipse_path, generate_spiral_path
from arguments import ModelParams, PipelineParams, RenderParams, get_combined_args
from gaussian_renderer import GaussianModel


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, render_depth):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{}'.format(view.image_name) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{}'.format(view.image_name) + ".png"))

        if render_depth: # Make sure you install the diff-gaussian-rasterization-confidence submodule
            depth_map = vis_depth(rendering['depth'][0].detach().cpu().numpy())
            np.save(os.path.join(render_path, view.image_name + '_depth.npy'), rendering['depth'][0].detach().cpu().numpy())
            cv2.imwrite(os.path.join(render_path, view.image_name + '_depth.png'), depth_map)


def render_video_func(source_path, model_path, iteration, views, gaussians, pipeline, background, classifier, fps=30):
    render_path = os.path.join(model_path, 'video', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = views[0]

    if source_path.find('llff') != -1:
        render_poses = generate_spiral_path(np.load(source_path + '/poses_bounds.npy'))
    else:
        render_poses = generate_ellipse_path(views)

    size = (view.original_image.shape[2], view.original_image.shape[1])
    # For rendering image + segmentation / depth / .etc
    # size = (view.original_image.shape[2] * 2, view.original_image.shape[1])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    final_video = cv2.VideoWriter(os.path.join(render_path, 'final_video.mp4'), fourcc, fps, size)

    for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        rendering = render(view, gaussians, pipeline, background)

        img = torch.clamp(rendering["render"], min=0., max=1.).cpu()
        # To save rendering image
        # torchvision.utils.save_image(img, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        video_img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]
        final_video.write(video_img)

    final_video.release()


def render_sets(dataset : ModelParams, pipeline : PipelineParams, render : RenderParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=render.iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not render.skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render.render_depth)

        if not render.skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,  render.render_depth)

        if render.render_video:
            render_video_func(dataset.source_path, dataset.model_path, scene.loaded_iter, scene.getTrainCameras(),
                         gaussians, pipeline, background, args.fps)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    render = RenderParams(parser)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), render.extract(args))