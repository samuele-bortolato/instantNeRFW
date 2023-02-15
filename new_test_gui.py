# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import os
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from wisp.app_utils import default_log_setup, args_to_log_format
from wisp.framework import WispState
from wisp.datasets import MultiviewDataset
from wisp.datasets.transforms import SampleRays
from wisp.trainers import MultiviewTrainer
from wisp.models.grids import OctreeGrid, HashGrid
from wisp.models.pipeline import Pipeline
from wisp.tracers import PackedRFTracer
from wisp.models.nefs import NeuralRadianceField

from extensions.dataset import MyMultiviewDataset
from extensions.trainer import Trainer
from extensions.tracer import Tracer
from extensions.nef import Nef
#from extensions.gui import DemoApp

from utils.parser import parse_args
from utils.default_params import *

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", help="Configuration file path.")
args = parser.parse_args()

if args.config_path is not None:
    try:
        exec('from ' + args.config_path + ' import *')
    except Exception as e:
        raise

default_log_setup(level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# NeRF is trained with a MultiviewDataset, which knows how to generate RGB rays from a set of images + cameras
train_dataset = MyMultiviewDataset(
    dataset_path=dataset_path,
    aabb_scale=aabb_scale,
    multiview_dataset_format=multiview_dataset_format,
    mip=mip,
    dataset_num_workers=dataset_num_workers,
    num_samples=num_samples
)

grid = HashGrid.from_geometric(feature_dim=feature_dim,
                               num_lods=num_lods,
                               multiscale_type=multiscale_type,
                               codebook_bitwidth=codebook_bitwidth,
                               min_grid_res=min_grid_res,
                               max_grid_res=max_grid_res,
                               blas_level=blas_level)

grid_t=HashGrid.from_geometric(feature_dim=t_feature_dim,
                                num_lods=t_num_lods,
                                multiscale_type=t_multiscale_type,
                                codebook_bitwidth=t_codebook_bitwidth,
                                min_grid_res=t_min_grid_res,
                                max_grid_res=t_max_grid_res,
                                blas_level=0).cuda()


appearence_emb=torch.randn(len(train_dataset), 2, device='cuda')*0.01

from wisp.models.nefs import NeuralRadianceField
nerf =  Nef(grid=grid,
            grid_t=grid_t,
            appearence_embedding=appearence_emb,
            view_embedder=view_embedder,
            view_multires=view_multires,
            direction_input=direction_input,
            hidden_dim = hidden_dim,
            prune_density_decay=prune_density_decay,
            prune_min_density = prune_min_density,
            steps_before_pruning=steps_before_pruning,
            max_samples = max_samples,
            trainable_background = trainable_background,
            starting_background=starting_background,
            starting_density_bias = starting_density_bias,
            render_radius = render_radius
            )

tracer = Tracer(raymarch_type='ray', num_steps=num_steps)
#tracer = PackedRFTracer(raymarch_type='ray', num_steps=1024)

from wisp.renderer.core.api.renderers_factory import register_neural_field_type
from wisp.renderer.core.renderers.radiance_pipeline_renderer import NeuralRadianceFieldPackedRenderer
register_neural_field_type(Nef, Tracer, NeuralRadianceFieldPackedRenderer)

if model_path is not None:
    pipeline = torch.load(model_path)
else:
    pipeline = Pipeline(nef=nerf, tracer=tracer).to('cuda')

# Joint trainer / app state, to allow the trainer and gui app to share updates
scene_state = WispState()

# TODO (operel): the trainer args really need to be simplified -_-

trainer = Trainer(pipeline=pipeline,
                dataset=train_dataset, #train_dataset
                num_epochs=epochs,
                batch_size=batch_size,           # 1 image per batch
                batch_accumulate=batch_accumulate,
                optim_cls=torch.optim.Adam,
                lr=lr,
                weight_decay=weight_decay,     # Weight decay, applied only to decoder weights.
                grid_lr_weight=100.0, # Relative learning rate weighting applied only for the grid parameters
                optim_params=dict(lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99), eps=1e-15),
                log_dir=log_dir,
                device=device,
                exp_name=exp_name,
                info='',
                extra_args=dict(  # TODO (operel): these should be optional..
                    dataset_path=dataset_path,
                    dataloader_num_workers=0,
                    num_lods=4,
                    grow_every=-1,
                    only_last=False,
                    resample=False,
                    resample_every=-1,
                    prune_every=len(train_dataset),#
                    random_lod=False,
                    rgb_loss=1.0,
                    camera_origin=camera_origin,
                    camera_lookat=camera_lookat,
                    camera_fov=camera_fov,
                    camera_clamp=[0, 10],
                    render_batch = render_batch,
                    bg_color='black',
                    valid_every=-1,
                    save_as_new=False,
                    model_format='full',
                    mip=1
                ),
                render_tb_every=render_tb_every,
                save_every=save_every,
                scene_state=scene_state,
                trainer_mode='train',
                using_wandb=using_wandb,
                trans_mult = trans_mult, 
                entropy_mult = entropy_mult, 
                empty_mult = empty_mult, 
                empty_selectivity = empty_selectivity)

torch.cuda.empty_cache()

#is_gui_mode = os.environ.get('WISP_HEADLESS') != '1'
if is_gui_mode: # is_gui_mode:
    scene_state.renderer.device = trainer.device  # Use same device for trainer and app renderer
    app = DemoApp(wisp_state=scene_state, background_task=trainer.iterate, trainer=trainer, window_name=exp_name) #trainer.iterate
    app.run()  # Interactive Mode runs here indefinitely
else:
    trainer.train()  # Headless mode runs all training epochs, then logs and quits