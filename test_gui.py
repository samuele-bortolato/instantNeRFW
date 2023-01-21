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
from wisp.app_utils import default_log_setup, args_to_log_format
from wisp.framework import WispState
from wisp.datasets import MultiviewDataset
from wisp.datasets.transforms import SampleRays
from wisp.trainers import MultiviewTrainer
from wisp.models.grids import OctreeGrid, HashGrid
from wisp.models.pipeline import Pipeline
from wisp.tracers import PackedRFTracer
from funny_neural_field import FunnyNeuralField
from gui import DemoApp


from dataset import dataset
from trainer import trainer
from tracer import tracer
from nef import nef


import numpy as np

# parser = argparse.ArgumentParser(description='An example app over kaolin-wisp to visualize internal layers '
#                                              'of the NeRF decoder.')
# parser.add_argument('--dataset-path', type=str,
#                     help='Path to NeRF dataset, in standard format.')
# parser.add_argument('--dataset-num-workers', type=int, default=16,
#                     help='Number of workers for dataset preprocessing, if it supports multiprocessing. '
#                          '-1 indicates no multiprocessing.')
# parser.add_argument('--epochs', type=int, default=100,
#                     help='Number of epochs to run the training.')
# args = parser.parse_args()


#dataset_path='C:/Users/Sam/Downloads/V8/V8_'
#model_path='C:/Sam/Scuola/02_UniBo/2_anno/ML4CV/instantNeRFW/_results/logs/runs/siggraph_2022_demo/20230118-171927/model.pth'
dataset_path='datasets/greendino/aabb4'
model_path=None
epochs=100

is_gui_mode=False



default_log_setup(level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# NeRF is trained with a MultiviewDataset, which knows how to generate RGB rays from a set of images + cameras
train_dataset = dataset(
    dataset_path=dataset_path,
    #multiview_dataset_format='rtmv',
    multiview_dataset_format='standard',
    mip=1,
    bg_color='white',
    dataset_num_workers=-1,
    transform=SampleRays(
        num_samples=2048
    )
)

grid = HashGrid.from_geometric(feature_dim=2,
                               num_lods=16,
                               multiscale_type='cat',
                               codebook_bitwidth=18,
                               min_grid_res=16,
                               max_grid_res=2048,
                               blas_level=6)

grid_t=[]
for i in range(len(train_dataset)):
    grid_t.append(HashGrid.from_geometric(feature_dim=2,
                                        num_lods=16,
                                        multiscale_type='cat',
                                        codebook_bitwidth=10,
                                        min_grid_res=16,
                                        max_grid_res=2048,
                                        blas_level=6).cuda())

appearence_emb=torch.randn(len(train_dataset), 48, device='cuda')*0.01

from wisp.models.nefs import NeuralRadianceField
nerf =  nef(grid=grid, 
            grid_t=grid_t,
            appearence_embedding=appearence_emb,
            view_embedder='positional',
            hidden_dim = 128,
            prune_density_decay=0.95,
            prune_min_density=0.01*1024/np.sqrt(3)
            )

tracer = tracer(raymarch_type='ray', num_steps=1024)

if model_path is not None:
    pipeline = torch.load(model_path)
else:
    pipeline = Pipeline(nef=nerf, tracer=tracer).to('cuda')

# Joint trainer / app state, to allow the trainer and gui app to share updates
scene_state = WispState()

# TODO (operel): the trainer args really need to be simplified -_-
lr = 0.001
weight_decay = 0
exp_name = 'siggraph_2022_demo'
trainer = trainer(pipeline=pipeline,
                           dataset=train_dataset, #train_dataset
                           num_epochs=epochs,
                           batch_size=1,    # 1 image per batch
                           optim_cls=torch.optim.RMSprop,
                           lr=0.001,
                           weight_decay=0.0,     # Weight decay, applied only to decoder weights.
                           grid_lr_weight=100.0, # Relative learning rate weighting applied only for the grid parameters
                           optim_params=dict(lr=lr, weight_decay=weight_decay),
                           log_dir='_results/logs/runs/',
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
                               prune_every=-1,
                               random_lod=False,
                               rgb_loss=1.0,
                               camera_origin=[-2.8, 2.8, -2.8],
                               camera_lookat=[0, 0, 0],
                               camera_fov=30,
                               camera_clamp=[0, 10],
                               render_batch=4000,
                               bg_color='black',
                               valid_every=-1,
                               save_as_new=False,
                               model_format='full',
                               mip=1
                           ),
                           render_tb_every=-1,
                           save_every=100,
                           scene_state=scene_state,
                           trainer_mode='train',
                           using_wandb=False)

torch.cuda.empty_cache()

#is_gui_mode = os.environ.get('WISP_HEADLESS') != '1'
if is_gui_mode: # is_gui_mode:
    scene_state.renderer.device = trainer.device  # Use same device for trainer and app renderer
    app = DemoApp(wisp_state=scene_state, background_task=trainer.iterate, window_name="SIGGRAPH 2022 Demo") #trainer.iterate
    app.run()  # Interactive Mode runs here indefinitely
else:
    trainer.train()  # Headless mode runs all training epochs, then logs and quits