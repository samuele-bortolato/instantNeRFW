import torch

is_gui_mode=True

## Load saved model (parameters of Grids and NeF have to match)
model_path=None

## Dataset
dataset_path='datasets/lego/'
aabb_scale=3
mip=0
multiview_dataset_format='standard'
dataset_num_workers=-1
num_samples=2048

## Grid
feature_dim=2
num_lods=16
multiscale_type='cat'
codebook_bitwidth=20
min_grid_res=16
max_grid_res=4096
blas_level=7
## Grid Transient
t_feature_dim=2
t_num_lods=16
t_multiscale_type='cat'
t_codebook_bitwidth=12
t_min_grid_res=16
t_max_grid_res=2048

## Neural Field
pos_embedder= 'none'
view_embedder= 'positional'
pos_multires= 10
view_multires= 2
position_input=False
direction_input= True
# decoder args
activation_type = 'relu'
layer_type = 'none'
hidden_dim = 128
num_layers = 1
beta_min = 0.01
# pruning args
prune_density_decay= 0.90
prune_min_density= 1e-2
steps_before_pruning = 1
# others
max_samples = 2**18 # size of the samples batch
starting_density_bias = -2

appearence_emb_dim = 5

## Tracer
num_steps=512
trainable_background = True
starting_background = None

## Rendering
rendering_threshold_density = 0.01
render_radius = 0.7
render_batch = 4000

## Training
epochs = 1000
batch_size = 1 # in images
batch_accumulate = 4
lr = 1e-3
weight_decay=0      # Weight decay, applied only to decoder weights.
grid_lr_weight = 100.0 # Relative learning rate weighting applied only for the grid parameters

trans_mult = 1e-4 
entropy_mult = 0 
empty_mult = 0
empty_selectivity = 50

# logs
camera_origin = [1.25, 1.25, 1.25]  
camera_lookat = [0, 0, 0]
camera_fov=30
render_tb_every=10 # tensorboard
save_every=10
using_wandb=False
log_dir = '_results/logs/runs/'
exp_name = 'NerfW for hand-held objects'
plot_grid=True