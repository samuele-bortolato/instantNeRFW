import argparse
from .default_params import *

def parse_args():
	parser = argparse.ArgumentParser(description="Train a NeRF for occluded/hand-held objects. [W] indicates the values to reduce in case of GPU memory problems!")

	parser.add_argument("--no_gui", action='store_false', help="Open GUI during training.")
        
	parser.add_argument("--dataset_path", default=dataset_path)
	parser.add_argument("--aabb_scale", default=aabb_scale, choices=[1, 2, 4, 8, 16, 32, 64, 128], help="Large scene scale factor. 1=scene fits in unit cube; power of 2 up to 128")
	parser.add_argument("--mip", default=mip, help="Downscaling factor for dataset images [W].")
	parser.add_argument("--multiview_dataset_format", default=multiview_dataset_format, choices=["standard", "nvcc"], help="Choose the format of the dataset.")
	parser.add_argument("--dataset_num_workers", default=dataset_num_workers, help="Number of workers used to load the dataset.")
	parser.add_argument("--num_samples", default=num_samples, help="Number of rays casted per image [W]")
        
	parser.add_argument("--feature_dim", default=feature_dim, help="The dimension of the features stored on the Hash grid.")
	parser.add_argument("--num_lods", default=num_lods, help="Number of resolution levels. [W]")
	parser.add_argument("--multiscale_type", default=multiscale_type, choices=["sum", "cat"], help="How to aggregate information from the Hash table.")
	parser.add_argument("--codebook_bitwidth", default=codebook_bitwidth, help="Hash table size. [W]")
	parser.add_argument("--min_grid_res", default=min_grid_res, help="Coarsest resolution.")
	parser.add_argument("--max_grid_res", default=max_grid_res, help="Finest resolution.")
	parser.add_argument("--blas_level", default=blas_level, help="The level of the octree to be used as the BLAS (bottom level acceleration structure).")
        
	parser.add_argument("--t_feature_dim", default=t_feature_dim, help="The dimension of the features stored on the Hash grid (transient).")
	parser.add_argument("--t_num_lods", default=t_num_lods, help="Number of resolution levels (transient). [W]")
	parser.add_argument("--t_multiscale_type", default=t_multiscale_type, choices=["sum", "cat"], help="How to aggregate information from the Hash table (transient).")
	parser.add_argument("--t_codebook_bitwidth", default=t_codebook_bitwidth, help="Hash table size (transient). [W]")
	parser.add_argument("--t_min_grid_res", default=t_min_grid_res, help="Coarsest resolution (transient).")
	parser.add_argument("--t_max_grid_res", default=t_max_grid_res, help="Finest resolution (transient).")
        
	parser.add_argument("--pos_embedder", default=pos_embedder, choices=["none", "identity", "positional"], help="Type of positional embedder to use for input coordinates.")
	parser.add_argument("--view_embedder", default=view_embedder, choices=["none", "identity", "positional"], help="Type of positional embedder to use for view directions.")
	parser.add_argument("--pos_multires", default=pos_multires, help="Number of frequencies used for 'positional' embedding of pos_embedder.")
	parser.add_argument("--view_multires", default=view_multires, help="Number of frequencies used for 'positional' embedding of view_embedder.")
	parser.add_argument("--position_input", action='store_true', help="If True, the input coordinates will be passed into the decoder.")
	parser.add_argument("--direction_input", action='store_true', help="If True, the input coordinates will be passed into the decoder.")
        
	parser.add_argument("--activation_type", default=activation_type, choices=['none', 'relu', 'sin', 'fullsort', 'minmax'], help="Type of activation function.")
	parser.add_argument("--layer_type", default=layer_type, choices=['none', 'linear', 'spectral_norm', 'frobenius_norm', 'l_1_norm', 'l_inf_norm'], help="Type of MLP layer.")
	parser.add_argument("--hidden_dim", default=hidden_dim, help="Number of neurons in hidden layers of both decoders.")
	parser.add_argument("--num_layers", default=num_layers, help="Number of hidden layers in both decoders.")
	parser.add_argument("--beta_min", default=beta_min, help="Minimum uncertainty for each transient location.")

	parser.add_argument("--prune_density_decay", default=prune_density_decay, help="Decay rate of density per 'prune step'.")
	parser.add_argument("--prune_min_density", default=prune_min_density, help="Minimal density allowed for 'cells' before they get pruned during a 'prune step'.")
	parser.add_argument("--steps_before_pruning", default=steps_before_pruning, help="")
        
	parser.add_argument("--max_samples", default=max_samples, help="Number of samples computed at the same time. [W]")
	parser.add_argument("--starting_density_bias", default=starting_density_bias, help="Initial value for densities.")
        
	parser.add_argument("--model_path", default=model_path, help="Continue to train a previous NeRF model.")
        
	parser.add_argument("--num_steps", default=num_steps, help="Number of samples generated per ray.")
	parser.add_argument("--trainable_background", action='store_true', help="Train to recognise (and remove) the background.")
	parser.add_argument("--starting_background", default=starting_background, help="Initial background color.")
        
	parser.add_argument("--rendering_threshold_density", default=rendering_threshold_density, help="Minimum density for rendered points.")
	parser.add_argument("--render_radius", default=render_radius, help="Radius of the spherical volume in which the scene is rendered.")
	parser.add_argument("--render_batch", default=render_batch, help="")
        
	parser.add_argument("--epochs", default=epochs, help="Number of training epochs.")
	parser.add_argument("--batch_size", default=batch_size, help="Number of images per batch.")
	parser.add_argument("--batch_accumulate", default=batch_accumulate, help="Number of gradient accumulations before parameter update.")
	parser.add_argument("--lr", default=lr, help="Learning rate.")
	parser.add_argument("--weight_decay", default=weight_decay, help="Weight decay, applied only to decoder weights.")
	parser.add_argument("--grid_lr_weight", default=grid_lr_weight, help="Relative learning rate weighting applied only for the grid parameters.")
	parser.add_argument("--trans_mult", default=trans_mult, help="Multiplier for the loss term involving the transient density.")
	parser.add_argument("--entropy_mult", default=entropy_mult, help="Multiplier for the loss term involving the entropy of the static density.")
	parser.add_argument("--empty_mult", default=empty_mult, help="")
	parser.add_argument("--empty_selectivity", default=empty_selectivity, help="")
        
	parser.add_argument("--camera_origin", default=camera_origin, help="")
	parser.add_argument("--camera_lookat", default=camera_lookat, help="")
	parser.add_argument("--camera_fov", default=camera_fov, help="")
	parser.add_argument("--render_tb_every", default=render_tb_every, help="")
	parser.add_argument("--save_every", default=save_every, help="")
	parser.add_argument("--using_wandb", action='store_true', help="")
	parser.add_argument("--log_dir", default=log_dir, help="")
	parser.add_argument("--exp_name", default=exp_name, help="")

	args = parser.parse_args()
	return args