import numpy as np

dataset_path='datasets/yellowdino_new_new/'
#model_path='_results/logs/runs/NerfW for hand-held objects/20230308-003325/model.pth'

epochs=500

trans_mult = 1e-2
entropy_mult = 1e-2 * 0
empty_mult = 1e-3 * 0
mask_mult = 1e-3

cameras_lr_weight = 1e-3

trainable_background = False
starting_background = [0.5,0.5,0.7]

plot_grid=False

is_gui_mode=True
mip=2

num_samples=2**12
max_samples = 2**18

num_lods=8
num_angles=1
camera_distance=2