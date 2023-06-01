import numpy as np

is_gui_mode=True

dataset_path='datasets/greendino_new/'
#model_path='_results/logs/runs/NerfW for hand-held objects/20230530-185943/model.pth'

view_multires=0

epochs=500

trans_mult = 1e-2
entropy_mult = 1e-2 * 0
empty_mult = 1e-3 * 0
mask_mult = 1e-2

cameras_lr_weight = 1e-2

trainable_background = True
starting_background = [0.5,0.5,0.7]

aabb_scale=4
mip=2

plot_grid=False

num_samples=2**12
max_samples = 2**18

num_lods=8
num_angles=1
camera_distance=2