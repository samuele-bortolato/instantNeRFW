import numpy as np

is_gui_mode=True

import numpy as np

dataset_path='datasets/greendino_new/'
#model_path='_results/logs/runs/NerfW for hand-held objects/20230530-185943/model.pth'

epochs=500

trans_mult = 1e-2
entropy_mult = 1e-2 * 0
empty_mult = 1e-3 * 0
mask_mult = 1e-3

cameras_lr_weight = 1e-3

trainable_background = False
starting_background = [0.5,0.5,0.7]

aabb_scale=4
mip=2

plot_grid=False