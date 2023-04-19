import numpy as np

dataset_path='datasets/greendino_new/'
#model_path='_results/logs/runs/NerfW for hand-held objects/20230308-003325/model.pth'

epochs=500

trans_mult = 1e-2
entropy_mult = 1e-2 * 0
empty_mult = 1e-3 * 0

trainable_background = False
starting_background = [0.5,0.5,0.7]

plot_grid=False

num_samples=2**12
max_samples = 2**20

num_angles=0
angles = np.pi * 0.1 * np.array(list(range(num_angles + 1)))
x = -2 * np.sin(angles)
y = [1.25] * (num_angles + 1)
z = -2 * np.cos(angles)
camera_origin = [[x[i], y[i], z[i]] for i in list(range(num_angles + 1))]