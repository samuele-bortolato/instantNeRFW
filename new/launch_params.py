dataset_path='../datasets/yellowdino_new_new/'
#model_path='_results/logs/runs/NerfW for hand-held objects/20230308-003325/model.pth'

epochs=500

trans_mult = 1e-4
entropy_mult = 1e-2
empty_mult = 1e-3

trainable_background = True
starting_background = [0.5,0.5,0.7]

plot_grid=False

num_samples=2**12
max_samples = 2**20