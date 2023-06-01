call conda activate wisp1
START tensorboard --logdir _results/logs/runs
timeout 3
start "" http://localhost:6006/

