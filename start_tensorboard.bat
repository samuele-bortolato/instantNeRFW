call conda activate wisp
START tensorboard --logdir _results/logs/runs
timeout 3
start "" http://localhost:6006/

