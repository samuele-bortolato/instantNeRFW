#!/bin/bash

tensorboard --logdir _results/logs/runs
timeout 3
xdg-open http://localhost:6006/