# Pretraining Pipeline for Speed-of-Sound Imaging

## Overview

This training script implements a pretraining pipeline for a neural network to reconstruct speed-of-sound (SoS) maps from time-shift data in ultrasound imaging.
A large-scale synthetic dataset is generated using a straight-ray (ray-tracing) model, with GPU-accelerated synthesis provided by modules in `../ray_tracing_synthesis/ray_tracing_synthesis.py`

## Key Features

- Uses GPU-accelerated ray-tracing synthesis for rapid generation of time-shift map and SoS pairs, via the `../ray_tracing_synthesis/ray_tracing_synthesis.py` module.
- Multiprocessing: Data preparation (image loading, synthesis, and batch assembly) is handled by multiple CPU processes running asynchronously in the background. This ensures training data is readily available for the GPU training loop, maximizing throughput and minimizing idle time.
- Multi-GPU training using PyTorch `DataParallel`.
- Periodic model checkpointing and validation.

## Usage

1. Adjust parameters at the top of the script (batch size, learning rate, GPUs, etc.) as needed.
2. Prepare input data: Place the required grayscale JPEG image files in the specified directory (see `test_data_parent_dir` in the script).
3. Run the script: `python pretraining_w_ray_tracing_synthesized_data_calling_module.py`

## Outputs
- Model checkpoints are saved periodically, with descriptive filenames.
- Training progress and validation error (RMSE) are printed to the console.
