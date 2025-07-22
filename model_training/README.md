# Training Scripts

## Overview

This directory contains two scripts for model training:

1. **Pretraining** using computationally inexpensive ray-tracing synthesized data to fully activate neural units.  
2. **Finetuning** using higher-fidelity full-wave simulation data to obtain the finalized model states.

## Dependencies and Structure

The training scripts reference components located in the project root directory `SoS_time_shift_DL/`:

- `model.py`: U-Net architecture definition.  
- `utils.py`: Utility functions for data handling and processing.  
- `pytorch_SSIM_module/`: Custom implementation of the Structural Similarity Index Measure (SSIM) loss, which is not available in the standard PyTorch library. This module is manually added to the Python path in the scripts.

## Note
- Multi-GPU training is supported via PyTorch’s DataParallel.
- Input and Output Design
	•	Input data are scaled up prior to training to compensate for their small original magnitude.
	•	The model output is defined as an additive correction to the beamformed SoS baseline. This allows the output to capture both positive and negative deviations and ensures the full dynamic range is utilized.
- Reconstruction Field-of-View
The reconstruction field is the central 50-pixel-wide lateral region, which corresponds to the area covered by all steered transmissions.

- Uses GPU-accelerated ray-tracing synthesis for rapid generation of time-shift map and SoS pairs, via the `../ray_tracing_synthesis/ray_tracing_synthesis.py` module.
- Data preparation (image loading, synthesis, and batch assembly) for 1.2 million ray-tracing data is handled by multiple CPU processes running asynchronously in the background. This ensures training data is readily available for the GPU training loop, maximizing throughput and minimizing idle time.


## Usage

1. Adjust parameters at the top of the script (batch size, learning rate, GPUs, etc.) as needed.
2. Prepare input data: Place the required grayscale JPEG image files in the specified directory (see `test_data_parent_dir` in the script).
3. Run the script: `python pretraining_w_ray_tracing_synthesized_data_calling_module.py`

