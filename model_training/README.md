# Training Scripts

## Overview

This directory contains two scripts for model training:

1. **Pretraining** using 1.2 million computationally inexpensive ray-tracing synthesized data to initialize the model with diverse input patterns.  
2. **Finetuning** using higher-fidelity full-wave simulation data to obtain the finalized model states.

## Dependencies and Structure

The training scripts call modules located in the project root directory `SoS_time_shift_DL/`:

- `model.py`: U-Net architecture definition.  
- `utils.py`: Utility functions for data processing.  
- `pytorch_SSIM_module/`: Implementation of the Structural Similarity Index Measure (SSIM) loss, which is not available in the standard PyTorch library. This module is manually added to the Python path in the scripts.

## Note
- Multi-GPU training is supported via PyTorch’s DataParallel.
- Input and Output Design
	•	Input data are scaled up prior to training to compensate for their small original magnitude.
	•	The model outputs an additive residual to the beamformed SoS baseline, enabling the network to represent both positive and negative deviations and to fully utilize the output dynamic range.
- Reconstruction Field-of-View
The reconstruction field is the central 50-pixel-wide lateral region, which corresponds to the area covered by all steered transmissions.

- Uses GPU-accelerated ray-tracing synthesis for rapid generation of time-shift map and SoS pairs, via the `../ray_tracing_synthesis/ray_tracing_synthesis.py` module.
- Data preparation (image loading, synthesis, and batch assembly) for 1.2 million ray-tracing data is handled by multiple CPU processes running asynchronously in the background. Data are buffered in an in-memory queue and consumed by the GPU training loop in real time.
- Adjustable parameters: Key hyper-parameters such as batch size, learning rate, and GPU configuration can be set at the top of the script. The U-Net model also accepts `zero_skip_s1` and `disable_bn` as optional arguments during instantiation.


