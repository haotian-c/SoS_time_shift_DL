# Deep Learning for SoS Imaging via Time Shift Maps


## Overview
This project presents a deep learning approach to pulse-echo speed-of-sound (SoS) imaging based on time-shift maps. It combines a neural network with the time-shift measurements that are physically informed and derived via common mid-angle (CMA) tracking. This integration enhances the model’s robustness and domain generalizability, while preserving its ability to learn imaging processes beyond traditional straight-ray assumptions. The repository provides the implementation of the neural network model, training scripts, and signal processing modules for computing time-shift maps from raw channel data.

This repository accompanies the paper  
[Chen, H., & Han, A. (2024). Robust deep learning for pulse-echo speed of sound imaging via time-shift maps. Authorea Preprints.](https://www.techrxiv.org/doi/full/10.36227/techrxiv.171709863.32880935)


## Directory Structure
```
/project_root
│── /CMA_tracking/                   # Common mid-angle tracking, including beamforming and phase shift tracking
│── /full-wave simulation/           # Full-wave simulation using k-Wave simulation toolbox and with GPU configuration
│── /ray_tracing_synthesis/          # GPU-accellerated ray-tracing synthesis to convert natural image to SoS map and then compute time-shift map
│── /pytorch_SSIM_module/            # PyTorch implementation of SSIM loss for training
│── /model_training/                 # pretraining using ray-tracing synthesized data and finetuning with full-wave simulation data
|── /demo_of_training_and_inference/ # Runable demo with minimal environment requirement
│── LICENSE                          # License information
│── README.md                        # Project documentation
│── model.py                         # Time-shift DL model implemented as U-Net
│── utils.py                         # Utility functions
│── environment.yml                  # Conda environment specification for PyTorch and dependencies
```



## Setup Instructions
This project uses 1) **MATLAB** for full-wave simulation, beamforming, and phase-shift tracking, and 2) **Python** for ray-tracing synthesis and deep learning model training.

### • Installing Python Dependencies
To set up the Python environment using Conda:
```bash
conda env create -f environment.yml
```


### • Installing k-Wave for MATLAB
This project relies on **k-Wave 1.4** for full-wave ultrasound simulation. Please follow these steps:

1) Download k-Wave from the official website: [http://www.k-wave.org/download.php](http://www.k-wave.org/download.php)
2) Add the k-Wave toolbox to your MATLAB path.
3) For GPU-accelerated simulation, install the C++/CUDA binaries provided on the k-Wave [download page](http://www.k-wave.org/download.php).

## Quick Startup
Run a hands-on demo of model training and inference

https://github.com/haotian-c/SoS_time_shift_DL/tree/main/demo_of_training_and_inference


## Notes
- The common-mid angle (CMA) tracking is implemented in MATLAB using Signal Processing Toolbox and Parallel Computing Toolbox.
- The deep learning is implemented in Python using PyTorch framework.


## Citation

If you use this work, please cite:

```bibtex
@article{chen2024robust,
  title={Robust deep learning for pulse-echo speed of sound imaging via time-shift maps},
  author={Chen, Haotian and Han, Aiguo},
  journal={Authorea Preprints},
  year={2024},
  publisher={Authorea}
}
```


## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or suggestions, please contact hc19@illinois.edu.
