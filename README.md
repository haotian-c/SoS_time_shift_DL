# SoS Imaging Using Time Shift Map (repository under construction)


## Overview
This repository accompanies the paper  
[Chen, H., & Han, A. (2024). Robust deep learning for pulse-echo speed of sound imaging via time-shift maps. Authorea Preprints.](https://www.techrxiv.org/doi/full/10.36227/techrxiv.171709863.32880935)

## Directory Structure
```
/project_root
│── /CMA_tracking             # Common mid-angle tracking, including beamforming and phase shift tracking
│── /full-wave simulation     # Full-wave simulation using k-Wave simulation toolbox and with GPU configuration
│── /ray_tracing_synthesis    # GPU-accellerated ray-tracing synthesize to convert natural image to SoS map and then compute time-shift map
│── /pytorch_SSIM_module      # PyTorch implementation SSIM loss for training
│── /model_training           # pretraining using ray-tracing synthesized data and finetuning with full-wave simulation data
│── LICENSE                   # License information
│── README.md                 # Project documentation
│── model.py                  # Time-shift DL model implemented as U-Net
│── utils.py                  # Utility functions
```

## Setup Instructions
### Installing Dependencies
To install the required dependencies, create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate my_env_name
```

### Running the Project
#### Running a Demo
To run a demonstration of the model:
```bash
jupyter notebook demo.ipynb
```


## Notes
- The common-mid angle (CMA) tracking is implemented in MATLAB using Signal Processing Toolbox for efficient processing.
- The inference is implemented in Python using PyTorch framework.

## Notes
- The common-mid angle (CMA) tracking is implemented in MATLAB using Signal Processing Toolbox for efficient processing.
- The inference is implemented in Python using PyTorch framework.

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

## Contributor
- **Haotian Chen** 

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or suggestions, please contact hc19@illinois.edu.
