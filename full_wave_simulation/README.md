# GPU-Based Ultrasound Simulation using k-Wave

This repository provides MATLAB scripts for performing GPU-accelerated 2D ultrasound wave propagation simulations using the [k-Wave Toolbox](http://www.k-wave.org/). The simulations generate received channel data from virtual phantoms under various transmit angles.

---


## 🚀 Getting Started

### 1. Prerequisites

- **MATLAB** R2020a or later
- **GPU with CUDA support** (e.g., NVIDIA RTX series)
- **[k-Wave Toolbox](http://www.k-wave.org/)**  
  Download and add to your MATLAB path:

```matlab
addpath('path_to_kWave');
savepath;
```


• Parallel Computing Toolbox
Required for GPU-based simulation (gpuArray support).
