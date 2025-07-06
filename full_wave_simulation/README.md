# GPU-Based Ultrasound Simulation using k-Wave

This repository provides MATLAB scripts for performing GPU-accelerated 2D ultrasound wave propagation simulations using the [k-Wave Toolbox](http://www.k-wave.org/). The simulations generate received channel data from virtual phantoms under various transmit angles.

---


## Getting Started

### 1. Prerequisites

- **MATLAB** R2020a or later
- **GPU with CUDA support** 
- **[k-Wave Toolbox](http://www.k-wave.org/)**  
  Download and add to your MATLAB path:

```matlab
addpath('path_to_kWave');
savepath;
```


- **Parallel Computing Toolbox**
Required for GPU-based simulation (gpuArray support).

### 2. Run Simulation

To run the simulation, call the following function in MATLAB:

```
run_simulation_on_GPU(GPU_i, list_object_index)
```


#### Input arguments:

- `GPU_i`: Integer specifying the GPU device index to use for simulation (e.g., `0` for the first GPU).
- `list_object_index`: Vector of object indices to simulate. Each index corresponds to a folder named `object_<index>` that contains:
  - `sos_phamton_gt_<index>.mat`
  - `density_phamton_gt_<index>.mat`

#### Output:

- `return_flag`: Returns `0` on success, or `-1` if a simulation instance fails.

---

### 3. Output file

For each object and transmit angle, the simulation produces:

- `object_<index>_received_at_angle_<angle>.mat`:  
  Contains channel data resampled to match Verasonics' 48 ns sampling interval.

---

## Multi-GPU Support

This script allows **dispatching jobs to a specific GPU** via the `GPU_i` argument. This enables:

- Efficient use of **multiple GPUs** on a shared machine
- Workload parallelism when generating large datasets
- Simulating different phantom indices independently across GPUs

### Example (multi-instance setup):

```matlab
% Instance 1 (GPU 0)
run_simulation_on_GPU(0, 1:50)

% Instance 2 (GPU 1)
run_simulation_on_GPU(1, 51:100)
```

You may run multiple MATLAB instances in parallel, each targeting different object indices and GPUs.

---

## Simulation Details

- Designed for the **GE 9L-D linear ultrasound probe**:
  - 192 elements
  - 0.23 mm pitch
- Angle sweep: from **−17.5° to +17.5°**, in 0.5° steps
- The receive aperture matches the transmit aperture
- **High-attenuation material is inserted between elements** to model kerf regions and reduce cross-talk  
  (see Section II-E of the manuscript)
- Grid spacing and time step are chosen to match a Verasonics-like sampling interval of **48 ns**

---

## Dependencies

- [k-Wave Toolbox](http://www.k-wave.org/)
- MATLAB with **Parallel Computing Toolbox**

---



