# Ray-Tracing-Based Time Shift Synthesis

This directory contains a GPU-accelerated implementation of ray-tracing synthesis, to derive time-shift maps from an arbitrary image pattern. It is part of the larger `SoS_time_shift_DL` project.

## Contents

- `ray_tracing_synthesis.py`: Python module that defines:
  - `SoSToTimeShiftTransformer`: a class for SoS-to-time-shift conversion using precomputed ray-tracing forward matrices
  - `image_to_sos_map`: a utility function to convert grayscale images into SoS maps

- `demo_ray_tracing_synthesis.ipynb`: Jupyter notebook demonstrating:
  - Image-to-SoS map conversion from `example.jpg`
  - Ray-tracing synthesis of time-shift maps using forward models corresponding to mid angles of 0°, +7.5°, and −7.5°
  - Computational performance measurement, with runtime under 0.01 seconds on an NVIDIA A40 GPU [Run time measurement provided in] https://github.com/haotian-c/SoS_time_shift_DL/blob/main/ray_tracing_synthesis/demo_ray_tracing_synthesis.ipynb

- `example.jpg`: Sample grayscale image used for demonstration

- `d11_11_psf0_forward_modelmatrix.mat`: Forward model matrix for 0° transmit angle

- `d11_11_7p5_forward_modelmatrix.mat`: Forward model matrix for +7.5° transmit angle

- `d11_11_minus7p5_forward_modelmatrix.mat`: Forward model matrix for −7.5° transmit angle

## Demo


```bash
jupyter notebook demo_ray_tracing_synthesis.ipynb
```



You can view the rendered demo notebook here:  
[demo_ray_tracing_synthesis.ipynb](./demo_ray_tracing_synthesis.ipynb)





![image](https://github.com/user-attachments/assets/37faa59a-30e1-46b5-a892-23173e618784)

![image](https://github.com/user-attachments/assets/426bd4cb-e8cd-4daf-9891-8fc301ec33c5)

![image](https://github.com/user-attachments/assets/e45acb11-2158-4001-a1dc-e2896a31e32f)
