# Quick demo of U-Net training and inference for SoS imaging

This project uses a U-Net model to transform time-shift maps into speed of sound (SoS) maps. While the complete project is more extensive, this directory provides a quick-start setup for model training and inference.

A runnable demo is provided in **`demo_of_training_and_inference.ipynb`**, which includes both the ray-tracing synthesis (using 1.2 million natural images) and the U-Net model pipeline.

## Quick Start

1. **Download the dataset:**  
   Download the compressed natural images (3.8 GB, already grayscaled and resized to 73×89) from  
   https://drive.google.com/file/d/1aEwPTJajCLqolF2QkslsZ3OPAIdrCnAO/view?usp=drive_link

2. **Place the data:**  
   Put the downloaded file in this directory (the same folder as `demo_of_training_and_inference.ipynb`).

3. **Run the demo:**  
   Open and run `demo_of_training_and_inference.ipynb`.  
   The notebook will automatically read and process the provided data.

   

> **Note:**  
> Please pull the entire repository and maintain the files in their original locations. There are dependencies for `demo_of_training_and_inference.ipynb` to function properly.
> For example, it reads the `ray_tracing_synthesis` module from `../ray_tracing_synthesis` and the U-Net model from `../` (project root directory)
   
