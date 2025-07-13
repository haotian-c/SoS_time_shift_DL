# Common-Mid Angle Processing

The common-mid angle processing is performed through two steps:

## 1. Delay-and-sum Beamforming with Dynamic Receive Focusing
The **function_beamforming_parallel.m** script performs dynamic receive focusing, confining the RX angle to be symmetric to the TX angle with respect to a prespecified mid-angle. 

- It reads from **BF_angle_combs.mat**, which specifies all the combinations of RX and TX angles for beamforming.
- To process all angle combinations in parallel, the script utilizes a parfor loop, which relies on MATLAB’s Parallel Computing Toolbox for distributed computation across multiple CPU cores.

## 2. Phase Shift Tracking
The **function_phase_shift_tracking.m** script performs common-mid angle phase shift tracking.

- It reads from **cells_phase_track_struct_lookup.mat**, which groups the beamformed RF data into three categories corresponding to mid-angles: **0°, 7.5°, and -7.5°**.
- To prevent signal decorrelation, phase shift tracking is performed incrementally over small angular steps. The phase shifts from each step are then added together to yield the total phase shift between the endpoints.
- The script executes phase shift tracking and summation, resulting in **three time-shift maps** that serve as inputs for a neural network used in **Speed of Sound (SoS) reconstruction**.
- Optionally, when attenuation is significant, consider using a depth-dependent center frequency to convert phase shift to time shift. The center frequency profile can be measured using the **function_calculate_depth_dependent_center_frequency.m** script, given a 2D beamformed RF dataset.

## Look-up Files

### **BF_angle_combs.mat**
- Specifies all the **TX and RX angle combinations** used in beamforming.

### **cells_phase_track_struct_lookup.mat**
-  Specifies all the **TX and RX angle combinations** used in phase-shift tracking.
-  Compared with **BF_angle_combs.mat**, which lists all angle combinations that can be processed in parallel, the **cells_phase_track_struct_lookup.mat** is grouped by mid angles (-7.5, 0, 7.5), and the order matters for each angular combination: whether tracking is performed from A to B or from B to A, since the direction will affect the sign in the later summation process.

### **workspace.mat**
- Stores experiment settings, including:
  - **TX sequences**
  - **Receive settings**
  - **Probe specifications**  
  These configurations were used in data collection with the **Verasonics Vantage 256** system.

