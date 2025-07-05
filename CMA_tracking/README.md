# Common-Mid Angle Processing

The common-mid angle processing is performed through two steps:

## 1. Dynamic Receiving with Beamforming
The `function_beamforming_parallel.m` script performs dynamic receiving, confining the RX angle to be symmetric to the TX angle with respect to a prespecified mid-angle. 

- It reads from `BF_angle_combs.mat`, which specifies all the combinations of RX and TX angles for beamforming.
- To efficiently compute all angle combinations, the script utilizes **parallel computing** in MATLAB.

## 2. Phase Shift Tracking
The `function_phase_shift_tracking.m` script performs common-mid angle phase shift tracking.

- It reads from `cells_phase_track_struct_lookup.mat`, which groups the beamformed RF data into three categories corresponding to mid-angles: **0°, 7.5°, and -7.5°**.
- The script executes phase shift tracking and summation, resulting in **three time-shift maps** that serve as inputs for a neural network used in **Speed of Sound (SoS) reconstruction**.

## Data Files

### **BF_angle_combs.mat**
- Specifies all the **TX and RX angle combinations** used in beamforming.

### **cells_phase_track_struct_lookup.mat**
- Defines the three **mid-angle groups** (0°, 7.5°, -7.5°) and the corresponding **TX and RX angle combinations**.

### **workspace.mat**
- Stores experiment settings, including:
  - **TX sequences**
  - **Receive settings**
  - **Probe specifications**  
  These configurations were used in data collection with the **Verasonics Vantage 256** system.

