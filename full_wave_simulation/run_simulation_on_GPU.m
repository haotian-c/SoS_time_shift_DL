function return_flag = run_simulation_on_GPU(GPU_i, list_object_index)
%RUN_SIMULATION_ON_GPU Simulates ultrasound wave propagation using k-Wave on GPU
%
%   return_flag = run_simulation_on_GPU(GPU_i, list_object_index)
%
%   Inputs:
%       GPU_i              - GPU device index
%       list_object_index  - Vector of object indices to simulate
%
%   Output:
%       return_flag        - 0 if all simulations succeeded, -1 if any failed
%
%   This function runs 2D k-Wave simulations of ultrasound propagation for a
%   list of phantoms. It simulates transmit-receive data at multiple steering
%   angles and saves the received data for each case.

% =========================================================================
% Simulation Parameters and Grid Setup
% =========================================================================

% Define constants
TARGET_SAMPLE_INTERVAL_in_s = 48e-9;    % Verasonics sample interval [s]
fc_trans = 5.2083e6;                    % Center frequency of transducer [Hz]
c0 = 1540;                              % Speed of sound in background [m/s]
plm_thickness = 20;                    % PML thickness [grid points]

% Grid size
Nx = 1024 - 2 * plm_thickness;         % Number of points in x-direction
Ny = Nx;                               % Number of points in y-direction

% Spatial resolution
pitch_grids = 5;
pitch_mm = 0.23;
dx = (pitch_mm * 1e-3) / pitch_grids;  % Grid spacing [m]
dy = dx;

% Create spatial and temporal grids
kgrid = kWaveGrid(Nx, dx, Ny, dy);
t_end = (Nx * dx) * 2.8 / c0;          % Simulation time duration [s]
kgrid.makeTime(c0, 0.2, t_end);        % Time array with CFL = 0.2

% =========================================================================
% Transducer Geometry and Sensor Mask
% =========================================================================

element_spacing = pitch_grids * dx;
kerf_grids_wing = 1;
num_elements = 192;
x_offset = 10;

source.p_mask = zeros(Nx, Ny);
sensor.mask = zeros(Nx, Ny);

start_index = Ny/2 - round((num_elements - 1) / 2 * element_spacing / dx);
mask_indices = start_index:round(element_spacing / dx):(start_index + round(element_spacing * (num_elements - 1) / dx));

for offset_i = -kerf_grids_wing:kerf_grids_wing
    source.p_mask(x_offset, offset_i + mask_indices) = 1;
    sensor.mask(x_offset, offset_i + mask_indices) = 1;
end


sampling_freq = 1 / kgrid.dt;
tone_burst_freq = fc_trans;
tone_burst_cycles = 3;

% =========================================================================
% insert high-attenuation materials between elements to simulate kerf regions and reduce cross-talk 
% =========================================================================


attenuation_map = zeros(Nx, Ny);
for xi = 1:Nx
    if source.p_mask(x_offset, xi) == 0
        attenuation_map((x_offset - 5):x_offset, xi) = 30;
    end
    attenuation_map(1:(x_offset - 5), xi) = 30;
end

% =========================================================================
% Iterate Through Each Phantom
% =========================================================================

for index_i = list_object_index
    dir_object = ['object_' int2str(index_i)];
    cd(dir_object);

    % Load phantom
    load(['sos_phamton_gt_', num2str(index_i), '.mat']);       % sound_speed_map
    load(['density_phamton_gt_', num2str(index_i), '.mat']);   % density_map

    medium.sound_speed = sound_speed_map;
    medium.density = density_map;
    medium.alpha_coeff = attenuation_map;
    medium.alpha_power = 1.5;

    % Stability check
    checkStability(kgrid, medium);

    % Simulation options
    input_args = {
        'PMLInside', false, ...
        'DataCast', 'gpuArray-single', ...
        'DataRecast', true, ...
        'PlotSim', true, ...
        'DeviceNum', GPU_i
    };

    % ============================================================
    % Sweep Transmit Angles
    % ============================================================

    for angle_i = -17.5:0.5:17.5
        filename_target = ['object_', num2str(index_i), ...
            '_received_at_angle_', num2str(angle_i), '.mat'];

        element_index = 0:(num_elements - 1);

        % Compute delay offsets
        if angle_i >= 0
            tone_burst_offset = 1 + element_spacing * element_index * ...
                sin(angle_i * pi / 180) / (c0 * kgrid.dt);
        else
            tone_burst_offset = 1 + element_spacing * (num_elements - 1 - element_index) * ...
                sin(abs(angle_i) * pi / 180) / (c0 * kgrid.dt);
        end

        % Expand tone burst to element width
        tone_burst_offset_consider_elementwidth = [];
        for tbo = tone_burst_offset
            tone_burst_offset_consider_elementwidth = ...
                [tone_burst_offset_consider_elementwidth; ...
                repmat(tbo, 2 * kerf_grids_wing + 1, 1)];
        end

        source.p = toneBurst(sampling_freq, tone_burst_freq, tone_burst_cycles, ...
            'SignalOffset', tone_burst_offset_consider_elementwidth);

        % Run simulation
        try
            sensor_data = kspaceFirstOrder2DG(kgrid, medium, source, sensor, input_args{:});
        catch
            warning('A simulation instance failed');
            return_flag = -1;
            return;
        end

        % Average sensor data across each element
        points_per_element = 2 * kerf_grids_wing + 1;
        sensor_data_averaged = zeros(num_elements, size(sensor_data, 2));
        for ei = 1:num_elements
            start_idx = (ei - 1) * points_per_element + 1;
            end_idx = ei * points_per_element;
            sensor_data_averaged(ei, :) = mean(sensor_data(start_idx:end_idx, :), 1);
        end
        sensor_data = sensor_data_averaged;

        % Resample to match Verasonics sampling rate
        SCALING_RATE = 1e13;
        sensor_data = resample(sensor_data', ...
            round(kgrid.dt * SCALING_RATE), ...
            round(TARGET_SAMPLE_INTERVAL_in_s * SCALING_RATE))';

        % Save result
        save(filename_target, 'sensor_data');
    end

    cd('..');
end

return_flag = 0;

end