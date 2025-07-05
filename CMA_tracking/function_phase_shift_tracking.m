function  function_phase_shift_tracking(dir_binary_data,source_dir_prefix,target_dir_prefix, ...
    PARAM_tracking_kernel_z,PARAM_tracking_kernel_y)

load(['phase_track_struct_lookup.mat']);
load([dir_binary_data 'workspace.mat']);

psf_dict = {'psf0','psf7p5','psfminus7p5'};
for psf_direction_idx =  1:length(psf_dict)
    psf_direction_i = psf_dict{psf_direction_idx};
    target_filename = [target_dir_prefix '_psf_' psf_direction_i 'compounded_phase_shift'  '_kernel_size_z_' int2str(PARAM_tracking_kernel_z)  '_kernel_size_y_' int2str(PARAM_tracking_kernel_y)] ;
    phase_shift_map = zeros(size(load([source_dir_prefix    '_RF_beamformed_win2p5degrees_compounded_at_Tx_' int2str(0*1000) '_Rx_' int2str(0*1000)  '.mat'  ]).compunded_before_b_mode));
for entry_i = 1:length(phase_track_struct.(psf_direction_i))

    tx_before_steer = phase_track_struct.(psf_direction_i){entry_i}(1,1);
    rx_before_steer = phase_track_struct.(psf_direction_i){entry_i}(1,2);
    tx_after_steer = phase_track_struct.(psf_direction_i){entry_i}(2,1);
    rx_after_steer = phase_track_struct.(psf_direction_i){entry_i}(2,2);

    RF_steering_angle_A = load([source_dir_prefix '_RF_beamformed_win2p5degrees_compounded_at_Tx_' int2str(tx_before_steer*1000) '_Rx_' int2str(rx_before_steer*1000)  '.mat'  ]).compunded_before_b_mode;
    RF_steering_angle_B =  load([source_dir_prefix  '_RF_beamformed_win2p5degrees_compounded_at_Tx_' int2str(tx_after_steer*1000) '_Rx_' int2str(rx_after_steer*1000)  '.mat'  ]).compunded_before_b_mode;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%% Compute Hermitian product
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
   RF_steering_angle_A = hilbert((RF_steering_angle_A));
   magnitudes = abs(RF_steering_angle_A);
  
    epsilon = 0.00000001;

   % Normalize the matrix
   RF_steering_angle_A = RF_steering_angle_A ./ (magnitudes+epsilon);
  
   RF_steering_angle_B = hilbert((RF_steering_angle_B));
   magnitudes = abs(RF_steering_angle_B);
  
   % Normalize the matrix
   RF_steering_angle_B = RF_steering_angle_B ./ (magnitudes+epsilon);   
  
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%% Compute Hermitian product
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  R_product = RF_steering_angle_A.*conj(RF_steering_angle_B);

    % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % % %%%%% Convolute with a tracking kernel
    % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Nx = PARAM_tracking_kernel_z;
    Ny = PARAM_tracking_kernel_y;

    wx = hann(Nx);
    wy = hann(Ny);
 
    tracking_kernel = wx* wy';
    
    R_product = conv2(R_product, tracking_kernel, 'same');
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %%%%% Calculate the time shift
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    

    delta_tao = atan2(imag(R_product), real(R_product));
    Nx = 1024-20*2;
    phase_shift_map = phase_shift_map+ delta_tao; 
    
end
phase_shift_map = phase_shift_map/2/(2*pi)/(5.208*1e6);

phase_shift_map = single(phase_shift_map);
save(target_filename,'phase_shift_map','-v7.3');

end