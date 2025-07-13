function function_beamforming_parallel(dir_binary_data, prefix_fiile, SoS_value, dir_to_save)
    load(['BF_angle_combs.mat']);
    dir_name = [dir_binary_data 'channel_data_extracted/'];

    parfor angle_combs_i = 1:size(angle_combs, 1)
        fun_beamforming_3psfs_WinSize123(angle_combs_i, dir_binary_data, dir_name, ...
            prefix_fiile, SoS_value, dir_to_save)
    end
end

function fun_beamforming_3psfs_WinSize123(angle_combs_i, dir_binary_data, dir_name, ...
    prefix_filename, beamforming_sos, dir_to_save)

    load([dir_binary_data 'workspace.mat']);
    load(['BF_angle_combs.mat']);

    DEFAULT_SOS = 1540;
    PITCH_IN_MM = 0.23;
    GRIDS_NUM_PER_PITCH = 5;

    INTERVAL = Receive(1).endSample; % Length of signal per Tx and Rx
    FACTOR_RESAMPLING_RATE = 8;

    FREQ_SAMPLING = 20.8333e6;
    FREQ_SAMPLING = FREQ_SAMPLING * FACTOR_RESAMPLING_RATE;

    ELEMENT_NUMBER = 192;
    STARTING_PIXEL_BEAMFORMING = 20;

    dx = (PITCH_IN_MM * 1e-3) / GRIDS_NUM_PER_PITCH;
    dz = dx;

    f_number = 3;

    Tx_angle_degree = angle_combs(angle_combs_i, 1);
    Rx_angle_degree = angle_combs(angle_combs_i, 2);

    Z_AXIS_LENGTH = 1100;
    WIDTH = GRIDS_NUM_PER_PITCH * (ELEMENT_NUMBER - 1) + 1;

    compunded_before_b_mode = zeros(Z_AXIS_LENGTH + 1, WIDTH);

    for angle_i_degree = (Tx_angle_degree - 2.5):0.5:(Tx_angle_degree + 2.5)
        sensor_data = load([dir_name 'object_' '1' 'received_RF_at_angle_' ...
            num2str(10000 * round(angle_i_degree, 4)) ]);
        sensor_data = sensor_data.sensor_data';

        sensor_data = resample(single(sensor_data'), FACTOR_RESAMPLING_RATE, 1)';
        theta_rad = angle_i_degree * pi / 180;
        effective_theta_rad_in_new_beamforming_sos = asin(sin(theta_rad) / DEFAULT_SOS * beamforming_sos);
        theta_rad = effective_theta_rad_in_new_beamforming_sos;

        pitch_grids = GRIDS_NUM_PER_PITCH;
        xi = pitch_grids * dx;
        c0 = beamforming_sos;
        dt = 1 / (FREQ_SAMPLING);

        res_ampli = zeros(Z_AXIS_LENGTH + 1, WIDTH);

        for z = 1 + 10:Z_AXIS_LENGTH
            for x = 1:WIDTH
                if z < STARTING_PIXEL_BEAMFORMING
                    res_ampli(z, x) = 0;
                else
                    center_of_rx_aperture = x + z * tand(-Rx_angle_degree);
                    for Rx = 1:ELEMENT_NUMBER
                        if abs(Rx * pitch_grids - pitch_grids / 2 - center_of_rx_aperture) > z / f_number / 2
                            continue
                        end

                        trx = 1 / c0 * sqrt((x * dx - xi * (Rx - 1))^2 + (z * dz)^2) / dt - 100;
                        if theta_rad >= 0
                            ttx = 1 / c0 * (z * dz * cos(theta_rad) + x * dx * sin(theta_rad)) / dt;
                        else
                            ttx = 1 / c0 * (z * dz * cos(abs(theta_rad)) + ...
                                ((ELEMENT_NUMBER - 1) * pitch_grids * dx - x * dx) * sin(abs(theta_rad))) / dt;
                        end
                        if round(trx + ttx) > INTERVAL * FACTOR_RESAMPLING_RATE
                            continue
                        end
                        res_ampli(z, x) = res_ampli(z, x) + sensor_data(Rx, round(trx + ttx));
                        compunded_before_b_mode(z, x) = compunded_before_b_mode(z, x) + sensor_data(Rx, round(trx + ttx));
                    end
                end
            end
        end
    end

    compunded_before_b_mode = single(compunded_before_b_mode);
    save([dir_to_save '_RF_beamformed_win2p5degrees_compounded_at_Tx_' num2str(round(Tx_angle_degree, 3) * 1000) ...
        '_Rx_' num2str(round(Rx_angle_degree, 3) * 1000)], 'compunded_before_b_mode');
end
