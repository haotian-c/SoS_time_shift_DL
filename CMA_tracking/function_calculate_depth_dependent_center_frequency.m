function function_calculate_depth_dependent_center_frequency(compunded_before_b_mode)
    rf_data = compunded_before_b_mode;
    figure;
    imagesc(rf_data);
    colormap('jet');
    colorbar;
    title('2D RF Data');
    xlabel('Samples (X-axis)');
    ylabel('Samples (Y-axis)');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% Step 1: Calculate depth-dependent center frequency along each vertical line
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    freq_all_A_lines = [];
    window_length = 16;
    half_win = floor(window_length/2);

    for column_i = 1:size(rf_data, 2)
        signal = mean(rf_data(:, column_i), 2);
        signal_length = length(signal);

        Fs = 1/(0.23e-3/5/1540)/2;
        NFFT = window_length;
        freq_axis = (0:NFFT-1)*(Fs/NFFT);

        center_freq_trace = zeros(signal_length, 1);
        win = hamming(window_length);

        % Loop through each depth to estimate the center frequency at that position
        for i = 1:(signal_length - window_length)
            seg = signal(i:(i + window_length - 1));
            seg = seg .* win;
            fft_seg = abs(fft(seg, NFFT));
            fft_half = fft_seg(1:floor(NFFT/2));
            [~, idx] = max(fft_half);
            center_freq = freq_axis(idx);
            center_idx = i + half_win;
            if center_idx <= signal_length
                center_freq_trace(center_idx) = center_freq;
            end
        end

        zero_idx = center_freq_trace == 0;
        center_freq_trace(zero_idx) = interp1(find(~zero_idx), center_freq_trace(~zero_idx), find(zero_idx), 'linear', 'extrap');

        % Median filter to remove spikes
        center_freq_trace = medfilt1(center_freq_trace, 21);

        % Savitzky-Golay smoothing for trend
        smoothed_center_freq = sgolayfilt(center_freq_trace, 3, 101);

        freq_all_A_lines = [freq_all_A_lines, smoothed_center_freq];
    end

    figure;
    imagesc(freq_all_A_lines);
    caxis([3.5e6, 5.5e6]);
    title('Single Line');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% Step 2: Average depth-dependent center frequency across all vertical lines
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    mean_freq_line = mean(freq_all_A_lines, 2);
    figure;
    plot(mean_freq_line);
    title('Averaged Line');

    window_size = 91;  
    smoothed_mean_freq_line = movmean(mean_freq_line, window_size, 'omitnan');
    figure;
    plot(smoothed_mean_freq_line);
    title('Smoothed Averaged Line');
    save('depth_dependent_freq', 'smoothed_mean_freq_line');
end
