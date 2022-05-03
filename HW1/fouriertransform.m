function [ft,freqs]=fouriertransform(Signal, Fs)
% Usage [ft,freqs]=fouriertransform(Signal, Fs)
% Signal: signal samples
% Fs: Sampling frequency

% Calculate Signal Length
L=length(Signal);
% Frequency Points
freqs=((1:L)-L/2)*2/L*Fs/2;
% Fourier Transform
ft=fftshift(fft(Signal));
