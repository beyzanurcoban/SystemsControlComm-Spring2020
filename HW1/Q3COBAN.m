

%% Conventional AM Example
% This documents describes/implements the AM modulation with carrier and
% demodulation of a song signal. 

%%
%     Prepared for ELEC 301

%%
%     by Beyzanur Çoban

%%
%     *01.04.2020*                                 



%% Program Initialization
%Clear Variables and Close All Figure Windows

% Clear all previous variables
clear
% Close all previous figure windows
close all

%% Read Song File
% *song.mat* contains  *song* variable containing Song samples and *Fs* which is
% the sampling frequency

% Load the song file
load song.mat
% song is the song samples
% Fs is the sampling frequency

% Transform the song to low rate sampling for listening (sound command
% requires sampling rate to be less than 44K
songlowrate=downsample(song,10);
% Listen to
sound(songlowrate,Fs/10);
% convert it to row array
song=reshape(song,1,length(song));
% Sampling Period
Ts=1/Fs;
% Sampling times
t=(0:1:(length(song)-1))*Ts;

%% Display the whole  song

% Display the whole song
figure(1)
plot(t,song);
grid
title('Whole song signal');
xlabel('Time (seconds)');


%% Generate Modulated Signal
% Generate carrier signal and multiply with the song signal to obtain
% DSB-S modulated waveform
%
% 
%

%% 
% Carrier frequency:

%%
% $f_c=60kHz$
fc=60e3; % 60 kHz;

%%
% Carrier signal:  _

%%
% $c(t)=cos(2\pi f_c t)$
c=cos(2*pi*fc*t);

%%
% Ac should be bigger than the minimum value of the signal to make the
% whole signal positive
%
% Find Ac for envelope detector
Ac = abs(min(song));

%%
% DSB-C Modulated waveform

%%
% $x(t)=s(t)c(t)$ + Ac*c(t)

x=(song.*c) + (Ac.*c);

%% Display the Segments of Signal and Modulated Signal
% Display small section of the original signal and then the DSB-C
% modulated version
figure(2)
% plot the song segment (for about 3000 samples)
subplot(2,1,1)
plot(t(1:3000)*1000, song(1:3000));
xlabel('Time (msecs)')
title('Song Signal Segment')
grid

subplot(2,1,2)
% plot the modulated signal
plot(t(1:3000)*1000,x(1:3000),'r');
hold on
% plot also positive and negative envelopes
p1=plot(t(1:3000)*1000,song(1:3000)+Ac,'k');
p2=plot(t(1:3000)*1000,-song(1:3000)-Ac,'k');
xlabel('Time (msecs)')
set(p1,'LineWidth',3)
set(p2,'LineWidth',3)
grid
title('DSB-SC Modulated Signal Segment')

%% The DSB-C Receiver Processing (Question 3)
% Coherent DSB-C Receiver operation,

%%
% Deciding RC constant
%RC = 1/(2*pi*fc);
fm = 15e3; % signal bandwidth

% 1/fc < RC < 1/fm
RC = ((1/fc) + (1/fm))/2;

%% 
% Constant 
a = exp(-Ts/(RC));

%%
% Defining Vin
Vin = x;


%%
% Defining Vout
Vout=(1:750000);

%% 
% The operation of the envelope detector at sample n

for n = 2:750000
   if Vout(n-1)*a > Vin(n)
       Vout(n) = a*Vout(n-1);
   else
       Vout(n) = Vin(n);
   end
    
end


%%
% Then, apply DC blocker to this signal
z = Vout - Ac;


%% Fourier Transforms of Song, Modulated and Demodulated Signals
% Calculate and Display the Fourier Transforms of the song,modulated and
% demodulated signals

%%
% Calculate the Fourier Transform of the song signal

[ftsong,freqs]=fouriertransform(song, Fs);

%%
% Calculate the Fourier Transform of the DSB-SC signal

[ftx,freqs]=fouriertransform(x,Fs);



%%
% Calculate Fourier Transform after receiver carrier multiplication

[fty,freqs]=fouriertransform(Vout,Fs);

%%
% Calculate Fourier Transform of the receiver output

[FTz,freqs]=fouriertransform(z,Fs);

%%
% Display these Fourier Transforms

figure(3)
subplot(3,1,1);
plot(freqs/1000, 20*log10(abs(ftsong)));
hold on
plot(freqs/1000, 20*log10(abs(ftx)),'r');
grid
legend('Message','Modulated','Location','Best')
xlabel('Frequency (kHz)');
title('Fourier Transform of Message and Modulated Signals')
axis([-Fs/2000 Fs/2000 -40 150])
subplot(3,1,2);
plot(freqs/1000, 20*log10(abs(fty)));
axis([-Fs/2000 Fs/2000 -40 100])
grid
xlabel('Frequency (kHz)');
title('FT of Receiver Signal After Multiplication with Carrier')
subplot(3,1,3)


plot(freqs/1000, 20*log10(abs(FTz)));
axis([-Fs/2000 Fs/2000 -40 100])
grid
xlabel('Frequency (kHz)')
title('FT of Receiver Demodulator Output')


%% Display the Original Song and the Receiver Output Segments
% They are hardly distinguishable!
figure(4)
plot(t(40000:190000)*1000,song(40000:190000))
hold on
plot(t(40000:190000)*1000,z(40000:190000),'r:')
grid
xlabel('Time (msec)');
ylabel('Waveform');
legend('Original','Received','Location','Best');

%% Display whole song single and demodulated version
figure(5)
plot(t*1000,song)
hold on
plot(t*1000,z,'r:')
grid
xlabel('Time (milisecond)');
ylabel('Waveform');
legend('Original','Received','Location','Best');



%% Play the demodulated sound
% Downsampling 
zlowrate=downsample(z,10);
% Listen to
sound(zlowrate,Fs/10);

% I could not figure out how to get exactly the same signal 

% I changed the RC constant to many different values, and this one is the
% best that I can find

% Since my demodulated signal includes noice, I hear a beep sound in the
% demodulated song, sorry :(


