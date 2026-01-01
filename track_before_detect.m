%% Initialization
clear; clc; close all;
rng(1); % set the seed to reproduce results

% Measurement grid (range bins)
r_min = 0;          % [m]
r_max = 200;        % [m] 
B = 150*10^6;       % [Hz] randar operating bandwidth
range_res = 3*10^8/(2*B); % 1m resolution cell
N = floor(r_max-r_min)/range_res;            % number of range bins from FFT
r = linspace(r_min, r_max, N).';   % bin centers (column)

% Time and motion (sensor moves with known constant velocity v_s)
v_s = 20;             % sensor velocity [m/s] -> 72km/h 
T   = 0.5*range_res/v_s;  % sample period [s] -> we want the displacement per sample to be smaller than the range resolution ->2samples 

% (Augumented) State definition: s_k = [x_k; 1]
F = [1, -v_s*T;
     0,  1];
G = [1; 0];         % process noise acts on position only 

% Process noise
sigma_w = 0.5;      % [m] -> half a resolution cellae
Qw = sigma_w^2;     % scalar -> only position is random

% Classes (PSF widths)
C = 2;
sigma_c = [1, 5];  % [m], class 1 narrow, class 2 wide -> expressed in terms of resolution cells

% Measurement noise and amplitude prior -> Gaussian 0 mean
sigma_n = 1;
sigma_a = 5;

% Particle filter settings
Np = 2000;

% Prior on initial relative position x0 (prior on initial state) -> assume
% Gaussian distribution
x0_mean = 150;      % [m]
x0_std  = 10;       % [m]

% Initialize particles and weights for each class
s_particles = zeros(2, Np, C);   % allocate particles for each class 
gamma = zeros(Np, C);            % allocate weights for each particle, per class

for c = 1:C
    x0 = x0_mean + x0_std*randn(1, Np);      % draw Np initial position samples-> Gaussian distribution
    s_particles(:,:,c) = [x0; ones(1, Np)];  % set initial state 
    gamma(:,c) = (1/Np)*ones(Np,1);          % set uniform weights
end

% Class prior probabilities
P_c = (1/C)*ones(C,1);   % maximum uncertainty -> equally probable classes


