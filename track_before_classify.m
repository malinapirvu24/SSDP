%% Initialization
clear; clc; close all;
rng(1); % set the seed to reproduce results

% Measurement grid (range bins)
r_min = 0;                         % [m]
r_max = 200;                       % [m] 
B = 150*10^6;                      % [Hz] randar operating bandwidth
range_res = 3*10^8/(2*B);          % 1m resolution cell
N = floor((r_max-r_min)/range_res);  % number of range bins from FFT
r = linspace(r_min, r_max, N).';   % bin centers (column)

% Time and motion (sensor moves with known constant velocity v_s)
v_s = 20;                 % sensor velocity [m/s] -> 72km/h 
T   = 0.5*range_res/v_s;  % sample period [s] -> we want the displacement per sample to be smaller than the range resolution ->2samples 

% (Augumented) State definition: s_k = [x_k; 1]
F = [1, -v_s*T;
     0,  1];
G = [1; 0];         % process noise acts on position only 

% Process noise
sigma_w = 0.5;      % [m] -> half a resolution cell
Qw = sigma_w^2;     % scalar -> only position is random

% Classes (PSF widths)
C = 2;
sigma_c = [1, 5];  % [m], class 1 narrow, class 2 wide -> expressed in terms of resolution cells

% Measurement noise and amplitude prior -> Gaussian 0 mean
sigma_n = 1;
sigma_a = 5;

% Particle filter settings
Np = 2000; % number of particles

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


%% Generate synthetic measurements

K = 100;                 % number of time steps (or measurements)
c_true = 2;              % true class (1=narrow, 2=wide)

% True state (augmented): s_k = [x_k; 1]
s_true = zeros(2, K);       % (x_1;1, x_2;1 ...) -> each s_k on each column
s_true(:,1) = [x0_mean; 1]; % true initial position -> chosen mean for convenience to initialize recursion

% Storage for measurements
z = zeros(N, K); % -> each measurement observes the response coming from all range bins

for k = 1:K  % propagation over time
    % Propagate true state (skip for k=1)
    if k > 1
        w = sigma_w * randn;                 % w_k -> N(0, Qw)
        s_true(:,k) = F*s_true(:,k-1) + G*w;  % s_{k} = F s_{k-1} + G w
    end

    xk = s_true(1,k); % first position in the augmented state

    % Class-dependent PSF (Gaussian-shaped, normalized)
    sig = sigma_c(c_true); % class dependent PSF
    hc = (1/(sqrt(2*pi)*sig)) * exp(-(r - xk).^2/(2*sig^2)); % energy spread across range bins

    % Draw amplitude and noise
    ak = sigma_a * randn;                     % a_k ~ N(0, sigma_a^2)
    nk = sigma_n * randn(N,1);                % n_k ~ N(0, sigma_n^2 I)

    % Measurement snapshot
    z(:,k) = ak*hc + nk;  % each column is the return enegy across all range bins
end


%% Parallel particle filters

% Storage for class-conditional likelihoods at each time
L_c = zeros(C, K);          % L_c(c,k) = p(z_k | Z_{k-1}, c) -> compute one each measurement
P_c_history = zeros(C, K);     % class probabilities over time

for k = 1:K  % -> for all measurements

    % Run PF for each class in parallel
    for c = 1:C % per clas, per drawn particle (via vector operation)

        % State prediction: propagate particles with state model 
        w = sigma_w * randn(1, Np);                 % scalar noise per particle
        s_particles(:,:,c) = F*s_particles(:,:,c) + G*w;

    end

    % Store current class probabilities (before update at time k)-> analysis
    P_c_history(:,k) = P_c;
    
    % Measurement update: PSF + likelihood (per class/per particle)
    for c = 1:C
        x = s_particles(1,:,c);          % 1 x Np, particle positions
        sig = sigma_c(c);

        % Build class-dependent PSF for all particles: H is N x Np
        H = 1/(sqrt(2*pi)*sig) * exp(-(r - x).^2/(2*sig^2));

        % Compute per-particle likelihood l_i = p(z_k | s_k^{i}, c)

        % Current measurement snapshot
        zk = z(:,k);

        % Constant over particles (for fixed k)
        zTz = zk.' * zk;

        % Particle-dependent inner products
        hTh = sum(H.^2, 1);          % 1 x Np, each is h^T h -> sum down the rows for each PSF (column by column)
        hTz = (H.' * zk).';          % 1 x Np, each is h^T z -> row vector result

        % Compute chi and xi (from lecture notes derivation)
        chi = (hTh/(sigma_n^2)) + (1/(sigma_a^2));          % 1 x Np
        xi  = (sigma_a^2/sigma_n^2) * hTh + 1;              % 1 x Np

        % Log-likelihood per particle (adapted from lecture notes derivation + error fix)
        lambda = -(N/2)*log(2*pi*sigma_n^2) ...
            - (zTz/(2*sigma_n^2)) ...
            - 0.5*log(xi) ...
            + 0.5 * ((hTz/(sigma_n^2)).^2) ./ chi;      % 1 x Np

        % Class-conditional log-likelihood L_c(use lecture trick)
        lam_max = max(lambda);  % across all particles, pick highest log-likelihood
        logL_c = log(mean(exp(lambda - lam_max))) + lam_max;
        L_c(c,k) = logL_c; % size C x K

        % Weight update for this class (use lecture trick)
        gamma_unnorm = 1/Np * exp(lambda - lam_max).';                  % Np x 1
        gamma(:,c) = gamma_unnorm / sum(gamma_unnorm);           % normalized Np x C

        % Check if the weights are correctly defined:
        % sum(gamma(:,c))

        % Resampling using the cumulative weights
        cdf = cumsum(gamma(:,c));     % cumulative weights, Np x 1 -> sum over rows
        u = rand(Np,1);               % uniform samples
        idx = arrayfun(@(x) find(cdf >= x, 1, 'first'), u); % find particle index using the cdf 

        s_particles(:,:,c) = s_particles(:,idx,c);     % resample particles 
        gamma(:,c) = (1/Np) * ones(Np,1);              % reset weights

    end


end




