% Training script -- SIM Navigation with DQN, SIM2 front-end + beam-scan
% diode readout + digital MAC Q-head.
%
% Architecture:
%   [Re(r); Im(r)]
%       -> SIM2 phase/propagation stack
%       -> last hidden M-plane field, M = M_x*M_y
%       -> fixed beam-scan propagation from each last-layer atom to one
%          beamformer phase center, including Sommerfeld path loss and
%          a directional receive-beam gain model
%       -> diodeReadoutLayer, used here as a hidden ReLU-like detector
%       -> digital MAC, implemented as fullyConnectedLayer(numAct) with
%          bias initialized to zero and bias learning disabled
%
% Physical interpretation of the readout:
%   The code evaluates all scanned beams in parallel for DQN training, but
%   this is mathematically equivalent to sequentially steering one beam,
%   passing the coherent detector output through one diode, storing the M
%   diode samples, and then applying a digital multiply-and-accumulate
%   block to produce the 9 Q-values, provided the channel/state is unchanged
%   over the scan.
%
% Main difference from the passive SIM2 9-port script:
%   This removes prop_M_to_Q -> output diode and replaces it with
%   prop_M_to_BF -> hidden diode -> trainable digital MAC.
%
% Required files in the MATLAB path:
%   Parameters.m
%   Calibration.m
%   simPhaseLayerCST.m
%   simPropagationLayer.m
%   diodeReadoutLayer.m
%   stepFunction_nav_CST_Aligned.m
%   resetFunction_nav_CST_Aligned.m

clc; clearvars; close all;
addingPathParentFolderByName('code');

Parameters;   % defines EnvPars, ActInfo, and base workspace variables

% -------------------------------------------------------------------------
% SIM2 hidden-plane size
% -------------------------------------------------------------------------
% For a 64-atom last hidden layer, use M_x = M_y = 8.
% To reproduce the previous Mx=25 geometry, set M_x = 25.
M_x = 7;
M_y = M_x;
M   = M_x * M_y;

%Definition of the FFT precision
N_x=4
N_y=N_x;
N=N_x*N_y;

EnvPars.N_x=N_x;
EnvPars.N_y=N_y;
EnvPars.N=N;

T_x=60
T_y=T_x; %accounting for a balanced error in the x an y axes of the Fourier transform
T=T_x.*T_y;

EnvPars.T = evalin('base','T');
EnvPars.T_x = evalin('base','T_x');
EnvPars.T_y = evalin('base','T_y');

% Meta-atom indexing — store inside EnvPars so closures don't depend on workspace
n = 1:EnvPars.N;
EnvPars.n_y = ceil(n ./ EnvPars.N_x);
EnvPars.n_x = n - (EnvPars.n_y - 1) .* EnvPars.N_x;

% Environment variables
n = (1:EnvPars.N);
n_y = ceil(n ./ EnvPars.N_x);
n_x = n - (n_y - 1) .* EnvPars.N_x;
n_psi = n;
n_psi_y = ceil(n_psi ./ EnvPars.N_x);
n_psi_x = n_psi - (n_y - 1) .* EnvPars.N_x;

%Recalculating the FFT
%Analytic FFT
% Kernel 2D-DFT == TO BE REPLACED BY SIM 1
G_func = @(n,n_psi) exp(-1i*2*pi*(n_psi_x(n_psi)-1)/EnvPars.N_x.*(n_x(n)-1)).* ...
    exp(-1i*2*pi*(n_psi_y(n_psi)-1)/EnvPars.N_y.*(n_y(n)-1));
[n_psi_grid, n_s_grid] = ndgrid(1:EnvPars.N, 1:EnvPars.N);
EnvPars.G = G_func(n_s_grid, n_psi_grid);

% Reclaculating the Upsilon matrix
EnvPars.U_func = @(n_, t_n_) exp(1i * ( ...
    -2*pi*(EnvPars.n_x(n_)-1) .* (EnvPars.t_x(t_n_)-1) / (EnvPars.N_x*EnvPars.T_x) ...
    -2*pi*(EnvPars.n_y(n_)-1) .* (EnvPars.t_y(t_n_)-1) / (EnvPars.N_y*EnvPars.T_y) ));

% CST SIM1 front-end. Keep commented unless those fields exist.
EnvPars.G      = EnvPars.G_CST;
EnvPars.U_func = EnvPars.U_func_CST;

Calibration;
%%


EnvPars.MaxEpisodes = EnvPars.N_cal * 50;

% -------------------------------------------------------------------------
% Observation override: aligned coherent Re/Im observation
% -------------------------------------------------------------------------
ObsInfo             = rlNumericSpec([2*EnvPars.N, 1]);
ObsInfo.Name        = 'observations';
ObsInfo.Description = 'Aligned coherent field [Re(r); Im(r)], no t_x/t_y in observation';
ObsInfo.LowerLimit  = -inf(2*EnvPars.N, 1);
ObsInfo.UpperLimit  =  inf(2*EnvPars.N, 1);

numObs = 2 * EnvPars.N;
numAct = EnvPars.n_actions;

% -------------------------------------------------------------------------
% SIM2 geometry: N-plane to M-plane and M-plane to M-plane propagation
% -------------------------------------------------------------------------
[xn, yn] = grid_coords_centered(N_x, N_y, d_x, d_y);
[xm, ym] = grid_coords_centered(M_x, M_y, s_x, s_y);


% First SIM2 propagation: N -> M
W0_SIM2 = zeros(M, N);
for m = 1:M
    for n = 1:N
        d = sqrt((xm(m)-xn(n))^2 + (ym(m)-yn(n))^2 + s_layer^2);
        cos_epsilon = s_layer / d;
        W0_SIM2(m,n) = (A_atom*cos_epsilon)/(2*pi*d^2) * ...
                       (1 - 1j*kappa*d) * exp(1j*kappa*d);
    end
end

% Hidden SIM2 propagation: M -> M. Reused as distinct matrix copies below.
W_M2M = zeros(M, M);
for m = 1:M
    for n = 1:M
        d = sqrt((xm(m)-xm(n))^2 + (ym(m)-ym(n))^2 + s_layer^2);
        cos_epsilon = s_layer / d;
        W_M2M(m,n) = (A_atom*cos_epsilon)/(2*pi*d^2) * ...
                     (1 - 1j*kappa*d) * exp(1j*kappa*d);
    end
end

fprintf('SIM2 hidden geometry: N=%d, M_x=%d, M_y=%d, M=%d\n', N, M_x, M_y, M);
fprintf('max(abs(W0_SIM2))=%.3e, max(abs(W_M2M))=%.3e\n', ...
        max(abs(W0_SIM2(:))), max(abs(W_M2M(:))));

% -------------------------------------------------------------------------
% Beamformer readout geometry: last M-plane -> M scanned measurements
% -------------------------------------------------------------------------
% One physical beamformer is placed a fixed distance away from the last
% SIM2 hidden plane. During scan k, it is steered/focused toward atom k.
% Row k of W_MBF contains the complex coupling from all last-layer atoms to
% the beamformer output while the beam points at atom k.
%
% Important: W_MBF includes both the Sommerfeld propagation/path loss and
% the receive-beam gain. Off-diagonal terms model finite beamwidth and
% sidelobe leakage. If you want an ideal scanner, set IncludeCrosstalk=false.
BF.Distance              = 20 * lambda;  % beamformer standoff from last layer
BF.x0                    = 0.0;          % beamformer phase-center x position
BF.y0                    = 0.0;          % beamformer phase-center y position
BF.PeakGain_dBi          = 18.0;         % realistic narrow-beam peak power gain
BF.HPBW_deg              = 12.0;         % full half-power beamwidth, degrees
BF.SideLobeLevel_dB      = -20.0;        % relative power sidelobe floor
BF.IncludeCrosstalk      = true;         % true: finite beam leakage, false: diagonal only
BF.PhaseReferenceTargets = true;         % make desired atom coefficient real positive
BF.NormalizeMatrix       = false;        % true only for debugging gradient scale

W_MBF = build_beam_scan_matrix(xm, ym, BF, A_atom, kappa);

fprintf('Beam-scan readout: W_MBF %dx%d, distance %.2f lambda, peak gain %.1f dBi, HPBW %.1f deg\n', ...
        size(W_MBF,1), size(W_MBF,2), BF.Distance/lambda, BF.PeakGain_dBi, BF.HPBW_deg);
fprintf('max(abs(W_MBF))=%.3e, diag/offdiag max ratio=%.2f dB\n', ...
        max(abs(W_MBF(:))), diag_to_offdiag_ratio_dB(W_MBF));

% -------------------------------------------------------------------------
% SIM2 layers
% -------------------------------------------------------------------------
load t_y_x.mat
[F_amp, phase_min_meas, phase_max_meas] = ...
    build_amplitude_interpolant(t_y_x_amp_dB, t_y_x_phase_deg);

sim2_layer1 = simPhaseLayerCST(N, F_amp, 'sim2_N');
prop_N_to_M = simPropagationLayer(W0_SIM2, 'prop_N_to_M');
sim2_layer2 = simPhaseLayerCST(M, F_amp, 'sim2_M1');

prop_M_to_M   = simPropagationLayer(W_M2M, 'prop_M_to_M_1');
sim2_layer3   = simPhaseLayerCST(M, F_amp, 'sim2_M2');

W_M2M_2       = W_M2M;
prop_M_to_M_2 = simPropagationLayer(W_M2M_2, 'prop_M_to_M_2');
sim2_layer4   = simPhaseLayerCST(M, F_amp, 'sim2_M3');

W_M2M_3       = W_M2M;
prop_M_to_M_3 = simPropagationLayer(W_M2M_3, 'prop_M_to_M_3');
sim2_layer5   = simPhaseLayerCST(M, F_amp, 'sim2_M4');

% Beam-scan readout as a fixed propagation layer.
% Input:  [Re(z_M); Im(z_M)]  size 2M x batch
% Output: [Re(y_scan); Im(y_scan)] size 2M x batch
prop_M_to_BF  = simPropagationLayer(W_MBF, 'prop_M_to_BF');

% Hidden diode: converts the M scanned coherent measurements to M
% nonnegative activations, analogous to the ReLU hidden layer in FC+ReLU+FC.
hidden_diode = diodeReadoutLayer('hidden_diode');

% Digital MAC Q-head. Bias is frozen at zero so the Q-head is only a
% multiply-and-accumulate operation: Q = W_MAC * h.
macNoBiasArgs = { ...
    'BiasInitializer',     'zeros', ...
    'BiasLearnRateFactor', 0, ...
    'BiasL2Factor',        0};

digital_mac = fullyConnectedLayer(numAct, ...
    'Name', 'digital_mac', ...
    macNoBiasArgs{:});

% -------------------------------------------------------------------------
% Critic network
% -------------------------------------------------------------------------
statePath = [
    featureInputLayer(numObs, 'Name', 'obs', 'Normalization', 'none')
    sim2_layer1
    prop_N_to_M
    sim2_layer2
    prop_M_to_M
    % sim2_layer3
    % prop_M_to_M_2
    %sim2_layer4
    %prop_M_to_M_3
    sim2_layer5
    prop_M_to_BF
    hidden_diode
    digital_mac];

criticNet = dlnetwork(layerGraph(statePath));

% Interface check before constructing/training the RL agent.
[obs0, logged0] = resetFunction_nav_CST_Aligned(EnvPars);
assert(isequal(size(obs0), [numObs, 1]), ...
    'Reset observation is %dx%d but expected %dx1.', size(obs0,1), size(obs0,2), numObs);

q0 = predict(criticNet, dlarray(single(obs0), 'CB'));
q0 = extractdata(q0);
assert(isequal(size(q0), [numAct, 1]), ...
    'Critic output is %dx%d but expected %dx1 Q-values.', size(q0,1), size(q0,2), numAct);

assertZeroBiases(criticNet, 'before training');

nPhaseParams = EnvPars.N + 4*M;
nMacWeights  = numAct * M;
fprintf('Critic OK: obs %dx1 -> hidden diode %dx1 -> Q %dx1\n', numObs, M, numAct);
fprintf('Trainable phase params: %d; trainable MAC weights: %d; MAC bias frozen at zero.\n', ...
        nPhaseParams, nMacWeights);

critic = rlVectorQValueFunction(criticNet, ObsInfo, ActInfo);

% -------------------------------------------------------------------------
% Environment
% -------------------------------------------------------------------------
env = rlFunctionEnv(ObsInfo, ActInfo, ...
      @(a,ls) stepFunction_nav_CST_Aligned(a, ls, EnvPars), ...
      @()     resetFunction_nav_CST_Aligned(EnvPars));

% -------------------------------------------------------------------------
% DQN agent
% -------------------------------------------------------------------------
agentOpts = rlDQNAgentOptions( ...
    'SampleTime',             1, ...
    'DiscountFactor',         EnvPars.DiscountFactor, ...
    'MiniBatchSize',          EnvPars.MiniBatchSize, ...
    'ExperienceBufferLength', EnvPars.ExperienceBufferLength, ...
    'TargetSmoothFactor',     EnvPars.TargetSmoothFactor, ...
    'CriticOptimizerOptions', rlOptimizerOptions('LearnRate', 5e-4));

agentOpts.EpsilonGreedyExploration.Epsilon      = 1.0;
agentOpts.EpsilonGreedyExploration.EpsilonMin   = 0.05;
agentOpts.EpsilonGreedyExploration.EpsilonDecay = EnvPars.EpsilonDecay;

agent = rlDQNAgent(critic, agentOpts);

% -------------------------------------------------------------------------
% Training
% -------------------------------------------------------------------------
trainOpts = rlTrainingOptions( ...
    'MaxEpisodes',                EnvPars.MaxEpisodes, ...
    'MaxStepsPerEpisode',         EnvPars.MaxStepsPerEpisode, ...
    'ScoreAveragingWindowLength', 50, ...
    'StopTrainingCriteria',       'AverageReward', ...
    'StopTrainingValue',          EnvPars.StopTrainingValue, ...
    'Verbose',                    true, ...
    'Plots',                      'none');

fprintf('=== SIM2 beam-scan + diode + digital MAC DQN training phase ===\n');
fprintf('N_cal=%d | MaxEpisodes=%d | MaxSteps=%d | n_actions=%d | obsDim=%d | M=%d\n', ...
        EnvPars.N_cal, EnvPars.MaxEpisodes, EnvPars.MaxStepsPerEpisode, EnvPars.n_actions, numObs, M);

trainingStats = train(agent, env, trainOpts);

% -------------------------------------------------------------------------
% Save
% -------------------------------------------------------------------------
save_dir = fullfile('..', 'Dataset');
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

save_path = fullfile(save_dir, sprintf( ...
    'dqn_agent_SIM2_BeamScanMAC_CST_2_layers_%d_atoms_L_%d_Nx_%d_Mx_%d_Tx_%d_Aligned.mat', M_x, L, EnvPars.N_x, EnvPars.M_x, EnvPars.T_x));

criticNet = getModel(getCritic(agent));
assertZeroBiases(criticNet, 'after training');

save(save_path, 'agent', 'trainingStats', 'criticNet', 'EnvPars', 'BF', 'W_MBF');
fprintf('Agent saved to %s\n', save_path);

% -------------------------------------------------------------------------
% Helper functions
% -------------------------------------------------------------------------
function addingPathParentFolderByName(targetName)
    currFolder = pwd;
    found = false;
    while true
        [parentFolder, currentName] = fileparts(currFolder);
        if strcmpi(currentName, targetName)
            found = true;
            break;
        end
        if isempty(parentFolder) || strcmp(currFolder, parentFolder)
            break;
        end
        currFolder = parentFolder;
    end
    if found
        addpath(genpath(currFolder));
        fprintf('Adding matlab path to: %s\n', currFolder);
    else
        error('Folder named "%s" not found in any parent directory.', targetName);
    end
end

function [x, y] = grid_coords_centered(Nx, Ny, dx, dy)
    Nloc = Nx * Ny;
    x = zeros(Nloc,1);
    y = zeros(Nloc,1);
    for n = 1:Nloc
        iy = ceil(n/Nx);
        ix = n - (iy-1)*Nx;
        x(n) = (ix - 1 - (Nx-1)/2) * dx;
        y(n) = ((Ny-1)/2 - (iy - 1)) * dy;
    end
end

function [F_amp, phase_min, phase_max] = build_amplitude_interpolant(mag_dB, phase_deg)
    phase_rad = deg2rad(phase_deg(:));
    phase_unwrapped = unwrap(phase_rad);
    mag_lin = 10.^(mag_dB(:)/20);

    [phase_sorted, idx] = sort(phase_unwrapped);
    mag_sorted = mag_lin(idx);

    phase_wrapped = mod(phase_sorted, 2*pi);
    [phase_wrapped, idx2] = sort(phase_wrapped);
    mag_sorted = mag_sorted(idx2);

    phase_min = phase_wrapped(1);
    phase_max = phase_wrapped(end);

    F_amp = griddedInterpolant(phase_wrapped, mag_sorted, 'pchip', 'nearest');
end

function W = build_beam_scan_matrix(xm, ym, BF, A_atom, kappa)
    % build_beam_scan_matrix
    %
    % Creates W, size M x M, where row k is the beamformer output when the
    % receive beam is steered to atom k and column m is the contribution of
    % last-layer atom m.
    %
    % The coefficient is:
    %   beam amplitude gain(k,m) * Sommerfeld path(atom m -> BF)
    % plus optional phase referencing so W(k,k) is real positive.

    Mloc = numel(xm);
    W = complex(zeros(Mloc, Mloc));

    rx = [BF.x0, BF.y0, BF.Distance];
    atoms = [xm(:), ym(:), zeros(Mloc,1)];

    % Unit vectors from receiver phase center toward each atom.
    u = zeros(Mloc, 3);
    dist = zeros(Mloc, 1);
    basePath = complex(zeros(Mloc,1));

    for m = 1:Mloc
        v_rx_to_atom = atoms(m,:) - rx;
        dist(m) = norm(v_rx_to_atom);
        u(m,:) = v_rx_to_atom ./ dist(m);

        % Sommerfeld coupling from atom to receiver phase center.
        % The normal separation is BF.Distance, so cos_epsilon models the
        % projected area/obliquity factor of the radiating atom.
        cos_epsilon = BF.Distance / dist(m);
        basePath(m) = (A_atom*cos_epsilon)/(2*pi*dist(m)^2) * ...
                      (1 - 1j*kappa*dist(m)) * exp(1j*kappa*dist(m));
    end

    peakAmpGain = 10^(BF.PeakGain_dBi/20);
    sidePowerFloor = 10^(BF.SideLobeLevel_dB/10);

    % Convert full HPBW to cosine-pattern exponent for power pattern.
    thetaHalf = deg2rad(BF.HPBW_deg/2);
    if thetaHalf <= 0 || thetaHalf >= pi/2
        error('BF.HPBW_deg must be in the interval (0, 180).');
    end
    qPower = log(0.5) / log(cos(thetaHalf));

    for k = 1:Mloc
        for m = 1:Mloc
            if BF.IncludeCrosstalk
                cosang = dot(u(k,:), u(m,:));
                cosang = max(min(cosang, 1), -1);
                mainPowerRel = max(cosang, 0)^qPower;
                powerRel = max(mainPowerRel, sidePowerFloor);
            else
                if k == m
                    powerRel = 1;
                else
                    powerRel = 0;
                end
            end

            ampGain = peakAmpGain * sqrt(powerRel);
            W(k,m) = ampGain * basePath(m);
        end

        % Coherent/homodyne reference for each scanned beam: make the target
        % atom coefficient W(k,k) real and positive before diode detection.
        if BF.PhaseReferenceTargets && abs(W(k,k)) > 0
            W(k,:) = W(k,:) * exp(-1j*angle(W(k,k)));
        end
    end

    if BF.NormalizeMatrix
        scale = max(abs(W(:)));
        if scale > 0
            W = W / scale;
        end
    end
end

function r_dB = diag_to_offdiag_ratio_dB(W)
    d = abs(diag(W));
    mask = true(size(W));
    mask(1:size(W,1)+1:end) = false;
    offMax = max(abs(W(mask)));
    if offMax == 0
        r_dB = Inf;
    else
        r_dB = 20*log10(max(d) / offMax);
    end
end

function assertZeroBiases(net, tag)
    L = net.Learnables;
    isBias = strcmp(string(L.Parameter), 'Bias');

    if ~any(isBias)
        fprintf('%s: no Bias parameters found.\n', tag);
        return;
    end

    rows = find(isBias).';
    for idx = rows
        b = extractdata(L.Value{idx});
        maxAbsBias = max(abs(b(:)));
        fprintf('%s: layer %s Bias max abs = %.3e\n', ...
            tag, string(L.Layer(idx)), maxAbsBias);
        assert(maxAbsBias < 1e-10, ...
            'Bias in layer %s is not zero after %s.', string(L.Layer(idx)), tag);
    end
end
