%% Training script — SIM Navigation with DQN, SIM2 Q-network variant
% Requires Parameters.m to have been run first (this script calls it)
%
% Built incrementally with Claude -- all four parts done, gradient-
% verified per-layer (see SIM2_GradientCheck_*.m, run separately):
%   Part 1 [DONE] -- SIM2 geometry & propagation matrices (W0_SIM2, W_M9)
%   Part 2 [DONE] -- custom layers: simPhaseLayerCST (N), simPhaseLayerCST (M)
%   Part 3 [DONE] -- realToComplexLayer, diodeReadoutLayer, dlnetwork assembly
%   Part 4 [DONE] -- ObsInfo override (2N-dim, local to this script),
%                     stepFunction_nav_CST.m / resetFunction_nav.m updated
%                     to emit [Re(r);Im(r)] (t_x,t_y tracked in silico,
%                     not observed)
%
% REVISED again after a real platform limitation: MATLAB custom layers
% cannot return complex values from predict()/forward() (confirmed via
% MathWorks doc, not just inferred from the error). Every layer
% (simPhaseLayerCST, simPropagationLayer, diodeReadoutLayer) now operates
% on REAL-STACKED [Re;Im] representations throughout -- see each layer
% file for the re-derived math. to_complex is GONE entirely: with
% everything real-stacked end to end, the observation [Re(r);Im(r)] feeds
% sim2_layer1 directly. realImagToComplexLayer.m and realToComplexLayer.m
% are both now historical/unused. Gradient checks were REDONE (not just
% re-run) against the new math -- see SIM2_GradientCheck_PhaseLayer.m,
% SIM2_GradientCheck_PropagationLayer.m, SIM2_GradientCheck_DiodeReadout.m.
%
% NOT yet done, separate from this architecture: EnvPars.U_func (v0) and
% G still use the idealized analytic models -- the CST amplitude-phase
% coupling on the SIM1 input layer, and loading the actual trained
% (CST-aware) G from SIM_Training_CST_SingleZeta_Parallel.m instead of
% the closed-form DFT kernel, are both still open from earlier in this
% project and independent of the SIM2 work above.
%
% STILL NEEDED before trusting a real training run:
%   - End-to-end check: does criticNet/predict() actually run on a real
%     observation without shape/format errors? (per-layer checks passed,
%     but the full chain hasn't been exercised together yet)
%   - Capacity sanity check: 241 trainable phases vs. the old FC
%     network's ~20k+ weights -- worth comparing against the original
%     training curve (Fig. 6) once this trains, not just checking that
%     it learns at all

clc; clear all; close all;
addingPathParentFolderByName('code');
Parameters;   % loads all base variables into workspace and EnvPars

%Update the G matrix to use the one computed with CST
EnvPars.G = EnvPars.G_CST;   % SIM-2 uses the CST-realistic SIM-1 front-end
EnvPars.U_func = EnvPars.U_func_CST;


Calibration

% In A_Training_Navigation — after Calibration runs
EnvPars.MaxEpisodes = EnvPars.N_cal * 400;

%% ── PART 4: ObsInfo OVERRIDE ────────────────────────────────────────────
% SIM2 observes only the N-dim field amplitude |r| -- never t_x/t_y.
% Parameters.m's own ObsInfo (N+2, used by the original FC script) is
% deliberately left untouched -- overriding it in place would silently
% break A_Training_Navigation_225_neurons_1_ReLU_CST.m. This local
% override shadows it for the rest of THIS script only.
%
% Must match what stepFunction_nav_CST.m / resetFunction_nav.m now
% actually return (observation = [real(r); imag(r)], 2N-dimensional,
% CONVENTION: first N rows real, next N rows imaginary).
ObsInfo             = rlNumericSpec([2*EnvPars.N, 1]);
ObsInfo.Name        = 'observations';
ObsInfo.Description = 'Coherent field [Re(r);Im(r)] (SIM1 output, phase preserved) -- t_x,t_y tracked in silico only, not observed';
ObsInfo.LowerLimit  = -inf(2*EnvPars.N, 1);   % real/imag parts can be negative -- not an amplitude anymore
ObsInfo.UpperLimit  =  inf(2*EnvPars.N, 1);

%% ── PART 1: SIM2 GEOMETRY & PROPAGATION MATRICES ──────────────────────────
% Computed fresh every run (not loaded from a saved .mat) so it always
% matches whatever is currently in Parameters.m -- avoids training against
% stale geometry if N_x/M_x/s_x/s_layer/etc. get tuned later and this
% block isn't rerun.
%
%   W0_SIM2 : N=16  -> M=225   (SIM2 layer 1 -> layer 2)
%   W_M9    : M=225 -> 9       (SIM2 layer 2 -> layer 3, passive 3x3 funnel)
%
% Same Sommerfeld kernel as SIM1's W0/W{l} in
% SIM_Training_CST_SingleZeta_Parallel.m, evaluated on different facing
% grids -- no new physics introduced, per our discussion.

% N-grid: SIM1's existing output plane = SIM2's input plane (same coords)
[xn, yn] = grid_coords_centered(N_x, N_y, d_x, d_y);

% M-grid: SIM2 layer 2 (15x15), same spacing as SIM1's intermediate layers
[xm, ym] = grid_coords_centered(M_x, M_y, s_x, s_y);

% 3x3 output grid: SIM2 layer 3 -- 9 ports, one per navigation action
Q_x = 3;
Q_y = 3;
[xq, yq] = grid_coords_centered(Q_x, Q_y, s_x, s_y);

M = M_x * M_y;
W0_SIM2 = zeros(M, N);
for m = 1:M
    for n = 1:N
        d = sqrt((xm(m)-xn(n))^2 + (ym(m)-yn(n))^2 + s_layer^2);
        cos_epsilon = s_layer/d;
        W0_SIM2(m,n) = (A_atom*cos_epsilon)/(2*pi*d^2) * (1-1j*kappa*d) * exp(1j*kappa*d);
    end
end

Q = Q_x * Q_y;   % = 9, must equal EnvPars.n_actions
W_M9 = zeros(Q, M);
for q = 1:Q
    for m = 1:M
        d = sqrt((xq(q)-xm(m))^2 + (yq(q)-ym(m))^2 + s_layer^2);
        cos_epsilon = s_layer/d;
        W_M9(q,m) = (A_atom*cos_epsilon)/(2*pi*d^2) * (1-1j*kappa*d) * exp(1j*kappa*d);
    end
end

assert(Q == EnvPars.n_actions, ...
    'Output port count (%d) does not match EnvPars.n_actions (%d) -- check the 3x3 grid against the actual action count before proceeding.', ...
    Q, EnvPars.n_actions);

fprintf('SIM2 geometry: W0_SIM2 %dx%d (N=%d->M=%d), W_M9 %dx%d (M=%d->%d ports)\n', ...
    size(W0_SIM2,1), size(W0_SIM2,2), N, M, size(W_M9,1), size(W_M9,2), M, Q);
fprintf('max(abs(W0_SIM2))=%.3e , max(abs(W_M9))=%.3e\n', max(abs(W0_SIM2(:))), max(abs(W_M9(:))));

%% ── PART 2: SIM2 CUSTOM LAYERS ──────────────────────────────────────────────
% Custom layers live in simPhaseLayerCST.m and simPropagationLayer.m
% (separate files -- MATLAB requires one classdef per file). Place both
% in your 'code' folder so addingPathParentFolderByName('code') finds them.
%
% IMPORTANT: run SIM2_GradientCheck_PhaseLayer.m once, separately, before
% trusting this in actual training -- it verifies simPhaseLayerCST's
% custom backward() against a brute-force finite difference on the loss.

% ----- CST amplitude-phase coupling (same source as SIM1's training) -----
load t_y_x.mat
[F_amp, phase_min_meas, phase_max_meas] = build_amplitude_interpolant(t_y_x_amp_dB, t_y_x_phase_deg);

% ----- SIM2 layers -----
sim2_layer1 = simPhaseLayerCST(N, F_amp, 'sim2_N');   % N=16,  learnable, CST-coupled
prop_N_to_M = simPropagationLayer(W0_SIM2, 'prop_N_to_M');
sim2_layer2 = simPhaseLayerCST(M, F_amp, 'sim2_M');   % M=225, learnable, CST-coupled
prop_M_to_Q = simPropagationLayer(W_M9, 'prop_M_to_Q');

% Layer 3 (the 9-port output) is confirmed to be the identity: Theta3=0,
% no F_amp lookup, so diag(exp(i*0)) = I. It contributes nothing beyond
% what prop_M_to_Q (W_M9, which already carries the Sommerfeld
% attenuation) does. No object instantiated for it -- W_M9's output feeds
% the readout directly in Part 3.

%% ── PART 3+4: DIODE READOUT + dlnetwork ASSEMBLY ───────────────────────────
% REVISED: MATLAB custom layers cannot return complex values from
% predict()/forward() ("Define Custom Deep Learning Layers" doc). Every
% layer now operates on REAL-STACKED [Re;Im] representations throughout
% -- see the layer files for the math. realImagToComplexLayer.m and
% realToComplexLayer.m are BOTH now unused/historical -- with everything
% real-stacked end to end, there's no separate "convert to complex" step
% needed: the observation [Re(r);Im(r)] feeds sim2_layer1 directly.

readout_layer = diodeReadoutLayer('readout');

statePath = [
    featureInputLayer(2*EnvPars.N, 'Name', 'obs', 'Normalization', 'none')
    sim2_layer1
    prop_N_to_M
    sim2_layer2
    prop_M_to_Q
    readout_layer];

criticNet = dlnetwork(layerGraph(statePath));
critic    = rlVectorQValueFunction(criticNet, ObsInfo, ActInfo);

%% ── 5. ENVIRONMENT ───────────────────────────────────────────────────────
env = rlFunctionEnv(ObsInfo, ActInfo, ...
      @(a,ls) stepFunction_nav_CST(a, ls, EnvPars), ...
      @()     resetFunction_nav_CST(EnvPars));

%% ── 7. AGENT ─────────────────────────────────────────────────────────────
agentOpts = rlDQNAgentOptions(...
    'SampleTime',                    1, ...
    'DiscountFactor',                EnvPars.DiscountFactor, ...
    'MiniBatchSize',                 EnvPars.MiniBatchSize, ...
    'ExperienceBufferLength',        EnvPars.ExperienceBufferLength, ...
    'TargetSmoothFactor',            EnvPars.TargetSmoothFactor, ...
    'CriticOptimizerOptions',        rlOptimizerOptions('LearnRate', 5e-4));

agentOpts.EpsilonGreedyExploration.Epsilon    = 1.0;
agentOpts.EpsilonGreedyExploration.EpsilonMin = 0.05;
agentOpts.EpsilonGreedyExploration.EpsilonDecay = EnvPars.EpsilonDecay;

agent = rlDQNAgent(critic, agentOpts);

%% ── 8. TRAINING ──────────────────────────────────────────────────────────
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',          EnvPars.MaxEpisodes, ...
    'MaxStepsPerEpisode',   EnvPars.MaxStepsPerEpisode, ...
    'ScoreAveragingWindowLength', 50, ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue',    EnvPars.StopTrainingValue, ...
    'Verbose',              true, ...
    'Plots',                'none');

fprintf('=== Training phase ===\n');
fprintf('N_cal=%d  |  MaxEpisodes=%d  |  MaxSteps=%d  |  n_actions=%d\n', ...
        N_cal, EnvPars.MaxEpisodes, EnvPars.MaxStepsPerEpisode, EnvPars.n_actions);

trainingStats = train(agent, env, trainOpts);

%% ── 9. SAVE ──────────────────────────────────────────────────────────────
% NOTE: filename no longer references "Neurons_per_layer" -- that variable
% (25*25=625) never matched any actual layer width (144=T_x*T_y, 225=M_x*M_y)
% even in the original script, and SIM2 doesn't have a single "neuron count"
% analog anyway (241 trainable phases across two layers + 1 frozen layer).
% Original line was also missing a closing paren on fullfile(); fixed here.
save_path = fullfile('..', 'Dataset', 'dqn_agent_navigation_225_neurons_CST.mat');
criticNet = getModel(getCritic(agent));
save(save_path, 'agent', 'trainingStats', 'criticNet', 'EnvPars');
fprintf('Agent saved to %s\n', save_path);

%% ======================= Functions =======================
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
    N = Nx*Ny;
    x = zeros(N,1);
    y = zeros(N,1);
    for n = 1:N
        iy = ceil(n/Nx);
        ix = n - (iy-1)*Nx;
        x(n) = (ix - 1 - (Nx-1)/2) * dx ;
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
