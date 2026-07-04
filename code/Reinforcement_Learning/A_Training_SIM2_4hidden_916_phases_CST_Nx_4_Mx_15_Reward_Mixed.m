%[text] # Training script — SIM Navigation with DQN, SIM2 Q-network (4 hidden M-layers)
%[text] VARIANT with FOUR hidden M-layers. Architecture: N -\> \[propN-\>M\] -\> M -\> \[propM-\>M\] -\> M -\> \[propM-\>M\] -\> M -\> \[propM-\>M\] -\> M -\> \[propM-\>9\] -\> diode 916 trainable phases (16+225+225+225+225) vs 691 (3-hidden). Saves to dqn\_agent\_navigation\_SIM2\_4hidden.mat -- all prior results untouched.
%[text] NOTE on the matrix-copy pattern: each M-\>M propagation hop MUST use its own distinct matrix object (W\_M2M, W\_M2M\_2, W\_M2M\_3, W\_M2M\_4) even though all are physically identical -- dlnetwork collapses structurally- identical layers during construction, silently dropping the downstream phase layer from the graph. This was confirmed empirically in the 3-hidden run. All four hops use explicit copies.
%[text] Requires Parameters.m to have been run first (this script calls it)
%[text] Built incrementally with Claude -- all four parts done, gradient- verified per-layer (see SIM2\_GradientCheck\_\*.m, run separately): Part 1 \[DONE\] -- SIM2 geometry & propagation matrices (W0\_SIM2, W\_M9) Part 2 \[DONE\] -- custom layers: simPhaseLayerCST (N), simPhaseLayerCST (M) Part 3 \[DONE\] -- realToComplexLayer, diodeReadoutLayer, dlnetwork assembly Part 4 \[DONE\] -- ObsInfo override (2N-dim, local to this script), stepFunction\_nav\_CST.m / resetFunction\_nav.m updated to emit \[Re(r);Im(r)\] (t\_x,t\_y tracked in silico, not observed)
%[text] REVISED again after a real platform limitation: MATLAB custom layers cannot return complex values from predict()/forward() (confirmed via MathWorks doc, not just inferred from the error). Every layer (simPhaseLayerCST, simPropagationLayer, diodeReadoutLayer) now operates on REAL-STACKED \[Re;Im\] representations throughout -- see each layer file for the re-derived math. to\_complex is GONE entirely: with everything real-stacked end to end, the observation \[Re(r);Im(r)\] feeds sim2\_layer1 directly. realImagToComplexLayer.m and realToComplexLayer.m are both now historical/unused. Gradient checks were REDONE (not just re-run) against the new math -- see SIM2\_GradientCheck\_PhaseLayer.m, SIM2\_GradientCheck\_PropagationLayer.m, SIM2\_GradientCheck\_DiodeReadout.m.
%[text] NOT yet done, separate from this architecture: EnvPars.U\_func (v0) and G still use the idealized analytic models -- the CST amplitude-phase coupling on the SIM1 input layer, and loading the actual trained (CST-aware) G from SIM\_Training\_CST\_SingleZeta\_Parallel.m instead of the closed-form DFT kernel, are both still open from earlier in this project and independent of the SIM2 work above.
%[text] STILL NEEDED before trusting a real training run: - End-to-end check: does criticNet/predict() actually run on a real observation without shape/format errors? (per-layer checks passed, but the full chain hasn't been exercised together yet) - Capacity sanity check: 241 trainable phases vs. the old FC network's ~20k+ weights -- worth comparing against the original training curve (Fig. 6) once this trains, not just checking that it learns at all
clc; clear all; close all;
addingPathParentFolderByName('code'); %[output:82f07755]
Parameters;   % loads all base variables into workspace and EnvPars %[output:40be3405] %[output:858cf39b] %[output:90d26340] %[output:17e0a73a] %[output:1d5b510b] %[output:5736db59] %[output:43b040c5] %[output:40fb16fc] %[output:0c12bb1c] %[output:23b20473]

%Update the number of atoms
M_x=15
M_y=M_x;
M=M_x*M_y;

%Update the G matrix to use the one computed with CST
% EnvPars.G = EnvPars.G_CST;   % SIM-2 uses the CST-realistic SIM-1 front-end
% EnvPars.U_func = EnvPars.U_func_CST;

Calibration %[output:9d8cb499]

% In A_Training_Navigation — after Calibration runs
EnvPars.MaxEpisodes = EnvPars.N_cal * 200;
%%
%[text] ## ── PART 4: ObsInfo OVERRIDE ────────────────────────────────────────────
%[text] SIM2 observes only the N-dim field amplitude `r` -- never t\_x/t\_y. Parameters.m's own ObsInfo (N+2, used by the original FC script) is deliberately left untouched -- overriding it in place would silently break A\_Training\_Navigation\_225\_neurons\_1\_ReLU\_CST.m. This local override shadows it for the rest of THIS script only.
%[text] Must match what stepFunction\_nav\_CST.m / resetFunction\_nav.m now actually return (observation = \[real(r); imag(r)\], 2N-dimensional, CONVENTION: first N rows real, next N rows imaginary).
ObsInfo             = rlNumericSpec([2*EnvPars.N, 1]);
ObsInfo.Name        = 'observations';
ObsInfo.Description = 'Coherent field [Re(r);Im(r)] (SIM1 output, phase preserved) -- t_x,t_y tracked in silico only, not observed';
ObsInfo.LowerLimit  = -inf(2*EnvPars.N, 1);   % real/imag parts can be negative -- not an amplitude anymore
ObsInfo.UpperLimit  =  inf(2*EnvPars.N, 1);
%%
%[text] ## ── PART 1: SIM2 GEOMETRY & PROPAGATION MATRICES ──────────────────────────
%[text] Computed fresh every run (not loaded from a saved .mat) so it always matches whatever is currently in Parameters.m -- avoids training against stale geometry if N\_x/M\_x/s\_x/s\_layer/etc. get tuned later and this block isn't rerun.
%[text] ```matlabCodeExample
%[text] W0_SIM2 : N=16  -> M=225   (SIM2 layer 1 -> layer 2)
%[text] W_M9    : M=225 -> 9       (SIM2 layer 2 -> layer 3, passive 3x3 funnel)
%[text] ```
%[text] Same Sommerfeld kernel as SIM1's W0/W{l} in SIM\_Training\_CST\_SingleZeta\_Parallel.m, evaluated on different facing grids -- no new physics introduced, per our discussion.
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

fprintf('SIM2 geometry: W0_SIM2 %dx%d (N=%d->M=%d), W_M9 %dx%d (M=%d->%d ports)\n', ... %[output:group:85682b55] %[output:0fb25d59]
    size(W0_SIM2,1), size(W0_SIM2,2), N, M, size(W_M9,1), size(W_M9,2), M, Q); %[output:group:85682b55] %[output:0fb25d59]
fprintf('max(abs(W0_SIM2))=%.3e , max(abs(W_M9))=%.3e\n', max(abs(W0_SIM2(:))), max(abs(W_M9(:)))); %[output:8083eefc]
%%
%[text] ## ── PART 2: SIM2 CUSTOM LAYERS ──────────────────────────────────────────────
%[text] Custom layers live in simPhaseLayerCST.m and simPropagationLayer.m (separate files -- MATLAB requires one classdef per file). Place both in your 'code' folder so addingPathParentFolderByName('code') finds them.
%[text] IMPORTANT: run SIM2\_GradientCheck\_PhaseLayer.m once, separately, before trusting this in actual training -- it verifies simPhaseLayerCST's custom backward() against a brute-force finite difference on the loss.
% ----- CST amplitude-phase coupling (same source as SIM1's training) -----
load t_y_x.mat
[F_amp, phase_min_meas, phase_max_meas] = build_amplitude_interpolant(t_y_x_amp_dB, t_y_x_phase_deg);

% ----- SIM2 layers -----
sim2_layer1 = simPhaseLayerCST(N, F_amp, 'sim2_N');   % N=16,  learnable, CST-coupled
prop_N_to_M = simPropagationLayer(W0_SIM2, 'prop_N_to_M');
sim2_layer2 = simPhaseLayerCST(M, F_amp, 'sim2_M');   % M=225, learnable, CST-coupled (hidden 1)

% ----- SECOND hidden M-layer (this variant's only architectural change) --
% A new M->M propagation hop carries the field from the first M-plane to
% the second, then a second learnable M-layer. Same s_layer standoff and
% same xm,ym grid as SIM-1's intermediate stack, so W_M2M is the exact
% W{l} construction from SIM_Training_CST_SingleZeta_Parallel.m.
% Capacity: 16+225+225 = 466 trainable phases (vs 241 in the single-M
% baseline). Name 'sim2_M2' is distinct from 'sim2_M' -- dlnetwork
% requires unique layer names.
W_M2M = zeros(M, M);
for m = 1:M
    for n = 1:M
        d = sqrt((xm(m)-xm(n))^2 + (ym(m)-ym(n))^2 + s_layer^2);
        cos_epsilon = s_layer/d;
        W_M2M(m,n) = (A_atom*cos_epsilon)/(2*pi*d^2) * (1-1j*kappa*d) * exp(1j*kappa*d);
    end
end
prop_M_to_M = simPropagationLayer(W_M2M, 'prop_M_to_M');
sim2_layer3 = simPhaseLayerCST(M, F_amp, 'sim2_M2');  % M=225, learnable (hidden 2)

% ----- THIRD hidden M-layer --------------------------------------------
% Physically the same M->M propagation as prop_M_to_M (identical geometry),
% but it MUST be a distinct matrix object, not the same W_M2M handle:
% dlnetwork collapses structurally-identical consecutive layers during
% construction, which silently dropped sim2_M3 from the graph when both
% hops shared one matrix. A copy keeps the physics identical while making
% the two layers distinct graph nodes.
W_M2M_2 = W_M2M;   % explicit distinct copy -- same values, separate object
prop_M_to_M_2 = simPropagationLayer(W_M2M_2, 'prop_M_to_M_2');
sim2_layer4   = simPhaseLayerCST(M, F_amp, 'sim2_M3');  % M=225, learnable (hidden 3)

% ----- FOURTH hidden M-layer -------------------------------------------
W_M2M_3 = W_M2M;   % distinct copy required (same reason as W_M2M_2)
prop_M_to_M_3 = simPropagationLayer(W_M2M_3, 'prop_M_to_M_3');
sim2_layer5   = simPhaseLayerCST(M, F_amp, 'sim2_M4');  % M=225, learnable (hidden 4)

prop_M_to_Q = simPropagationLayer(W_M9, 'prop_M_to_Q');

% Layer 3 (the 9-port output) is confirmed to be the identity: Theta3=0,
% no F_amp lookup, so diag(exp(i*0)) = I. It contributes nothing beyond
% what prop_M_to_Q (W_M9, which already carries the Sommerfeld
% attenuation) does. No object instantiated for it -- W_M9's output feeds
% the readout directly in Part 3.
%%
%[text] ## ── PART 3+4: DIODE READOUT + dlnetwork ASSEMBLY ───────────────────────────
%[text] REVISED: MATLAB custom layers cannot return complex values from predict()/forward() ("Define Custom Deep Learning Layers" doc). Every layer now operates on REAL-STACKED \[Re;Im\] representations throughout -- see the layer files for the math. realImagToComplexLayer.m and realToComplexLayer.m are BOTH now unused/historical -- with everything real-stacked end to end, there's no separate "convert to complex" step needed: the observation \[Re(r);Im(r)\] feeds sim2\_layer1 directly.
readout_layer = diodeReadoutLayer('readout');

statePath = [
    featureInputLayer(2*EnvPars.N, 'Name', 'obs', 'Normalization', 'none')
    sim2_layer1
    prop_N_to_M
    sim2_layer2
    prop_M_to_M
    sim2_layer3
    prop_M_to_M_2
    sim2_layer4
    prop_M_to_M_3
    sim2_layer5
    prop_M_to_Q
    readout_layer];

criticNet = dlnetwork(layerGraph(statePath));
critic    = rlVectorQValueFunction(criticNet, ObsInfo, ActInfo);
%%
%[text] ## ── 5. ENVIRONMENT ───────────────────────────────────────────────────────
env = rlFunctionEnv(ObsInfo, ActInfo, ...
      @(a,ls) stepFunction_nav_CST_Reward_Mixed(a, ls, EnvPars), ...
      @()     resetFunction_nav_CST_Reward_Mixed(EnvPars));
%%
%[text] ## ── 7. AGENT ─────────────────────────────────────────────────────────────
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
%%
%[text] ## ── 8. TRAINING ──────────────────────────────────────────────────────────
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',          EnvPars.MaxEpisodes, ...
    'MaxStepsPerEpisode',   EnvPars.MaxStepsPerEpisode, ...
    'ScoreAveragingWindowLength', 50, ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue',    EnvPars.StopTrainingValue, ...
    'Verbose',              true, ...
    'Plots',                'none', ...
    'SaveAgentCriteria',    'EpisodeCount', ...
    'SaveAgentValue',       500, ...
    'SaveAgentDirectory',   fullfile('..', 'Dataset', 'SIM_Nx_4_Mx_15_Reward_Mixed_checkpoints'));

fprintf('=== Training phase ===\n'); %[output:6854e8b0]
fprintf('N_cal=%d  |  MaxEpisodes=%d  |  MaxSteps=%d  |  n_actions=%d\n', ... %[output:group:4ce3be05] %[output:48fdfca4]
        N_cal, EnvPars.MaxEpisodes, EnvPars.MaxStepsPerEpisode, EnvPars.n_actions); %[output:group:4ce3be05] %[output:48fdfca4]

trainingStats = train(agent, env, trainOpts); %[output:58b08826]
%%
%[text] ## ── 9. SAVE ──────────────────────────────────────────────────────────────
%[text] NOTE: filename no longer references "Neurons\_per\_layer" -- that variable (25\*25=625) never matched any actual layer width (144=T\_x\*T\_y, 225=M\_x\*M\_y) even in the original script, and SIM2 doesn't have a single "neuron count" analog anyway (241 trainable phases across two layers + 1 frozen layer). Original line was also missing a closing paren on fullfile(); fixed here.
save_path = fullfile('..', 'Dataset', 'dqn_agent_navigation_4_hidden_916_phases_CST_Nx_4_Mx_15_Reward_Mixed.mat');
criticNet = getModel(getCritic(agent));
save(save_path, 'agent', 'trainingStats', 'criticNet', 'EnvPars');
fprintf('Agent saved to %s\n', save_path);
%%
%[text] ## ======================= Functions =======================
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

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright"}
%---
%[output:82f07755]
%   data: {"dataType":"text","outputData":{"text":"Adding matlab path to: G:\\My Drive\\Work\\Research\\SIM\\code\n","truncated":false}}
%---
%[output:40be3405]
%   data: {"dataType":"textualVariable","outputData":{"name":"total_iteration","value":"1"}}
%---
%[output:858cf39b]
%   data: {"dataType":"text","outputData":{"text":"Wireless packet type: SC\n","truncated":false}}
%---
%[output:90d26340]
%   data: {"dataType":"textualVariable","outputData":{"name":"N","value":"16"}}
%---
%[output:17e0a73a]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_x","value":"15"}}
%---
%[output:1d5b510b]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_y","value":"15"}}
%---
%[output:5736db59]
%   data: {"dataType":"matrix","outputData":{"columns":6,"name":"zeta","rows":1,"type":"double","value":[["0.9850","0.9860","0.9870","0.9880","0.9890","0.9900"]]}}
%---
%[output:43b040c5]
%   data: {"dataType":"textualVariable","outputData":{"name":"T_coh","value":"0.0038"}}
%---
%[output:40fb16fc]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_packets_coh","value":"12"}}
%---
%[output:0c12bb1c]
%   data: {"dataType":"textualVariable","outputData":{"name":"SNR_dB","value":"36.5005"}}
%---
%[output:23b20473]
%   data: {"dataType":"text","outputData":{"text":"Loaded CST SIM-1 G_CST. Deviation from analytic G: 1.806%n","truncated":false}}
%---
%[output:9d8cb499]
%   data: {"dataType":"text","outputData":{"text":"=== Calibration phase ===\nGrid: 10 x 10 = 100 calibration positions\nCalibration complete.\n\n","truncated":false}}
%---
%[output:0fb25d59]
%   data: {"dataType":"text","outputData":{"text":"SIM2 geometry: W0_SIM2 225x16 (N=16->M=225), W_M9 9x225 (M=225->9 ports)\n","truncated":false}}
%---
%[output:8083eefc]
%   data: {"dataType":"text","outputData":{"text":"max(abs(W0_SIM2))=2.628e-01 , max(abs(W_M9))=2.748e-01\n","truncated":false}}
%---
%[output:6854e8b0]
%   data: {"dataType":"text","outputData":{"text":"=== Training phase ===\n","truncated":false}}
%---
%[output:48fdfca4]
%   data: {"dataType":"text","outputData":{"text":"N_cal=100  |  MaxEpisodes=20000  |  MaxSteps=289  |  n_actions=9\n","truncated":false}}
%---
%[output:58b08826]
%   data: {"dataType":"text","outputData":{"text":"Episode:   1\/20000 | Episode reward:    18.12 | Episode steps:   30 | Average reward:    18.12 | Step Count:   30 | Episode Q0:    78.69\nEpisode:   2\/20000 | Episode reward:     2.22 | Episode steps:  289 | Average reward:    10.17 | Step Count:  319 | Episode Q0:    16.26\nEpisode:   3\/20000 | Episode reward:    16.01 | Episode steps:  289 | Average reward:    12.11 | Step Count:  608 | Episode Q0:     5.44\nEpisode:   4\/20000 | Episode reward:     7.36 | Episode steps:    9 | Average reward:    10.92 | Step Count:  617 | Episode Q0:    58.57\nEpisode:   5\/20000 | Episode reward:    31.78 | Episode steps:  289 | Average reward:    15.10 | Step Count:  906 | Episode Q0:    48.02\nEpisode:   6\/20000 | Episode reward:    20.72 | Episode steps:  289 | Average reward:    16.03 | Step Count: 1195 | Episode Q0:    65.87\nEpisode:   7\/20000 | Episode reward:    50.67 | Episode steps:  289 | Average reward:    20.98 | Step Count: 1484 | Episode Q0:    59.01\nEpisode:   8\/20000 | Episode reward:    29.06 | Episode steps:  238 | Average reward:    21.99 | Step Count: 1722 | Episode Q0:    25.10\nEpisode:   9\/20000 | Episode reward:     7.29 | Episode steps:  289 | Average reward:    20.36 | Step Count: 2011 | Episode Q0:    15.39\nEpisode:  10\/20000 | Episode reward:     4.51 | Episode steps:    6 | Average reward:    18.77 | Step Count: 2017 | Episode Q0:    69.51\nEpisode:  11\/20000 | Episode reward:    21.27 | Episode steps:  289 | Average reward:    19.00 | Step Count: 2306 | Episode Q0:    59.05\nEpisode:  12\/20000 | Episode reward:     1.21 | Episode steps:  289 | Average reward:    17.52 | Step Count: 2595 | Episode Q0:    33.72\nEpisode:  13\/20000 | Episode reward:    24.52 | Episode steps:  289 | Average reward:    18.06 | Step Count: 2884 | Episode Q0:    31.41\nEpisode:  14\/20000 | Episode reward:    11.45 | Episode steps:   57 | Average reward:    17.59 | Step Count: 2941 | Episode Q0:    69.05\nEpisode:  15\/20000 | Episode reward:    15.51 | Episode steps:  209 | Average reward:    17.45 | Step Count: 3150 | Episode Q0:    29.68\nEpisode:  16\/20000 | Episode reward:     6.99 | Episode steps:  289 | Average reward:    16.79 | Step Count: 3439 | Episode Q0:    23.41\nEpisode:  17\/20000 | Episode reward:     1.85 | Episode steps:    2 | Average reward:    15.91 | Step Count: 3441 | Episode Q0:    23.69\nEpisode:  18\/20000 | Episode reward:    17.48 | Episode steps:  289 | Average reward:    16.00 | Step Count: 3730 | Episode Q0:    93.79\nEpisode:  19\/20000 | Episode reward:    10.62 | Episode steps:  289 | Average reward:    15.72 | Step Count: 4019 | Episode Q0:    26.09\nEpisode:  20\/20000 | Episode reward:     7.87 | Episode steps:   44 | Average reward:    15.33 | Step Count: 4063 | Episode Q0:    50.42\nEpisode:  21\/20000 | Episode reward:     9.47 | Episode steps:   12 | Average reward:    15.05 | Step Count: 4075 | Episode Q0:    27.42\nEpisode:  22\/20000 | Episode reward:     4.94 | Episode steps:    6 | Average reward:    14.59 | Step Count: 4081 | Episode Q0:    29.95\nEpisode:  23\/20000 | Episode reward:     9.04 | Episode steps:  289 | Average reward:    14.35 | Step Count: 4370 | Episode Q0:    35.95\nEpisode:  24\/20000 | Episode reward:     4.56 | Episode steps:  289 | Average reward:    13.94 | Step Count: 4659 | Episode Q0:    26.38\nEpisode:  25\/20000 | Episode reward:     4.32 | Episode steps:  289 | Average reward:    13.55 | Step Count: 4948 | Episode Q0:    51.79\nEpisode:  26\/20000 | Episode reward:     3.23 | Episode steps:  289 | Average reward:    13.16 | Step Count: 5237 | Episode Q0:     0.00\nEpisode:  27\/20000 | Episode reward:     8.20 | Episode steps:  289 | Average reward:    12.97 | Step Count: 5526 | Episode Q0:    15.45\nEpisode:  28\/20000 | Episode reward:    30.87 | Episode steps:  165 | Average reward:    13.61 | Step Count: 5691 | Episode Q0:    23.54\nEpisode:  29\/20000 | Episode reward:     4.60 | Episode steps:  289 | Average reward:    13.30 | Step Count: 5980 | Episode Q0:    11.68\nEpisode:  30\/20000 | Episode reward:     4.80 | Episode steps:    5 | Average reward:    13.02 | Step Count: 5985 | Episode Q0:    13.70\nEpisode:  31\/20000 | Episode reward:     1.97 | Episode steps:    2 | Average reward:    12.66 | Step Count: 5987 | Episode Q0:    20.70\nEpisode:  32\/20000 | Episode reward:     1.44 | Episode steps:  289 | Average reward:    12.31 | Step Count: 6276 | Episode Q0:     1.89\nEpisode:  33\/20000 | Episode reward:     9.58 | Episode steps:  289 | Average reward:    12.23 | Step Count: 6565 | Episode Q0:     0.00\nEpisode:  34\/20000 | Episode reward:    10.42 | Episode steps:  131 | Average reward:    12.18 | Step Count: 6696 | Episode Q0:     0.00\n","truncated":false}}
%---
