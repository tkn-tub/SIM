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
Parameters;   % loads all base variables into workspace and EnvPars %[output:40be3405] %[output:858cf39b] %[output:90d26340] %[output:17e0a73a] %[output:1d5b510b] %[output:12fad53e] %[output:5736db59] %[output:43b040c5] %[output:40fb16fc] %[output:0c12bb1c] %[output:50f59f5c] %[output:46de169c]

%Update the number of atoms
M_x=15 %[output:9a2bbabc]
M_y=M_x;
M=M_x*M_y;

%Update the G matrix to use the one computed with CST
% EnvPars.G = EnvPars.G_CST;   % SIM-2 uses the CST-realistic SIM-1 front-end
% EnvPars.U_func = EnvPars.U_func_CST;

Calibration %[output:23b20473]

% In A_Training_Navigation — after Calibration runs
EnvPars.MaxEpisodes = EnvPars.N_cal * 20;
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
      @(a,ls) stepFunction_nav_CST_Aligned(a, ls, EnvPars), ...
      @()     resetFunction_nav_CST_Aligned(EnvPars));
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
%   data: {"dataType":"text","outputData":{"text":"Adding matlab path to: D:\\code\\SIM\\code\n","truncated":false}}
%---
%[output:40be3405]
%   data: {"dataType":"textualVariable","outputData":{"name":"total_iteration","value":"1"}}
%---
%[output:858cf39b]
%   data: {"dataType":"text","outputData":{"text":"Wireless packet type: SC\n","truncated":false}}
%---
%[output:90d26340]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_x","value":"4"}}
%---
%[output:17e0a73a]
%   data: {"dataType":"textualVariable","outputData":{"name":"N","value":"16"}}
%---
%[output:1d5b510b]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_x","value":"15"}}
%---
%[output:12fad53e]
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
%   data: {"dataType":"textualVariable","outputData":{"name":"T_x","value":"40"}}
%---
%[output:50f59f5c]
%   data: {"dataType":"textualVariable","outputData":{"name":"SNR_dB","value":"36.5005"}}
%---
%[output:46de169c]
%   data: {"dataType":"textualVariable","outputData":{"header":"struct with fields:","name":"EnvPars","value":"                     N: 16\n                   N_x: 4\n                   N_y: 4\n                     T: 1600\n                   T_x: 40\n                   T_y: 40\n                SNR_dB: 36.5005\n             theta_min: 1.8485\n             theta_max: 4.4347\n                    fc: 2.8000e+10\n                lambda: 0.0107\n               Ptx_dBm: 23\n               Gtx_dBi: 14\n               Grx_dBi: 8\n               txArray: [1×1 struct]\n                   cdl: [1×1 nrCDLChannel]\n          var_noise_dB: -110.9794\n                     r: 0\n                   d_x: 0.0054\n               pos_SIM: [5 5 4]\n                pos_MU: [1.0204 5.7613 1.5000]\n                   n_y: [1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4]\n                   n_x: [1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4]\n                   t_y: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 … ] (1×1600 double)\n                   t_x: [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 … ] (1×1600 double)\n                  h_MU: 1.5000\n                L_hall: 10\n                W_hall: 10\n                 N_cal: 100\n             MU_margin: 0.5000\n           MaxEpisodes: 5000\n                 psi_x: 0\n                 psi_y: 0\n    MaxStepsPerEpisode: 120\n             tolerance: 0.0393\n     StopTrainingValue: 114\n       episode_counter: 0\n           delta_moves: [9×2 double]\n             n_actions: 9\n        DiscountFactor: 0.9500\n"}}
%---
%[output:9a2bbabc]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_x","value":"15"}}
%---
%[output:23b20473]
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
%   data: {"dataType":"text","outputData":{"text":"N_cal=100  |  MaxEpisodes=20000  |  MaxSteps=120  |  n_actions=9\n","truncated":false}}
%---
%[output:58b08826]
%   data: {"dataType":"text","outputData":{"text":"Episode:   1\/20000 | Episode reward:     7.16 | Episode steps:  120 | Average reward:     7.16 | Step Count:  120 | Episode Q0:    17.42\nEpisode:   2\/20000 | Episode reward:     4.33 | Episode steps:  120 | Average reward:     5.75 | Step Count:  240 | Episode Q0:    38.63\nEpisode:   3\/20000 | Episode reward:     8.46 | Episode steps:  120 | Average reward:     6.65 | Step Count:  360 | Episode Q0:    39.68\nEpisode:   4\/20000 | Episode reward:     7.59 | Episode steps:  120 | Average reward:     6.89 | Step Count:  480 | Episode Q0:    38.64\nEpisode:   5\/20000 | Episode reward:     8.59 | Episode steps:  120 | Average reward:     7.23 | Step Count:  600 | Episode Q0:    58.17\nEpisode:   6\/20000 | Episode reward:     5.33 | Episode steps:  120 | Average reward:     6.91 | Step Count:  720 | Episode Q0:    12.27\nEpisode:   7\/20000 | Episode reward:     8.61 | Episode steps:  120 | Average reward:     7.15 | Step Count:  840 | Episode Q0:    40.06\nEpisode:   8\/20000 | Episode reward:    10.74 | Episode steps:  120 | Average reward:     7.60 | Step Count:  960 | Episode Q0:    32.34\nEpisode:   9\/20000 | Episode reward:     8.15 | Episode steps:  120 | Average reward:     7.66 | Step Count: 1080 | Episode Q0:    42.75\nEpisode:  10\/20000 | Episode reward:     6.43 | Episode steps:  120 | Average reward:     7.54 | Step Count: 1200 | Episode Q0:    38.61\nEpisode:  11\/20000 | Episode reward:     5.14 | Episode steps:  120 | Average reward:     7.32 | Step Count: 1320 | Episode Q0:    44.21\nEpisode:  12\/20000 | Episode reward:     6.57 | Episode steps:  120 | Average reward:     7.26 | Step Count: 1440 | Episode Q0:    66.17\nEpisode:  13\/20000 | Episode reward:     6.36 | Episode steps:  120 | Average reward:     7.19 | Step Count: 1560 | Episode Q0:    16.30\nEpisode:  14\/20000 | Episode reward:     6.36 | Episode steps:  120 | Average reward:     7.13 | Step Count: 1680 | Episode Q0:    80.97\nEpisode:  15\/20000 | Episode reward:    10.84 | Episode steps:  120 | Average reward:     7.38 | Step Count: 1800 | Episode Q0:    52.07\nEpisode:  16\/20000 | Episode reward:     9.12 | Episode steps:  120 | Average reward:     7.49 | Step Count: 1920 | Episode Q0:    80.65\nEpisode:  17\/20000 | Episode reward:     6.61 | Episode steps:  120 | Average reward:     7.44 | Step Count: 2040 | Episode Q0:    65.02\nEpisode:  18\/20000 | Episode reward:     7.32 | Episode steps:  120 | Average reward:     7.43 | Step Count: 2160 | Episode Q0:    13.97\nEpisode:  19\/20000 | Episode reward:     6.49 | Episode steps:  120 | Average reward:     7.38 | Step Count: 2280 | Episode Q0:    82.26\nEpisode:  20\/20000 | Episode reward:     9.68 | Episode steps:  120 | Average reward:     7.49 | Step Count: 2400 | Episode Q0:    45.28\nEpisode:  21\/20000 | Episode reward:     4.49 | Episode steps:  120 | Average reward:     7.35 | Step Count: 2520 | Episode Q0:    42.11\nEpisode:  22\/20000 | Episode reward:     5.20 | Episode steps:  120 | Average reward:     7.25 | Step Count: 2640 | Episode Q0:    62.35\nEpisode:  23\/20000 | Episode reward:     9.43 | Episode steps:  120 | Average reward:     7.35 | Step Count: 2760 | Episode Q0:    23.15\nEpisode:  24\/20000 | Episode reward:     5.09 | Episode steps:  120 | Average reward:     7.25 | Step Count: 2880 | Episode Q0:    15.65\nEpisode:  25\/20000 | Episode reward:     8.91 | Episode steps:  120 | Average reward:     7.32 | Step Count: 3000 | Episode Q0:    75.16\nEpisode:  26\/20000 | Episode reward:     6.21 | Episode steps:  120 | Average reward:     7.28 | Step Count: 3120 | Episode Q0:     8.63\nEpisode:  27\/20000 | Episode reward:     5.06 | Episode steps:  120 | Average reward:     7.20 | Step Count: 3240 | Episode Q0:     9.34\nEpisode:  28\/20000 | Episode reward:     7.95 | Episode steps:  120 | Average reward:     7.22 | Step Count: 3360 | Episode Q0:    18.56\nEpisode:  29\/20000 | Episode reward:     3.88 | Episode steps:  120 | Average reward:     7.11 | Step Count: 3480 | Episode Q0:    77.32\nEpisode:  30\/20000 | Episode reward:     9.33 | Episode steps:  120 | Average reward:     7.18 | Step Count: 3600 | Episode Q0:    61.77\nEpisode:  31\/20000 | Episode reward:     7.43 | Episode steps:  120 | Average reward:     7.19 | Step Count: 3720 | Episode Q0:     6.22\nEpisode:  32\/20000 | Episode reward:     4.22 | Episode steps:  120 | Average reward:     7.10 | Step Count: 3840 | Episode Q0:    26.57\nEpisode:  33\/20000 | Episode reward:     8.66 | Episode steps:  120 | Average reward:     7.14 | Step Count: 3960 | Episode Q0:     6.72\nEpisode:  34\/20000 | Episode reward:    10.04 | Episode steps:  120 | Average reward:     7.23 | Step Count: 4080 | Episode Q0:    30.97\nEpisode:  35\/20000 | Episode reward:    10.62 | Episode steps:  120 | Average reward:     7.33 | Step Count: 4200 | Episode Q0:     8.95\nEpisode:  36\/20000 | Episode reward:    46.90 | Episode steps:   71 | Average reward:     8.43 | Step Count: 4271 | Episode Q0:    23.64\nEpisode:  37\/20000 | Episode reward:     6.40 | Episode steps:  120 | Average reward:     8.37 | Step Count: 4391 | Episode Q0:    24.45\nEpisode:  38\/20000 | Episode reward:     8.06 | Episode steps:  120 | Average reward:     8.36 | Step Count: 4511 | Episode Q0:    17.67\nEpisode:  39\/20000 | Episode reward:    50.38 | Episode steps:  116 | Average reward:     9.44 | Step Count: 4627 | Episode Q0:    34.08\nEpisode:  40\/20000 | Episode reward:     5.40 | Episode steps:  120 | Average reward:     9.34 | Step Count: 4747 | Episode Q0:    64.44\nEpisode:  41\/20000 | Episode reward:    10.58 | Episode steps:  120 | Average reward:     9.37 | Step Count: 4867 | Episode Q0:    15.16\nEpisode:  42\/20000 | Episode reward:     8.10 | Episode steps:  120 | Average reward:     9.34 | Step Count: 4987 | Episode Q0:     7.45\nEpisode:  43\/20000 | Episode reward:     9.70 | Episode steps:  120 | Average reward:     9.35 | Step Count: 5107 | Episode Q0:    27.61\nEpisode:  44\/20000 | Episode reward:     8.75 | Episode steps:  120 | Average reward:     9.33 | Step Count: 5227 | Episode Q0:    21.95\nEpisode:  45\/20000 | Episode reward:     4.63 | Episode steps:  120 | Average reward:     9.23 | Step Count: 5347 | Episode Q0:    28.28\nEpisode:  46\/20000 | Episode reward:    10.20 | Episode steps:  120 | Average reward:     9.25 | Step Count: 5467 | Episode Q0:    11.53\nEpisode:  47\/20000 | Episode reward:    41.62 | Episode steps:   17 | Average reward:     9.94 | Step Count: 5484 | Episode Q0:    13.10\nEpisode:  48\/20000 | Episode reward:     7.03 | Episode steps:  120 | Average reward:     9.88 | Step Count: 5604 | Episode Q0:    42.32\nEpisode:  49\/20000 | Episode reward:    11.02 | Episode steps:  120 | Average reward:     9.90 | Step Count: 5724 | Episode Q0:    16.06\nEpisode:  50\/20000 | Episode reward:     5.34 | Episode steps:  120 | Average reward:     9.81 | Step Count: 5844 | Episode Q0:    14.36\nEpisode:  51\/20000 | Episode reward:     8.30 | Episode steps:  120 | Average reward:     9.83 | Step Count: 5964 | Episode Q0:    32.15\nEpisode:  52\/20000 | Episode reward:     4.47 | Episode steps:  120 | Average reward:     9.84 | Step Count: 6084 | Episode Q0:    13.64\nEpisode:  53\/20000 | Episode reward:     7.38 | Episode steps:  120 | Average reward:     9.81 | Step Count: 6204 | Episode Q0:    34.72\nEpisode:  54\/20000 | Episode reward:     4.68 | Episode steps:  120 | Average reward:     9.76 | Step Count: 6324 | Episode Q0:    23.29\nEpisode:  55\/20000 | Episode reward:    10.27 | Episode steps:  120 | Average reward:     9.79 | Step Count: 6444 | Episode Q0:    30.43\nEpisode:  56\/20000 | Episode reward:     7.67 | Episode steps:  120 | Average reward:     9.84 | Step Count: 6564 | Episode Q0:    29.36\nEpisode:  57\/20000 | Episode reward:     9.65 | Episode steps:  120 | Average reward:     9.86 | Step Count: 6684 | Episode Q0:    26.42\nEpisode:  58\/20000 | Episode reward:    10.60 | Episode steps:  120 | Average reward:     9.85 | Step Count: 6804 | Episode Q0:    38.62\nEpisode:  59\/20000 | Episode reward:     4.29 | Episode steps:  120 | Average reward:     9.78 | Step Count: 6924 | Episode Q0:     6.18\nEpisode:  60\/20000 | Episode reward:     5.87 | Episode steps:  120 | Average reward:     9.77 | Step Count: 7044 | Episode Q0:    17.50\nEpisode:  61\/20000 | Episode reward:     7.89 | Episode steps:  120 | Average reward:     9.82 | Step Count: 7164 | Episode Q0:    27.16\nEpisode:  62\/20000 | Episode reward:     9.26 | Episode steps:  120 | Average reward:     9.87 | Step Count: 7284 | Episode Q0:    10.48\nEpisode:  63\/20000 | Episode reward:     7.12 | Episode steps:  120 | Average reward:     9.89 | Step Count: 7404 | Episode Q0:    47.76\nEpisode:  64\/20000 | Episode reward:     6.12 | Episode steps:  120 | Average reward:     9.88 | Step Count: 7524 | Episode Q0:    22.27\nEpisode:  65\/20000 | Episode reward:    44.96 | Episode steps:   51 | Average reward:    10.57 | Step Count: 7575 | Episode Q0:    12.17\nEpisode:  66\/20000 | Episode reward:     7.66 | Episode steps:  120 | Average reward:    10.54 | Step Count: 7695 | Episode Q0:    24.96\nEpisode:  67\/20000 | Episode reward:     9.49 | Episode steps:  120 | Average reward:    10.60 | Step Count: 7815 | Episode Q0:    14.81\nEpisode:  68\/20000 | Episode reward:     5.92 | Episode steps:  120 | Average reward:    10.57 | Step Count: 7935 | Episode Q0:     8.37\nEpisode:  69\/20000 | Episode reward:     7.13 | Episode steps:  120 | Average reward:    10.58 | Step Count: 8055 | Episode Q0:    14.92\nEpisode:  70\/20000 | Episode reward:     7.08 | Episode steps:  120 | Average reward:    10.53 | Step Count: 8175 | Episode Q0:    27.09\nEpisode:  71\/20000 | Episode reward:     7.56 | Episode steps:  120 | Average reward:    10.59 | Step Count: 8295 | Episode Q0:     6.56\nEpisode:  72\/20000 | Episode reward:    10.08 | Episode steps:  120 | Average reward:    10.69 | Step Count: 8415 | Episode Q0:    14.29\nEpisode:  73\/20000 | Episode reward:     7.73 | Episode steps:  120 | Average reward:    10.65 | Step Count: 8535 | Episode Q0:    10.27\nEpisode:  74\/20000 | Episode reward:     5.73 | Episode steps:  120 | Average reward:    10.67 | Step Count: 8655 | Episode Q0:    28.19\nEpisode:  75\/20000 | Episode reward:     5.88 | Episode steps:  120 | Average reward:    10.61 | Step Count: 8775 | Episode Q0:     7.54\nEpisode:  76\/20000 | Episode reward:     8.32 | Episode steps:  120 | Average reward:    10.65 | Step Count: 8895 | Episode Q0:     9.74\nEpisode:  77\/20000 | Episode reward:     3.90 | Episode steps:  120 | Average reward:    10.62 | Step Count: 9015 | Episode Q0:    26.16\nEpisode:  78\/20000 | Episode reward:    11.16 | Episode steps:  120 | Average reward:    10.69 | Step Count: 9135 | Episode Q0:    18.16\nEpisode:  79\/20000 | Episode reward:     7.43 | Episode steps:  120 | Average reward:    10.76 | Step Count: 9255 | Episode Q0:     8.40\nEpisode:  80\/20000 | Episode reward:     7.63 | Episode steps:  120 | Average reward:    10.73 | Step Count: 9375 | Episode Q0:    22.51\nEpisode:  81\/20000 | Episode reward:     8.31 | Episode steps:  120 | Average reward:    10.74 | Step Count: 9495 | Episode Q0:    20.14\nEpisode:  82\/20000 | Episode reward:     8.35 | Episode steps:  120 | Average reward:    10.83 | Step Count: 9615 | Episode Q0:    15.13\nEpisode:  83\/20000 | Episode reward:     4.52 | Episode steps:  120 | Average reward:    10.74 | Step Count: 9735 | Episode Q0:    10.65\nEpisode:  84\/20000 | Episode reward:     6.21 | Episode steps:  120 | Average reward:    10.67 | Step Count: 9855 | Episode Q0:    16.67\nEpisode:  85\/20000 | Episode reward:     6.23 | Episode steps:  120 | Average reward:    10.58 | Step Count: 9975 | Episode Q0:    26.61\nEpisode:  86\/20000 | Episode reward:     5.92 | Episode steps:  120 | Average reward:     9.76 | Step Count: 10095 | Episode Q0:    23.00\nEpisode:  87\/20000 | Episode reward:     5.77 | Episode steps:  120 | Average reward:     9.75 | Step Count: 10215 | Episode Q0:    20.11\nEpisode:  88\/20000 | Episode reward:     5.69 | Episode steps:  120 | Average reward:     9.70 | Step Count: 10335 | Episode Q0:    18.75\nEpisode:  89\/20000 | Episode reward:     8.33 | Episode steps:  120 | Average reward:     8.86 | Step Count: 10455 | Episode Q0:    18.19\nEpisode:  90\/20000 | Episode reward:    10.85 | Episode steps:  120 | Average reward:     8.97 | Step Count: 10575 | Episode Q0:    12.83\nEpisode:  91\/20000 | Episode reward:     6.80 | Episode steps:  120 | Average reward:     8.89 | Step Count: 10695 | Episode Q0:    13.31\nEpisode:  92\/20000 | Episode reward:     5.94 | Episode steps:  120 | Average reward:     8.85 | Step Count: 10815 | Episode Q0:    17.30\nEpisode:  93\/20000 | Episode reward:     5.50 | Episode steps:  120 | Average reward:     8.77 | Step Count: 10935 | Episode Q0:    18.12\nEpisode:  94\/20000 | Episode reward:     6.58 | Episode steps:  120 | Average reward:     8.72 | Step Count: 11055 | Episode Q0:    16.28\nEpisode:  95\/20000 | Episode reward:     8.58 | Episode steps:  120 | Average reward:     8.80 | Step Count: 11175 | Episode Q0:    17.60\nEpisode:  96\/20000 | Episode reward:     6.27 | Episode steps:  120 | Average reward:     8.72 | Step Count: 11295 | Episode Q0:    10.31\nEpisode:  97\/20000 | Episode reward:     7.84 | Episode steps:  120 | Average reward:     8.05 | Step Count: 11415 | Episode Q0:    16.54\nEpisode:  98\/20000 | Episode reward:     4.29 | Episode steps:  120 | Average reward:     7.99 | Step Count: 11535 | Episode Q0:    12.59\n","truncated":false}}
%---
