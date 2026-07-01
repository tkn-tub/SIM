%[text] # Training script — SIM Navigation with DQN, SIM2 Q-network (3 hidden M-layers)
%[text] VARIANT with THREE hidden M-layers. Architecture: N -\> \[propN-\>M\] -\> M -\> \[propM-\>M\] -\> M -\> \[propM-\>M\] -\> M -\> \[propM-\>9\] -\> diode 691 trainable phases (16+225+225+225) vs 466 (2-hidden) and 241 (1-hidden). Saves to dqn\_agent\_navigation\_SIM2\_3hidden.mat -- the 1- and 2-hidden results are left untouched, giving a clean 1/2/3 M-layer ablation sweep.
%[text] CAVEAT, now stronger: three phase masks separated only by fixed propagation, with the ONLY nonlinearity at the diode. The deeper this goes the more likely training is to stall or plateau no higher than shallower versions -- watch the gradient norms (verify script) and the learning curve. If 3-hidden does not beat 2-hidden, that is the reportable finding (diminishing/negative returns from all-phase depth), not a bug to fix.
%[text] Requires Parameters.m to have been run first (this script calls it)
%[text] Built incrementally with Claude -- all four parts done, gradient- verified per-layer (see SIM2\_GradientCheck\_\*.m, run separately): Part 1 \[DONE\] -- SIM2 geometry & propagation matrices (W0\_SIM2, W\_M9) Part 2 \[DONE\] -- custom layers: simPhaseLayerCST (N), simPhaseLayerCST (M) Part 3 \[DONE\] -- realToComplexLayer, diodeReadoutLayer, dlnetwork assembly Part 4 \[DONE\] -- ObsInfo override (2N-dim, local to this script), stepFunction\_nav\_CST.m / resetFunction\_nav.m updated to emit \[Re(r);Im(r)\] (t\_x,t\_y tracked in silico, not observed)
%[text] REVISED again after a real platform limitation: MATLAB custom layers cannot return complex values from predict()/forward() (confirmed via MathWorks doc, not just inferred from the error). Every layer (simPhaseLayerCST, simPropagationLayer, diodeReadoutLayer) now operates on REAL-STACKED \[Re;Im\] representations throughout -- see each layer file for the re-derived math. to\_complex is GONE entirely: with everything real-stacked end to end, the observation \[Re(r);Im(r)\] feeds sim2\_layer1 directly. realImagToComplexLayer.m and realToComplexLayer.m are both now historical/unused. Gradient checks were REDONE (not just re-run) against the new math -- see SIM2\_GradientCheck\_PhaseLayer.m, SIM2\_GradientCheck\_PropagationLayer.m, SIM2\_GradientCheck\_DiodeReadout.m.
%[text] NOT yet done, separate from this architecture: EnvPars.U\_func (v0) and G still use the idealized analytic models -- the CST amplitude-phase coupling on the SIM1 input layer, and loading the actual trained (CST-aware) G from SIM\_Training\_CST\_SingleZeta\_Parallel.m instead of the closed-form DFT kernel, are both still open from earlier in this project and independent of the SIM2 work above.
%[text] STILL NEEDED before trusting a real training run: - End-to-end check: does criticNet/predict() actually run on a real observation without shape/format errors? (per-layer checks passed, but the full chain hasn't been exercised together yet) - Capacity sanity check: 241 trainable phases vs. the old FC network's ~20k+ weights -- worth comparing against the original training curve (Fig. 6) once this trains, not just checking that it learns at all
clc; clear all; close all;
addingPathParentFolderByName('code'); %[output:14bce44f]
Parameters;   % loads all base variables into workspace and EnvPars %[output:1c4cfd7b] %[output:97099cad] %[output:37db94e7] %[output:5dcf8dd9] %[output:4aec8d3a] %[output:5c6a35a7] %[output:2a5ba87f] %[output:6fbdd6c7] %[output:97ef5efe] %[output:7a48328b]

%Update the G matrix to use the one computed with CST
% EnvPars.G = EnvPars.G_CST;   % SIM-2 uses the CST-realistic SIM-1 front-end
% EnvPars.U_func = EnvPars.U_func_CST;
M_x = 17;
M_y = M_x;

Calibration %[output:05a58d39]

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

fprintf('SIM2 geometry: W0_SIM2 %dx%d (N=%d->M=%d), W_M9 %dx%d (M=%d->%d ports)\n', ... %[output:group:86e462a9] %[output:7cbb7271]
    size(W0_SIM2,1), size(W0_SIM2,2), N, M, size(W_M9,1), size(W_M9,2), M, Q); %[output:group:86e462a9] %[output:7cbb7271]
fprintf('max(abs(W0_SIM2))=%.3e , max(abs(W_M9))=%.3e\n', max(abs(W0_SIM2(:))), max(abs(W_M9(:)))); %[output:23628d1d]
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
    prop_M_to_Q
    readout_layer];

criticNet = dlnetwork(layerGraph(statePath));
critic    = rlVectorQValueFunction(criticNet, ObsInfo, ActInfo);
%%
%[text] ## ── 5. ENVIRONMENT ───────────────────────────────────────────────────────
env = rlFunctionEnv(ObsInfo, ActInfo, ...
      @(a,ls) stepFunction_nav_CST(a, ls, EnvPars), ...
      @()     resetFunction_nav_CST(EnvPars));
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
    'Plots',                'none');

fprintf('=== Training phase ===\n'); %[output:7c8e9a6e]
fprintf('N_cal=%d  |  MaxEpisodes=%d  |  MaxSteps=%d  |  n_actions=%d\n', ... %[output:group:52f4958b] %[output:73462c30]
        N_cal, EnvPars.MaxEpisodes, EnvPars.MaxStepsPerEpisode, EnvPars.n_actions); %[output:group:52f4958b] %[output:73462c30]

trainingStats = train(agent, env, trainOpts); %[output:2b841e88]
%%
%[text] ## ── 9. SAVE ──────────────────────────────────────────────────────────────
%[text] NOTE: filename no longer references "Neurons\_per\_layer" -- that variable (25\*25=625) never matched any actual layer width (144=T\_x\*T\_y, 225=M\_x\*M\_y) even in the original script, and SIM2 doesn't have a single "neuron count" analog anyway (241 trainable phases across two layers + 1 frozen layer). Original line was also missing a closing paren on fullfile(); fixed here.
save_path = fullfile('..', 'Dataset', 'dqn_agent_navigation_775_phases_CST_Nx_8_Mx_17.mat');
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
%[output:14bce44f]
%   data: {"dataType":"text","outputData":{"text":"Adding matlab path to: G:\\My Drive\\Work\\Research\\SIM\\code\n","truncated":false}}
%---
%[output:1c4cfd7b]
%   data: {"dataType":"textualVariable","outputData":{"name":"total_iteration","value":"1"}}
%---
%[output:97099cad]
%   data: {"dataType":"text","outputData":{"text":"Wireless packet type: SC\n","truncated":false}}
%---
%[output:37db94e7]
%   data: {"dataType":"textualVariable","outputData":{"name":"N","value":"16"}}
%---
%[output:5dcf8dd9]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_x","value":"15"}}
%---
%[output:4aec8d3a]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_y","value":"15"}}
%---
%[output:5c6a35a7]
%   data: {"dataType":"matrix","outputData":{"columns":6,"name":"zeta","rows":1,"type":"double","value":[["0.9850","0.9860","0.9870","0.9880","0.9890","0.9900"]]}}
%---
%[output:2a5ba87f]
%   data: {"dataType":"textualVariable","outputData":{"name":"T_coh","value":"0.0038"}}
%---
%[output:6fbdd6c7]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_packets_coh","value":"12"}}
%---
%[output:97ef5efe]
%   data: {"dataType":"textualVariable","outputData":{"name":"SNR_dB","value":"36.5005"}}
%---
%[output:7a48328b]
%   data: {"dataType":"text","outputData":{"text":"Loaded CST SIM-1 G_CST. Deviation from analytic G: 1.806%n","truncated":false}}
%---
%[output:05a58d39]
%   data: {"dataType":"text","outputData":{"text":"=== Calibration phase ===\nGrid: 10 x 10 = 100 calibration positions\nCalibration complete.\n\n","truncated":false}}
%---
%[output:7cbb7271]
%   data: {"dataType":"text","outputData":{"text":"SIM2 geometry: W0_SIM2 289x16 (N=16->M=289), W_M9 9x289 (M=289->9 ports)\n","truncated":false}}
%---
%[output:23628d1d]
%   data: {"dataType":"text","outputData":{"text":"max(abs(W0_SIM2))=2.628e-01 , max(abs(W_M9))=2.748e-01\n","truncated":false}}
%---
%[output:7c8e9a6e]
%   data: {"dataType":"text","outputData":{"text":"=== Training phase ===\n","truncated":false}}
%---
%[output:73462c30]
%   data: {"dataType":"text","outputData":{"text":"N_cal=100  |  MaxEpisodes=20000  |  MaxSteps=289  |  n_actions=9\n","truncated":false}}
%---
%[output:2b841e88]
%   data: {"dataType":"text","outputData":{"text":"Episode:   1\/20000 | Episode reward:    50.52 | Episode steps:  289 | Average reward:    50.52 | Step Count:  289 | Episode Q0:   108.53\nEpisode:   2\/20000 | Episode reward:    16.13 | Episode steps:  145 | Average reward:    33.32 | Step Count:  434 | Episode Q0:    92.39\nEpisode:   3\/20000 | Episode reward:    17.63 | Episode steps:  289 | Average reward:    28.09 | Step Count:  723 | Episode Q0:    64.22\nEpisode:   4\/20000 | Episode reward:     1.37 | Episode steps:  289 | Average reward:    21.41 | Step Count: 1012 | Episode Q0:    20.50\nEpisode:   5\/20000 | Episode reward:     4.56 | Episode steps:  289 | Average reward:    18.04 | Step Count: 1301 | Episode Q0:    78.42\nEpisode:   6\/20000 | Episode reward:    16.33 | Episode steps:  289 | Average reward:    17.76 | Step Count: 1590 | Episode Q0:    71.87\nEpisode:   7\/20000 | Episode reward:     8.26 | Episode steps:  289 | Average reward:    16.40 | Step Count: 1879 | Episode Q0:     4.04\nEpisode:   8\/20000 | Episode reward:     3.88 | Episode steps:   27 | Average reward:    14.83 | Step Count: 1906 | Episode Q0:   111.95\nEpisode:   9\/20000 | Episode reward:    13.37 | Episode steps:  289 | Average reward:    14.67 | Step Count: 2195 | Episode Q0:    92.21\nEpisode:  10\/20000 | Episode reward:    13.37 | Episode steps:  289 | Average reward:    14.54 | Step Count: 2484 | Episode Q0:   104.71\nEpisode:  11\/20000 | Episode reward:    39.32 | Episode steps:  289 | Average reward:    16.79 | Step Count: 2773 | Episode Q0:    89.79\nEpisode:  12\/20000 | Episode reward:     8.41 | Episode steps:  289 | Average reward:    16.09 | Step Count: 3062 | Episode Q0:    72.84\nEpisode:  13\/20000 | Episode reward:    10.73 | Episode steps:  289 | Average reward:    15.68 | Step Count: 3351 | Episode Q0:    51.39\nEpisode:  14\/20000 | Episode reward:     1.95 | Episode steps:  289 | Average reward:    14.70 | Step Count: 3640 | Episode Q0:    58.17\nEpisode:  15\/20000 | Episode reward:    28.90 | Episode steps:   94 | Average reward:    15.65 | Step Count: 3734 | Episode Q0:    84.39\nEpisode:  16\/20000 | Episode reward:    24.56 | Episode steps:   64 | Average reward:    16.21 | Step Count: 3798 | Episode Q0:    60.95\nEpisode:  17\/20000 | Episode reward:    58.26 | Episode steps:  289 | Average reward:    18.68 | Step Count: 4087 | Episode Q0:     6.92\nEpisode:  18\/20000 | Episode reward:     8.41 | Episode steps:  289 | Average reward:    18.11 | Step Count: 4376 | Episode Q0:    99.77\nEpisode:  19\/20000 | Episode reward:     6.90 | Episode steps:  289 | Average reward:    17.52 | Step Count: 4665 | Episode Q0:    98.00\nEpisode:  20\/20000 | Episode reward:    23.36 | Episode steps:  155 | Average reward:    17.81 | Step Count: 4820 | Episode Q0:     7.83\nEpisode:  21\/20000 | Episode reward:    43.65 | Episode steps:  231 | Average reward:    19.04 | Step Count: 5051 | Episode Q0:    44.86\nEpisode:  22\/20000 | Episode reward:    16.28 | Episode steps:  124 | Average reward:    18.92 | Step Count: 5175 | Episode Q0:    48.58\nEpisode:  23\/20000 | Episode reward:    22.90 | Episode steps:  289 | Average reward:    19.09 | Step Count: 5464 | Episode Q0:    41.93\nEpisode:  24\/20000 | Episode reward:     0.99 | Episode steps:  289 | Average reward:    18.34 | Step Count: 5753 | Episode Q0:    46.75\nEpisode:  25\/20000 | Episode reward:    13.67 | Episode steps:  228 | Average reward:    18.15 | Step Count: 5981 | Episode Q0:    46.23\nEpisode:  26\/20000 | Episode reward:    20.61 | Episode steps:   83 | Average reward:    18.24 | Step Count: 6064 | Episode Q0:    46.30\nEpisode:  27\/20000 | Episode reward:    14.09 | Episode steps:  289 | Average reward:    18.09 | Step Count: 6353 | Episode Q0:    40.13\nEpisode:  28\/20000 | Episode reward:    14.00 | Episode steps:  192 | Average reward:    17.94 | Step Count: 6545 | Episode Q0:    36.59\nEpisode:  29\/20000 | Episode reward:    18.31 | Episode steps:  289 | Average reward:    17.96 | Step Count: 6834 | Episode Q0:    55.23\nEpisode:  30\/20000 | Episode reward:    23.42 | Episode steps:  289 | Average reward:    18.14 | Step Count: 7123 | Episode Q0:    75.18\nEpisode:  31\/20000 | Episode reward:     3.74 | Episode steps:  113 | Average reward:    17.67 | Step Count: 7236 | Episode Q0:    58.31\nEpisode:  32\/20000 | Episode reward:     7.50 | Episode steps:  289 | Average reward:    17.36 | Step Count: 7525 | Episode Q0:    37.23\nEpisode:  33\/20000 | Episode reward:    10.02 | Episode steps:   80 | Average reward:    17.13 | Step Count: 7605 | Episode Q0:    43.80\nEpisode:  34\/20000 | Episode reward:    11.01 | Episode steps:   67 | Average reward:    16.95 | Step Count: 7672 | Episode Q0:    52.95\nEpisode:  35\/20000 | Episode reward:    72.76 | Episode steps:  289 | Average reward:    18.55 | Step Count: 7961 | Episode Q0:    25.57\nEpisode:  36\/20000 | Episode reward:    22.16 | Episode steps:  289 | Average reward:    18.65 | Step Count: 8250 | Episode Q0:    21.22\nEpisode:  37\/20000 | Episode reward:     2.50 | Episode steps:  289 | Average reward:    18.21 | Step Count: 8539 | Episode Q0:    55.94\nEpisode:  38\/20000 | Episode reward:    15.70 | Episode steps:   59 | Average reward:    18.15 | Step Count: 8598 | Episode Q0:    60.71\nEpisode:  39\/20000 | Episode reward:    31.93 | Episode steps:  289 | Average reward:    18.50 | Step Count: 8887 | Episode Q0:    14.05\nEpisode:  40\/20000 | Episode reward:    24.34 | Episode steps:  289 | Average reward:    18.65 | Step Count: 9176 | Episode Q0:    24.31\nEpisode:  41\/20000 | Episode reward:    19.03 | Episode steps:  279 | Average reward:    18.65 | Step Count: 9455 | Episode Q0:    44.08\nEpisode:  42\/20000 | Episode reward:    19.75 | Episode steps:  122 | Average reward:    18.68 | Step Count: 9577 | Episode Q0:    35.88\nEpisode:  43\/20000 | Episode reward:    32.99 | Episode steps:  224 | Average reward:    19.01 | Step Count: 9801 | Episode Q0:    51.97\nEpisode:  44\/20000 | Episode reward:    14.32 | Episode steps:  215 | Average reward:    18.91 | Step Count: 10016 | Episode Q0:    30.13\nEpisode:  45\/20000 | Episode reward:     7.77 | Episode steps:   48 | Average reward:    18.66 | Step Count: 10064 | Episode Q0:    35.23\nEpisode:  46\/20000 | Episode reward:    24.89 | Episode steps:  289 | Average reward:    18.80 | Step Count: 10353 | Episode Q0:    12.78\nEpisode:  47\/20000 | Episode reward:     9.82 | Episode steps:  289 | Average reward:    18.60 | Step Count: 10642 | Episode Q0:    27.26\nEpisode:  48\/20000 | Episode reward:     0.43 | Episode steps:  289 | Average reward:    18.23 | Step Count: 10931 | Episode Q0:    19.50\nEpisode:  49\/20000 | Episode reward:    10.61 | Episode steps:  132 | Average reward:    18.07 | Step Count: 11063 | Episode Q0:    45.25\nEpisode:  50\/20000 | Episode reward:     4.19 | Episode steps:  289 | Average reward:    17.79 | Step Count: 11352 | Episode Q0:    32.73\nEpisode:  51\/20000 | Episode reward:    19.95 | Episode steps:  289 | Average reward:    17.18 | Step Count: 11641 | Episode Q0:    23.53\nEpisode:  52\/20000 | Episode reward:     2.38 | Episode steps:  289 | Average reward:    16.91 | Step Count: 11930 | Episode Q0:    25.09\nEpisode:  53\/20000 | Episode reward:    64.83 | Episode steps:  289 | Average reward:    17.85 | Step Count: 12219 | Episode Q0:     6.98\nEpisode:  54\/20000 | Episode reward:    24.33 | Episode steps:  289 | Average reward:    18.31 | Step Count: 12508 | Episode Q0:     8.99\nEpisode:  55\/20000 | Episode reward:    13.87 | Episode steps:  289 | Average reward:    18.50 | Step Count: 12797 | Episode Q0:    37.79\nEpisode:  56\/20000 | Episode reward:     2.71 | Episode steps:   29 | Average reward:    18.22 | Step Count: 12826 | Episode Q0:    25.60\nEpisode:  57\/20000 | Episode reward:     7.16 | Episode steps:   34 | Average reward:    18.20 | Step Count: 12860 | Episode Q0:    25.31\nEpisode:  58\/20000 | Episode reward:     8.72 | Episode steps:  289 | Average reward:    18.30 | Step Count: 13149 | Episode Q0:    30.46\nEpisode:  59\/20000 | Episode reward:    31.89 | Episode steps:  241 | Average reward:    18.67 | Step Count: 13390 | Episode Q0:    11.13\nEpisode:  60\/20000 | Episode reward:    25.20 | Episode steps:  149 | Average reward:    18.91 | Step Count: 13539 | Episode Q0:    16.83\nEpisode:  61\/20000 | Episode reward:     0.68 | Episode steps:  289 | Average reward:    18.13 | Step Count: 13828 | Episode Q0:    12.60\nEpisode:  62\/20000 | Episode reward:    26.90 | Episode steps:  289 | Average reward:    18.50 | Step Count: 14117 | Episode Q0:    26.67\nEpisode:  63\/20000 | Episode reward:     9.01 | Episode steps:   48 | Average reward:    18.47 | Step Count: 14165 | Episode Q0:    22.52\nEpisode:  64\/20000 | Episode reward:    14.15 | Episode steps:  289 | Average reward:    18.71 | Step Count: 14454 | Episode Q0:     9.14\nEpisode:  65\/20000 | Episode reward:    29.31 | Episode steps:  122 | Average reward:    18.72 | Step Count: 14576 | Episode Q0:    19.83\nEpisode:  66\/20000 | Episode reward:    12.95 | Episode steps:   21 | Average reward:    18.49 | Step Count: 14597 | Episode Q0:    13.42\nEpisode:  67\/20000 | Episode reward:    20.53 | Episode steps:  289 | Average reward:    17.73 | Step Count: 14886 | Episode Q0:    16.13\nEpisode:  68\/20000 | Episode reward:    30.25 | Episode steps:  227 | Average reward:    18.17 | Step Count: 15113 | Episode Q0:    12.84\nEpisode:  69\/20000 | Episode reward:    21.93 | Episode steps:  289 | Average reward:    18.47 | Step Count: 15402 | Episode Q0:    10.91\nEpisode:  70\/20000 | Episode reward:     5.35 | Episode steps:  152 | Average reward:    18.11 | Step Count: 15554 | Episode Q0:    30.21\nEpisode:  71\/20000 | Episode reward:    20.87 | Episode steps:  289 | Average reward:    17.65 | Step Count: 15843 | Episode Q0:    16.32\nEpisode:  72\/20000 | Episode reward:    11.85 | Episode steps:  140 | Average reward:    17.57 | Step Count: 15983 | Episode Q0:    18.68\nEpisode:  73\/20000 | Episode reward:     4.54 | Episode steps:  289 | Average reward:    17.20 | Step Count: 16272 | Episode Q0:     3.68\nEpisode:  74\/20000 | Episode reward:    21.75 | Episode steps:  289 | Average reward:    17.61 | Step Count: 16561 | Episode Q0:     9.29\nEpisode:  75\/20000 | Episode reward:    29.98 | Episode steps:  289 | Average reward:    17.94 | Step Count: 16850 | Episode Q0:    23.41\nEpisode:  76\/20000 | Episode reward:     5.82 | Episode steps:   58 | Average reward:    17.64 | Step Count: 16908 | Episode Q0:    12.87\nEpisode:  77\/20000 | Episode reward:    11.41 | Episode steps:  289 | Average reward:    17.59 | Step Count: 17197 | Episode Q0:    20.20\nEpisode:  78\/20000 | Episode reward:    13.06 | Episode steps:  289 | Average reward:    17.57 | Step Count: 17486 | Episode Q0:     7.81\nEpisode:  79\/20000 | Episode reward:    34.13 | Episode steps:  289 | Average reward:    17.89 | Step Count: 17775 | Episode Q0:     0.00\nEpisode:  80\/20000 | Episode reward:    15.65 | Episode steps:   83 | Average reward:    17.73 | Step Count: 17858 | Episode Q0:     8.55\nEpisode:  81\/20000 | Episode reward:     6.55 | Episode steps:  147 | Average reward:    17.79 | Step Count: 18005 | Episode Q0:    17.25\nEpisode:  82\/20000 | Episode reward:    22.68 | Episode steps:  289 | Average reward:    18.09 | Step Count: 18294 | Episode Q0:     9.40\nEpisode:  83\/20000 | Episode reward:     0.46 | Episode steps:  289 | Average reward:    17.90 | Step Count: 18583 | Episode Q0:     5.11\nEpisode:  84\/20000 | Episode reward:    30.32 | Episode steps:  289 | Average reward:    18.29 | Step Count: 18872 | Episode Q0:    19.84\nEpisode:  85\/20000 | Episode reward:    17.27 | Episode steps:  289 | Average reward:    17.18 | Step Count: 19161 | Episode Q0:     4.17\nEpisode:  86\/20000 | Episode reward:    45.40 | Episode steps:  289 | Average reward:    17.64 | Step Count: 19450 | Episode Q0:     6.61\nEpisode:  87\/20000 | Episode reward:    38.01 | Episode steps:  167 | Average reward:    18.35 | Step Count: 19617 | Episode Q0:    14.24\nEpisode:  88\/20000 | Episode reward:    36.13 | Episode steps:  254 | Average reward:    18.76 | Step Count: 19871 | Episode Q0:     8.26\nEpisode:  89\/20000 | Episode reward:    36.42 | Episode steps:  289 | Average reward:    18.85 | Step Count: 20160 | Episode Q0:    10.31\nEpisode:  90\/20000 | Episode reward:     4.75 | Episode steps:  289 | Average reward:    18.46 | Step Count: 20449 | Episode Q0:     5.72\nEpisode:  91\/20000 | Episode reward:    21.41 | Episode steps:  289 | Average reward:    18.51 | Step Count: 20738 | Episode Q0:    12.02\nEpisode:  92\/20000 | Episode reward:    18.02 | Episode steps:  289 | Average reward:    18.47 | Step Count: 21027 | Episode Q0:     9.76\nEpisode:  93\/20000 | Episode reward:     5.05 | Episode steps:  289 | Average reward:    17.91 | Step Count: 21316 | Episode Q0:     8.74\nEpisode:  94\/20000 | Episode reward:    12.77 | Episode steps:   82 | Average reward:    17.88 | Step Count: 21398 | Episode Q0:    16.55\nEpisode:  95\/20000 | Episode reward:    18.03 | Episode steps:  109 | Average reward:    18.09 | Step Count: 21507 | Episode Q0:     8.97\nEpisode:  96\/20000 | Episode reward:    15.36 | Episode steps:  289 | Average reward:    17.90 | Step Count: 21796 | Episode Q0:     8.74\nEpisode:  97\/20000 | Episode reward:     9.10 | Episode steps:  289 | Average reward:    17.88 | Step Count: 22085 | Episode Q0:    19.65\nEpisode:  98\/20000 | Episode reward:     8.91 | Episode steps:   79 | Average reward:    18.05 | Step Count: 22164 | Episode Q0:    12.21\nEpisode:  99\/20000 | Episode reward:    24.40 | Episode steps:   73 | Average reward:    18.33 | Step Count: 22237 | Episode Q0:    12.88\nEpisode: 100\/20000 | Episode reward:    14.39 | Episode steps:   98 | Average reward:    18.53 | Step Count: 22335 | Episode Q0:     5.42\nEpisode: 101\/20000 | Episode reward:     7.35 | Episode steps:   91 | Average reward:    18.28 | Step Count: 22426 | Episode Q0:    11.64\nEpisode: 102\/20000 | Episode reward:    34.79 | Episode steps:  289 | Average reward:    18.93 | Step Count: 22715 | Episode Q0:     4.66\nEpisode: 103\/20000 | Episode reward:     4.72 | Episode steps:    8 | Average reward:    17.73 | Step Count: 22723 | Episode Q0:     5.91\nEpisode: 104\/20000 | Episode reward:    19.24 | Episode steps:  289 | Average reward:    17.62 | Step Count: 23012 | Episode Q0:     0.20\nEpisode: 105\/20000 | Episode reward:     4.87 | Episode steps:  289 | Average reward:    17.44 | Step Count: 23301 | Episode Q0:    11.91\nEpisode: 106\/20000 | Episode reward:    18.32 | Episode steps:  181 | Average reward:    17.76 | Step Count: 23482 | Episode Q0:     6.48\nEpisode: 107\/20000 | Episode reward:     4.39 | Episode steps:  289 | Average reward:    17.70 | Step Count: 23771 | Episode Q0:     5.73\nEpisode: 108\/20000 | Episode reward:     5.31 | Episode steps:  289 | Average reward:    17.63 | Step Count: 24060 | Episode Q0:     3.41\nEpisode: 109\/20000 | Episode reward:    21.82 | Episode steps:  289 | Average reward:    17.43 | Step Count: 24349 | Episode Q0:     5.61\nEpisode: 110\/20000 | Episode reward:     1.61 | Episode steps:    2 | Average reward:    16.96 | Step Count: 24351 | Episode Q0:     8.75\nEpisode: 111\/20000 | Episode reward:     7.26 | Episode steps:  289 | Average reward:    17.09 | Step Count: 24640 | Episode Q0:    13.35\nEpisode: 112\/20000 | Episode reward:     1.00 | Episode steps:    1 | Average reward:    16.57 | Step Count: 24641 | Episode Q0:     3.49\nEpisode: 113\/20000 | Episode reward:     4.28 | Episode steps:  289 | Average reward:    16.48 | Step Count: 24930 | Episode Q0:     8.75\nEpisode: 114\/20000 | Episode reward:     6.15 | Episode steps:  203 | Average reward:    16.32 | Step Count: 25133 | Episode Q0:     5.27\nEpisode: 115\/20000 | Episode reward:     0.00 | Episode steps:  289 | Average reward:    15.73 | Step Count: 25422 | Episode Q0:     9.87\nEpisode: 116\/20000 | Episode reward:    11.30 | Episode steps:  103 | Average reward:    15.70 | Step Count: 25525 | Episode Q0:    28.55\nEpisode: 117\/20000 | Episode reward:     8.65 | Episode steps:  289 | Average reward:    15.46 | Step Count: 25814 | Episode Q0:     3.07\nEpisode: 118\/20000 | Episode reward:     1.00 | Episode steps:    1 | Average reward:    14.88 | Step Count: 25815 | Episode Q0:     3.33\nEpisode: 119\/20000 | Episode reward:    62.53 | Episode steps:  109 | Average reward:    15.69 | Step Count: 25924 | Episode Q0:     5.23\nEpisode: 120\/20000 | Episode reward:    15.64 | Episode steps:   58 | Average reward:    15.89 | Step Count: 25982 | Episode Q0:    12.76\nEpisode: 121\/20000 | Episode reward:    61.71 | Episode steps:  289 | Average reward:    16.71 | Step Count: 26271 | Episode Q0:     1.83\nEpisode: 122\/20000 | Episode reward:    29.44 | Episode steps:  289 | Average reward:    17.06 | Step Count: 26560 | Episode Q0:     1.76\nEpisode: 123\/20000 | Episode reward:    35.52 | Episode steps:  289 | Average reward:    17.68 | Step Count: 26849 | Episode Q0:     5.08\nEpisode: 124\/20000 | Episode reward:    50.69 | Episode steps:  289 | Average reward:    18.26 | Step Count: 27138 | Episode Q0:     4.45\nEpisode: 125\/20000 | Episode reward:    11.15 | Episode steps:  289 | Average reward:    17.89 | Step Count: 27427 | Episode Q0:     7.52\nEpisode: 126\/20000 | Episode reward:     9.31 | Episode steps:   49 | Average reward:    17.95 | Step Count: 27476 | Episode Q0:     0.00\nEpisode: 127\/20000 | Episode reward:    18.58 | Episode steps:  229 | Average reward:    18.10 | Step Count: 27705 | Episode Q0:     4.48\nEpisode: 128\/20000 | Episode reward:     5.33 | Episode steps:   66 | Average reward:    17.94 | Step Count: 27771 | Episode Q0:     2.04\nEpisode: 129\/20000 | Episode reward:    23.75 | Episode steps:  289 | Average reward:    17.74 | Step Count: 28060 | Episode Q0:     1.95\nEpisode: 130\/20000 | Episode reward:    43.84 | Episode steps:  289 | Average reward:    18.30 | Step Count: 28349 | Episode Q0:     3.04\nEpisode: 131\/20000 | Episode reward:    19.66 | Episode steps:   54 | Average reward:    18.56 | Step Count: 28403 | Episode Q0:     1.88\nEpisode: 132\/20000 | Episode reward:     3.49 | Episode steps:   61 | Average reward:    18.18 | Step Count: 28464 | Episode Q0:     0.26\nEpisode: 133\/20000 | Episode reward:     7.08 | Episode steps:   92 | Average reward:    18.31 | Step Count: 28556 | Episode Q0:     6.79\nEpisode: 134\/20000 | Episode reward:    21.32 | Episode steps:  197 | Average reward:    18.13 | Step Count: 28753 | Episode Q0:     3.58\nEpisode: 135\/20000 | Episode reward:     2.28 | Episode steps:  289 | Average reward:    17.83 | Step Count: 29042 | Episode Q0:     4.00\nEpisode: 136\/20000 | Episode reward:    17.01 | Episode steps:   49 | Average reward:    17.26 | Step Count: 29091 | Episode Q0:     4.99\nEpisode: 137\/20000 | Episode reward:    20.50 | Episode steps:  156 | Average reward:    16.91 | Step Count: 29247 | Episode Q0:     9.55\nEpisode: 138\/20000 | Episode reward:    42.34 | Episode steps:  289 | Average reward:    17.04 | Step Count: 29536 | Episode Q0:     5.43\nEpisode: 139\/20000 | Episode reward:    20.86 | Episode steps:  289 | Average reward:    16.73 | Step Count: 29825 | Episode Q0:     2.99\n","truncated":false}}
%---
