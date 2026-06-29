%[text] # Training script — SIM Navigation with DQN, SIM2 Q-network (3 hidden M-layers)
%[text] VARIANT with THREE hidden M-layers. Architecture: N -\> \[propN-\>M\] -\> M -\> \[propM-\>M\] -\> M -\> \[propM-\>M\] -\> M -\> \[propM-\>9\] -\> diode 691 trainable phases (16+225+225+225) vs 466 (2-hidden) and 241 (1-hidden). Saves to dqn\_agent\_navigation\_SIM2\_3hidden.mat -- the 1- and 2-hidden results are left untouched, giving a clean 1/2/3 M-layer ablation sweep.
%[text] CAVEAT, now stronger: three phase masks separated only by fixed propagation, with the ONLY nonlinearity at the diode. The deeper this goes the more likely training is to stall or plateau no higher than shallower versions -- watch the gradient norms (verify script) and the learning curve. If 3-hidden does not beat 2-hidden, that is the reportable finding (diminishing/negative returns from all-phase depth), not a bug to fix.
%[text] Requires Parameters.m to have been run first (this script calls it)
%[text] Built incrementally with Claude -- all four parts done, gradient- verified per-layer (see SIM2\_GradientCheck\_\*.m, run separately): Part 1 \[DONE\] -- SIM2 geometry & propagation matrices (W0\_SIM2, W\_M9) Part 2 \[DONE\] -- custom layers: simPhaseLayerCST (N), simPhaseLayerCST (M) Part 3 \[DONE\] -- realToComplexLayer, diodeReadoutLayer, dlnetwork assembly Part 4 \[DONE\] -- ObsInfo override (2N-dim, local to this script), stepFunction\_nav\_CST.m / resetFunction\_nav.m updated to emit \[Re(r);Im(r)\] (t\_x,t\_y tracked in silico, not observed)
%[text] REVISED again after a real platform limitation: MATLAB custom layers cannot return complex values from predict()/forward() (confirmed via MathWorks doc, not just inferred from the error). Every layer (simPhaseLayerCST, simPropagationLayer, diodeReadoutLayer) now operates on REAL-STACKED \[Re;Im\] representations throughout -- see each layer file for the re-derived math. to\_complex is GONE entirely: with everything real-stacked end to end, the observation \[Re(r);Im(r)\] feeds sim2\_layer1 directly. realImagToComplexLayer.m and realToComplexLayer.m are both now historical/unused. Gradient checks were REDONE (not just re-run) against the new math -- see SIM2\_GradientCheck\_PhaseLayer.m, SIM2\_GradientCheck\_PropagationLayer.m, SIM2\_GradientCheck\_DiodeReadout.m.
%[text] NOT yet done, separate from this architecture: EnvPars.U\_func (v0) and G still use the idealized analytic models -- the CST amplitude-phase coupling on the SIM1 input layer, and loading the actual trained (CST-aware) G from SIM\_Training\_CST\_SingleZeta\_Parallel.m instead of the closed-form DFT kernel, are both still open from earlier in this project and independent of the SIM2 work above.
%[text] STILL NEEDED before trusting a real training run: - End-to-end check: does criticNet/predict() actually run on a real observation without shape/format errors? (per-layer checks passed, but the full chain hasn't been exercised together yet) - Capacity sanity check: 241 trainable phases vs. the old FC network's ~20k+ weights -- worth comparing against the original training curve (Fig. 6) once this trains, not just checking that it learns at all
clc; clear all; close all;
addingPathParentFolderByName('code'); %[output:5c82a229]
Parameters;   % loads all base variables into workspace and EnvPars %[output:7b67a6af] %[output:5066e828] %[output:60704393] %[output:8cbe96c6] %[output:979744fe] %[output:1c40a258] %[output:24f74f12] %[output:580d000f] %[output:301a220d] %[output:4c101c86]

%Update the G matrix to use the one computed with CST
EnvPars.G = EnvPars.G_CST;   % SIM-2 uses the CST-realistic SIM-1 front-end
EnvPars.U_func = EnvPars.U_func_CST;

Calibration %[output:15820be8]

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

fprintf('SIM2 geometry: W0_SIM2 %dx%d (N=%d->M=%d), W_M9 %dx%d (M=%d->%d ports)\n', ... %[output:group:874bddc9] %[output:6f58218e]
    size(W0_SIM2,1), size(W0_SIM2,2), N, M, size(W_M9,1), size(W_M9,2), M, Q); %[output:group:874bddc9] %[output:6f58218e]
fprintf('max(abs(W0_SIM2))=%.3e , max(abs(W_M9))=%.3e\n', max(abs(W0_SIM2(:))), max(abs(W_M9(:)))); %[output:4f387be9]
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

fprintf('=== Training phase ===\n'); %[output:559c5493]
fprintf('N_cal=%d  |  MaxEpisodes=%d  |  MaxSteps=%d  |  n_actions=%d\n', ... %[output:group:91a85f37] %[output:6c4b9b6a]
        N_cal, EnvPars.MaxEpisodes, EnvPars.MaxStepsPerEpisode, EnvPars.n_actions); %[output:group:91a85f37] %[output:6c4b9b6a]

trainingStats = train(agent, env, trainOpts); %[output:56826432]
%%
%[text] ## ── 9. SAVE ──────────────────────────────────────────────────────────────
%[text] NOTE: filename no longer references "Neurons\_per\_layer" -- that variable (25\*25=625) never matched any actual layer width (144=T\_x\*T\_y, 225=M\_x\*M\_y) even in the original script, and SIM2 doesn't have a single "neuron count" analog anyway (241 trainable phases across two layers + 1 frozen layer). Original line was also missing a closing paren on fullfile(); fixed here.
save_path = fullfile('..', 'Dataset', 'dqn_agent_navigation_775_phases_CST_Nx_8_Mx_15.mat');
criticNet = getModel(getCritic(agent));
save(save_path, 'agent', 'trainingStats', 'criticNet', 'EnvPars');
fprintf('Agent saved to %s\n', save_path); %[output:5cd9bb71]
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
%[output:5c82a229]
%   data: {"dataType":"text","outputData":{"text":"Adding matlab path to: G:\\My Drive\\Work\\Research\\SIM\\code\n","truncated":false}}
%---
%[output:7b67a6af]
%   data: {"dataType":"textualVariable","outputData":{"name":"total_iteration","value":"1"}}
%---
%[output:5066e828]
%   data: {"dataType":"text","outputData":{"text":"Wireless packet type: SC\n","truncated":false}}
%---
%[output:60704393]
%   data: {"dataType":"textualVariable","outputData":{"name":"N","value":"16"}}
%---
%[output:8cbe96c6]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_x","value":"15"}}
%---
%[output:979744fe]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_y","value":"15"}}
%---
%[output:1c40a258]
%   data: {"dataType":"matrix","outputData":{"columns":6,"name":"zeta","rows":1,"type":"double","value":[["0.9850","0.9860","0.9870","0.9880","0.9890","0.9900"]]}}
%---
%[output:24f74f12]
%   data: {"dataType":"textualVariable","outputData":{"name":"T_coh","value":"0.0038"}}
%---
%[output:580d000f]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_packets_coh","value":"12"}}
%---
%[output:301a220d]
%   data: {"dataType":"textualVariable","outputData":{"name":"SNR_dB","value":"36.5005"}}
%---
%[output:4c101c86]
%   data: {"dataType":"text","outputData":{"text":"Loaded CST SIM-1 G_CST. Deviation from analytic G: 1.806%n","truncated":false}}
%---
%[output:15820be8]
%   data: {"dataType":"text","outputData":{"text":"=== Calibration phase ===\nGrid: 10 x 10 = 100 calibration positions\nCalibration complete.\n\n","truncated":false}}
%---
%[output:6f58218e]
%   data: {"dataType":"text","outputData":{"text":"SIM2 geometry: W0_SIM2 225x16 (N=16->M=225), W_M9 9x225 (M=225->9 ports)\n","truncated":false}}
%---
%[output:4f387be9]
%   data: {"dataType":"text","outputData":{"text":"max(abs(W0_SIM2))=2.628e-01 , max(abs(W_M9))=2.748e-01\n","truncated":false}}
%---
%[output:559c5493]
%   data: {"dataType":"text","outputData":{"text":"=== Training phase ===\n","truncated":false}}
%---
%[output:6c4b9b6a]
%   data: {"dataType":"text","outputData":{"text":"N_cal=100  |  MaxEpisodes=100  |  MaxSteps=289  |  n_actions=9\n","truncated":false}}
%---
%[output:56826432]
%   data: {"dataType":"text","outputData":{"text":"Episode:   1\/100 | Episode reward:    12.33 | Episode steps:   42 | Average reward:    12.33 | Step Count:   42 | Episode Q0:    60.00\nEpisode:   2\/100 | Episode reward:    15.78 | Episode steps:  289 | Average reward:    14.05 | Step Count:  331 | Episode Q0:    71.00\nEpisode:   3\/100 | Episode reward:    16.60 | Episode steps:  118 | Average reward:    14.90 | Step Count:  449 | Episode Q0:    63.07\nEpisode:   4\/100 | Episode reward:    14.83 | Episode steps:  136 | Average reward:    14.89 | Step Count:  585 | Episode Q0:    53.73\nEpisode:   5\/100 | Episode reward:    45.51 | Episode steps:  289 | Average reward:    21.01 | Step Count:  874 | Episode Q0:    28.91\nEpisode:   6\/100 | Episode reward:     7.89 | Episode steps:  289 | Average reward:    18.82 | Step Count: 1163 | Episode Q0:   123.89\nEpisode:   7\/100 | Episode reward:     5.78 | Episode steps:  289 | Average reward:    16.96 | Step Count: 1452 | Episode Q0:    18.09\nEpisode:   8\/100 | Episode reward:    35.00 | Episode steps:  151 | Average reward:    19.21 | Step Count: 1603 | Episode Q0:   122.92\nEpisode:   9\/100 | Episode reward:    63.36 | Episode steps:  289 | Average reward:    24.12 | Step Count: 1892 | Episode Q0:    30.72\nEpisode:  10\/100 | Episode reward:    10.22 | Episode steps:   98 | Average reward:    22.73 | Step Count: 1990 | Episode Q0:   112.54\nEpisode:  11\/100 | Episode reward:     4.27 | Episode steps:  289 | Average reward:    21.05 | Step Count: 2279 | Episode Q0:    85.21\nEpisode:  12\/100 | Episode reward:    30.56 | Episode steps:  221 | Average reward:    21.84 | Step Count: 2500 | Episode Q0:    44.79\nEpisode:  13\/100 | Episode reward:    10.76 | Episode steps:  289 | Average reward:    20.99 | Step Count: 2789 | Episode Q0:    19.27\nEpisode:  14\/100 | Episode reward:     1.20 | Episode steps:  289 | Average reward:    19.58 | Step Count: 3078 | Episode Q0:    80.14\nEpisode:  15\/100 | Episode reward:    21.37 | Episode steps:  283 | Average reward:    19.70 | Step Count: 3361 | Episode Q0:    74.89\nEpisode:  16\/100 | Episode reward:    36.10 | Episode steps:  252 | Average reward:    20.72 | Step Count: 3613 | Episode Q0:   126.63\nEpisode:  17\/100 | Episode reward:     1.89 | Episode steps:  289 | Average reward:    19.61 | Step Count: 3902 | Episode Q0:    50.29\nEpisode:  18\/100 | Episode reward:    11.99 | Episode steps:   63 | Average reward:    19.19 | Step Count: 3965 | Episode Q0:    43.80\nEpisode:  19\/100 | Episode reward:     8.51 | Episode steps:  289 | Average reward:    18.63 | Step Count: 4254 | Episode Q0:    87.96\nEpisode:  20\/100 | Episode reward:    12.73 | Episode steps:  289 | Average reward:    18.33 | Step Count: 4543 | Episode Q0:    46.58\nEpisode:  21\/100 | Episode reward:    30.40 | Episode steps:  282 | Average reward:    18.91 | Step Count: 4825 | Episode Q0:    46.60\nEpisode:  22\/100 | Episode reward:    47.36 | Episode steps:  282 | Average reward:    20.20 | Step Count: 5107 | Episode Q0:    39.19\nEpisode:  23\/100 | Episode reward:    16.03 | Episode steps:  289 | Average reward:    20.02 | Step Count: 5396 | Episode Q0:    79.39\nEpisode:  24\/100 | Episode reward:    28.35 | Episode steps:  257 | Average reward:    20.37 | Step Count: 5653 | Episode Q0:   154.18\nEpisode:  25\/100 | Episode reward:    12.53 | Episode steps:   35 | Average reward:    20.05 | Step Count: 5688 | Episode Q0:    62.48\nEpisode:  26\/100 | Episode reward:    10.48 | Episode steps:  289 | Average reward:    19.69 | Step Count: 5977 | Episode Q0:    47.50\nEpisode:  27\/100 | Episode reward:     6.15 | Episode steps:   10 | Average reward:    19.18 | Step Count: 5987 | Episode Q0:    30.50\nEpisode:  28\/100 | Episode reward:     8.20 | Episode steps:  289 | Average reward:    18.79 | Step Count: 6276 | Episode Q0:    24.08\nEpisode:  29\/100 | Episode reward:    14.09 | Episode steps:  164 | Average reward:    18.63 | Step Count: 6440 | Episode Q0:    26.72\nEpisode:  30\/100 | Episode reward:     0.26 | Episode steps:  289 | Average reward:    18.02 | Step Count: 6729 | Episode Q0:    64.93\nEpisode:  31\/100 | Episode reward:    28.89 | Episode steps:  289 | Average reward:    18.37 | Step Count: 7018 | Episode Q0:    79.13\nEpisode:  32\/100 | Episode reward:    21.20 | Episode steps:  289 | Average reward:    18.46 | Step Count: 7307 | Episode Q0:    67.03\nEpisode:  33\/100 | Episode reward:    13.01 | Episode steps:  289 | Average reward:    18.29 | Step Count: 7596 | Episode Q0:    72.26\nEpisode:  34\/100 | Episode reward:     6.24 | Episode steps:   17 | Average reward:    17.94 | Step Count: 7613 | Episode Q0:    47.25\nEpisode:  35\/100 | Episode reward:    20.68 | Episode steps:  289 | Average reward:    18.02 | Step Count: 7902 | Episode Q0:    70.45\nEpisode:  36\/100 | Episode reward:    13.42 | Episode steps:   39 | Average reward:    17.89 | Step Count: 7941 | Episode Q0:    50.15\nEpisode:  37\/100 | Episode reward:     3.61 | Episode steps:    5 | Average reward:    17.50 | Step Count: 7946 | Episode Q0:    24.65\nEpisode:  38\/100 | Episode reward:    21.41 | Episode steps:  289 | Average reward:    17.60 | Step Count: 8235 | Episode Q0:    63.72\nEpisode:  39\/100 | Episode reward:     6.12 | Episode steps:  222 | Average reward:    17.31 | Step Count: 8457 | Episode Q0:    56.09\nEpisode:  40\/100 | Episode reward:    28.50 | Episode steps:  255 | Average reward:    17.59 | Step Count: 8712 | Episode Q0:    51.24\nEpisode:  41\/100 | Episode reward:    37.92 | Episode steps:  134 | Average reward:    18.09 | Step Count: 8846 | Episode Q0:    32.24\nEpisode:  42\/100 | Episode reward:    18.60 | Episode steps:   86 | Average reward:    18.10 | Step Count: 8932 | Episode Q0:    55.01\nEpisode:  43\/100 | Episode reward:    31.57 | Episode steps:  279 | Average reward:    18.41 | Step Count: 9211 | Episode Q0:    91.92\nEpisode:  44\/100 | Episode reward:    15.21 | Episode steps:  289 | Average reward:    18.34 | Step Count: 9500 | Episode Q0:     0.00\nEpisode:  45\/100 | Episode reward:    44.75 | Episode steps:  289 | Average reward:    18.93 | Step Count: 9789 | Episode Q0:     0.00\nEpisode:  46\/100 | Episode reward:    54.31 | Episode steps:  233 | Average reward:    19.69 | Step Count: 10022 | Episode Q0:    55.98\nEpisode:  47\/100 | Episode reward:    36.33 | Episode steps:  289 | Average reward:    20.05 | Step Count: 10311 | Episode Q0:    46.11\nEpisode:  48\/100 | Episode reward:     7.18 | Episode steps:   37 | Average reward:    19.78 | Step Count: 10348 | Episode Q0:    33.97\nEpisode:  49\/100 | Episode reward:     0.12 | Episode steps:  289 | Average reward:    19.38 | Step Count: 10637 | Episode Q0:    47.48\nEpisode:  50\/100 | Episode reward:     7.51 | Episode steps:  289 | Average reward:    19.14 | Step Count: 10926 | Episode Q0:    50.98\nEpisode:  51\/100 | Episode reward:    32.31 | Episode steps:  273 | Average reward:    19.54 | Step Count: 11199 | Episode Q0:    54.09\nEpisode:  52\/100 | Episode reward:     8.77 | Episode steps:   56 | Average reward:    19.40 | Step Count: 11255 | Episode Q0:    41.89\nEpisode:  53\/100 | Episode reward:    33.66 | Episode steps:  289 | Average reward:    19.74 | Step Count: 11544 | Episode Q0:    41.52\nEpisode:  54\/100 | Episode reward:     5.35 | Episode steps:   28 | Average reward:    19.55 | Step Count: 11572 | Episode Q0:    22.29\nEpisode:  55\/100 | Episode reward:    10.97 | Episode steps:   43 | Average reward:    18.86 | Step Count: 11615 | Episode Q0:    19.60\nEpisode:  56\/100 | Episode reward:    21.55 | Episode steps:  289 | Average reward:    19.14 | Step Count: 11904 | Episode Q0:    42.18\nEpisode:  57\/100 | Episode reward:     4.19 | Episode steps:   13 | Average reward:    19.10 | Step Count: 11917 | Episode Q0:    19.81\nEpisode:  58\/100 | Episode reward:    51.82 | Episode steps:  289 | Average reward:    19.44 | Step Count: 12206 | Episode Q0:     8.15\nEpisode:  59\/100 | Episode reward:     2.30 | Episode steps:  289 | Average reward:    18.22 | Step Count: 12495 | Episode Q0:    34.16\nEpisode:  60\/100 | Episode reward:    16.27 | Episode steps:  289 | Average reward:    18.34 | Step Count: 12784 | Episode Q0:    36.90\nEpisode:  61\/100 | Episode reward:    29.20 | Episode steps:  192 | Average reward:    18.84 | Step Count: 12976 | Episode Q0:    23.27\nEpisode:  62\/100 | Episode reward:    26.81 | Episode steps:  289 | Average reward:    18.76 | Step Count: 13265 | Episode Q0:    34.51\nEpisode:  63\/100 | Episode reward:     5.33 | Episode steps:   18 | Average reward:    18.65 | Step Count: 13283 | Episode Q0:    32.98\nEpisode:  64\/100 | Episode reward:    36.03 | Episode steps:  289 | Average reward:    19.35 | Step Count: 13572 | Episode Q0:    27.33\nEpisode:  65\/100 | Episode reward:     7.92 | Episode steps:  289 | Average reward:    19.08 | Step Count: 13861 | Episode Q0:     7.43\nEpisode:  66\/100 | Episode reward:    48.71 | Episode steps:  112 | Average reward:    19.33 | Step Count: 13973 | Episode Q0:    33.00\nEpisode:  67\/100 | Episode reward:    14.63 | Episode steps:  289 | Average reward:    19.59 | Step Count: 14262 | Episode Q0:    24.02\nEpisode:  68\/100 | Episode reward:     0.00 | Episode steps:  289 | Average reward:    19.35 | Step Count: 14551 | Episode Q0:    32.39\nEpisode:  69\/100 | Episode reward:     0.82 | Episode steps:  289 | Average reward:    19.20 | Step Count: 14840 | Episode Q0:    35.99\nEpisode:  70\/100 | Episode reward:    19.31 | Episode steps:  289 | Average reward:    19.33 | Step Count: 15129 | Episode Q0:     6.23\nEpisode:  71\/100 | Episode reward:     4.86 | Episode steps:   13 | Average reward:    18.82 | Step Count: 15142 | Episode Q0:    28.89\nEpisode:  72\/100 | Episode reward:     6.64 | Episode steps:   49 | Average reward:    18.00 | Step Count: 15191 | Episode Q0:    21.47\nEpisode:  73\/100 | Episode reward:    11.91 | Episode steps:  289 | Average reward:    17.92 | Step Count: 15480 | Episode Q0:    25.14\nEpisode:  74\/100 | Episode reward:    22.39 | Episode steps:  279 | Average reward:    17.80 | Step Count: 15759 | Episode Q0:    11.18\nEpisode:  75\/100 | Episode reward:    15.05 | Episode steps:  289 | Average reward:    17.85 | Step Count: 16048 | Episode Q0:    13.90\nEpisode:  76\/100 | Episode reward:    59.08 | Episode steps:  289 | Average reward:    18.82 | Step Count: 16337 | Episode Q0:    29.27\nEpisode:  77\/100 | Episode reward:    30.17 | Episode steps:  289 | Average reward:    19.30 | Step Count: 16626 | Episode Q0:    25.51\nEpisode:  78\/100 | Episode reward:    27.63 | Episode steps:  289 | Average reward:    19.69 | Step Count: 16915 | Episode Q0:    22.51\nEpisode:  79\/100 | Episode reward:     6.68 | Episode steps:   15 | Average reward:    19.54 | Step Count: 16930 | Episode Q0:     7.57\nEpisode:  80\/100 | Episode reward:    17.28 | Episode steps:  289 | Average reward:    19.88 | Step Count: 17219 | Episode Q0:    19.95\nEpisode:  81\/100 | Episode reward:    26.81 | Episode steps:  289 | Average reward:    19.84 | Step Count: 17508 | Episode Q0:    15.95\nEpisode:  82\/100 | Episode reward:    35.61 | Episode steps:  289 | Average reward:    20.13 | Step Count: 17797 | Episode Q0:    15.97\nEpisode:  83\/100 | Episode reward:    34.73 | Episode steps:  289 | Average reward:    20.57 | Step Count: 18086 | Episode Q0:     1.19\nEpisode:  84\/100 | Episode reward:    23.86 | Episode steps:  289 | Average reward:    20.92 | Step Count: 18375 | Episode Q0:    16.29\nEpisode:  85\/100 | Episode reward:    42.76 | Episode steps:  289 | Average reward:    21.36 | Step Count: 18664 | Episode Q0:    14.15\nEpisode:  86\/100 | Episode reward:     5.07 | Episode steps:   33 | Average reward:    21.19 | Step Count: 18697 | Episode Q0:    23.70\nEpisode:  87\/100 | Episode reward:    19.61 | Episode steps:  289 | Average reward:    21.51 | Step Count: 18986 | Episode Q0:    12.31\nEpisode:  88\/100 | Episode reward:     9.19 | Episode steps:  289 | Average reward:    21.27 | Step Count: 19275 | Episode Q0:    10.01\nEpisode:  89\/100 | Episode reward:     5.91 | Episode steps:    7 | Average reward:    21.26 | Step Count: 19282 | Episode Q0:     8.46\nEpisode:  90\/100 | Episode reward:    24.46 | Episode steps:  289 | Average reward:    21.18 | Step Count: 19571 | Episode Q0:    19.62\nEpisode:  91\/100 | Episode reward:     1.41 | Episode steps:  289 | Average reward:    20.45 | Step Count: 19860 | Episode Q0:    10.30\nEpisode:  92\/100 | Episode reward:     2.40 | Episode steps:  289 | Average reward:    20.13 | Step Count: 20149 | Episode Q0:     0.00\nEpisode:  93\/100 | Episode reward:    10.51 | Episode steps:  289 | Average reward:    19.71 | Step Count: 20438 | Episode Q0:     8.37\nEpisode:  94\/100 | Episode reward:    86.35 | Episode steps:  248 | Average reward:    21.13 | Step Count: 20686 | Episode Q0:    14.12\nEpisode:  95\/100 | Episode reward:    30.69 | Episode steps:  289 | Average reward:    20.85 | Step Count: 20975 | Episode Q0:    10.26\nEpisode:  96\/100 | Episode reward:    27.52 | Episode steps:  173 | Average reward:    20.31 | Step Count: 21148 | Episode Q0:    12.62\nEpisode:  97\/100 | Episode reward:    64.50 | Episode steps:  289 | Average reward:    20.88 | Step Count: 21437 | Episode Q0:     4.10\nEpisode:  98\/100 | Episode reward:    23.78 | Episode steps:   48 | Average reward:    21.21 | Step Count: 21485 | Episode Q0:     8.15\nEpisode:  99\/100 | Episode reward:    20.42 | Episode steps:  195 | Average reward:    21.61 | Step Count: 21680 | Episode Q0:    13.94\nEpisode: 100\/100 | Episode reward:    73.83 | Episode steps:  289 | Average reward:    22.94 | Step Count: 21969 | Episode Q0:    14.49\n","truncated":false}}
%---
%[output:5cd9bb71]
%   data: {"dataType":"text","outputData":{"text":"Agent saved to ..\\Dataset\\dqn_agent_navigation_775_phases_CST_Nx_8_Mx_15.mat\n","truncated":false}}
%---
