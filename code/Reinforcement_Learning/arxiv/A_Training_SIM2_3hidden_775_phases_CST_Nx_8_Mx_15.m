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
Parameters;   % loads all base variables into workspace and EnvPars %[output:7b67a6af] %[output:5066e828] %[output:60704393] %[output:8cbe96c6] %[output:979744fe] %[output:1c40a258] %[output:24f74f12] %[output:580d000f] %[output:301a220d]

%Update the G matrix to use the one computed with CST
% EnvPars.G = EnvPars.G_CST;   % SIM-2 uses the CST-realistic SIM-1 front-end
% EnvPars.U_func = EnvPars.U_func_CST;

Calibration %[output:67a785f4]

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

fprintf('SIM2 geometry: W0_SIM2 %dx%d (N=%d->M=%d), W_M9 %dx%d (M=%d->%d ports)\n', ... %[output:group:9ba73d97] %[output:894153d4]
    size(W0_SIM2,1), size(W0_SIM2,2), N, M, size(W_M9,1), size(W_M9,2), M, Q); %[output:group:9ba73d97] %[output:894153d4]
fprintf('max(abs(W0_SIM2))=%.3e , max(abs(W_M9))=%.3e\n', max(abs(W0_SIM2(:))), max(abs(W_M9(:)))); %[output:4c5e79ab]
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

fprintf('=== Training phase ===\n'); %[output:874d7ac0]
fprintf('N_cal=%d  |  MaxEpisodes=%d  |  MaxSteps=%d  |  n_actions=%d\n', ... %[output:group:46f33b65] %[output:2d1accb5]
        N_cal, EnvPars.MaxEpisodes, EnvPars.MaxStepsPerEpisode, EnvPars.n_actions); %[output:group:46f33b65] %[output:2d1accb5]

trainingStats = train(agent, env, trainOpts); %[output:48f05990]
%%
%[text] ## ── 9. SAVE ──────────────────────────────────────────────────────────────
%[text] NOTE: filename no longer references "Neurons\_per\_layer" -- that variable (25\*25=625) never matched any actual layer width (144=T\_x\*T\_y, 225=M\_x\*M\_y) even in the original script, and SIM2 doesn't have a single "neuron count" analog anyway (241 trainable phases across two layers + 1 frozen layer). Original line was also missing a closing paren on fullfile(); fixed here.
save_path = fullfile('..', 'Dataset', 'dqn_agent_navigation_775_phases_CST_Nx_8_Mx_15.mat');
criticNet = getModel(getCritic(agent));
save(save_path, 'agent', 'trainingStats', 'criticNet', 'EnvPars');
fprintf('Agent saved to %s\n', save_path); %[output:38a5c96e]
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
%   data: {"dataType":"textualVariable","outputData":{"name":"N","value":"64"}}
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
%[output:67a785f4]
%   data: {"dataType":"text","outputData":{"text":"=== Calibration phase ===\nGrid: 10 x 10 = 100 calibration positions\nCalibration complete.\n\n","truncated":false}}
%---
%[output:894153d4]
%   data: {"dataType":"text","outputData":{"text":"SIM2 geometry: W0_SIM2 225x64 (N=64->M=225), W_M9 9x225 (M=225->9 ports)\n","truncated":false}}
%---
%[output:4c5e79ab]
%   data: {"dataType":"text","outputData":{"text":"max(abs(W0_SIM2))=2.743e-01 , max(abs(W_M9))=2.748e-01\n","truncated":false}}
%---
%[output:874d7ac0]
%   data: {"dataType":"text","outputData":{"text":"=== Training phase ===\n","truncated":false}}
%---
%[output:2d1accb5]
%   data: {"dataType":"text","outputData":{"text":"N_cal=100  |  MaxEpisodes=50  |  MaxSteps=289  |  n_actions=9\n","truncated":false}}
%---
%[output:48f05990]
%   data: {"dataType":"text","outputData":{"text":"Episode:   1\/ 50 | Episode reward:    13.25 | Episode steps:   64 | Average reward:    13.25 | Step Count:   64 | Episode Q0:   437.12\nEpisode:   2\/ 50 | Episode reward:     3.11 | Episode steps:  289 | Average reward:     8.18 | Step Count:  353 | Episode Q0:   445.73\nEpisode:   3\/ 50 | Episode reward:    11.31 | Episode steps:   59 | Average reward:     9.22 | Step Count:  412 | Episode Q0:   265.92\nEpisode:   4\/ 50 | Episode reward:    31.23 | Episode steps:  289 | Average reward:    14.72 | Step Count:  701 | Episode Q0:   347.03\nEpisode:   5\/ 50 | Episode reward:    25.21 | Episode steps:  289 | Average reward:    16.82 | Step Count:  990 | Episode Q0:   163.82\nEpisode:   6\/ 50 | Episode reward:    13.96 | Episode steps:  289 | Average reward:    16.34 | Step Count: 1279 | Episode Q0:   165.58\nEpisode:   7\/ 50 | Episode reward:     2.15 | Episode steps:  289 | Average reward:    14.32 | Step Count: 1568 | Episode Q0:   244.55\nEpisode:   8\/ 50 | Episode reward:     5.19 | Episode steps:  289 | Average reward:    13.18 | Step Count: 1857 | Episode Q0:    67.86\nEpisode:   9\/ 50 | Episode reward:    14.93 | Episode steps:   45 | Average reward:    13.37 | Step Count: 1902 | Episode Q0:   385.17\nEpisode:  10\/ 50 | Episode reward:     2.66 | Episode steps:  289 | Average reward:    12.30 | Step Count: 2191 | Episode Q0:    52.20\nEpisode:  11\/ 50 | Episode reward:     7.97 | Episode steps:  289 | Average reward:    11.91 | Step Count: 2480 | Episode Q0:   336.37\nEpisode:  12\/ 50 | Episode reward:    13.81 | Episode steps:  237 | Average reward:    12.07 | Step Count: 2717 | Episode Q0:   102.47\nEpisode:  13\/ 50 | Episode reward:    19.00 | Episode steps:   69 | Average reward:    12.60 | Step Count: 2786 | Episode Q0:   383.81\nEpisode:  14\/ 50 | Episode reward:     1.14 | Episode steps:  289 | Average reward:    11.78 | Step Count: 3075 | Episode Q0:    73.57\nEpisode:  15\/ 50 | Episode reward:    18.93 | Episode steps:   37 | Average reward:    12.26 | Step Count: 3112 | Episode Q0:   216.48\nEpisode:  16\/ 50 | Episode reward:     0.92 | Episode steps:  289 | Average reward:    11.55 | Step Count: 3401 | Episode Q0:   294.47\nEpisode:  17\/ 50 | Episode reward:    23.52 | Episode steps:  289 | Average reward:    12.25 | Step Count: 3690 | Episode Q0:   439.40\nEpisode:  18\/ 50 | Episode reward:     2.88 | Episode steps:  289 | Average reward:    11.73 | Step Count: 3979 | Episode Q0:   356.84\nEpisode:  19\/ 50 | Episode reward:    21.66 | Episode steps:  289 | Average reward:    12.25 | Step Count: 4268 | Episode Q0:   152.91\nEpisode:  20\/ 50 | Episode reward:    12.64 | Episode steps:  289 | Average reward:    12.27 | Step Count: 4557 | Episode Q0:   184.70\nEpisode:  21\/ 50 | Episode reward:    23.67 | Episode steps:  100 | Average reward:    12.82 | Step Count: 4657 | Episode Q0:   172.73\nEpisode:  22\/ 50 | Episode reward:     8.86 | Episode steps:  289 | Average reward:    12.64 | Step Count: 4946 | Episode Q0:   498.86\nEpisode:  23\/ 50 | Episode reward:    19.44 | Episode steps:  289 | Average reward:    12.93 | Step Count: 5235 | Episode Q0:   367.88\nEpisode:  24\/ 50 | Episode reward:    15.72 | Episode steps:  289 | Average reward:    13.05 | Step Count: 5524 | Episode Q0:   360.37\nEpisode:  25\/ 50 | Episode reward:    37.89 | Episode steps:  184 | Average reward:    14.04 | Step Count: 5708 | Episode Q0:    63.77\nEpisode:  26\/ 50 | Episode reward:    16.52 | Episode steps:   52 | Average reward:    14.14 | Step Count: 5760 | Episode Q0:   266.19\nEpisode:  27\/ 50 | Episode reward:    23.43 | Episode steps:  289 | Average reward:    14.48 | Step Count: 6049 | Episode Q0:   147.51\nEpisode:  28\/ 50 | Episode reward:     4.21 | Episode steps:   61 | Average reward:    14.12 | Step Count: 6110 | Episode Q0:   142.77\nEpisode:  29\/ 50 | Episode reward:     5.93 | Episode steps:    7 | Average reward:    13.83 | Step Count: 6117 | Episode Q0:   177.40\nEpisode:  30\/ 50 | Episode reward:     6.72 | Episode steps:  289 | Average reward:    13.60 | Step Count: 6406 | Episode Q0:   129.75\nEpisode:  31\/ 50 | Episode reward:    11.58 | Episode steps:   64 | Average reward:    13.53 | Step Count: 6470 | Episode Q0:   120.56\nEpisode:  32\/ 50 | Episode reward:    15.23 | Episode steps:  200 | Average reward:    13.58 | Step Count: 6670 | Episode Q0:     0.00\nEpisode:  33\/ 50 | Episode reward:     8.17 | Episode steps:   25 | Average reward:    13.42 | Step Count: 6695 | Episode Q0:   181.36\nEpisode:  34\/ 50 | Episode reward:     0.00 | Episode steps:  289 | Average reward:    13.02 | Step Count: 6984 | Episode Q0:   151.81\nEpisode:  35\/ 50 | Episode reward:    17.79 | Episode steps:  289 | Average reward:    13.16 | Step Count: 7273 | Episode Q0:   187.50\nEpisode:  36\/ 50 | Episode reward:    17.81 | Episode steps:  289 | Average reward:    13.29 | Step Count: 7562 | Episode Q0:   178.21\nEpisode:  37\/ 50 | Episode reward:     0.00 | Episode steps:  289 | Average reward:    12.93 | Step Count: 7851 | Episode Q0:   164.05\nEpisode:  38\/ 50 | Episode reward:     8.06 | Episode steps:  289 | Average reward:    12.80 | Step Count: 8140 | Episode Q0:   108.91\nEpisode:  39\/ 50 | Episode reward:    48.48 | Episode steps:   93 | Average reward:    13.72 | Step Count: 8233 | Episode Q0:   181.08\nEpisode:  40\/ 50 | Episode reward:    27.16 | Episode steps:  110 | Average reward:    14.05 | Step Count: 8343 | Episode Q0:   160.92\nEpisode:  41\/ 50 | Episode reward:    34.18 | Episode steps:  289 | Average reward:    14.54 | Step Count: 8632 | Episode Q0:   261.56\nEpisode:  42\/ 50 | Episode reward:    17.55 | Episode steps:  289 | Average reward:    14.62 | Step Count: 8921 | Episode Q0:   234.44\nEpisode:  43\/ 50 | Episode reward:    17.20 | Episode steps:  289 | Average reward:    14.68 | Step Count: 9210 | Episode Q0:    91.57\nEpisode:  44\/ 50 | Episode reward:    14.95 | Episode steps:  289 | Average reward:    14.68 | Step Count: 9499 | Episode Q0:   127.95\nEpisode:  45\/ 50 | Episode reward:    37.38 | Episode steps:  289 | Average reward:    15.19 | Step Count: 9788 | Episode Q0:    58.59\nEpisode:  46\/ 50 | Episode reward:    29.75 | Episode steps:  289 | Average reward:    15.50 | Step Count: 10077 | Episode Q0:   205.85\nEpisode:  47\/ 50 | Episode reward:    26.47 | Episode steps:   46 | Average reward:    15.74 | Step Count: 10123 | Episode Q0:   127.44\nEpisode:  48\/ 50 | Episode reward:    21.49 | Episode steps:  289 | Average reward:    15.86 | Step Count: 10412 | Episode Q0:    97.18\nEpisode:  49\/ 50 | Episode reward:    17.46 | Episode steps:   26 | Average reward:    15.89 | Step Count: 10438 | Episode Q0:    86.39\nEpisode:  50\/ 50 | Episode reward:    10.41 | Episode steps:  289 | Average reward:    15.78 | Step Count: 10727 | Episode Q0:   116.44\n","truncated":false}}
%---
%[output:38a5c96e]
%   data: {"dataType":"text","outputData":{"text":"Agent saved to ..\\Dataset\\dqn_agent_navigation_775_phases_CST_Nx_8_Mx_15.mat\n","truncated":false}}
%---
