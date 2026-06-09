%% Training script — SIM Navigation with DQN
% Requires Parameters.mlx to have been run first
clc; clear all; close all;
addingPathParentFolderByName('code'); %[output:084ff5bb]
Parameters;   % loads all base variables into workspace and EnvPars %[output:1b3a6c37] %[output:7e2be390] %[output:94b50f6a] %[output:54970599] %[output:060c976e] %[output:24212b63] %[output:516e2386] %[output:77aaf974] %[output:4d6c3649] %[output:00757844]

Calibration %[output:75e1cdaa]

% In A_Training_Navigation.mlx — after Calibration runs
EnvPars.MaxEpisodes = EnvPars.N_cal * 200;
%%
%% ── 5. ENVIRONMENT ───────────────────────────────────────────────────────
env = rlFunctionEnv(ObsInfo, ActInfo, ...
      @(a,ls) stepFunction_nav(a, ls, EnvPars), ...
      @()     resetFunction_nav(EnvPars));

%% ── 6. Q-NETWORK ─────────────────────────────────────────────────────────
% Input: N+2 dimensional observation
% Output: n_actions Q-values
statePath = [
    featureInputLayer(EnvPars.N + 2, 'Name', 'obs')
    fullyConnectedLayer(144, 'Name', 'fc1')
    % reluLayer('Name', 'relu1')
    fullyConnectedLayer(144, 'Name', 'fc2')
    % reluLayer('Name', 'relu2')
    fullyConnectedLayer(144, 'Name', 'fc3')
    fullyConnectedLayer(EnvPars.n_actions, 'Name', 'output')];

criticNet = dlnetwork(layerGraph(statePath));
critic    = rlVectorQValueFunction(criticNet, ObsInfo, ActInfo);

% plot(criticNet);

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

% trainOpts.UseParallel = true;
%trainOpts.ParallelizationOptions.Mode = "async";

fprintf('=== Training phase ===\n'); %[output:7b222876]
fprintf('N_cal=%d  |  MaxEpisodes=%d  |  MaxSteps=%d  |  n_actions=%d\n', ... %[output:group:8a399eba] %[output:20c546ca]
        N_cal, EnvPars.MaxEpisodes, EnvPars.MaxStepsPerEpisode, EnvPars.n_actions); %[output:group:8a399eba] %[output:20c546ca]

trainingStats = train(agent, env, trainOpts); %[output:9eff4c86]

%% ── 9. SAVE ──────────────────────────────────────────────────────────────
save_path = fullfile('..', 'Dataset', 'dqn_agent_navigation_3layer_144neurons_norelu_decay1.mat');
criticNet = getModel(getCritic(agent));
save(save_path, 'agent', 'trainingStats', 'criticNet', 'EnvPars');
fprintf('Agent saved to %s\n', save_path); %[output:9699bef4]

%[text] #### Functions
function addingPathParentFolderByName(targetName)
    %Place the matlab directory at the right position
    % Start from the current directory
    currFolder = pwd;
    found = false;
    
    % Continue searching until you reach the root folder
    while true
        % Get the parent folder
        [parentFolder, currentName] = fileparts(currFolder);
        
        % Check if the current folder's name is the target
        if strcmpi(currentName, targetName)
            found = true;
            break;
        end
        
        % If we've reached the root or no change, exit the loop
        if isempty(parentFolder) || strcmp(currFolder, parentFolder)
            break;
        end
        
        % Move one level up
        currFolder = parentFolder;
    end

    if found
        addpath(genpath(currFolder));
        fprintf('Adding matlab path to: %s\n', currFolder);
    else
        error('Folder named "%s" not found in any parent directory.', targetName);
    end
end



%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":40}
%---
%[output:084ff5bb]
%   data: {"dataType":"text","outputData":{"text":"Adding matlab path to: G:\\My Drive\\Work\\Research\\SIM\\code\n","truncated":false}}
%---
%[output:1b3a6c37]
%   data: {"dataType":"textualVariable","outputData":{"name":"total_iteration","value":"1"}}
%---
%[output:7e2be390]
%   data: {"dataType":"text","outputData":{"text":"Wireless packet type: SC\n","truncated":false}}
%---
%[output:94b50f6a]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_y","value":"4"}}
%---
%[output:54970599]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_x","value":"15"}}
%---
%[output:060c976e]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_y","value":"15"}}
%---
%[output:24212b63]
%   data: {"dataType":"matrix","outputData":{"columns":20,"name":"zeta","rows":1,"type":"double","value":[["0.9800","0.9810","0.9820","0.9830","0.9840","0.9850","0.9860","0.9870","0.9880","0.9890","0.9900","0.9910","0.9920","0.9930","0.9940","0.9950","0.9960","0.9970","0.9980","0.9990"]]}}
%---
%[output:516e2386]
%   data: {"dataType":"textualVariable","outputData":{"name":"T_coh","value":"0.0021"}}
%---
%[output:77aaf974]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_packets_coh","value":"9"}}
%---
%[output:4d6c3649]
%   data: {"dataType":"textualVariable","outputData":{"name":"T","value":"144"}}
%---
%[output:00757844]
%   data: {"dataType":"textualVariable","outputData":{"name":"SNR_dB","value":"11.6443"}}
%---
%[output:75e1cdaa]
%   data: {"dataType":"text","outputData":{"text":"=== Calibration phase ===\nGrid: 10 x 5 = 50 calibration positions\nCalibration complete.\n\n","truncated":false}}
%---
%[output:7b222876]
%   data: {"dataType":"text","outputData":{"text":"=== Training phase ===\n","truncated":false}}
%---
%[output:20c546ca]
%   data: {"dataType":"text","outputData":{"text":"N_cal=50  |  MaxEpisodes=100  |  MaxSteps=144  |  n_actions=9\n","truncated":false}}
%---
%[output:9eff4c86]
%   data: {"dataType":"text","outputData":{"text":"Episode:   1\/100 | Episode reward:     3.94 | Episode steps:   37 | Average reward:     3.94 | Step Count:   37 | Episode Q0:  1505.89\nEpisode:   2\/100 | Episode reward:     3.02 | Episode steps:   26 | Average reward:     3.48 | Step Count:   63 | Episode Q0:  1079.58\nEpisode:   3\/100 | Episode reward:     5.37 | Episode steps:    8 | Average reward:     4.11 | Step Count:   71 | Episode Q0:   777.84\nEpisode:   4\/100 | Episode reward:    14.91 | Episode steps:  123 | Average reward:     6.81 | Step Count:  194 | Episode Q0:  1024.30\nEpisode:   5\/100 | Episode reward:     6.10 | Episode steps:  132 | Average reward:     6.67 | Step Count:  326 | Episode Q0:   606.05\nEpisode:   6\/100 | Episode reward:     1.51 | Episode steps:  144 | Average reward:     5.81 | Step Count:  470 | Episode Q0:   720.48\nEpisode:   7\/100 | Episode reward:    13.10 | Episode steps:   49 | Average reward:     6.85 | Step Count:  519 | Episode Q0:  1279.34\nEpisode:   8\/100 | Episode reward:     7.61 | Episode steps:  128 | Average reward:     6.95 | Step Count:  647 | Episode Q0:  1051.41\nEpisode:   9\/100 | Episode reward:     4.52 | Episode steps:   35 | Average reward:     6.68 | Step Count:  682 | Episode Q0:   576.84\nEpisode:  10\/100 | Episode reward:     6.99 | Episode steps:   36 | Average reward:     6.71 | Step Count:  718 | Episode Q0:   618.47\nEpisode:  11\/100 | Episode reward:     4.44 | Episode steps:  144 | Average reward:     6.50 | Step Count:  862 | Episode Q0:   448.17\nEpisode:  12\/100 | Episode reward:     0.29 | Episode steps:  144 | Average reward:     5.98 | Step Count: 1006 | Episode Q0:   558.44\nEpisode:  13\/100 | Episode reward:     2.09 | Episode steps:  144 | Average reward:     5.68 | Step Count: 1150 | Episode Q0:   222.93\nEpisode:  14\/100 | Episode reward:     1.58 | Episode steps:  144 | Average reward:     5.39 | Step Count: 1294 | Episode Q0:    90.52\nEpisode:  15\/100 | Episode reward:     6.90 | Episode steps:  144 | Average reward:     5.49 | Step Count: 1438 | Episode Q0:     3.88\nEpisode:  16\/100 | Episode reward:     2.57 | Episode steps:    4 | Average reward:     5.31 | Step Count: 1442 | Episode Q0:   504.25\nEpisode:  17\/100 | Episode reward:     4.64 | Episode steps:  123 | Average reward:     5.27 | Step Count: 1565 | Episode Q0:   230.82\nEpisode:  18\/100 | Episode reward:     5.74 | Episode steps:  144 | Average reward:     5.30 | Step Count: 1709 | Episode Q0:   -74.49\nEpisode:  19\/100 | Episode reward:    13.75 | Episode steps:  106 | Average reward:     5.74 | Step Count: 1815 | Episode Q0:   133.11\nEpisode:  20\/100 | Episode reward:     2.84 | Episode steps:   40 | Average reward:     5.60 | Step Count: 1855 | Episode Q0:   668.95\nEpisode:  21\/100 | Episode reward:     0.00 | Episode steps:  144 | Average reward:     5.33 | Step Count: 1999 | Episode Q0:   -97.31\nEpisode:  22\/100 | Episode reward:     8.03 | Episode steps:   17 | Average reward:     5.45 | Step Count: 2016 | Episode Q0:   554.37\nEpisode:  23\/100 | Episode reward:     2.28 | Episode steps:    3 | Average reward:     5.31 | Step Count: 2019 | Episode Q0:  -143.60\nEpisode:  24\/100 | Episode reward:     1.10 | Episode steps:  144 | Average reward:     5.14 | Step Count: 2163 | Episode Q0:    16.07\nEpisode:  25\/100 | Episode reward:     8.71 | Episode steps:   64 | Average reward:     5.28 | Step Count: 2227 | Episode Q0:   176.86\nEpisode:  26\/100 | Episode reward:    18.38 | Episode steps:   83 | Average reward:     5.79 | Step Count: 2310 | Episode Q0:  -189.57\nEpisode:  27\/100 | Episode reward:     2.54 | Episode steps:  144 | Average reward:     5.66 | Step Count: 2454 | Episode Q0:   -17.82\nEpisode:  28\/100 | Episode reward:     6.60 | Episode steps:  144 | Average reward:     5.70 | Step Count: 2598 | Episode Q0:    -2.78\nEpisode:  29\/100 | Episode reward:    10.69 | Episode steps:   25 | Average reward:     5.87 | Step Count: 2623 | Episode Q0:   439.31\nEpisode:  30\/100 | Episode reward:     3.42 | Episode steps:   30 | Average reward:     5.79 | Step Count: 2653 | Episode Q0:   183.80\nEpisode:  31\/100 | Episode reward:     4.64 | Episode steps:   18 | Average reward:     5.75 | Step Count: 2671 | Episode Q0:     4.61\nEpisode:  32\/100 | Episode reward:    10.21 | Episode steps:  144 | Average reward:     5.89 | Step Count: 2815 | Episode Q0:  -159.39\nEpisode:  33\/100 | Episode reward:     6.79 | Episode steps:  144 | Average reward:     5.92 | Step Count: 2959 | Episode Q0:    41.43\nEpisode:  34\/100 | Episode reward:    14.83 | Episode steps:  144 | Average reward:     6.18 | Step Count: 3103 | Episode Q0:  -225.99\nEpisode:  35\/100 | Episode reward:     6.47 | Episode steps:  144 | Average reward:     6.19 | Step Count: 3247 | Episode Q0:   157.03\nEpisode:  36\/100 | Episode reward:     2.87 | Episode steps:  144 | Average reward:     6.10 | Step Count: 3391 | Episode Q0:   175.44\nEpisode:  37\/100 | Episode reward:    22.02 | Episode steps:  144 | Average reward:     6.53 | Step Count: 3535 | Episode Q0:   314.02\nEpisode:  38\/100 | Episode reward:     8.08 | Episode steps:   46 | Average reward:     6.57 | Step Count: 3581 | Episode Q0:    90.49\nEpisode:  39\/100 | Episode reward:    12.69 | Episode steps:  144 | Average reward:     6.72 | Step Count: 3725 | Episode Q0:   139.16\nEpisode:  40\/100 | Episode reward:    12.45 | Episode steps:  144 | Average reward:     6.87 | Step Count: 3869 | Episode Q0:   133.93\nEpisode:  41\/100 | Episode reward:     6.18 | Episode steps:   19 | Average reward:     6.85 | Step Count: 3888 | Episode Q0:   148.18\nEpisode:  42\/100 | Episode reward:     3.20 | Episode steps:  144 | Average reward:     6.76 | Step Count: 4032 | Episode Q0:   225.31\nEpisode:  43\/100 | Episode reward:     7.75 | Episode steps:  144 | Average reward:     6.79 | Step Count: 4176 | Episode Q0:   110.41\nEpisode:  44\/100 | Episode reward:    17.89 | Episode steps:   86 | Average reward:     7.04 | Step Count: 4262 | Episode Q0:    41.87\nEpisode:  45\/100 | Episode reward:     8.17 | Episode steps:   26 | Average reward:     7.06 | Step Count: 4288 | Episode Q0:    26.21\nEpisode:  46\/100 | Episode reward:    11.15 | Episode steps:   49 | Average reward:     7.15 | Step Count: 4337 | Episode Q0:   -15.26\nEpisode:  47\/100 | Episode reward:     4.57 | Episode steps:  103 | Average reward:     7.10 | Step Count: 4440 | Episode Q0:   -41.02\nEpisode:  48\/100 | Episode reward:     8.14 | Episode steps:  144 | Average reward:     7.12 | Step Count: 4584 | Episode Q0:   -33.90\nEpisode:  49\/100 | Episode reward:     6.51 | Episode steps:   44 | Average reward:     7.11 | Step Count: 4628 | Episode Q0:   206.33\nEpisode:  50\/100 | Episode reward:     5.86 | Episode steps:  144 | Average reward:     7.08 | Step Count: 4772 | Episode Q0:   -32.89\nEpisode:  51\/100 | Episode reward:     3.16 | Episode steps:   37 | Average reward:     7.07 | Step Count: 4809 | Episode Q0:     8.23\nEpisode:  52\/100 | Episode reward:    14.91 | Episode steps:  144 | Average reward:     7.31 | Step Count: 4953 | Episode Q0:    68.65\nEpisode:  53\/100 | Episode reward:    13.40 | Episode steps:   56 | Average reward:     7.47 | Step Count: 5009 | Episode Q0:   126.87\nEpisode:  54\/100 | Episode reward:    15.10 | Episode steps:   76 | Average reward:     7.47 | Step Count: 5085 | Episode Q0:    91.14\nEpisode:  55\/100 | Episode reward:     6.67 | Episode steps:  144 | Average reward:     7.48 | Step Count: 5229 | Episode Q0:    17.38\nEpisode:  56\/100 | Episode reward:     1.91 | Episode steps:  144 | Average reward:     7.49 | Step Count: 5373 | Episode Q0:    51.23\nEpisode:  57\/100 | Episode reward:     9.61 | Episode steps:  144 | Average reward:     7.42 | Step Count: 5517 | Episode Q0:   -59.26\nEpisode:  58\/100 | Episode reward:     0.28 | Episode steps:  144 | Average reward:     7.27 | Step Count: 5661 | Episode Q0:   -46.53\nEpisode:  59\/100 | Episode reward:    13.04 | Episode steps:   86 | Average reward:     7.44 | Step Count: 5747 | Episode Q0:   -13.54\nEpisode:  60\/100 | Episode reward:     5.75 | Episode steps:  144 | Average reward:     7.42 | Step Count: 5891 | Episode Q0:    33.77\nEpisode:  61\/100 | Episode reward:    10.59 | Episode steps:  144 | Average reward:     7.54 | Step Count: 6035 | Episode Q0:   -33.53\nEpisode:  62\/100 | Episode reward:     9.52 | Episode steps:   19 | Average reward:     7.73 | Step Count: 6054 | Episode Q0:   -13.10\nEpisode:  63\/100 | Episode reward:     7.39 | Episode steps:  144 | Average reward:     7.83 | Step Count: 6198 | Episode Q0:    33.53\nEpisode:  64\/100 | Episode reward:     3.16 | Episode steps:   31 | Average reward:     7.86 | Step Count: 6229 | Episode Q0:    48.06\nEpisode:  65\/100 | Episode reward:     5.60 | Episode steps:  120 | Average reward:     7.84 | Step Count: 6349 | Episode Q0:    48.58\nEpisode:  66\/100 | Episode reward:     5.91 | Episode steps:   69 | Average reward:     7.90 | Step Count: 6418 | Episode Q0:   -21.26\nEpisode:  67\/100 | Episode reward:    13.10 | Episode steps:   85 | Average reward:     8.07 | Step Count: 6503 | Episode Q0:   -48.32\nEpisode:  68\/100 | Episode reward:    10.41 | Episode steps:  144 | Average reward:     8.17 | Step Count: 6647 | Episode Q0:    14.06\nEpisode:  69\/100 | Episode reward:    21.06 | Episode steps:  117 | Average reward:     8.31 | Step Count: 6764 | Episode Q0:    31.51\nEpisode:  70\/100 | Episode reward:     3.20 | Episode steps:   34 | Average reward:     8.32 | Step Count: 6798 | Episode Q0:    12.95\nEpisode:  71\/100 | Episode reward:    11.35 | Episode steps:   60 | Average reward:     8.55 | Step Count: 6858 | Episode Q0:    74.18\nEpisode:  72\/100 | Episode reward:    22.39 | Episode steps:  144 | Average reward:     8.83 | Step Count: 7002 | Episode Q0:    33.13\nEpisode:  73\/100 | Episode reward:     6.87 | Episode steps:   64 | Average reward:     8.93 | Step Count: 7066 | Episode Q0:    22.77\nEpisode:  74\/100 | Episode reward:    12.88 | Episode steps:  100 | Average reward:     9.16 | Step Count: 7166 | Episode Q0:    46.74\nEpisode:  75\/100 | Episode reward:    19.62 | Episode steps:  144 | Average reward:     9.38 | Step Count: 7310 | Episode Q0:   114.66\nEpisode:  76\/100 | Episode reward:     8.30 | Episode steps:  106 | Average reward:     9.18 | Step Count: 7416 | Episode Q0:    42.00\nEpisode:  77\/100 | Episode reward:    11.21 | Episode steps:  144 | Average reward:     9.35 | Step Count: 7560 | Episode Q0:     5.00\nEpisode:  78\/100 | Episode reward:     9.84 | Episode steps:  144 | Average reward:     9.42 | Step Count: 7704 | Episode Q0:    35.04\nEpisode:  79\/100 | Episode reward:    13.44 | Episode steps:   53 | Average reward:     9.47 | Step Count: 7757 | Episode Q0:    55.07\nEpisode:  80\/100 | Episode reward:    11.50 | Episode steps:  144 | Average reward:     9.63 | Step Count: 7901 | Episode Q0:     8.98\nEpisode:  81\/100 | Episode reward:    21.18 | Episode steps:  144 | Average reward:     9.96 | Step Count: 8045 | Episode Q0:   -10.40\nEpisode:  82\/100 | Episode reward:    20.95 | Episode steps:  142 | Average reward:    10.18 | Step Count: 8187 | Episode Q0:    -4.35\nEpisode:  83\/100 | Episode reward:    16.79 | Episode steps:  144 | Average reward:    10.38 | Step Count: 8331 | Episode Q0:     6.68\nEpisode:  84\/100 | Episode reward:     5.75 | Episode steps:  144 | Average reward:    10.20 | Step Count: 8475 | Episode Q0:    -6.32\nEpisode:  85\/100 | Episode reward:    20.25 | Episode steps:  144 | Average reward:    10.47 | Step Count: 8619 | Episode Q0:   -19.33\nEpisode:  86\/100 | Episode reward:    11.02 | Episode steps:   70 | Average reward:    10.64 | Step Count: 8689 | Episode Q0:    -0.44\nEpisode:  87\/100 | Episode reward:     9.92 | Episode steps:  123 | Average reward:    10.39 | Step Count: 8812 | Episode Q0:     9.09\nEpisode:  88\/100 | Episode reward:     6.16 | Episode steps:  144 | Average reward:    10.35 | Step Count: 8956 | Episode Q0:    26.25\nEpisode:  89\/100 | Episode reward:    12.76 | Episode steps:  144 | Average reward:    10.36 | Step Count: 9100 | Episode Q0:    14.80\nEpisode:  90\/100 | Episode reward:     4.87 | Episode steps:  144 | Average reward:    10.20 | Step Count: 9244 | Episode Q0:    49.22\nEpisode:  91\/100 | Episode reward:    14.26 | Episode steps:   71 | Average reward:    10.37 | Step Count: 9315 | Episode Q0:     4.63\nEpisode:  92\/100 | Episode reward:     2.81 | Episode steps:   47 | Average reward:    10.36 | Step Count: 9362 | Episode Q0:    11.51\nEpisode:  93\/100 | Episode reward:     4.27 | Episode steps:  144 | Average reward:    10.29 | Step Count: 9506 | Episode Q0:     2.04\nEpisode:  94\/100 | Episode reward:     1.87 | Episode steps:  144 | Average reward:     9.97 | Step Count: 9650 | Episode Q0:    19.30\nEpisode:  95\/100 | Episode reward:    10.08 | Episode steps:  144 | Average reward:    10.01 | Step Count: 9794 | Episode Q0:    -3.28\nEpisode:  96\/100 | Episode reward:    14.77 | Episode steps:  108 | Average reward:    10.08 | Step Count: 9902 | Episode Q0:    -3.51\nEpisode:  97\/100 | Episode reward:    11.24 | Episode steps:  117 | Average reward:    10.21 | Step Count: 10019 | Episode Q0:    -8.38\nEpisode:  98\/100 | Episode reward:    12.71 | Episode steps:  144 | Average reward:    10.30 | Step Count: 10163 | Episode Q0:     1.44\nEpisode:  99\/100 | Episode reward:    17.81 | Episode steps:  144 | Average reward:    10.53 | Step Count: 10307 | Episode Q0:    44.70\nEpisode: 100\/100 | Episode reward:    16.34 | Episode steps:  144 | Average reward:    10.74 | Step Count: 10451 | Episode Q0:    17.34\n","truncated":false}}
%---
%[output:9699bef4]
%   data: {"dataType":"text","outputData":{"text":"Agent saved to ..\\Dataset\\dqn_agent_navigation_81_neurons_norelu.mat\n","truncated":false}}
%---
