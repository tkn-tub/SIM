%% Training script — SIM Navigation with DQN
% Requires Parameters.mlx to have been run first
clc; clear all; close all;
addingPathParentFolderByName('code');
Parameters;   % loads all base variables into workspace and EnvPars

Calibration

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
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(81, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(EnvPars.n_actions, 'Name', 'output')];

criticNet = dlnetwork(layerGraph(statePath));
critic    = rlVectorQValueFunction(criticNet, ObsInfo, ActInfo);

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

fprintf('=== Training phase ===\n');
fprintf('N_cal=%d  |  MaxEpisodes=%d  |  MaxSteps=%d  |  n_actions=%d\n', ...
        N_cal, EnvPars.MaxEpisodes, EnvPars.MaxStepsPerEpisode, EnvPars.n_actions);

trainingStats = train(agent, env, trainOpts);

%% ── 9. SAVE ──────────────────────────────────────────────────────────────
save_path = fullfile('..', 'Dataset', 'dqn_agent_navigation_144_neurons.mat');
criticNet = getModel(getCritic(agent));
save(save_path, 'agent', 'trainingStats', 'criticNet', 'EnvPars');
fprintf('Agent saved to %s\n', save_path);

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
