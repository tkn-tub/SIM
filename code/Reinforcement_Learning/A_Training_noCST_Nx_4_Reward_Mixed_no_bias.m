% Training script -- DQN navigation with an ordinary FC Q-network baseline.
%
% This file keeps the same observation contract as the SIM2 critic:
%   observation = [real(Ral(:)); imag(Ral(:))]
% where Ral is the argmax-aligned, phase-referenced 4x4 coherent field.
%
% The SIM2 propagation/phase/readout layers are intentionally removed here.
% Use this as a diagnostic:
%   - FC learns, SIM2 does not  -> likely SIM2 capacity/scale/readout issue.
%   - FC also does not learn    -> debug reward, observation, or DQN settings.

clc; clear; close all;

addingPathParentFolderByName('code');
Parameters;      % defines EnvPars, ActInfo, and the default N+2 ObsInfo
Calibration;     % fills EnvPars.psi_*_cal, global_max_cal, best_tx/ty_cal

% Use the same episode count convention as the SIM2 run.
EnvPars.MaxEpisodes = EnvPars.N_cal * 30;

%% Observation override: coherent Re/Im field, no (t_x,t_y) in the observation
ObsInfo             = rlNumericSpec([2*EnvPars.N, 1]);
ObsInfo.Name        = 'observations';
ObsInfo.Description = 'Coherent field [Re(r); Im(r)] after argmax alignment and phase reference';
ObsInfo.LowerLimit  = -inf(2*EnvPars.N, 1);
ObsInfo.UpperLimit  =  inf(2*EnvPars.N, 1);

%% FC critic: [Re(r); Im(r)] -> one Q-value per navigation action
numObs = 2 * EnvPars.N;
numAct = EnvPars.n_actions;

fcWidth1 = 8;
fcWidth2 = fcWidth1;

% Freeze all FC biases at exactly zero.
% Weights still train normally.
noBiasArgs = { ...
    'BiasInitializer',      'zeros', ...
    'BiasLearnRateFactor',  0, ...
    'BiasL2Factor',         0};

criticLayers = [
    featureInputLayer(numObs, 'Name', 'obs', 'Normalization', 'none')

    fullyConnectedLayer(fcWidth1, ...
        'Name', 'fc1', ...
        noBiasArgs{:})

    reluLayer('Name', 'relu2')

    fullyConnectedLayer(numAct, ...
        'Name', 'q_values', ...
        noBiasArgs{:})];

%%ReLU located at the final layer
% fcWidth1 = 32;
% fcWidth2 = 9;
% 
% % Freeze all FC biases at exactly zero.
% % Weights still train normally.
% noBiasArgs = { ...
%     'BiasInitializer',      'zeros', ...
%     'BiasLearnRateFactor',  0, ...
%     'BiasL2Factor',         0};
% 
% criticLayers = [
%     featureInputLayer(numObs, 'Name', 'obs', 'Normalization', 'none')
% 
%     fullyConnectedLayer(fcWidth1, ...
%         'Name', 'fc1', ...
%         noBiasArgs{:})
% 
%     fullyConnectedLayer(numAct, ...
%         'Name', 'q_values', ...
%         noBiasArgs{:})
% 
%     reluLayer('Name', 'relu2')
%     ];

criticNet = dlnetwork(layerGraph(criticLayers));

% Check before training
assertZeroBiases(criticNet, "before training");

critic = rlVectorQValueFunction(criticNet, ObsInfo, ActInfo);

%% Fast interface check before training
[obs0, logged0] = resetFunction_nav_CST_Aligned(EnvPars); %#ok<ASGLU>
assert(isequal(size(obs0), [numObs, 1]), ...
    'Reset observation is %dx%d but expected %dx1.', size(obs0,1), size(obs0,2), numObs);

q0 = predict(criticNet, dlarray(single(obs0), 'CB'));
q0 = extractdata(q0);
assert(isequal(size(q0), [numAct, 1]), ...
    'Critic output is %dx%d but expected %dx1 Q-values.', size(q0,1), size(q0,2), numAct);

fprintf('FC critic interface OK: observation %dx1 -> Q-values %dx1\n', numObs, numAct);

%% Environment: keep the same aligned Re/Im observation as SIM2
% stepFunction_nav_CST_Aligned and resetFunction_nav_CST_Aligned both emit
% observation = [real(Ral(:)); imag(Ral(:))].
env = rlFunctionEnv(ObsInfo, ActInfo, ...
      @(a,ls) stepFunction_nav_CST_Aligned(a, ls, EnvPars), ...
      @()     resetFunction_nav_CST_Aligned(EnvPars));

%% DQN agent
agentOpts = rlDQNAgentOptions(...
    'SampleTime',                    1, ...
    'DiscountFactor',                EnvPars.DiscountFactor, ...
    'MiniBatchSize',                 EnvPars.MiniBatchSize, ...
    'ExperienceBufferLength',        EnvPars.ExperienceBufferLength, ...
    'TargetSmoothFactor',            EnvPars.TargetSmoothFactor, ...
    'CriticOptimizerOptions',        rlOptimizerOptions('LearnRate', 5e-4));

agentOpts.EpsilonGreedyExploration.Epsilon      = 1.0;
agentOpts.EpsilonGreedyExploration.EpsilonMin   = 0.05;
agentOpts.EpsilonGreedyExploration.EpsilonDecay = EnvPars.EpsilonDecay;

agent = rlDQNAgent(critic, agentOpts);

%% Training
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',                 EnvPars.MaxEpisodes, ...
    'MaxStepsPerEpisode',          EnvPars.MaxStepsPerEpisode, ...
    'ScoreAveragingWindowLength',  50, ...
    'StopTrainingCriteria',        'AverageReward', ...
    'StopTrainingValue',           EnvPars.StopTrainingValue, ...
    'Verbose',                     true, ...
    'Plots',                       'none');

fprintf('=== FC Re/Im DQN training phase ===\n');
fprintf('N_cal=%d | MaxEpisodes=%d | MaxSteps=%d | n_actions=%d | obsDim=%d\n', ...
        EnvPars.N_cal, EnvPars.MaxEpisodes, EnvPars.MaxStepsPerEpisode, EnvPars.n_actions, numObs);

trainingStats = train(agent, env, trainOpts);

criticNet = getModel(getCritic(agent));

% Check after training
assertZeroBiases(criticNet, "after training");

%% Save
save_dir = fullfile('..', 'Dataset');
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

save_path = fullfile(save_dir, 'A_Training_noCST_FC_1_ReLU_1_end_Neurons_8_Nx_4_Reward_Mixed_no_bias.mat');
criticNet = getModel(getCritic(agent));
save(save_path, 'agent', 'trainingStats', 'criticNet', 'EnvPars');
fprintf('Agent saved to %s\n', save_path);

%% Helpers
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

function assertZeroBiases(net, tag)
    L = net.Learnables;
    isBias = strcmp(string(L.Parameter), "Bias");

    if ~any(isBias)
        fprintf('%s: no Bias parameters found.\n', tag);
        return;
    end

    biasRows = find(isBias).';

    for k = biasRows
        b = extractdata(L.Value{k});
        maxAbsBias = max(abs(b(:)));

        fprintf('%s: layer %s Bias max abs = %.3e\n', ...
            tag, string(L.Layer(k)), maxAbsBias);

        assert(maxAbsBias < 1e-10, ...
            'Bias in layer %s is not zero after %s.', ...
            string(L.Layer(k)), tag);
    end
end