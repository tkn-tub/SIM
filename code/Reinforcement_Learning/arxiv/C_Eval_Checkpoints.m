%% C_Eval_Checkpoints.m
% Evaluates each saved checkpoint by running a fresh batch of episodes
% and computing average reward. Produces a curve of policy quality vs
% training progress -- more meaningful than raw training reward (which
% is polluted by epsilon-greedy exploration noise).
%
% Assumes SaveAgentDirectory in A_Training_SIM2_..._Reward_Mixed.m
% points at fullfile('..','Dataset','SIM2_checkpoints') and files are
% named AgentN.mat, each containing 'saved_agent'.

clc; clear all; close all;
addingPathParentFolderByName('code');
Parameters;

Mx=25
My=Mx
M=Mx*My

% % CST overrides -- must MATCH what training used, or evaluation is on the
% % wrong signal model
% EnvPars.G      = EnvPars.G_CST;
% EnvPars.U_func = EnvPars.U_func_CST;
Calibration;

% Environment (must match training's step/reset function selection)
env = rlFunctionEnv(rlNumericSpec([2*EnvPars.N,1]), rlFiniteSetSpec(1:EnvPars.n_actions), ...
      @(a,ls) stepFunction_nav_CST_Reward_Mixed(a, ls, EnvPars), ...
      @()     resetFunction_nav_CST_Reward_Mixed(EnvPars));

%% Find all checkpoints
ckpt_dir = fullfile('..','Dataset','SIM_Nx_4_Mx_25_Reward_Mixed_checkpoints');
files    = dir(fullfile(ckpt_dir, 'Agent*.mat'));

% Sort by episode number embedded in filename (Agent500.mat -> 500)
ep_nums = zeros(numel(files),1);
for i = 1:numel(files)
    tok = regexp(files(i).name, 'Agent(\d+)\.mat', 'tokens', 'once');
    ep_nums(i) = str2double(tok{1});
end
[ep_nums, order] = sort(ep_nums);
files = files(order);

fprintf('Found %d checkpoints, from episode %d to %d.\n', ...
    numel(files), ep_nums(1), ep_nums(end));

%% Evaluate each checkpoint
n_eval_episodes = 50;   % per checkpoint -- tune based on time budget
avg_reward      = zeros(numel(files),1);
std_reward      = zeros(numel(files),1);

for i = 1:numel(files)
    S = load(fullfile(ckpt_dir, files(i).name));
    if isfield(S, 'saved_agent')
        agent = S.saved_agent;
    elseif isfield(S, 'agent')
        agent = S.agent;
    else
        warning('Skipping %s: no agent variable found.', files(i).name);
        continue;
    end

    % Force greedy evaluation (no exploration) so we measure the actual
    % learned policy, not the epsilon-greedy training behavior. Different
    % MATLAB releases expose this differently; setting Epsilon=0 works on
    % rlDQNAgent across the versions in this project.
    agent.AgentOptions.EpsilonGreedyExploration.Epsilon = 0;

    ep_rewards = zeros(n_eval_episodes,1);
    for k = 1:n_eval_episodes
        exp = sim(env, agent, ...
            rlSimulationOptions('MaxSteps', EnvPars.MaxStepsPerEpisode));
        ep_rewards(k) = sum(exp.Reward);
    end

    avg_reward(i) = mean(ep_rewards);
    std_reward(i) = std(ep_rewards);
    fprintf('Ep %6d: avg reward = %8.2f  (+/- %.2f, n=%d)\n', ...
        ep_nums(i), avg_reward(i), std_reward(i), n_eval_episodes);
end

%% Plot
figure;
errorbar(ep_nums, avg_reward, std_reward, 'LineWidth', 1.5);
xlabel('Training episode', 'Interpreter','latex');
ylabel('Avg reward (greedy eval, %d episodes)', ...
    'Interpreter','latex'); ylabel(sprintf('Avg reward (%d greedy episodes)', n_eval_episodes));
title('SIM-2 policy quality vs training progress', 'Interpreter','latex');
grid on;

save(fullfile('..','Dataset','SIM2_partial_reward_curve.mat'), ...
    'ep_nums', 'avg_reward', 'std_reward', 'n_eval_episodes');
fprintf('Saved curve data.\n');

%%
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
