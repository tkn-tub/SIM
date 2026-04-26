%% Training script for a DQN agent

clc;
clear all;
close all;

% =========================================================================
% Progress indicator: record wall-clock start time
% =========================================================================
t_start = tic;
fprintf('=============================================================\n');
fprintf('  DQN Training Started: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf('=============================================================\n');

% Initial parameters

%including all parent folders up to the file 'code', this makes visible all
%files within this code
addingPathParentFolderByName('code');
Parameters; % Script containing necessary variables 

% Environment: 2DFT as evaluated by the SIM 1 physics
% Description: The environment is defined within the cell EnvPars following 
% two variables
%% 
% * G: Reflects the 2D-DFT matrix and calculated with the G_func.
% * U: Reflects the angular shift introduced by the first layer of SIM 1 and 
% is calculated with U_func.
%% 
% These two variables are defined to evaluate later the observation upon a given 
% action by the RL agent, these are defined as follows
%% 
% * Observations: Given by the waverform amplitude delivered by SIM 1.
% * Actions: Phase shifts introduced at the first layer of the SIM 1. This phase 
% shift is introduced through the U variable.
%% 
% The oberservation and actions are implemented on each train step within the 
% following two functions
%% 
% * |resetFunction|: This function is called at the beginning of each episode 
% and defines the initial environment where the RL agent is re-trained. On each 
% episode we randomly locate the MU.
% * |stepFunction|: This function is called on each step, following the beginning 
% of each episode.

% Meta-atom indexing
n = (1:EnvPars.N);
n_y = ceil(n ./ EnvPars.N_x);
n_x = n - (n_y - 1) .* EnvPars.N_x;

% Psi indexing
n_psi = n;
n_psi_y = ceil(n_psi ./ EnvPars.N_x);
n_psi_x = n_psi - (n_psi_y - 1) .* EnvPars.N_x;

% Kernel 2D-DFT == TO BE REPLACED BY SIM 1
G_func = @(n,n_psi) exp(-1i*2*pi*(n_psi_x(n_psi)-1)/EnvPars.N_x.*(n_x(n)-1)).* ...
    exp(-1i*2*pi*(n_psi_y(n_psi)-1)/EnvPars.N_y.*(n_y(n)-1));
[n_psi_grid, n_s_grid] = ndgrid(1:EnvPars.N, 1:EnvPars.N);
EnvPars.G = G_func(n_s_grid, n_psi_grid);

%% Environment wiring

% A handle-class counter that persists across reset calls
EnvPars.episode_counter = EpisodeCounter();%calling to the constructor
StepHandle = @(Action, LoggedSignals) stepFunction(Action, LoggedSignals, EnvPars);
% Random reset for training. This function is called on each episode
ResetHandle = @() resetFunction_random(EnvPars);
%Environment
EnvPars.episode_counter = EpisodeCounter();
env = rlFunctionEnv(ObsInfo, ActInfo, StepHandle, ResetHandle);


fprintf('Environment created successfully.\n');

% Creation of the DQN Agent and Training

% Neural network (Critic)
statePath = [
    featureInputLayer(EnvPars.N, 'Normalization', 'none', 'Name', 'observations')
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(128, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(numel(ActInfo.Elements), 'Name', 'qout')];

criticOpts = rlRepresentationOptions('UseDevice', 'gpu');
critic = rlQValueRepresentation(statePath, ObsInfo, ActInfo, 'Observation', {'observations'}, criticOpts);

fprintf('Critic network built.\n');

%% Agent options
agentOpts = rlDQNAgentOptions(...
    'UseDoubleDQN', true, ...              % Use Double DQN
    'TargetSmoothFactor', 1e-3, ...          
    'TargetUpdateFrequency', 1, ...
    'ExperienceBufferLength', 1e5, ...      
    'MiniBatchSize', 128, ...
    'DiscountFactor', 0.99);

agentOpts.EpsilonGreedyExploration = rl.option.EpsilonGreedyExploration(...
    'Epsilon', 1.0, 'EpsilonMin', 0.05, 'EpsilonDecay', 1e-3);

agentOpts.CriticOptimizerOptions.LearnRate = 5e-4;
agentOpts.CriticOptimizerOptions.GradientThreshold = 10;

agent = rlDQNAgent(critic, agentOpts);

fprintf('DQN agent created. Starting training...\n');
fprintf('  Max episodes        : %d\n', MaxEpisodes);
fprintf('  Max steps/episode   : %d\n', EnvPars.maxSteps);
fprintf('  Stop criterion      : AverageReward >= 1000\n');
fprintf('  Avg window length   : 50 episodes\n');
fprintf('-------------------------------------------------------------\n');
fprintf('  [Verbose output below — one line per episode]\n');
fprintf('-------------------------------------------------------------\n');

%% Training options
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', MaxEpisodes, ...               
    'MaxStepsPerEpisode', EnvPars.maxSteps, ...%total of iterations in a single episode
    'ScoreAveragingWindowLength', 50, ... % Check this
    'Verbose', true, ...
    'Plots', 'none', ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', 1000.0); % Stop if the average reward is high
 
trainingStats = train(agent, env, trainOpts);

%% Progress summary after training
elapsed = toc(t_start);
fprintf('-------------------------------------------------------------\n');
fprintf('  Training finished: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf('  Total elapsed time: %.1f s (%.2f min)\n', elapsed, elapsed/60);
fprintf('  Episodes completed: %d\n', numel(trainingStats.EpisodeReward));
fprintf('  Final episode reward    : %.4f\n', trainingStats.EpisodeReward(end));
fprintf('  Final average reward    : %.4f\n', trainingStats.AverageReward(end));
fprintf('=============================================================\n');

%% Saving the results
path_agent= fullfile('..', 'Dataset', 'dqn_agent_random.mat');
save(path_agent, 'agent','trainingStats');

fprintf('Agent and training stats saved to: %s\n', path_agent);
%% Functions

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
% Evaluation

% numEpisodesToTest = 1; % Number of test episodes you want to run
% EnvPars.tolerance = pi/25;
% 
% for episode = 1:numEpisodesToTest
% 
%     % Initialize episode using your resetFunction to obtain LoggedSignals
%     [observation, LoggedSignals] = resetFunction(EnvPars);
% 
%     % Preallocate/collect episode data
%     actions = [];
%     rewards = [];
%     psi_x_est_history = [];
%     psi_y_est_history = [];
% 
%     isDone = false;
%     stepCount = 0;
% 
%     while ~isDone && stepCount < EnvPars.maxSteps
%         % Choose action from agent policy
%         [action, ~] = getAction(agent, observation);
% 
%         % If action is a cell, extract its contents
%         if iscell(action)
%             action = action{1};
%         end
% 
%         % Environment step using pre-defined stepFunction, keeping LoggedSignals
%         [nextObs, reward, isDone, LoggedSignals] = stepFunction(action, LoggedSignals, EnvPars);
% 
%         % Log
%         stepCount = stepCount + 1;
%         actions(stepCount,1) = double(action);
%         rewards(stepCount,1) = reward;
%         psi_x_est_history(stepCount,1) = LoggedSignals.psi_x_est;
%         psi_y_est_history(stepCount,1) = LoggedSignals.psi_y_est;
% 
%         % Advance
%         observation = nextObs;
%     end
% end
% 
% % Final info comes from LoggedSignals
% psi_x_true = LoggedSignals.psi_x_true;
% psi_y_true = LoggedSignals.psi_y_true;
% finalError = LoggedSignals.error_sum;
% totalReward = sum(rewards);
% 
% fprintf('Episodio %d:\n', episode);
% fprintf('  - Ground Truth (psi_x, psi_y): (%.3f, %.3f)\n', psi_x_true, psi_y_true);
% fprintf('  - Estimation:         (%.3f, %.3f)\n', psi_x_est_history(end), psi_y_est_history(end));
% fprintf('  - Numbers of iterations: %d / %d\n', stepCount, EnvPars.maxSteps);
% fprintf('  - Error: %.4f\n', finalError);
% fprintf('  - Cumulative Reward: %.2f\n', totalReward);
% fprintf('----------------------------------------\n');
% 
% Figures

% % Plots
% figure
% subplot(2,1,1);
% plot(1:stepCount, psi_x_est_history, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 4); hold on;
% yline(psi_x_true, 'r--', 'LineWidth', 2);
% title(['\psi_x vs. steps (episode ' num2str(episode) ')']);
% xlabel('step'); ylabel('\psi_x [rad]'); legend('Estimated angle','Real angle'); grid on;
% 
% subplot(2,1,2);
% plot(1:stepCount, psi_y_est_history, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 4); hold on;
% yline(psi_y_true, 'r--', 'LineWidth', 2);
% title(['\psi_y vs. steps (episode ' num2str(episode) ')']);
% xlabel('Step'); ylabel('\psi_y [rad]'); legend('Estimated angle','Real angle'); grid on;
% 
% figure
% stem(1:stepCount, actions, 'filled'); title('Sequence of actions (t_{\psi})');
% xlabel('Step'); ylabel('t_{\psi} index'); grid on;
% 
% function addingPathParentFolderByName(targetName)
%     % Start from the current directory
%     currFolder = pwd;
%     found = false;
% 
%     % Continue searching until you reach the root folder
%     while true
%         % Get the parent folder
%         [parentFolder, currentName] = fileparts(currFolder);
% 
%         % Check if the current folder's name is the target
%         if strcmpi(currentName, targetName)
%             found = true;
%             break;
%         end
% 
%         % If we've reached the root or no change, exit the loop
%         if isempty(parentFolder) || strcmp(currFolder, parentFolder)
%             break;
%         end
% 
%         % Move one level up
%         currFolder = parentFolder;
%     end
% 
%     if found
%         addpath(genpath(currFolder));
%         fprintf('Adding matlab path to: %s\n', currFolder);
%     else
%         error('Folder named "%s" not found in any parent directory.', targetName);
%     end
% end


% Plot 2D-DFT
% figure                
% R_psi_x_y = fliplr(flipud(reshape(abs(r_psi_x_y_t{max_t_psi(i_N,i_t,i_itr),i_t}), [N_x(i_N), N_y(i_N)])))';
% %coordinates with the electric angle indexes k_x and k_y as defined in
% %the Parameters.mlx file
% % [X,Y] = meshgrid(linspace(1,N_x(i_N),N_x(i_N))+(max_t_psi_x(i_N,i_t,i_itr)-1)/T_x,linspace(1,N_y(i_N),N_y(i_N))+(max_t_psi_y(i_N,i_t,i_itr)-1)/T_y);
% % %coordinates with the electric angle
% [X,Y] = meshgrid(2*pi/N_x(i_N)*(linspace(1,N_x(i_N),N_x(i_N))+(max_t_psi_x(i_N,i_t,i_itr))/T_x),...
%     2*pi/N_y(i_N)*(linspace(1,N_y(i_N),N_y(i_N))+(max_t_psi_y(i_N,i_t,i_itr))/T_y));
% 
% contourf(X,Y,abs(R_psi_x_y));
% % contourf(abs(R_psi_x_y));
% colorbar
% grid on;
% xlabel('$\psi_\mathrm{x}$ [rad]','Interpreter','latex');
% ylabel('$\psi_\mathrm{y}$ [rad]','Interpreter','latex');
% %Tick angles
% X=2*pi/N_x(i_N)*(linspace(1,N_x(i_N),N_x(i_N))+(max_t_psi_x(i_N,i_t,i_itr)-1)/T_x);
% set(gca, 'XTick', X);
% X_Tick_text=cell(1,length(X));
% for j=1:length(X)
%     X_Tick_text{j}=strcat('$',num2str((X(j))/pi,2),'\pi$');
% end
% set(gca, 'XTickLabel', [X_Tick_text], 'TickLabelInterpreter', 'latex');
% 
% Y=2*pi/N_y(i_N)*(linspace(1,N_y(i_N),N_y(i_N))+(max_t_psi_y(i_N,i_t,i_itr))/T_y);
% set(gca, 'YTick', Y);
% Y_Tick_text=cell(1,length(Y));
% for j=1:length(Y)
%     Y_Tick_text{j}=strcat('$',num2str((Y(j))/pi,2),'\pi$');
% end
% set(gca, 'YTickLabel', [Y_Tick_text], 'TickLabelInterpreter', 'latex');
% % 
% % set(gca, 'YTick', [0:0.25:2]*pi);
% % set(gca, 'YTickLabel', [{"0","$0.25\pi$","$0.5\pi$","$0.75\pi$","\pi","$1.25\pi$","1.5\pi","$1.75\pi$","2\pi"}], 'TickLabelInterpreter', 'latex');
% % axis([0 2*pi 0 2*pi])
% set(gca,'FontSize',font,'GridColor','white');
% legend_text={strcat('SNR$=',num2str(SNR_dB,3),'$ dB,',' $N=$',num2str(N_x(i_N)),'$\times$',num2str(N_y(i_N)),', $T=$',num2str(T_x),'$\times$',num2str(T_y)),...
%     strcat('$\bar{\psi}_x=',num2str(psi_x(i_N,i_t,i_itr)/pi,2),'\pi$ rad, ',' $\hat{\psi}_x=',num2str(psi_x_est(i_N,i_t,i_itr)/pi,3),'\pi$ rad')...
%     strcat('$\bar{\psi}_y=',num2str(psi_y(i_N,i_t,i_itr)/pi,2),'\pi$ rad, ',' $\hat{\psi}_y=',num2str(psi_y_est(i_N,i_t,i_itr)/pi,3),'\pi$ rad')};
% % Get current axis limits
% x_limits = xlim;
% y_limits = ylim;
% % Define position (upper-right corner)
% x_pos = x_limits(2) - 0.89 * diff(x_limits); % Slightly inside the right edge
% y_pos = y_limits(1) + 0.85 * diff(y_limits); % Slightly inside the top edge
% text(x_pos,y_pos,legend_text,'Interpreter','latex','BackgroundColor','w','FontSize',font);
% set(gca,'FontSize',font);