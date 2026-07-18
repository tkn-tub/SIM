%[text] # SIM Ablation Study -- Visualization of all four parameters' impact
%[text] Loads the already-computed results from SIM_Ablation_FullGrid_Training.m
%[text] (or its _parallel variant) and produces four complementary views:
%[text]   1. Parallel coordinates plot -- all 4 parameters + loss in one figure
%[text]   2. 2D heatmap -- interaction between a chosen parameter PAIR (not just
%[text]      one-factor-at-a-time slices, which can hide interaction effects)
%[text]   3. Full convergence curves overlaid per round (like the zeta plot you
%[text]      already have) -- shows convergence SPEED, not just final loss
%[text]   4. N-way ANOVA -- quantifies which parameter explains the most variance
%[text]      in final loss across the grid (requires Statistics and Machine
%[text]      Learning Toolbox -- check availability before relying on this one)
%[text]
%[text] No sweep is re-run here; this only reads the saved .mat file.

clc;
clear all;
close all;

addingPathParentFolderByName('code');
Parameters   % needed for lambda, to express T_SIM/s_x in physically meaningful units

load(fullfile('..', 'Dataset', 'Ablation_FullGrid_Training.mat'));
% expects: results (table), loss_hist_all (cell), T_SIM_sweep, L_sweep,
% M_sweep, s_x_sweep, zeta_fixed, maxIter, i_T_base, i_L_base, i_M_base, i_s_base

T_SIM_base = T_SIM_sweep(i_T_base);
L_base     = L_sweep(i_L_base);
M_base     = M_sweep(i_M_base);
s_x_base   = s_x_sweep(i_s_base);

%% ===================== 1. Parallel coordinates plot =====================
% All four parameters + loss_final in one figure, one line per combo,
% colored by loss. parallelplot is base MATLAB (R2020a+), no extra toolbox.
%
% NOTE: parallelplot's 'Color' only accepts color specs tied to a
% CATEGORICAL 'GroupVariable' -- it does not support continuous coloring
% by a numeric column directly. Binning loss_final into quintiles below
% as a workaround; coarser than true continuous color, but shows the
% low-to-high trend across combos.
results_disp = results;
results_disp.T_SIM = results.T_SIM / lambda;   % display in units of lambda
results_disp.s_x   = results.s_x / lambda;
results_disp.Properties.VariableNames{'T_SIM'} = 'T_SIM_over_lambda';
results_disp.Properties.VariableNames{'s_x'}   = 's_x_over_lambda';

nBins = 5;
edges = quantile(results_disp.loss_final, linspace(0,1,nBins+1));
edges(1) = -inf; edges(end) = inf;   % make sure min/max points are captured
bin_idx = discretize(results_disp.loss_final, edges);
bin_labels = arrayfun(@(i) sprintf('Q%d', i), 1:nBins, 'UniformOutput', false);
results_disp.loss_bin = categorical(bin_idx, 1:nBins, bin_labels, 'Ordinal', true);

cmap = parula(nBins);   % Q1 = lowest loss, Q5 = highest loss

figure;
parallelplot(results_disp, 'CoordinateVariables', ...
    {'T_SIM_over_lambda','L','M','s_x_over_lambda','loss_final'}, ...
    'GroupVariable', 'loss_bin', 'Color', cmap);
title('Parallel coordinates: all 4 parameters vs. converged loss (Q1=lowest loss, Q5=highest)');

%% ---- 1b. Ground-truth check: don't trust colors, read the actual rows ----
% Most reliable answer to "which direction should each parameter move":
% just sort by loss_final and look at the extremes directly.
sorted_results = sortrows(results_disp, 'loss_final');
fprintf('\n--- 10 BEST combos (lowest loss) ---\n');
disp(sorted_results(1:10, {'T_SIM_over_lambda','L','M','s_x_over_lambda','loss_final'}));
fprintf('\n--- 10 WORST combos (highest loss) ---\n');
disp(sorted_results(end-9:end, {'T_SIM_over_lambda','L','M','s_x_over_lambda','loss_final'}));

%% ---- 1c. Decluttered parallel plot: only the two extreme quintiles ----
% 81 lines across 5 overlapping colors is hard to read reliably by eye.
% Showing only Q1 (best) vs Q5 (worst) -- roughly 32 lines, 2 colors --
% makes "where do the best lines sit on each axis" much easier to see,
% and sidesteps the legend-ordering ambiguity above since there are only
% two groups to keep straight.
extremes = results_disp(results_disp.loss_bin=='Q1' | results_disp.loss_bin=='Q5', :);
extremes.loss_bin = removecats(extremes.loss_bin);

cmap_extremes = cmap([1 nBins], :);   % reuse the same parula(nBins) defined above, just the two ends

figure;
parallelplot(extremes, 'CoordinateVariables', ...
    {'T_SIM_over_lambda','L','M','s_x_over_lambda','loss_final'}, ...
    'GroupVariable', 'loss_bin', 'Color', cmap_extremes);
title('Best (Q1) vs. worst (Q5) combos only -- decluttered view');

%% ===================== 2. 2D heatmap for a parameter pair =====================
% Reveals interaction effects that one-factor-at-a-time round plots can hide.
% Example: M x L interaction, with T_SIM and s_x held at baseline.
% Swap which two variables you pivot on to inspect other pairs.
mask = results.T_SIM==T_SIM_base & results.s_x==s_x_base;
sub = results(mask,:);

[M_vals, ~, M_idx] = unique(sub.M);
[L_vals, ~, L_idx] = unique(sub.L);
lossMat = nan(numel(M_vals), numel(L_vals));
for k = 1:height(sub)
    lossMat(M_idx(k), L_idx(k)) = sub.loss_final(k);
end

figure;
h = heatmap(string(L_vals), string(M_vals), lossMat);
h.Title = sprintf('Loss(M,L) at T\\_SIM=%.1f\\lambda, s\\_x=%.2f\\lambda', T_SIM_base/lambda, s_x_base/lambda);
h.XLabel = 'L'; h.YLabel = 'M';
h.Colormap = parula;

%% ===================== 3. Full convergence curves per round =====================
% Same style as your zeta plot, but for the M round: one curve per M value,
% loss vs iteration, instead of collapsing each combo down to its final loss.
mask = results.T_SIM==T_SIM_base & results.L==L_base & results.s_x==s_x_base;
idx_round = find(mask);
[~, order] = sort(results.M(idx_round));
idx_round = idx_round(order);

figure; hold on; grid on;
legend_labels = strings(numel(idx_round),1);
for k = 1:numel(idx_round)
    semilogy(loss_hist_all{idx_round(k)}, 'LineWidth', 1.5);
    legend_labels(k) = sprintf('M = %d', results.M(idx_round(k)));
end
set(gca, 'YScale', 'log', 'FontSize', font);
xlabel('Iterations', 'Interpreter','latex');
ylabel('$\mathcal{L}=\|\beta G-F\|^2$', 'Interpreter','latex');
legend(legend_labels, 'Location', 'best');
title(sprintf('Convergence curves, Round M (T\\_SIM=%.1f$\\lambda$, L=%d, s\\_x=%.2f$\\lambda$)', ...
    T_SIM_base/lambda, L_base, s_x_base/lambda), 'Interpreter','latex');

% Same pattern works for any round -- swap the mask/sort variable to L,
% T_SIM, or s_x to get the equivalent convergence-curve view for that round.

%% ===================== 4. Sensitivity ranking via N-way ANOVA =====================
% Quantifies how much of the variance in loss_final each parameter (and
% pairwise interactions) explains across the FULL grid -- not just the
% one-factor-at-a-time baseline slices used in views 2 and 3 above.
% Requires Statistics and Machine Learning Toolbox; check before relying on it:
if license('test', 'Statistics_Toolbox')
    [p, tbl, stats] = anovan(results.loss_final, ...
        {results.T_SIM, results.L, results.M, results.s_x}, ...
        'model', 'interaction', ...
        'varnames', {'T_SIM','L','M','s_x'}, ...
        'display', 'off');

    fprintf('\n--- ANOVA: contribution to loss_final variance ---\n');
    disp(tbl);
    fprintf('p-values (main effects): T_SIM=%.4f, L=%.4f, M=%.4f, s_x=%.4f\n', p(1:4));
else
    warning('Statistics and Machine Learning Toolbox not found -- skipping ANOVA. Views 1-3 above (parallelplot, heatmap, semilogy) are base MATLAB and do not need it.');
end

%% ===================== 5. T_SIM x L interaction plot =====================
% ANOVA found T_SIM:L significant (p=0.026) and larger than L's own main
% effect (p=0.047) -- meaning L's effect on loss isn't a single direction,
% it depends on T_SIM. This plots mean loss_final for each (T_SIM, L) cell,
% averaged over M and s_x. Non-parallel/crossing lines = real interaction
% (as opposed to T_SIM:M, T_SIM:s_x, L:M, L:s_x, M:s_x, none of which were
% significant, so those pairs would look closer to parallel if plotted).
[uT, ~, iT] = unique(results.T_SIM);
[uL, ~, iL] = unique(results.L);
meanGrid = accumarray([iT iL], results.loss_final, [], @mean);   % nT x nL

figure; hold on; grid on;
colors = lines(numel(uT));
for i = 1:numel(uT)
    plot(uL, meanGrid(i,:), '-o', 'LineWidth', 1.5, 'Color', colors(i,:), ...
        'DisplayName', sprintf('T\\_SIM = %.1f$\\lambda$', uT(i)/lambda));
end
set(gca, 'FontSize', font);
xlabel('$L$', 'Interpreter','latex');
ylabel('Mean loss\_final (over M, s\_x)', 'Interpreter','latex');
legend('Interpreter','latex', 'Location','best');
title('T\_SIM $\times$ L interaction (non-parallel lines = real interaction)', 'Interpreter','latex');

%% ===================== 6. T_SIM x L interaction, faceted by M =====================
% The plot above averages over M too (9 raw combos per point), and M
% explains 36% of total variance -- far more than T_SIM, L, or their
% interaction. That averaging could be manufacturing or masking the
% apparent T_SIM:L pattern. Faceting by M isolates it: if the spike shows
% up at every M level, it's a real 2-way effect; if it only appears at one
% M (most likely M=9, the most resource-starved case), that's evidence of
% an unmodeled 3-way T_SIM:L:M interaction the 'interaction'-only ANOVA
% model can't capture -- and a likely contributor to the 32.8% Error term.
uM = unique(results.M);
figure;
for j = 1:numel(uM)
    subplot(1, numel(uM), j); hold on; grid on;
    sub = results(results.M==uM(j), :);
    [uT2, ~, iT2] = unique(sub.T_SIM);
    [uL2, ~, iL2] = unique(sub.L);
    meanGrid2 = accumarray([iT2 iL2], sub.loss_final, [], @mean);

    colors = lines(numel(uT2));
    for i = 1:numel(uT2)
        plot(uL2, meanGrid2(i,:), '-o', 'LineWidth', 1.5, 'Color', colors(i,:), ...
            'DisplayName', sprintf('T\\_SIM=%.1f$\\lambda$', uT2(i)/lambda));
    end
    xlabel('$L$', 'Interpreter','latex');
    if j == 1
        ylabel('Mean loss\_final (over s\_x)', 'Interpreter','latex');
    end
    title(sprintf('M = %d', uM(j)));
    if j == numel(uM)
        legend('Interpreter','latex', 'Location','best');
    end
    set(gca, 'FontSize', font);
end
sgtitle('T\_SIM $\times$ L interaction, faceted by M -- checks for a 3-way effect', 'Interpreter','latex');

%% ===================== 7. Production-relevant slice: M=81 only =====================
% The faceted plot showed the T_SIM:L interaction is strongest at the
% largest M tested -- i.e. the most production-relevant slice. Pulling
% exact numbers here rather than reading them off the chart.
M_target = 81;
sub81 = sortrows(results_disp(results_disp.M==M_target, :), 'loss_final');
fprintf('\n--- All M=%d combos, sorted by loss (production-relevant slice) ---\n', M_target);
disp(sub81(:, {'T_SIM_over_lambda','L','s_x_over_lambda','loss_final'}));
