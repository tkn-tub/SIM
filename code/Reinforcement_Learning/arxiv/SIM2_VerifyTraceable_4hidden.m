%% SIM2 (4 hidden layers) -- traceable-layer verification
% Same three gates as SIM2_VerifyTraceable.m, but Gate 3 now checks
% gradient flow through ALL THREE phase layers, not just one. In a deeper
% all-phase stack the EARLIEST layer (sim2_N, furthest from the loss) is
% the one most at risk of a vanishing/severed gradient -- so it's checked
% explicitly. Run in place of the criticNet = dlnetwork(...) line.

fprintf('\n===== SIM2 (4 hidden) traceable-layer verification =====\n');

%% Gate 1 -- constructs
try
    criticNet = dlnetwork(layerGraph(statePath));
    fprintf('[GATE 1 PASS] dlnetwork constructed.\n');
catch ME
    fprintf('[GATE 1 FAIL] %s\n', ME.message);
    return;
end

%% Gate 2 -- forward pass valid
try
    x_test = dlarray(rand(2*EnvPars.N, 1), 'CB');
    q_val  = extractdata(predict(criticNet, x_test));
    if isequal(size(q_val),[EnvPars.n_actions,1]) && isreal(q_val) && all(isfinite(q_val))
        fprintf('[GATE 2 PASS] forward pass -> %dx1 real finite Q-vector.\n', EnvPars.n_actions);
    else
        fprintf('[GATE 2 FAIL] invalid output (shape/real/finite).\n'); return;
    end
catch ME
    fprintf('[GATE 2 FAIL] %s\n', ME.message); return;
end

%% Gate 3 -- gradient flows to ALL THREE phase layers
phase_layers = {'sim2_N','sim2_M','sim2_M2','sim2_M3','sim2_M4'};
try
    lossFcn = @(net,x) sum(predict(net,x),'all');
    x_test  = dlarray(rand(2*EnvPars.N, 1), 'CB');
    grads   = dlfeval(@(net,x) dlgradient(lossFcn(net,x), net.Learnables), criticNet, x_test);

    all_ok = true;
    for i = 1:numel(phase_layers)
        nm  = phase_layers{i};
        idx = find(strcmp(grads.Layer,nm) & strcmp(grads.Parameter,'Theta'));
        if isempty(idx)
            fprintf('[GATE 3 FAIL] no gradient entry for %s/Theta.\n', nm); all_ok=false; continue;
        end
        gv = grads.Value{idx};
        if isa(gv,'dlarray'), g = extractdata(gv); else, g = gv; end
        gnorm = norm(double(g(:)));
        if gnorm > 0 && all(isfinite(g(:)))
            fprintf('  [%s] ||grad|| = %.3e  OK\n', nm, gnorm);
        else
            fprintf('  [%s] ||grad|| = %.3e  PROBLEM (zero/non-finite)\n', nm, gnorm);
            all_ok = false;
        end
    end
    if all_ok
        fprintf('[GATE 3 PASS] gradient flows to all three phase layers.\n');
        fprintf('\n===== ALL GATES PASS -- 4-hidden network OK, proceed to train(). =====\n');
    else
        fprintf('[GATE 3 FAIL] at least one phase layer has no usable gradient.\n');
        fprintf('  If sim2_N (the earliest) is the one failing, that is a vanishing-\n');
        fprintf('  gradient symptom of the deeper stack, not a wiring bug.\n');
    end
catch ME
    fprintf('[GATE 3 ERROR in harness] %s\n', ME.message);
    fprintf('  If Gates 1-2 passed, this is likely a bug in THIS test, not the layers.\n');
end
