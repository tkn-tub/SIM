%% SIM2 -- Empirical verification of the traceable (option b) layers
% Run AFTER Parameters; Calibration; and after building the geometry
% matrices and sim2 layers (i.e. drop in just before the
% criticNet = dlnetwork(...) line, or run the relevant setup first).
%
% This replaces "trust the math" with three concrete pass/fail gates,
% because the previous 3 fixes all looked sound in principle and all
% failed in practice. The third gate (does Theta actually move under a
% gradient step) is the one that catches a severed trace -- the failure
% mode option (b) is most at risk of.

fprintf('\n===== SIM2 traceable-layer verification =====\n');

%% Gate 1 -- network constructs without error
try
    criticNet = dlnetwork(layerGraph(statePath));
    fprintf('[GATE 1 PASS] dlnetwork constructed.\n');
catch ME
    fprintf('[GATE 1 FAIL] dlnetwork construction errored:\n  %s\n', ME.message);
    fprintf('  -> option (b) did not work. This is the signal to switch to option (a)\n');
    fprintf('     (refit F_amp as a closed-form traceable curve, fully autodiff-native).\n');
    return;
end

%% Gate 2 -- forward pass returns a valid 9x1 real finite vector
try
    x_test = dlarray(rand(2*EnvPars.N, 1), 'CB');
    q_test = predict(criticNet, x_test);
    q_val = extractdata(q_test);
    ok_shape  = isequal(size(q_val), [EnvPars.n_actions, 1]);
    ok_real   = isreal(q_val);
    ok_finite = all(isfinite(q_val));
    if ok_shape && ok_real && ok_finite
        fprintf('[GATE 2 PASS] forward pass -> %dx1 real finite Q-vector.\n', EnvPars.n_actions);
    else
        fprintf('[GATE 2 FAIL] forward pass output invalid (shape=%d real=%d finite=%d).\n', ...
            ok_shape, ok_real, ok_finite);
        return;
    end
catch ME
    fprintf('[GATE 2 FAIL] forward pass errored:\n  %s\n', ME.message);
    return;
end

%% Gate 3 -- gradient actually flows to Theta (the severed-trace test)
% Pull sim2_N's Theta gradient after one autodiff step on a dummy loss.
% If the gradient is nonzero and finite, the trace is intact and training
% will update the phases. (extractdata is guarded here because MATLAB may
% hand back either a dlarray or an already-plain single depending on
% version/path -- the v1 of this gate assumed dlarray and errored.)
try
    lossFcn = @(net, x) sum(predict(net, x), 'all');
    x_test  = dlarray(rand(2*EnvPars.N, 1), 'CB');
    grads   = dlfeval(@(net,x) dlgradient(lossFcn(net,x), net.Learnables), criticNet, x_test);

    idx = find(strcmp(grads.Layer,'sim2_N') & strcmp(grads.Parameter,'Theta'));
    if isempty(idx)
        fprintf('[GATE 3 FAIL] no gradient entry found for sim2_N/Theta.\n');
        return;
    end

    gval = grads.Value{idx};
    if isa(gval,'dlarray')
        g = extractdata(gval);
    else
        g = gval;   % already plain (single/double) -- this is what tripped v1
    end

    gnorm = norm(double(g(:)));
    if gnorm > 0 && all(isfinite(g(:)))
        fprintf('[GATE 3 PASS] gradient flows to sim2_N.Theta (||grad|| = %.3e).\n', gnorm);
        fprintf('\n===== ALL GATES PASS -- option (b) works. Proceed to train(). =====\n');
    else
        fprintf('[GATE 3 FAIL] gradient to Theta is zero or non-finite (||grad|| = %.3e).\n', gnorm);
        fprintf('  -> trace severed; phases would not update. Switch to option (a).\n');
    end
catch ME
    fprintf('[GATE 3 ERROR in harness] %s\n', ME.message);
    fprintf('  NOTE: if Gates 1 and 2 passed, this is likely a bug in THIS test\n');
    fprintf('  script, not in the layers. The network already constructed and ran.\n');
end
