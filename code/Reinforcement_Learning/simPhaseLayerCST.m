classdef simPhaseLayerCST < nnet.layer.Layer & nnet.layer.Acceleratable
    % simPhaseLayerCST  Learnable per-meta-atom phase mask with CST
    % amplitude-phase coupling -- REAL-STACKED, TRACEABLE (option b).
    %
    % Operates on dlarray X and dlarray Theta DIRECTLY -- no extractdata
    % in the forward path -- so dlnetwork's autodiff tracer can follow the
    % computation. No custom backward(): autodiff handles it.
    %
    % The ONLY non-traceable piece is F_amp (a griddedInterpolant, which
    % cannot accept dlarray). It is evaluated on a plain-double copy of
    % theta and injected as a FROZEN per-pass constant amplitude. The
    % phase term (cos/sin of the TRACED Theta) flows through autodiff.
    %
    % GRADIENT CONSEQUENCE (deliberate): because amplitude is frozen per
    % pass, the optimizer does NOT differentiate through dAmp/dTheta. The
    % forward pass is still fully CST-realistic (amplitude coupling is
    % applied every step); only the gradient treats amplitude as a fixed
    % hardware response. This differs from the earlier hand-written
    % backward (which included a finite-difference dAmp/dTheta term) by
    % exactly that term -- a stop-gradient on the amplitude nonideality.
    %
    %   X = [x; y]  (x=Re(in), y=Im(in), each numAtoms x batch)
    %   amp = F_amp(theta)            [frozen constant, per pass]
    %   a = amp.*cos(theta), b = amp.*sin(theta)   [a,b traced via theta]
    %   Re(out) = a.*x - b.*y ;  Im(out) = b.*x + a.*y
    %   Z = [Re(out); Im(out)]

    properties
        F_amp
    end

    properties (Learnable)
        Theta
    end

    methods
        function layer = simPhaseLayerCST(numAtoms, F_amp, name)
            layer.Name = name;
            layer.Description = sprintf('SIM2 learnable phase layer (%d atoms, CST-coupled, traceable)', numAtoms);
            layer.F_amp = F_amp;
            layer.Theta = dlarray(2*pi*rand(numAtoms,1));
        end

        function Z = predict(layer, X)
            numAtoms = size(X,1)/2;
            x = X(1:numAtoms, :);
            y = X(numAtoms+1:2*numAtoms, :);

            % theta stays TRACED for the phase term; a plain copy is used
            % ONLY as the lookup key into the (non-traceable) interpolant.
            theta = layer.Theta;
            theta_plain = extractdata(theta);
            amp = layer.F_amp(mod(theta_plain, 2*pi));   % frozen constant, numAtoms x 1

            % cos/sin operate on the TRACED theta -> differentiable.
            a = amp .* cos(theta);
            b = amp .* sin(theta);

            Re_out = a.*x - b.*y;
            Im_out = b.*x + a.*y;
            Z = [Re_out; Im_out];
        end
    end
end
