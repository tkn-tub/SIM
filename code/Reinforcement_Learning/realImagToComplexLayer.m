classdef realImagToComplexLayer < nnet.layer.Layer
    % realImagToComplexLayer  Reconstructs a genuine complex field from
    % its real and imaginary parts, stacked: X = [Re(r); Im(r)], a real
    % 2N-vector. Z = X(1:N) + 1i*X(N+1:2N), a complex N-vector.
    %
    % Coherent-r SIM1->SIM2 interface: phase carried through intact.
    % CONVENTION: input layout is [Re(r); Im(r)] -- must match
    % stepFunction_nav_CST.m / resetFunction_nav.m exactly.
    %
    % predict() (1 output) for inference; forward() (2 outputs, with
    % memory) for training, paired with backward().

    methods
        function layer = realImagToComplexLayer(name)
            layer.Name = name;
            layer.Description = 'Real [Re(r);Im(r)] (2N) -> complex field (N), phase preserved';
        end

        function Z = predict(layer, X)
            Z = layer.computeZ(X);
        end

        function [Z, memory] = forward(layer, X)
            [Z, memory] = layer.computeZ(X);
        end

        function dLdX = backward(layer, X, Z, dLdZ, memory)
            % Z = Re_part + i*Im_part => dL/dRe_part=Re(dLdZ), dL/dIm_part=Im(dLdZ)
            dLdZ_d = layer.safeExtract(dLdZ);
            dLdX = dlarray([real(dLdZ_d); imag(dLdZ_d)], 'CB');
        end
    end

    methods (Access = private)
        function [Z, memory] = computeZ(layer, X)
            X_d = layer.safeExtract(X);
            N2 = size(X_d, 1);
            N  = N2 / 2;
            Z = dlarray(complex(X_d(1:N,:), X_d(N+1:2*N,:)), 'CB');
            if nargout > 1
                memory = struct('N', N);
            end
        end

        function y = safeExtract(~, x)
            if isa(x, 'dlarray')
                y = extractdata(x);
            else
                y = x;
            end
        end
    end
end
