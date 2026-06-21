classdef simPropagationLayer < nnet.layer.Layer & nnet.layer.Acceleratable
    % simPropagationLayer  Fixed complex propagation as a real block
    % matrix -- REAL-STACKED, TRACEABLE (option b).
    %
    % Operates on dlarray X directly (no extractdata) so autodiff traces
    % through the matrix multiply. No learnable parameters, no custom
    % backward() -- autodiff handles the gradient.
    %
    % P (complex, outDim x inDim) -> real 2*outDim x 2*inDim block:
    %   BigP = [Re(P) -Im(P); Im(P) Re(P)]
    % BigP*[x;y] = [Re(P)x-Im(P)y; Im(P)x+Re(P)y] = [Re(Z);Im(Z)]
    % for Z = P*(x+iy).

    properties
        BigP   % real, [2*outDim x 2*inDim], constant
    end

    methods
        function layer = simPropagationLayer(P, name)
            layer.Name = name;
            layer.Description = sprintf('SIM2 fixed propagation (%d -> %d, traceable)', size(P,2), size(P,1));
            A = real(P); B = imag(P);
            layer.BigP = [A, -B; B, A];
        end

        function Z = predict(layer, X)
            Z = layer.BigP * X;   % BigP plain double, X traced dlarray -> traced output
        end
    end
end
