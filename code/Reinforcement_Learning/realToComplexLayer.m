classdef realToComplexLayer < nnet.layer.Layer
    % realToComplexLayer  Embeds a real-valued amplitude vector into the
    % complex domain with zero phase: Z = X + 0i. NOT used in the current
    % (coherent-r) pipeline -- superseded by realImagToComplexLayer.m --
    % kept correct in case the zero-phase design is revisited.
    %
    % predict() (1 output) for inference; forward() (2 outputs, with
    % memory) for training, paired with backward().

    methods
        function layer = realToComplexLayer(name)
            layer.Name = name;
            layer.Description = 'Real amplitude -> complex field, phase = 0';
        end

        function Z = predict(layer, X)
            Z = layer.computeZ(X);
        end

        function [Z, memory] = forward(layer, X)
            [Z, memory] = layer.computeZ(X);
        end

        function dLdX = backward(layer, X, Z, dLdZ, memory)
            dLdZ_d = layer.safeExtract(dLdZ);
            dLdX = dlarray(real(dLdZ_d), 'CB');
        end
    end

    methods (Access = private)
        function [Z, memory] = computeZ(layer, X)
            X_d = layer.safeExtract(X);
            Z = dlarray(complex(X_d, zeros(size(X_d), 'like', X_d)), 'CB');
            if nargout > 1
                memory = [];
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
