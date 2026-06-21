classdef diodeReadoutLayer < nnet.layer.Layer & nnet.layer.Acceleratable
    % diodeReadoutLayer  Down-conversion (real-part half) + diode
    % rectification -- REAL-STACKED input, TRACEABLE (option b).
    %
    % Operates on dlarray X directly (no extractdata). max(.,0) and
    % indexing are dlarray-native, so autodiff traces through. No custom
    % backward().
    %
    %   Input X = [Re(r_Q); Im(r_Q)], real, 2Q x batch
    %   y = X(1:Q,:)            (homodyne in-phase down-conversion)
    %   Z = max(y, 0)            (diode, linear regime) -> Q-values

    methods
        function layer = diodeReadoutLayer(name)
            layer.Name = name;
            layer.Description = 'Down-conversion + diode rectification (Q-values), traceable';
        end

        function Z = predict(layer, X)
            Q = size(X,1)/2;
            y = X(1:Q, :);
            Z = max(y, 0);
        end
    end
end
