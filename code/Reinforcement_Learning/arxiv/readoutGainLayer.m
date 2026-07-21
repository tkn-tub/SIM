classdef readoutGainLayer < nnet.layer.Layer & nnet.layer.Acceleratable
    % readoutGainLayer  Single learnable scalar gain after the diode
    % readout: Z = Gain .* X. Physically: a digitally-programmable
    % readout gain (VGA/AGC before the ADC) -- standard receiver
    % hardware, one parameter.
    %
    % WHY: SIM-2 is passive -- phases redistribute energy but cannot
    % amplify, so the network's output magnitude is capped by the fixed
    % propagation physics, while the Q-targets under the current reward
    % reach ~40 (peak_bonus). If that passive ceiling is below the
    % target scale, TD targets are unreachable and learning saturates.
    % Initialized at Gain = 1, so the network is IDENTICAL to the
    % no-gain version at start: the LEARNED value of Gain is itself the
    % diagnostic. Gain ~ 1 after training -> scale was never a problem.
    % Gain >> 1 -> the passive ceiling was real and this one parameter
    % fixed it.
    %
    % Pure dlarray ops, no extractdata -- fully traceable, no custom
    % backward needed.

    properties (Learnable)
        Gain
    end

    methods
        function layer = readoutGainLayer(name)
            layer.Name = name;
            layer.Description = 'Learnable scalar readout gain (VGA/AGC), init 1';
            layer.Gain = dlarray(1);
        end

        function Z = predict(layer, X)
            Z = layer.Gain .* X;
        end
    end
end
