function [r, r_signal] = computeSIM1ReceivedSignal(t_psi, h, snr_dB, EnvPars)
%COMPUTESIM1RECEIVEDSIGNAL Evaluate the noisy SIM-1 received vector.
%
%   r = sqrt(SNR) * G * Upsilon0 * h + u
%
% where:
%   h ~ channel/steering vector
%   u ~ CN(0,I)
%
% Inputs:
%   t_psi  : scalar snapshot index
%   h      : N-by-1 complex channel vector
%   snr_dB : scalar SNR in dB
%   EnvPars: simulation parameter structure
%
% Outputs:
%   r_signal : noiseless received vector
%   r        : noisy received vector

    %% Basic input checks
    validateattributes(t_psi, {'numeric'}, ...
        {'scalar', 'integer', 'positive', '<=', EnvPars.T}, ...
        mfilename, 't_psi');

    validateattributes(snr_dB, {'numeric'}, ...
        {'scalar', 'real', 'finite'}, ...
        mfilename, 'snr_dB');

    % Guarantee a column vector
    h = h(:);

    assert(numel(h) == EnvPars.N, ...
        'computeSIM1ReceivedSignal:InvalidChannelSize', ...
        'The channel h has %d elements, but EnvPars.N = %d.', ...
        numel(h), EnvPars.N);

    
    %% Input-layer coefficients
    v0 = EnvPars.U_func(1:EnvPars.N, t_psi);

    % Preserve the conjugate-transpose convention used by your current code
    Upsilon0 = diag(v0');

    %% Noiseless received vector
    snr_linear = db2pow(snr_dB);

    r_signal = sqrt(snr_linear) * ...
               EnvPars.G * Upsilon0 * h;

    %% Unit-power circular complex Gaussian noise
    % E{|u_n|^2} = 1
    u = (randn(EnvPars.N, 1) + ...
         1i*randn(EnvPars.N, 1)) / sqrt(2);

    %% Noisy observation
    r = r_signal + u;
end