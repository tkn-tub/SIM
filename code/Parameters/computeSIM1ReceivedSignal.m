function [r, r_signal] = computeSIM1ReceivedSignal(t_psi, h, SNR_dB, EnvPars)
%COMPUTESIM1RECEIVEDSIGNAL Implements Eq. (5):
%
% r = sqrt(SNR) * G * Upsilon0 * h + u
%
% h is assumed to satisfy mean(abs(h).^2) = 1.
% u ~ CN(0,I).

    arguments
        t_psi (1,1) double {mustBeInteger, mustBePositive}
        h (:,1) double
        snr_dB (1,1) double
        EnvPars struct
    end

    assert(numel(h) == EnvPars.N, ...
        'Channel vector h must contain EnvPars.N elements.');

    v0 = EnvPars.U_func(1:EnvPars.N, t_psi);

    % Preserve the convention used during training.
    % Verify separately whether the conjugation introduced by v0'
    % is intended in your phase convention.
    Upsilon0 = diag(v0');

    SNR_linear = 10.^(SNR_dB/10);

    r_signal = sqrt(SNR_linear) * ...
        EnvPars.G * Upsilon0 * h;

    % Unit-variance circular complex Gaussian noise:
    % E{|u_n|^2} = 1
    u = (randn(EnvPars.N,1) + ...
         1i*randn(EnvPars.N,1)) / sqrt(2);

    r = r_signal + u;
end