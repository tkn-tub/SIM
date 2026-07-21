function [psi_x_est, psi_y_est, dbg] = refineAngleParabolic(h, t_x0, t_y0, EnvPars, opts)
%REFINEANGLEPARABOLIC  Sub-bin refinement of the SIM-2 steering estimate.
%
% POST-PROCESSING ONLY. Runs after the episode terminates. Never touches the
% reward. Probes m-spaced neighbours of the terminal SIM-2 configuration,
% fits a 3-point parabola to the PEAK DIODE POWER along each axis (separable),
% and refines the fine steering coordinate  t -> t + delta.  The coarse SIM-1
% DFT bin (n) is read once at the terminal config, using the SAME convention
% (flipud/fliplr + mod 2*pi) that the psi/position probe validated. Only t is
% refined, because that is the axis the quantization floor lives on.
%
% INPUTS
%   h        : frozen episode channel (N x 1) -- MU does not move within episode
%   t_x0,t_y0: terminal SIM-2 config (integer, 1-indexed)
%   EnvPars  : must carry G, U_func, SNR_dB, N, N_x, N_y, T_x, T_y
%   opts.m         : sample spacing in t-steps (default 3)
%   opts.noiseless : true for the quantization-floor test (default false)
%
% OUTPUT
%   psi_x_est,psi_y_est : refined electrical angles, [0,2*pi), same convention
%   dbg                 : struct with delta_x, delta_y (t-step offsets) and
%                         rejected flags (edge-guard fired -> fell back to int)

    if nargin < 5, opts = struct; end
    if ~isfield(opts,'m'),         opts.m = 3;             end
    if ~isfield(opts,'noiseless'), opts.noiseless = false; end
    m  = opts.m;
    Tx = EnvPars.T_x;  Ty = EnvPars.T_y;

    % ---- field at an (integer) SIM-2 config, toroidal wrap, optional noise ----
    function rr = fieldAt(tx, ty)
        tx = mod(tx-1, Tx) + 1;                 % wrap only the PHYSICAL config
        ty = mod(ty-1, Ty) + 1;
        tpsi = (ty-1)*Tx + tx;
        v0   = EnvPars.U_func(1:EnvPars.N, tpsi);
        rr   = sqrt(db2pow(EnvPars.SNR_dB)) * EnvPars.G * diag(v0') * h;
        if ~opts.noiseless
            rr = rr + (randn(EnvPars.N,1) + 1i*randn(EnvPars.N,1))/sqrt(2);
        end
    end

    % ---- center config: coarse DFT bin (verbatim estimator convention) ----
    r0 = fieldAt(t_x0, t_y0);
    R  = flipud(fliplr(reshape(abs(r0), [EnvPars.N_x, EnvPars.N_y])))';
    [~, linear_idx] = max(R, [], 'all');
    n_psi_x_max = ceil(linear_idx/EnvPars.N_x);
    n_psi_y_max = linear_idx-(n_psi_x_max-1)*EnvPars.N_x;

    P_0 = max(abs(r0).^2);                       % shared center power

    % ---- 3-point peak-power samples, separable (x holds t_y0, y holds t_x0) ----
    P_xm = max(abs(fieldAt(t_x0 - m, t_y0)).^2);
    P_xp = max(abs(fieldAt(t_x0 + m, t_y0)).^2);
    P_ym = max(abs(fieldAt(t_x0, t_y0 - m)).^2);
    P_yp = max(abs(fieldAt(t_x0, t_y0 + m)).^2);

    [delta_x, rej_x] = parabolicVertex(P_xm, P_0, P_xp, m);
    [delta_y, rej_y] = parabolicVertex(P_ym, P_0, P_yp, m);

    % ---- refined estimate: SAME convention, integer t replaced by t + delta ----
    % delta is applied to the UN-wrapped t0 (n from the center holds), so no
    % wrap arithmetic enters the psi formula; |delta| <= m keeps it in-sector.
    psi_x_est = mod(2*pi * (n_psi_x_max + (t_x0 + delta_x)/Tx) / EnvPars.N_x, 2*pi);
    psi_y_est = mod(2*pi * (n_psi_y_max + (t_y0 + delta_y)/Ty) / EnvPars.N_y, 2*pi);

    dbg = struct('delta_x',delta_x,'delta_y',delta_y, ...
                 'rejected_x',rej_x,'rejected_y',rej_y);
end

function [delta, rejected] = parabolicVertex(y_m, y_0, y_p, spacing)
% 3-point parabolic peak offset, equal spacing (in t-steps).
% Edge-guard: the center must be the (weak) max AND the parabola concave,
% else the samples don't bracket an interior peak -> return 0 (keep integer).
    denom = (y_m - 2*y_0 + y_p);
    if (y_0 >= y_m) && (y_0 >= y_p) && (denom < 0)
        delta = 0.5 * spacing * (y_m - y_p) / denom;
        if abs(delta) > spacing            % vertex outside bracket -> reject
            delta = 0; rejected = true;
        else
            rejected = false;
        end
    else
        delta = 0; rejected = true;        % not a clean interior peak
    end
end