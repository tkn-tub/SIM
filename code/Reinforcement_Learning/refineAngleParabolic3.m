function [psi_x_est, psi_y_est, dbg] = refineAngleParabolic3(h, t_x0, t_y0, EnvPars, opts)
%REFINEANGLEPARABOLIC3  Sub-bin refinement, v3: recentering + 2D quadratic LS.
%
% v3 vs v2 (same hardware, same conventions, (t-1) bias fix included):
%   1) RECENTERING: 3x3 scan at spacing m around the given centre BEFORE any
%      fitting; if a neighbour beats the centre, the patch moves there (up to
%      opts.n_recenter hops). Widens capture from +-m to +-(1+n_recenter)*m,
%      attacking the timeout-driven reject rate.
%   2) 2D QUADRATIC LEAST-SQUARES on the 3x3 patch (9 samples, 6 params):
%      overdetermined, so multipath ripple and noise AVERAGE OUT instead of
%      steering a 3-point vertex. Guard = negative-definite Hessian + vertex
%      inside the patch.
%   3) PROBE CACHE: every probed config stored; recentre hops reuse the
%      overlapping patch points. dbg reports exact probe/packet counts.
%
% POST-PROCESSING ONLY: never touches the reward. Coarse SIM-1 bin read at
% the FINAL (wrapped) centre with the validated flip + mod 2*pi convention.
% Only the fine steering coordinate t is refined.
%
% INPUTS
%   h        : frozen episode channel (N x 1)
%   t_x0,t_y0: initial centre (integer, 1-indexed) -- pass BEST-VISITED config
%   EnvPars  : G, U_func, SNR_dB, N, N_x, N_y, T_x, T_y
%   opts.m          : patch spacing in t-steps            (default 12)
%   opts.n_avg      : packets averaged per probed config  (default 8)
%   opts.n_recenter : max recentre hops                   (default 2)
%   opts.noiseless  : true for quantization-floor tests   (default false)
%
% OUTPUT
%   psi_x_est, psi_y_est : refined electrical angles, [0,2*pi)
%   dbg : delta_x/y, rejected (+ rejected_x/y mirrors so the v2 call-site
%         line 'dbg.rejected_x || dbg.rejected_y' works unchanged),
%         recenter_dx/dy, n_configs, n_packets, contrast

    if nargin < 5, opts = struct; end
    if ~isfield(opts,'m'),          opts.m = 12;            end
    if ~isfield(opts,'n_avg'),      opts.n_avg = 8;         end
    if ~isfield(opts,'n_recenter'), opts.n_recenter = 2;    end
    if ~isfield(opts,'noiseless'),  opts.noiseless = false; end
    m  = opts.m;
    Tx = EnvPars.T_x;  Ty = EnvPars.T_y;
    snr_amp = sqrt(db2pow(EnvPars.SNR_dB));

    % ---------- cached probe: per-port power, averaged over n_avg packets --
    cacheK = {};  cacheV = {};
    n_probed = 0;
    function Pvec = powerAt(tx, ty)
        tx = mod(tx-1, Tx) + 1;  ty = mod(ty-1, Ty) + 1;   % physical wrap
        key = sprintf('%d_%d', tx, ty);
        idx = find(strcmp(cacheK, key), 1);
        if ~isempty(idx), Pvec = cacheV{idx}; return; end
        tpsi = (ty-1)*Tx + tx;
        v0   = EnvPars.U_func(1:EnvPars.N, tpsi);
        rsig = snr_amp * EnvPars.G * diag(v0') * h;
        if opts.noiseless
            Pvec = abs(rsig).^2;
        else
            Pvec = zeros(EnvPars.N,1);
            for p = 1:opts.n_avg
                u = (randn(EnvPars.N,1) + 1i*randn(EnvPars.N,1))/sqrt(2);
                Pvec = Pvec + abs(rsig + u).^2;
            end
            Pvec = Pvec / opts.n_avg;
        end
        cacheK{end+1} = key;  cacheV{end+1} = Pvec;  n_probed = n_probed + 1;
    end

    % ---------- 3x3 patch of PEAK powers around a centre -------------------
    function P9 = patchAt(cx_, cy_)
        P9 = zeros(3,3);
        for iu = -1:1
            for iv = -1:1
                P9(iu+2, iv+2) = max(powerAt(cx_ + iu*m, cy_ + iv*m));
            end
        end
    end

    % ---------- recentering -------------------------------------------------
    cx = t_x0;  cy = t_y0;              % unwrapped walk; powerAt wraps physically
    P9 = patchAt(cx, cy);
    for hop = 1:opts.n_recenter
        [~, mi] = max(P9(:));
        [iu, iv] = ind2sub([3 3], mi);
        if iu == 2 && iv == 2, break; end
        cx = cx + (iu-2)*m;
        cy = cy + (iv-2)*m;
        P9 = patchAt(cx, cy);           % cache makes the overlap free
    end
    recenter_dx = cx - t_x0;
    recenter_dy = cy - t_y0;
    % Wrap the final centre BEFORE the psi formula: the coarse bin n is read
    % at the wrapped config, and (n, wrapped t) is the consistent anchor pair.
    cx = mod(cx-1, Tx) + 1;
    cy = mod(cy-1, Ty) + 1;

    % ---------- 2D quadratic LS fit on the 3x3 patch ------------------------
    % Coordinates (u,v) in {-1,0,1}. Orthogonal basis on the 9-point stencil:
    %   b = S(uP)/6, c = S(vP)/6, d = S((u^2-2/3)P)/2, e = S((v^2-2/3)P)/2,
    %   f = S(uvP)/4.   Vertex: [2d f; f 2e][u*;v*] = -[b;c].
    [U, V] = ndgrid(-1:1, -1:1);
    b = sum(sum(U .* P9)) / 6;
    c = sum(sum(V .* P9)) / 6;
    d = sum(sum((U.^2 - 2/3) .* P9)) / 2;
    e = sum(sum((V.^2 - 2/3) .* P9)) / 2;
    f = sum(sum((U .* V) .* P9)) / 4;

    detH = 4*d*e - f^2;
    rejected = ~(d < 0 && detH > 0);    % Hessian negative definite (true 2D max)
    u_star = 0;  v_star = 0;
    if ~rejected
        u_star = (-2*e*b + f*c) / detH;
        v_star = (-2*d*c + f*b) / detH;
        if abs(u_star) > 1 || abs(v_star) > 1   % vertex outside the patch
            rejected = true;  u_star = 0;  v_star = 0;
        end
    end
    delta_x = m * u_star;
    delta_y = m * v_star;

    % ---------- coarse SIM-1 bin at the FINAL centre (validated convention) -
    Pvec0 = powerAt(cx, cy);            % cached: no extra probe
    R = flipud(fliplr(reshape(Pvec0, [EnvPars.N_x, EnvPars.N_y])))';
    [~, linear_idx] = max(R, [], 'all');
    n_psi_x_max = ceil(linear_idx/EnvPars.N_x);
    n_psi_y_max = linear_idx-(n_psi_x_max-1)*EnvPars.N_x;

    % ---------- refined estimate, (t-1) bias fix included -------------------
    psi_x_est = mod(2*pi * (n_psi_x_max + (cx - 1 + delta_x)/Tx) / EnvPars.N_x, 2*pi);
    psi_y_est = mod(2*pi * (n_psi_y_max + (cy - 1 + delta_y)/Ty) / EnvPars.N_y, 2*pi);

    if opts.noiseless, n_packets = n_probed; else, n_packets = n_probed * opts.n_avg; end
    P_0  = P9(2,2);
    ring = P9([1 2 3 4 6 7 8 9]);
    dbg = struct( ...
        'delta_x',delta_x, 'delta_y',delta_y, ...
        'rejected',rejected, 'rejected_x',rejected, 'rejected_y',rejected, ...
        'recenter_dx',recenter_dx, 'recenter_dy',recenter_dy, ...
        'n_configs',n_probed, 'n_packets',n_packets, ...
        'contrast',(P_0 - mean(ring)) / max(P_0, eps));
end
