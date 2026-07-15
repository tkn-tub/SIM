function h = generateChannel(pos_MU, EnvPars)
%GENERATECHANNEL Generate one realization of the spatial channel
%
% The realization contains:
%   - one LoS component;
%   - Nc NLoS clusters;
%   - Mray rays per cluster.
%
% Large-scale received power is absorbed into the controlled SNR.
% The complex coefficients therefore primarily represent phase and
% small-scale spatial interference.


    KR = 10.^(EnvPars.KR_dB/10);

    Nc   = EnvPars.Nc;
    Mray = EnvPars.Mray;

    kappa = 2*pi/EnvPars.lambda;

    %% LoS geometry

    delta = pos_MU(:) - EnvPars.pos_SIM(:);
    distance = norm(delta);

    if distance <= 0
        error('MU and SIM positions must be different.');
    end

    % Elevation measured from the downward SIM boresight:
    % theta = 0 means directly below the SIM.
    horizontalDistance = hypot(delta(1), delta(2));

    theta0 = atan2(horizontalDistance, -delta(3));
    phi0   = atan2(delta(2), delta(1));

    a0 = steeringVector(theta0, phi0, EnvPars);

    % Approximate receive-element amplitude response
    F0 = max(cos(theta0), 0).^EnvPars.elementCosinePower;

    % Free-propagation phase of the LoS component.
    % Path loss is absorbed into SNR for the controlled-SNR experiment.
    alphaL = exp(-1i*kappa*distance);

    h_los = sqrt(KR/(KR + 1)) * F0 * alphaL * a0;

    %% Cluster powers Pc

    if isempty(EnvPars.Pc)

        % Temporary exponentially decaying profile.
        % Replace with Pc generated according to 38.901 for the final model.
        clusterIndex = (0:Nc-1).';

        Pc = exp(-clusterIndex/EnvPars.clusterPowerDecay);

    else
        Pc = EnvPars.Pc(:);

        assert(numel(Pc) == Nc, ...
            'EnvPars.eq6.Pc must contain exactly %d entries.', Nc);
    end

    Pc = Pc / sum(Pc);

    %% NLoS clustered component

    sigmaClusterTheta = deg2rad(EnvPars.clusterThetaSpread_deg);
    sigmaClusterPhi   = deg2rad(EnvPars.clusterPhiSpread_deg);
    sigmaRayTheta     = deg2rad(EnvPars.rayThetaSpread_deg);
    sigmaRayPhi       = deg2rad(EnvPars.rayPhiSpread_deg);

    h_nlos = complex(zeros(EnvPars.N, 1));

    for c = 1:Nc

        % Cluster-center direction
        thetaCluster = theta0 + sigmaClusterTheta*randn;
        phiCluster   = phi0   + sigmaClusterPhi*randn;

        for m = 1:Mray

            % Ray direction around its cluster center
            thetaRay = thetaCluster + sigmaRayTheta*randn;
            phiRay   = phiCluster   + sigmaRayPhi*randn;

            % Restrict arrivals to the visible half-space of the SIM
            thetaRay = min(max(thetaRay, 0), pi/2 - 1e-6);

            aRay = steeringVector( ...
                thetaRay, phiRay, EnvPars);

            FRay = max(cos(thetaRay), 0).^ ...
                   EnvPars.elementCosinePower;

            % Random phase represents free-propagation phase and the
            % fixed Doppler phase within this channel-coherence interval.
            alphaRay = exp(1i*2*pi*rand);

            h_nlos = h_nlos + ...
                sqrt(Pc(c)/Mray) * ...
                FRay * alphaRay * aRay;
        end
    end

    %% Equation (6)

    h = h_los + sqrt(1/(KR + 1))*h_nlos;

    %% Controlled-SNR normalization

    if EnvPars.normalizeH
        channelPower = mean(abs(h).^2);

        if channelPower <= eps
            error('Generated channel has approximately zero power.');
        end

        h = h / sqrt(channelPower);
    end
end


function a = steeringVector(theta, phi, EnvPars)

    kappa = 2*pi/EnvPars.lambda;

    psi_x = kappa * EnvPars.d_x * ...
            sin(theta) * cos(phi);

    psi_y = kappa * EnvPars.d_y * ...
            sin(theta) * sin(phi);

    % Same sign convention as calibration, reset, step,
    % and the agent's original training environment
    ax = exp( ...
        -1i*psi_x*(0:EnvPars.N_x-1)).';

    ay = exp( ...
        -1i*psi_y*(0:EnvPars.N_y-1)).';

    a = kron(ay, ax);
end

%[appendix]{"version":"1.0"}
%---
