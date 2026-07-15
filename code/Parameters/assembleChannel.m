function h = assembleChannel(KR, los, nlos, EnvPars)
% los fields:
%   theta, phi, alpha, F
%
% nlos fields:
%   Pc                 [Nc x 1]
%   theta              [Nc x Mray]
%   phi                [Nc x Mray]
%   alpha              [Nc x Mray]
%   F                  [Nc x Mray]

    KR = double(KR);

    a_los = steeringVector2D( ...
        los.theta, los.phi, EnvPars);

    h_los = sqrt(KR/(KR + 1)) * ...
        los.F * los.alpha * a_los;

    Nc   = numel(nlos.Pc);
    Mray = size(nlos.alpha, 2);

    h_nlos = zeros(EnvPars.N, 1);

    for c = 1:Nc
        for m = 1:Mray

            a_cm = steeringVector2D( ...
                nlos.theta(c,m), ...
                nlos.phi(c,m), ...
                EnvPars);

            h_nlos = h_nlos + ...
                sqrt(nlos.Pc(c)/Mray) * ...
                nlos.F(c,m) * ...
                nlos.alpha(c,m) * ...
                a_cm;
        end
    end

    h = h_los + sqrt(1/(KR + 1)) * h_nlos;

    % Controlled-SNR convention:
    % absorb large-scale attenuation into SNR
    h = h / sqrt(mean(abs(h).^2));
end

function a = steeringVector2D(theta, phi, EnvPars)

    psi_x = 2*pi/EnvPars.lambda * ...
        EnvPars.d_x * sin(theta) * cos(phi);

    psi_y = 2*pi/EnvPars.lambda * ...
        EnvPars.d_x * sin(theta) * sin(phi);

    ax = exp(1i*psi_x*((0:EnvPars.N_x-1).'));
    ay = exp(1i*psi_y*((0:EnvPars.N_y-1).'));

    a = kron(ay, ax);
end