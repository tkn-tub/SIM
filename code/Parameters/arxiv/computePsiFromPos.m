function [psi_x, psi_y] = computePsiFromPos(pos_MU, pos_SIM, lambda,d_x,d_y)
    % Geometry: vector from SIM to MU. SIM on the ceiling, array in the
    % horizontal (xy) plane, z-axis pointing down toward the floor.
    
    delta = pos_MU(:) - pos_SIM(:);   % column vector
    rng   = norm(delta);
    
    % Direction cosines along the array's x and y axes
    u_x = delta(1) / rng;
    u_y = delta(2) / rng;

    %elevation evaluation
    sin_theta=norm(pos_MU(1:2)-pos_SIM(1:2))/norm(pos_MU-pos_SIM);
    %azimuth evaluation
    sin_psi=norm(pos_MU(2)-pos_SIM(2))/norm(pos_MU(1:2)-pos_SIM(1:2));
    cos_psi=norm(pos_MU(1)-pos_SIM(1))/norm(pos_MU(1:2)-pos_SIM(1:2));
    
    % Electrical angles as follows from Eqs. (1) and (2) in [1]
    psi_x = 2*pi/lambda * d_x * sin_theta * cos_psi;
    psi_y = 2*pi/lambda * d_y * sin_theta * cos_psi;
    % psi_x = mod(2*pi/lambda * d_x * u_x, 2*pi);
    % psi_y = mod(2*pi/lambda * d_y * u_y, 2*pi);
end

%References
%[1] J. An et al., "Two-Dimensional Direction-of-Arrival Estimation Using Stacked Intelligent Metasurfaces," IEEE Journal on Selected Areas in Communications, vol. 42, no. 10, pp. 2786–2802, Oct. 2024, doi: 10.1109/JSAC.2024.3414613.
