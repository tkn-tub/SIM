function pos_MU_est = estimatePosFromAngles(psi_x_est, psi_y_est, height_MU,pos_SIM, lambda,d_x,d_y)
    % 1. Shift phases back to principal domain [-pi, pi]
    psi_x_est = mod(psi_x_est + pi, 2*pi) - pi;
    psi_y_est = mod(psi_y_est + pi, 2*pi) - pi;

    %azimuth angle
    phi_est = atan2(-psi_y_est, -psi_x_est);
    %elevation angle
    theta_est = asin(lambda/(2*pi)*sqrt(psi_x_est^2/d_x^2+psi_y_est^2/d_y^2));

    pos_MU_est(1) = (pos_SIM(3)-height_MU)/tan(theta_est)*cos(phi_est)+pos_SIM(1);
    pos_MU_est(2) = (pos_SIM(3)-height_MU)/tan(theta_est)*sin(phi_est)+pos_SIM(2);
    pos_MU_est(3) = height_MU3;
end

%[appendix]{"version":"1.0"}
%---
