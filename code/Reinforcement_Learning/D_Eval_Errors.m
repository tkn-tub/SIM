clc
clear all
close all

% Add required codebase folders to path
addingPathParentFolderByName('code');

% Load Parameters
Parameters;  %[output:09906f83] %[output:2aa8f679] %[output:7a5dc32e] %[output:4bb309be] %[output:04b7d3eb] %[output:024432db] %[output:6c8907f3] %[output:812d5799] %[output:734ad9e3] %[output:1fc2a7db]

% pos_MU = pos_SIM+[4, 4, (-pos_SIM(3)+1.5)] 
pos_MU = [2, 4, 1.5];

[psi_x_true, psi_y_true] = computePsiFromPos(pos_MU, EnvPars) %[output:40a6f55d] %[output:04baf914] %[output:7e03b266] %[output:9f0363de]

pos_MU_est = estimatePosFromAngles(psi_x_true, psi_y_true, EnvPars, pos_MU) %[output:5b763191] %[output:81ea9bc2]

error = norm(pos_MU-pos_MU_est) %[output:3189eb6f]

%%

function [psi_x_true, psi_y_true] = computePsiFromPos(pos_MU, EnvPars)
    % Geometry: vector from SIM to MU. SIM on the ceiling, array in the
    % horizontal (xy) plane, z-axis pointing down toward the floor.

    % Vector from SIM to MU
    v = pos_MU - EnvPars.pos_SIM;

    % Compute azimuth (phi) and elevation (theta)
    phi   = atan2(v(2), v(1))                 % azimuth angle [rad]
    theta = atan2(v(3), norm(v(1:2)))          % elevation angle [rad]
    % Antenna parameters
    d_x = EnvPars.d_x;   % SIM element spacing in x (meters)
    d_y = d_x;           % SIM element spacing in y (meters)

    % Raw electrical angles (phase progression per element spacing)
    psi_x = (2*pi*d_x/EnvPars.lambda) * sin(theta) * cos(phi);
    psi_y = (2*pi*d_y/EnvPars.lambda) * sin(theta) * sin(phi);

    % Wrapped electrical angles [0, 2\pi] 
    psi_x_true = mod(psi_x, 2*pi);
    psi_y_true = mod(psi_y, 2*pi);
end

function pos_MU_est = estimatePosFromAngles(psi_x_est, psi_y_est, EnvPars, pos_MU_true)
    % 1. Shift phases back to principal domain [-pi, pi]
    psi_x_est = mod(psi_x_est + pi, 2*pi) - pi;
    psi_y_est = mod(psi_y_est + pi, 2*pi) - pi;

    phi_est = atan2(-psi_y_est, -psi_x_est);
    % phi_est = atan(psi_y_est*EnvPars.d_x/(psi_x_est*EnvPars.d_x))
    theta_est = asin(EnvPars.lambda/(2*pi)*sqrt(psi_x_est^2/EnvPars.d_x^2+psi_y_est^2/EnvPars.d_x^2))

    pos_MU_est(1) = (EnvPars.pos_SIM(3)-pos_MU_true(3))/tan(theta_est)*cos(phi_est)+EnvPars.pos_SIM(1);
    pos_MU_est(2) = (EnvPars.pos_SIM(3)-pos_MU_true(3))/tan(theta_est)*sin(phi_est)+EnvPars.pos_SIM(2);
    pos_MU_est(3) = pos_MU_true(3);
end

% function [psi_x, psi_y] = computePsiFromPos(pos_MU, EnvPars)
%     % Geometry: vector from SIM to MU. SIM on the ceiling, array in the
%     % horizontal (xy) plane, z-axis pointing down toward the floor.
% 
%     delta = pos_MU(:) - EnvPars.pos_SIM(:);   % column vector
%     rng   = norm(delta);
% 
%     % Direction cosines along the array's x and y axes
%     u_x = delta(1) / rng;
%     u_y = delta(2) / rng;
% 
%     % Electrical angles (assuming d_y = d_x)
%     psi_x = mod(2*pi * EnvPars.d_x * u_x / EnvPars.lambda, 2*pi);
%     psi_y = mod(2*pi * EnvPars.d_x * u_y / EnvPars.lambda, 2*pi);
% end
% 
% function pos_MU_est = estimatePosFromAngles(psi_x_est, psi_y_est, EnvPars, pos_MU_true)
%     % 1. Unwrap phases from [0, 2pi) back to the principal domain [-pi, pi]
%     % This is required because physical phase differences cannot exceed pi 
%     % (assuming lambda/2 spacing).
%     psi_x_w = mod(psi_x_est + pi, 2*pi) - pi;
%     psi_y_w = mod(psi_y_est + pi, 2*pi) - pi;
% 
%     % 2. Recover direction cosines u_x and u_y
%     % Derived from: psi = 2 * pi * d * u / lambda
%     u_x = (psi_x_w * EnvPars.lambda) / (2 * pi * EnvPars.d_x);
%     u_y = (psi_y_w * EnvPars.lambda) / (2 * pi * EnvPars.d_x);
% 
%     % 3. Safety Clamp for noisy DQN estimations
%     % u_x^2 + u_y^2 + u_z^2 = 1. Therefore, u_x^2 + u_y^2 must be <= 1.
%     % If the agent outputs bad angles, this prevents imaginary numbers.
%     sum_u2 = u_x^2 + u_y^2;
%     if sum_u2 > 1
%         u_x = u_x / sqrt(sum_u2);
%         u_y = u_y / sqrt(sum_u2);
%         sum_u2 = 1;
%     end
% 
%     % 4. Calculate the magnitude of u_z
%     abs_u_z = sqrt(1 - sum_u2);
% 
%     % 5. Intersect the 3D vector with the known Z-plane of the MU
%     delta_z = pos_MU_true(3) - EnvPars.pos_SIM(3);
% 
%     % Calculate absolute distance (Range) from SIM to MU
%     if abs_u_z < 1e-6
%         R = EnvPars.L_hall/2; % Prevent division by zero if angles point exactly horizontal
%     else
%         R = abs(delta_z) / abs_u_z;
%     end
% 
%     % 6. Reconstruct the 3D position
%     pos_MU_est = zeros(3, 1); % Ensure it's a column vector
%     pos_MU_est(1) = EnvPars.pos_SIM(1) + R * u_x;
%     pos_MU_est(2) = EnvPars.pos_SIM(2) + R * u_y;
%     pos_MU_est(3) = pos_MU_true(3);
% end

function addingPathParentFolderByName(targetName)
    % Recursively places target folders into the MATLAB path
    currFolder = pwd;
    found = false;
    while true
        [parentFolder, currentName] = fileparts(currFolder);
        if strcmpi(currentName, targetName)
            found = true;
            break;
        end
        if isempty(parentFolder) || strcmp(currFolder, parentFolder)
            break;
        end
        currFolder = parentFolder;
    end
    if found
        addpath(genpath(currFolder));
    else
        warning('Folder named "%s" not found in any parent directory.', targetName);
    end
end

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright"}
%---
%[output:09906f83]
%   data: {"dataType":"textualVariable","outputData":{"name":"total_iteration","value":"1"}}
%---
%[output:2aa8f679]
%   data: {"dataType":"text","outputData":{"text":"Wireless packet type: SC\n","truncated":false}}
%---
%[output:7a5dc32e]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_y","value":"4"}}
%---
%[output:4bb309be]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_x","value":"15"}}
%---
%[output:04b7d3eb]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_y","value":"15"}}
%---
%[output:024432db]
%   data: {"dataType":"matrix","outputData":{"columns":20,"name":"zeta","rows":1,"type":"double","value":[["0.9800","0.9810","0.9820","0.9830","0.9840","0.9850","0.9860","0.9870","0.9880","0.9890","0.9900","0.9910","0.9920","0.9930","0.9940","0.9950","0.9960","0.9970","0.9980","0.9990"]]}}
%---
%[output:6c8907f3]
%   data: {"dataType":"textualVariable","outputData":{"name":"T_coh","value":"0.0045"}}
%---
%[output:812d5799]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_packets_coh","value":"14"}}
%---
%[output:734ad9e3]
%   data: {"dataType":"textualVariable","outputData":{"name":"T","value":"144"}}
%---
%[output:1fc2a7db]
%   data: {"dataType":"textualVariable","outputData":{"name":"SNR_dB","value":"18.2642"}}
%---
%[output:40a6f55d]
%   data: {"dataType":"textualVariable","outputData":{"name":"phi","value":"1.8925"}}
%---
%[output:04baf914]
%   data: {"dataType":"textualVariable","outputData":{"name":"theta","value":"-0.5639"}}
%---
%[output:7e03b266]
%   data: {"dataType":"textualVariable","outputData":{"name":"psi_x_true","value":"0.5310"}}
%---
%[output:9f0363de]
%   data: {"dataType":"textualVariable","outputData":{"name":"psi_y_true","value":"4.6901"}}
%---
%[output:5b763191]
%   data: {"dataType":"textualVariable","outputData":{"name":"theta_est","value":"0.5639"}}
%---
%[output:81ea9bc2]
%   data: {"dataType":"matrix","outputData":{"columns":3,"name":"pos_MU_est","rows":1,"type":"double","value":[["2.0000","4.0000","1.5000"]]}}
%---
%[output:3189eb6f]
%   data: {"dataType":"textualVariable","outputData":{"name":"error","value":"0"}}
%---
