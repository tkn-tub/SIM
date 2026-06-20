%[text] # SIM optimization (2D-DFT)
%[text] Description: The code follows the algorithm introduced in reference \[1, Section II\] and \[2, Section III\], to find the SIM elements that approximates the 2D-DFT. The code includes the amplitude vs phase as finded out with the cell element simulation in CST. This code evaluates the phase shift $\\xi\_{l,m} \\in\[0,2 \\pi)$ of each meta-atom in the SIM (only for the intermediate layers). Intermediate layers correspond to all the layers beneath the input layer, see Fig. 1. 
%[text] The propagation of the waveform through the SIM is modeled with the attenuation coefficients between SIM layers, and the transmission coefficient for each meta-atom, given by $\\Gamma\_{l,m}=e^{j\\xi\_{l,m}}$. In practice, the transmission coefficients are adjusted by controlling the bias voltage on each meta-atom. The attenuation coefficients are modeled with the Rayleigh-Sommerfeld difraction equation \[2, Eq. (7)\], and in practice they are fixed, as they depend on the geometry of the SIM and the frequency of the waveform.
%[text] The coefficient which defines the transmission parameter of each single SIM element.
%[text] The optimization problem to solve is the following
%[text] $\\min\_{\\xi\_{l,m}} \\;\\; \\mathcal L=\\|\\beta G - F\\|\_F^2$ (1)
%[text] $\\text{s.t. }$
%[text] $G = W\_L \\Upsilon\_L \\cdots W\_1\\Upsilon\_1 W\_0,$ $\\mathbf{G}\\,\\in\\,\\mathbb{C}^{N\_x\\times N\_y}$               (2)
%[text] $\\Upsilon\_l=\\mathrm{diag}(e^{j\\xi\_{l,1}},\\dots,e^{j\\xi\_{l,M}}) \\\]$                                (3)
%[text] $\\xi\_{l, m} \\in\[0,2 \\pi), \\quad m=1, \\cdots, M,\\quad l=1, \\cdots, L$    (4)
%[text] where 
%[text] - $l$ is the intermediate index layer
%[text] - $L$ is the number of intermediate layers
%[text] - $\\beta \\in \\mathbb{C}$ is a scaling factor and evaluated as \[Eq. (21), 1\] \
%[text]  $\\beta=(g^{H}g)^{-1}g^{H}f$                           (5)
%[text]              where $g=\\text{vec}{(G)}$ and $f=\\text{vec}{(F)}$. The $\\text{vec}(\\cdot)$ operation refers to concatenating each column of the matrix to construct a vector.
%[text] Besides
%[text] - $m$ is the meta-atom index in the layer $l$. This index encodes linearly the 2D position of the meta-atom as follows:
%[text] - $m\_{y}= \\left \\lceil {{ m/M\_{{x}} }}\\right \\rceil$                  (6)
%[text] - $m\_{x}= m-\\left( m\_{y}-1\\right) M\_{x}$     (7)
%[text] - $M$ is the number of meta-atoms on each intermediate layer \
%[text] $F$ is the matrix that represents the 2D-DFT as
%[text] $\\left \[{{ \\textbf{F} }}\\right \]\_{n,{n\_\\psi}}\\triangleq e^{-j2\\pi \\frac {\\left ({{ n\_{\\psi\_{\\textrm {x}}}-1 }}\\right) \\left ({{ n\_{\\textrm {x}}-1 }}\\right)}{N\_{\\textrm {x}}}}e^{-j2\\pi \\frac {\\left ({{ n\_{\\psi\_{\\textrm {y}}}-1 }}\\right)\\left ({{ n\_{\\textrm {y}}-1 }}\\right)}{N\_{\\textrm {y}}}}, \\mathbf{F}\\,\\in\\,\\mathbb{C}^{N\_x\\times N\_y}$   (8)
%[text] where $n\_x$ and $n\_y$ refers to the coordinates of each patch element in the input layer of the SIM, and $n\_{\\psi\_x}$ and $n\_{\\psi\_y}$ refer to the coordinates on the transform domain (angular), where the electric angles are estimated.
%[text] ![](text:image:2509)
%[text] Fig. 1: Coordinates of each meta-atom in the input layer of the SIM
%[text] 
%[text] $W\_l\\in\\mathbb{C}^{M\\times M}$ is the matrix that evaluates the EM propation between atoms from adjacent layers, where $W\_L, W\_0 \\, \\in\\mathbb{C}^{N\\times N}$. These matrices are evaluated based on the Rayleigh-Sommerfeld diffraction equation as follows \[Eq. (7), 1\]:
%[text] $\[\\mathbf{W}\_{l}\]\_{n, \\breve{n}}=\\frac{A\_{\\text {meta-atom }} s\_{\\text {layer }}}{2 \\pi d\_{n, \\breve{n}}^{3}}(1-j \\kappa d\_{n, \\breve{n}}) e^{j \\kappa d\_{n, \\breve{n}}}$   (9)
%[text] where $A\_{\\text {meta-atom }$ is the area of each meta-atom, $k=\\frac{2\\pi}{\\lambda}$ is the waveform number. The variable $d$ refers to the propagation distance between the atoms coordinates $n$ and $\\breve n$
%[text] $d\_{n, \\breve{n}}=\\sqrt{(n\_{\\mathrm{x}}-\\breve{n}\_{\\mathrm{x}})^{2} s\_{\\mathrm{x}}^{2}+(n\_{\\mathrm{y}}-\\breve{n}\_{\\mathrm{y}})^{2} s\_{\\mathrm{y}}^{2}+s\_{\\text {layer }}^{2}}$  (10)
%[text] where $n$ is the coordinates of the atom in the $l$ layer and $\\breve{n}$ is the coordinate of the atom in the $(l+1)$ layer. Their $(x,y)$ coordinates are evaluated as indicated in Eqs. (5)-(6).
%[text] where $s\_x$, $s\_y$, and $s\_\\text{layer}$ are the distances between elements in $x$-direction, $y$-direction, and between consecutive layers, see Fig. 2
%[text] ![](text:image:3778)
%[text] Fig. 2: Dimensions and distances parameters between meta-atoms.
%[text] 
%[text] The $W\_L\\in\\mathbb{C}^{M\\times N}$matrix, where $N$are the number of atoms in the input layer is also calculated with the Rayleight-Sommerfeld equation in Eq. (9).
%[text] 
%[text] The purpose of this code is to find the variables ${\\xi\_{l,m}$ for each patch element $m$ on each intermediate layer $m$. This code follows the Gradient Descent algorithm as described in \[Eq. (19), 1\] as follows
%[text] $\\xi\_{l,m}^{(k)}\\leftarrow\\xi\_{l,m}^{(k-1)}-\\eta\\nabla\_{\\xi\_{l,m}^{(k-1)}}\\mathcal{L}$ (11)
%[text] where $\\eta$ is a dynamic learning rate that is updated on each $k$-th iteration according to \[2, Eq. (20)\]
%[text] $\\eta^{(k)} \\leftarrow \\eta^{(k-1)} \\frac{\\zeta \\pi}{ \\max \_{l=1,2, \\cdots, L}\\{\\max \\nabla\_{{\\xi}\_{l}} \\mathcal{L}\\}}$ (12)
%[text] The parameter $\\zeta$ controls the maximum phase rotation per iteration as 
%[text] $\\max\_{l,m}|\\Delta \\epsilon\_{l,m}|\\leq\\zeta \\pi$     (13)
%[text] Smaller values of $\\zeta$ means faster convergence.
%[text] The iterative process in $k$ runs till the following condition is met
%[text] $\\frac{|\\mathcal{L}^{(k)}-\\mathcal{L}^{(k-1)}|}{ \\max \\{\\mathcal{L}^{(k-1)}\\}}\< \\text{tol}$   (14)
%[text] which tells about the control on the variability of $\\mathcal{L}$ in Eq. (1) above.
%[text] The $m$-th entry of gradient of $\\mathcal{L}$ is calculated as follows \[Eq. (15), 1\]:
%[text] $\\nabla\_{{\\xi}\_{l}} \\mathcal{L}=2 \\sum\_{n=1}^{N} \\Im\\left\\{\\beta^{\*} {\\Upsilon}\_{l}^{H} {P}\_{l, n}^{H}(\\beta {g}\_{n}-{f}\_{n})\\right\\}$     (15)
%[text] where $g=\\text{vec}{(G)}$ and $f=\\text{vec}{(F)}$, $e^H\_m$ is the $m$-th row of the identity matrix $I\_M$, and ${P}\_{l, n}$ is given by \[Eq. (16), 1\]
%[text] ${P}\_{l, n}={W}\_{L} {\\Upsilon}\_{L} {W}\_{L-1} \\cdots {W}\_{l+1} {\\Upsilon}\_{l+1} {W}\_{l} \\text{diag}({q}\_{l, n})$ (16)
%[text] and ${q}\_{l, n}$ is given by \[Eq. (17), 1\]
%[text] ${q}\_{l, n}={W}\_{l-1} {\\Upsilon}\_{l-1} {W}\_{l-2} \\cdots {W}\_{2} {\\Upsilon}\_{2} {W}\_{1} {\\Upsilon}\_{1} {w}\_{0, n}$      (17)
%[text] and $w\_{0,n}$ is the $n$-th column of $W\_0$.
clc;
clear all;
close all;

%including all parent folders up to the file 'code', this makes visible all
%files within this code
addingPathParentFolderByName('code'); %[output:1b7a169d]

Parameters %[output:73309dbf] %[output:44eeed17] %[output:1c38dc4a] %[output:6b8cd335] %[output:0c285bac] %[output:726ab75d] %[output:1bcb5afe] %[output:477a7ee4] %[output:21114024] %[output:23397e9b]

%%

%Loading the amplitude and phase calculated by CST
load t_y_x.mat
%Interpolate the transmission coefficient t_y_x for the amplitude
[F_amp, phase_min_meas, phase_max_meas] = build_amplitude_interpolant(t_y_x_amp_dB, t_y_x_phase_deg);
gap_frac_hist = zeros(maxIter,1);   % fraction of trained phases outside measured CST range, per iteration

% Sanity-check plot: interpolated amplitude vs phase (degrees)
% Run after build_amplitude_interpolant has produced F_amp, phase_min_meas, phase_max_meas,
% and after t_y_x.mat is loaded (mag_dB, phase_deg) so raw points can be overlaid.

xi_query = linspace(0, 2*pi, 2000);
amp_query = F_amp(xi_query);

figure; %[output:180eda8a]
plot(rad2deg(xi_query), 20*log10(amp_query), '--','Color', [0.5 0.5 0.5], 'LineWidth', 1.5); hold on; grid on; %[output:180eda8a]

% Overlay raw CST samples, wrapped into [0,360) to match the query axis
phase_meas_wrapped = mod(t_y_x_phase_deg, 360);

plot(phase_meas_wrapped, t_y_x_amp_dB, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 4); %[output:180eda8a]

% Shade the unmeasured region (where F_amp is extrapolating, not interpolating)
pmin_deg = rad2deg(phase_min_meas);
pmax_deg = rad2deg(phase_max_meas);
yl = ylim;
if pmin_deg <= pmax_deg
    patch([pmax_deg 360 360 pmax_deg], [yl(1) yl(1) yl(2) yl(2)], [1 0.6 0.6], 'FaceAlpha', 0.2, 'EdgeColor', 'none'); %[output:180eda8a]
    patch([0 pmin_deg pmin_deg 0], [yl(1) yl(1) yl(2) yl(2)], [1 0.6 0.6], 'FaceAlpha', 0.2, 'EdgeColor', 'none'); %[output:180eda8a]
else
    patch([pmax_deg pmin_deg pmin_deg pmax_deg], [yl(1) yl(1) yl(2) yl(2)], [1 0.6 0.6], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
end
ylim(yl); %[output:180eda8a]

xlabel('$\angle t_{yx}$ [degree]', 'Interpreter', 'latex'); %[output:180eda8a]
ylabel('$|t_{yx}|$ [dB]', 'Interpreter', 'latex'); %[output:180eda8a]
legend({'Interpolant', 'CST samples', 'Unmeasured region'}, 'Interpreter', 'latex', 'Location', 'best'); %[output:180eda8a]
set(gca, 'FontSize', font); %[output:180eda8a]
xlim([0 360]); %[output:180eda8a]


% Evaluating target 2D-DFT matrix F in Eq. (8) as
F = dft2_matrix(N_x, N_y);   % N_xN_y x N_xN_y, matches paper definition. [Eq. (11), 1]

% TRAIN_SIM_TO_DFT  Gradient descent training of SIM so G ~ beta^{-1} F
% Follows [Sec. III-B, Eqs. (18)-(21), (19)-(20), 1]
%
% Key dimensions:
%   N = N_x*N_y    (input/output layer meta-atoms & receiver probes)
%   M = M_x*M_y    (meta-atoms per intermediate layer)
%   L            (number of intermediate layers, layers 1..L)
%
% NOTE: Input layer phases (layer 0) are NOT trained here (paper keeps them for protocol later).
%
% Returns trained xi{l} vectors, and beta history.

rng(seed);

% -------------------- Grid Coordinates in the 2D --------------------
% ---- input grid coords (N = Nx*Ny) ----
[xn, yn] = grid_coords_centered(N_x, N_y, d_x, d_y);   % N x 1

% ---- metasurface grid coords (M = Mx*My) ----
[xm, ym] = grid_coords_centered(M_x, M_y, s_x, s_y);   % M x 1


% -------------------- Build propagation matrices W0, Wl, WL --------------------

%Input layer
% W0: M x N (input layer -> layer 1), Eq. (6)
W0=zeros(M,N);
for m=1:M%loop on the arriving atom
    for n=1:N %loop on the departing atom
        %evaluating the distance between meta-atoms along layer 0 and layer 1
        d  = sqrt((xm(m) - xn(n))^2 + (ym(m) - yn(n))^2 + s_layer^2);
        %
        cos_epsilon = s_layer/d; % as follows from Fig. 2 and defined in paragraph after Eq. (10) in [1]
        %
        W0(m,n) = (A_atom * cos_epsilon) .* (1./(2*pi*d.^2)) .* (1 - 1j*kappa.*d) .* exp(1j*kappa.*d);
    end
end
% %Testing
% figure;
% plot(abs(W0(:,3)),'LineWidth',2); grid on;
% xlabel('$m$','Interpreter','latex');
% ylabel('$|W0_{m,3}|$','Interpreter','latex');
% set(gca,'FontSize',font);
% axis([1 M 0 1.1*max(abs(W0(:,3)))]);

%Intermediate layer
% Wl: M x M for l=1..L-1 (between intermediate layers)
W = cell(L-1,1);
for l=1:(L-1)%loop on the layer index
    for m=1:M%loop on the arriving atom
        for n=1:M %loop on the departing atom
            %evaluating the distance between meta-atoms along layer 0 and layer 1
            d  = sqrt((xm(m) - xm(n))^2 + (ym(m) - ym(n))^2 + s_layer^2);
            %
            cos_epsilon = s_layer/d; % as follows from Fig. 2 and defined in paragraph after Eq. (10) in [1]
            %
            W_matrix(m,n) = (A_atom * cos_epsilon) .* (1./(2*pi*d.^2)) .* (1 - 1j*kappa.*d) .* exp(1j*kappa.*d);
            
        end
    end
    W{l}=W_matrix;
end
 
% %Testing
% figure;
% plot(abs(W{1}(:,M_y)),'LineWidth',2); grid on;
% xlabel('$m$','Interpreter','latex');
% ylabel('$|W1_{m,11}|$','Interpreter','latex');
% set(gca,'FontSize',font);
% axis([1 M 0 1.1*max(abs(W0(:,3)))]);

%Output layer
% WL: N x M (layer L -> output layer)
WL = W0.';   % paper in [2] before Eq. (14) notes WL = W0^T under isomorphic arrangement
% %Testing
% figure;
% plot(abs(WL(3,:)),'LineWidth',2); grid on;
% xlabel('$\breve{m}$','Interpreter','latex');
% ylabel('$|WL_{3,\breve{m}}|$','Interpreter','latex');
% set(gca,'FontSize',font);
% axis([1 M 0 1.1*max(abs(W0(:,3)))]);


% -------------------- Transmission matrix --------------------

%Phases are randomly initialized on each layer l of a total of L layers
xi = cell(L,1);%phases per layer
Upsilon = cell(L-1,1);%transmission matrix per layer
for l = 1:L
    xi{l} = 2*pi*rand(M,1);  % in [0,2pi)

    Upsilon{l}=diag(exp(1i*xi{l}));
end


%%



eta = eta0;  %initial learning rate
beta = 1+0j; %initial scaling factor


beta_hist = zeros(maxIter,1);

% ----------- Numeric gradient settings ------------------------------------
h = 1e-5; % finite difference step in radians (try 1e-4..1e-6)

G_vs_eta=cell(length(eta),1);%save the approximations given by G versus the parameter zeta
loss_hist = cell(length(eta),1);%save the evaluation of the loss function

for i_zeta=1:length(zeta)
    disp(i_zeta/length(zeta)*100)

    eta = eta0;  %initial learning rate
    beta = 1+0j; %initial scaling factor


    for it = 1:maxIter    
        % ----- Compute G = WL*Upsilon_L*W_{L-1}*...*Upsilon_1*W0  [Eq. (2)] ----- 
        G_intermediate = eye(M,M);%initializing the intermidieate variable
        %Computing first the intermediate terms Upsilon_l*W_{l-1}
        for l=2:(L)
            G_intermediate=Upsilon{l}*W{l-1}*G_intermediate;
        end
    
        %Including the remaining terms
        G=WL*G_intermediate*Upsilon{1}*W0;
    
        % ----- Update beta by LS -----
        g = G(:);
        f = F(:);
        % beta = (g' * g) \ (g' * f);%as follows from [Eq. (24), 1]
        beta = inv(g' * g) * (g' * f);%as follows from [Eq. (24), 1]
    
        % ----- Loss -----
        E = beta*G - F;
        Lval = norm(E,'fro')^2;
        loss_hist{i_zeta}(it) = Lval;
        beta_hist(it) = beta;
    
        % % stopping criteria
        % if it > 1 && abs(loss_hist{i_zeta}(it)-loss_hist{i_zeta}(it-1))/max(1,loss_hist{i_zeta}(it-1)) < tol
        %     loss_hist{i_zeta} = loss_hist{i_zeta}(1:it);
        %     beta_hist = beta_hist(1:it);
        %     break;
        % end
    
        
        % ===== Numerical gradient evaluation avoiding to compute Eq. (15) ===
        % grads{l}(m) ≈ (L(xi+ h e_{l,m}) - L(xi)) / (h)
        % ============================================================
        grads = cell(L,1);
        for l = 1:L
            grads{l} = zeros(M,1);
        end
    
        for l = 1:L
            for m = 1:M
    
                %Evaluating the gradient per layer and atom
                % --- xi plus ---
                xi_p = xi{l}(m)+h;
                Upsilon_p=Upsilon;
                Upsilon_p{l}(m,m)=exp(1i*xi_p);
    
                % ----- Compute G = WL*Upsilon_L*W_{L-1}*...*Upsilon_1*W0  [Eq. (2)] ----- 
                G_intermediate = eye(M,M);%initializing the intermidieate variable
                %Computing first the intermediate terms Upsilon_l*W_{l-1}
                for l_int=2:L
                    G_intermediate=Upsilon_p{l_int}*W{l_int-1}*G_intermediate;
                end
    
                %Including the remaining terms
                G_p=WL*G_intermediate*Upsilon_p{1}*W0;
                g_p=G_p(:);
                beta_p=inv(g_p' * g_p) * (g_p' * f);%as follows from [Eq. (24), 1]
    
                E_p = beta_p*G_p - F;
                L_p = norm(E_p,'fro')^2;           
    
                % --- central difference ---
                grads{l}(m) = (L_p - Lval) / (h);
    
            end
        end
    
        % ----- Update phases xi_l <- xi_l - eta * grad -----
        
    
        for l = 1:L
            for m=1:M
                xi{l}(m) = xi{l}(m) - eta * grads{l}(m);            
            end
            xi{l} = mod(xi{l}, 2*pi);
        end
    
        %Updating the new transmission matrix
        for l = 1:L        
            Upsilon{l}=diag(exp(1i*xi{l}));
        end
    
        % learning-rate schedule
        eta = eta*zeta(i_zeta); %as follows from [Eq. (20), 1]
        % % optional: keep your adaptive eta schedule, but usually start with fixed eta first
        % maxGrad = 0;
        % for l = 1:L
        %     maxGrad = max(maxGrad, max(abs(grads{l})));
        % end
        % eta = eta * zeta * pi / max(1e-12, maxGrad); % as follows from Eq. (12): eta <- eta * zeta * pi / max_l, following [Eq. (20), 2]
        
    
        % fprintf('it=%d, loss=%.3e, |beta|=%.3f, eta=%.3e\n', it, Lval, abs(beta), eta);
    end

    G_vs_eta{i_zeta}=G;

end
%%
% figure;
% % semilogy(loss_hist{1},'LineWidth',2); grid on; hold on;
% semilogy(loss_hist{2},'LineWidth',2); grid on; hold on;
% % semilogy(loss_hist{3},'LineWidth',2);
% semilogy(loss_hist{4},'LineWidth',2);
% % semilogy(loss_hist{5},'LineWidth',2);
% semilogy(loss_hist{6},'LineWidth',2);
% % semilogy(loss_hist{7},'LineWidth',2);
% semilogy(loss_hist{8},'LineWidth',2);
% % semilogy(loss_hist{9},'LineWidth',2);
% xlabel('Iterations','Interpreter','latex');
% ylabel('$\mathcal{L}=||\beta G-F||^2$','Interpreter','latex');
% legend({'$\zeta=0.2$','$\zeta=0.4$','$\zeta=0.6$','$\zeta=0.8$'},'Interpreter','latex')
% set(gca,'FontSize',font);
%%
%saving on the current folder
% save G_vs_eta_N_4 G_vs_eta M_x M_y N_x N_y L M N loss_hist beta_hist
% save G_vs_eta_N_16_Zeta_0_8_to_1 G_vs_eta M_x M_y N_x N_y L M N loss_hist beta_hist zeta
save G_vs_eta_N_16_Zeta_0_98_to_0_99 G_vs_eta M_x M_y N_x N_y L M N loss_hist beta_hist zeta
%saving on the Dataset folder
path = fullfile('..', 'Dataset', 'G_vs_eta_N_16_Zeta_0_98_to_0_99.mat');
save(path, 'G_vs_eta','M_x','M_y', 'N_x', 'N_y', 'L', 'M', 'N', 'loss_hist', 'beta_hist', 'zeta');

%%
% ======================= Helpers =======================
%%
function [x, y] = grid_coords_centered(Nx, Ny, dx, dy)
%This function evaluates the coordinates for each atom and assumes that the
% center (0,0) is at the center of the geometry, see Fig. 1 above.
% % n=1 indexes the bottom-right atom, n increases upward (x direction),
% then moves left column-by-column.
%

% Input Variables:
% Nx and Ny denote the total of atoms in the layer
% dx and dy denote the separation of each element
% Output Variables:
% x and y denotes the spatial coordinates

    N = Nx*Ny;

    x = zeros(N,1);
    y = zeros(N,1);
    nx = zeros(N,1);
    ny = zeros(N,1);    

    for n = 1:N
        %Evaluating the indexes nx and ny
        %See Fig. 1 to interpret n vs nx and ny
        iy = ceil(n/Nx); % 0..Ny-1 right->left
        ix = n - (iy-1)*Nx; % 0..Nx-1 bottom->top
        
        %Evaluating the spatial coordinates per atom
        x(n) = (ix - 1 - (Nx-1)/2) * dx ;        
        y(n) = ((Ny-1)/2 - (iy - 1)) * dy;  
    end
end
function [Lval, beta, G] = loss_SIM(Xi, WL, W, W0, F)
% Xi: MxL real phases
% Returns L = || beta*G - F ||_F^2 with optimal beta (LS)

    [G, Gamma] = forward_G_from_Xi(WL, W, W0, Xi);

    g = G(:);  f = F(:);
    beta = (g'*g) \ (g'*f);

    E = beta*G - F;
    Lval = norm(E,'fro')^2;
end

function [G, Gamma] = forward_G_from_Xi(WL, W, W0, Xi)
    [M,L] = size(Xi);
    Gamma = cell(L,1);
    for l = 1:L
        Gamma{l} = diag(exp(1j*Xi(:,l)));
    end

    X = W0;
    for l = 1:(L-1)
        X = W{l} * Gamma{l} * X;
    end
    X = Gamma{L} * X;
    G = WL * X;
end

function grad = numgrad_central(Xi, WL, W, W0, F, h)
% Central finite-difference gradient of loss w.r.t. Xi (MxL)
    [M,L] = size(Xi);
    grad = zeros(M,L);

    % baseline loss (optional)
    % L0 = loss_SIM(Xi, WL, W, W0, F);

    for l = 1:L
        for m = 1:M
            Xi_p = Xi; Xi_p(m,l) = Xi_p(m,l) + h;
            Xi_m = Xi; Xi_m(m,l) = Xi_m(m,l) - h;

            Lp = loss_SIM(Xi_p, WL, W, W0, F);
            Lm = loss_SIM(Xi_m, WL, W, W0, F);

            grad(m,l) = (Lp - Lm) / (2*h);
        end
    end
end


function [F_amp, phase_min, phase_max] = build_amplitude_interpolant(mag_dB, phase_deg)
% Builds a phase -> amplitude lookup from the CST sweep.
% Outside the measured arc [phase_min, phase_max], holds the boundary
% value constant (flat) rather than wrapping/blending across the gap --
% periodic wrap-around produced unphysical >0 dB (gain) in the unmeasured
% region, so flat-hold is the physically defensible choice here.

    phase_rad = deg2rad(phase_deg(:));
    phase_unwrapped = unwrap(phase_rad);
    mag_lin = 10.^(mag_dB(:)/20);

    [phase_sorted, idx] = sort(phase_unwrapped);
    mag_sorted = mag_lin(idx);

    % Bring into [0,2*pi) to match how xi is queried elsewhere in the code.
    % Safe as a uniform shift as long as the whole sweep sits within one
    % 360-degree window (true here), so monotonicity is preserved.
    phase_wrapped = mod(phase_sorted, 2*pi);
    [phase_wrapped, idx2] = sort(phase_wrapped);
    mag_sorted = mag_sorted(idx2);

    phase_min = phase_wrapped(1);
    phase_max = phase_wrapped(end);

    F_amp = griddedInterpolant(phase_wrapped, mag_sorted, 'pchip', 'nearest');
end

function F = dft2_matrix(Nx, Ny)
% Matches eq. (11): F(n,nhat) = exp(-j2pi (nx-1)(nhx-1)/Nx) * exp(-j2pi (ny-1)(nhy-1)/Ny)
    N = Nx*Ny;
    F = zeros(N,N);
    for n = 1:N
        [nx, ny] = ind2sub([Nx, Ny], n); % careful: Matlab ind2sub uses column-major; we want nx fastest
        % We'll define our own mapping: nx = mod(n-1,Nx)+1, ny = floor((n-1)/Nx)+1
        nx = mod(n-1, Nx) + 1;
        ny = floor((n-1)/Nx) + 1;
        for nh = 1:N
            nhx = mod(nh-1, Nx) + 1;
            nhy = floor((nh-1)/Nx) + 1;
            F(n,nh) = exp(-1j*2*pi*(nx-1)*(nhx-1)/Nx) * exp(-1j*2*pi*(ny-1)*(nhy-1)/Ny);
        end
    end
end


function [G, Gamma] = forward_G(WL, W, W0, xi)
    % This function evaluates all the products of Gamma_l*W_l as follows from Eq. (2)
    % Construct Gamma matrices and forward multiply to get G.
    L = numel(xi);
    Gamma = cell(L,1);
    
    %Evaluates the Gamma matrix per layer as follows from Eq. (3)
    for l = 1:L
        Gamma{l} = diag(exp(1j*xi{l}));
    end
    
    %Evaluates the multiplication Gamma_l*W_l
    X = W0;
    for l = 1:(L-1)
        X = W{l} * Gamma{l} * X;
    end
    X = Gamma{L} * X;
    G = WL * X;
    
    
end

function Wmm = build_W_MM(Mx, My, sx, sy, slayer, kappa, Ameta)
% Builds W^l (M x M) using Eq. (9)
%pre-allocating space
M = Mx*My;
Wmm = zeros(M,M);
for m = 1:M %loop in the (l+1)-layer
    [mx, my] = idx_to_xy(m, Mx);%evaluates the x and y indexes for the meta-atom m as follows from Eqs. (6)-(7).
    for mh = 1:M %loop in the l-layer
        [mhx, mhy] = idx_to_xy(mh, Mx);
        dxh = (mx - mhx)*sx;
        dyh = (my - mhy)*sy;
        d = sqrt(dxh^2 + dyh^2 + slayer^2);
        Wmm(m,mh) = (Ameta * slayer) / (2*pi*d^3) * (1 - 1j*kappa*d) * exp(1j*kappa*d);
    end
end
end

function post = backward_post_products(WL, W, Gamma)
%This function evaluates the product in Eq. (14)
% post{l} = WL*Gamma_L*W_{L-1}*Gamma_{L-1}*...*W_l   (N x M)
    L = numel(Gamma);
    post = cell(L,1);
    
    % Start from WL*Gamma_L
    P = WL * Gamma{L};          % N x M
    post{L} = P;            % for l=L, interpret W_L is identity here (no W_L term in between layer L and output aside from WL)
    
    for l = (L-1):-1:1
        % include W_l then Gamma_l
        P = P * W{l};       % N x M
        post{l} = P;        % this matches WL*Gamma_L*...*W_l
        P = P * Gamma{l};       % prepare for next
    end
end

function q = forward_q_all_layers(W, Gamma, w0_n)
% This function evaluate the vector q for all values of l of a total of L
    L = numel(Gamma);
    q = cell(L,1);
    
    % x = w0_n;           % enters layer1 before Gamma1
    % x = Gamma{1} * x;       % after layer1 phase
    % q{1} = x;
    x = w0_n;           % enters layer1 before Gamma1
    q{1} = w0_n;
    
    for l = 2:L
        % x = W{l-1} * x;     % propagate to next layer
        % x = Gamma{l} * x;       % apply phases of layer l
        % q{l} = x;
    
        % propagate from layer (l-1) to layer l, after applying Gamma_{l-1}
        x = W{l-1} * (Gamma{l-1} * x);
        q{l} = x;  % still BEFORE Gamma_l
        
    end
end

function [x, y] = idx_to_xy(m, Mx)
    % m -> (mx,my) with mx fastest, matches paper mapping idea. :contentReference[oaicite:14]{index=14}
    y = ceil(m/Mx);
    x = m - (y-1)*Mx;
end

function addingPathParentFolderByName(targetName)
    % Start from the current directory
    currFolder = pwd;
    found = false;
    
    % Continue searching until you reach the root folder
    while true
        % Get the parent folder
        [parentFolder, currentName] = fileparts(currFolder);
        
        % Check if the current folder's name is the target
        if strcmpi(currentName, targetName)
            found = true;
            break;
        end
        
        % If we've reached the root or no change, exit the loop
        if isempty(parentFolder) || strcmp(currFolder, parentFolder)
            break;
        end
        
        % Move one level up
        currFolder = parentFolder;
    end

    if found
        addpath(genpath(currFolder));
        fprintf('Adding matlab path to: %s\n', currFolder);
    else
        error('Folder named "%s" not found in any parent directory.', targetName);
    end
end

%[text] ### References
%[text] \[1\] J. An et al., "Stacked Intelligent Metasurface Performs a 2D DFT in the Wave Domain for DOA Estimation," in ICC 2024 - IEEE International Conference on Communications, June 2024, pp. 3445–3451. doi: 10.1109/ICC51166.2024.10622963.  
%[text] \[2\] J. An et al., "Two-Dimensional Direction-of-Arrival Estimation Using Stacked Intelligent Metasurfaces," IEEE Journal on Selected Areas in Communications, vol. 42, no. 10, pp. 2786–2802, Oct. 2024, doi: 10.1109/JSAC.2024.3414613.  
%[text]  
%[text] 

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":27.8}
%---
%[text:image:2509]
%   data: {"align":"baseline","height":242,"src":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlcAAAJsCAYAAADQhobqAAAA4WlDQ1BzUkdCAAAYlWNgYDzNAARMDgwMuXklRUHuTgoRkVEKDEggMbm4gAE3YGRg+HYNRDIwXNYNLGHlx6MWG+AsAloIpD8AsUg6mM3IAmInQdgSIHZ5SUEJkK0DYicXFIHYQBcz8BSFBDkD2T5AtkI6EjsJiZ2SWpwMZOcA2fEIv+XPZ2Cw+MLAwDwRIZY0jYFhezsDg8QdhJjKQgYG\/lYGhm2XEWKf\/cH+ZRQ7VJJaUQIS8dN3ZChILEoESzODAjQtjYHh03IGBt5IBgbhCwwMXNEQd4ABazEwoEkMJ0IAAHLYNoSjH0ezAAAACXBIWXMAAA7DAAAOwwHHb6hkAAAgAElEQVR4nOzdf3hU5Z3\/\/1e6Vl0XTCbj7rc\/+KGZoFDR8IEBLD9kQyWJwdpSWTNhqbWChqSt1a1ADWXXLsI2YG2r2yQo2BZdMlEoFiUmYOGLCaAQKAFUWpiIQLfuLsMk4odFd9t8\/qAzzpmZJDOTMzkzk+fjunJd58ycOeedE01e3Pd97jujq6urSwAAADDFJ6wuAAAAIJ0QrgAAAExEuAIAADAR4QoAAMBEhCsAAAATEa4AAABMRLgCkpTX65Xb7ZbL5VJFRYXV5aAPGhoaNH78eC1ZskTt7e1WlwMgwTKY5wpILl6vV2vXrtUPfvAD+Xw+SZLNZtPZs2ctrgzxqqioUE1NTWC\/pKREDz\/8sPLy8iysCkCi0HIFJJGGhgaNGDFCixcvls\/nk81mU2VlpVpbW6M+R0tLizIyMgxfLS0tCax6YGhoaFBubq4yMjJi\/uyyZctUW1srh8MhSaqvr9eYMWO0ZMkSeb1es0sFYDHCFZAEvF6vKioqNHPmzEBrVUlJiVpbW7V8+XLl5OREfa7du3eHvTZlyhTTah1o2tvbVVRUpJkzZ8rj8cR1DrvdrrKyMh0\/fly1tbWy2WySpBUrVqioqEhtbW1mlgzAYoQri1RUVARaFbKzs9XQ0GB1SbCI1+tVUVFRoNvIZrOprq5Obrc7plDlt337dsN+YWGhKXUONF6vVytXrpTD4VBTU5Np5y0rK9OOHTvkdDolSa2trcrPzydgAWmEcGWR4PEXPp9PR44csbAaWMUfrPzdfjabTTt27JDL5Yr7nHv37jXsT58+vU81DkRut1sTJ07U4sWLE3L+vLw8NTY2BgKWz+cjYAFphHAFWGjp0qWG8VTPPfdcnwY5t7W1BboV\/UaPHh33+cxUVFRkGAeWjNra2lRUVKTS0tK4uwCjZbfb1djYGBiH5fP5dMcddzAGC0gDhCvAIm6329CCWVVVpeLi4j6dMy8vT11dXYavvp5zIPB6vVqyZInGjBkT6AK02WyBsVGJYrfbtXHjxsC+x+PRN77xjYReE0DiEa4AC\/gHsPs5HA4tWrTIwooGLrfbrREjRmjFihWSLo5Ra25u1tmzZ3Xs2LGEB6y8vDxVVVUF9uvr6xmDCaQ4whVggaVLlxq67x599FELqxm4\/CHX5\/PJ4XBoy5YtamxsDDxdabfbNWHChITXMW\/ePEOIu\/\/++xN+TQCJQ7gC+ll7e7uhO9DhcPRpADviZ7fbVV1drcrKSr3xxhuWdaHa7XZ997vfDex7PB653W5LagHQd4QroJ899thjhv2FCxdaVAkkyeVyafny5bLb7ZbWMXv2bMP+9773PYsqAdBXhCugH\/nXCwwW+kc1FuPHjw+bjT14\/jSkjpycHMOcZB6Ph7FXQIoiXKWxhoYGLVmyJOwR+IyMDBUVFamioqLXX94tLS3Kzs4OfG78+PFxPSrucrlinjS1paUlYv3Z2dkqKirSypUrY14E1+12G84T\/PmWlha5XK7A95udnS2Xy2Xq0jHbtm0zjLUqLCzsU4vJvn37dPDgwcB8ScEKCgriPi+sMWvWLMP+yy+\/bFElAPqkC5aQZPiqqqoy5bxnzpzpqqqq6rLZbGHX6O7L6XR2HTx4MOL5Dh48GHZ8XV1dTDV5PJ6wc3R3va6urq7m5uauwsLCqOuvrKzsOnPmTFS1VFVVGT7b3Nwc8fXgr8LCwpi+356UlJQk5Ofe3NycsP+mzBL6M00Vkf5bTJTQ\/99sNlvCrgUgcWi5SiNtbW2GRX+j1dPyG3l5eYFJDv1efPHFmOratm2bYd\/hcHQ7Uabb7dbUqVNjWm7Evz5bvJMvVlRUJGwm7lBbt2417E+aNClh10rkuZEYeXl5hqcGfT4fi24DKYhwlUYihSqbzaaSkhJVVVUFvkpKSsLm7vHPDh1J6IDr+vr6mLrj1qxZY9gvKSmJeJzb7VZpaWlY\/ZWVldqyZYuam5vV3Nys2trasPXyWltb4wpY69evNzy5J0lOp1OFhYWmz28UafZ0sxZUPn36dNhrLNacmkKnfoi0EDeAJGd109lApQR04QR3XxQWFnZt2bKl22PPnDnTVV5eHlZHpM9E6tarra2NqqZI3Yoejyeq40pKSnrs7qurq4v5PvbU\/Repe3TLli09dmHGIrReh8Nhynm7usK\/L6fTadq5zUK3YHRCf5YlJSUJvR4A89FylUbGjRsnh8Oh5uZmNTY29jhnj39+n9Auv127doUdm5OTE9baFNoa1Z3nn3\/esO90OpWTkxN2XGi3XGFhodxud4+DvV0ul2praw2v\/eAHP4ire9DpdKqxsTGsu7K4uLhPa\/0FO3nypGE\/NzfXlPNK0vbt2w37DGZPH4le4xCA+QhXaWT58uU6fvx4TN1B9913n2F\/\/\/79EY\/78pe\/bNhvbW2NOEYrVH19vWF\/\/vz5Yce0t7eHjbGqrq7u9dySVFZWZgiIPp8vbIxXb2w2mxobGxM+z1FoAIoUMuO1d+9ew\/4NN9xg2rnRv0LHygUv7A0gNRCuBrjRo0dHdZzL5QobgxTaKhWqra0t7F\/dkeZ02rBhg2G\/pKQkpuARGhAPHz4c9WclafPmzZZMIHn11Vebcp5IY7n6Y8kW9J94H9YAYA3C1QB35ZVXRn1s6BItoa1SoULDV0lJScQQE9qik5+fH3VNkjRs2DDDfnetb5GUl5f328DvRP2BfPvttw37DofD1FaxUCtXrux24tKevkJbJ+M5R1FRUcK+r2QW+jMGkNwusboA9J9Ij3S\/+eabUX++rKzM8GSdx+NRS0tLt+EkNHyFdi36RZp2IZbHzzs7Ow37sYQYs1qPopGo7p3XXnvNsB9pQlEAQP8hXKWp9vZ2bdu2TZs2bdLevXtjmveqO\/45r4K7+tavXx8xXLW0tBiOs9lsUS9OvGDBgj7VOdDGqOzbt8+wH2vLHwDAXHQLphmv16uKigo5HA4tWLBATU1NpgQrv9A5r0LXyfNbv369Yb+7YBXNoPhYhT4Bmc68Xm9YmLz++usTes1Fixapq6sr5q\/QucniOUdjY2NCvzcAMAPhKo20tbVp4sSJYZNimil0QLrP54u4TmBo6JozZ07E8507d8684v7s0UcfNf2cySrSWBwmDwUAaxGu0sj8+fPDns5zOp2qra1Vc3NzxJaA5ubmmK5ht9vD5rxat26dYb+hocHQWuZwOLr9gx\/p9e5qjfYr2u7H\/paIsVChs3eHtg4BAPof4SpNuN3usO6h2tpa7du3T2VlZaa2Ztx1112G\/fr6esMg8pdfftnwfnfL3XQn0lIu6SAR0z2EPmk5btw4068B640aNcrqEgDEgHCVJn7+858b9gsLC1VWVpaQaxUXF4fNeRU8cWdol+C8efN6PF\/oGKnQmczTVWgwikfo5KGTJ0\/u8zlhrffffz\/sNSvmYQMQP8JVmjh+\/Lhhf\/r06Qm9Xnl5uWH\/hz\/8oaTwLsHulrsJFrpUywsvvGBSlcnF7J9Je3t72MMKI0eONPUa6H9Hjhwx7A+kBzSAdEG4ShPxrj8WyzxXwe68807Dfmtrq9rb28O6BL\/zne\/0eq6bb7457FyxzHOVKkInOw1tdYrVv\/\/7v4e9lsjJQ9E\/Tpw4Ydhn3jIg9RCu0kToQObQX9CRuN3uuOeUysvLC\/ulv2HDhrAnFWfMmNHruVwuV9i\/zu++++6YZzRvaWmJ+ORisggdN+Pz+fo0a3toMI70R7itrU1FRUUsn5JC2tvbDftjx461qBIA8SJcpYnQgcw1NTU9Bo2VK1eqtLS0T9cMXYR58eLFhv3ulruJJHT6BI\/HoxEjRnQ7j5af1+uV2+3W+PHjNXXqVM2dOzeq61khLy8vbKxaX5Y1CZ2ZPlRbW5vy8\/PV1NSktWvXxn2dgS60y11K7Fp\/oSsWhC7kDCD5MUN7knjqqadiGuCclZVlCB7z5s3TihUrDMfMnDlT5eXluu222wJrCL7yyiuqr6+Puxsx2OzZs3ts+Qp9qrAnLpdLr732mqHly+fzqbS0VBUVFSooKDD8C76zs1Nbt24Ne0LSzAlTE6GgoMCwLNDu3bvjfpIztJuxtbVVbW1tysvL0+rVq\/Xwww\/L5\/OpvLxcixYt6lPdA1XoSgN+GzZsSMgDI5Em1WXeMiAFdcESkvr85fF4DOesq6uL+Rzl5eVdhYWFgf3CwsKYvo+SkpKI57XZbHHdl\/Ly8j7dk5KSkh7PX1VVZTi+qqoqrjrjFfozcjqdcZ\/L4\/FE9fNNFsH\/nSXzr54zZ850NTc3d1VWVnbZbLYe721zc3PXmTNnTLt26H+fvf33DCA50S1oETMGqYbObu5yuVRXVxfW9RSJzWZTbW2tqqur+1RDd61T8U7kWV1drbq6upifkHI4HKqtre21G9FqM2bMMPx8\/A8CxCMnJ0eVlZXdvm\/Gz3egqKioUEZGhjIyMnTVVVdp6tSpWrFiRY8toTU1NZo6daquuuqqwGdzc3P7VEfok7KxtP4CSB6EK4tE8xRdT2w2m\/Ly8sJed7lcOnbsmGpra1VYWGj4Q26z2VRSUqLa2lodO3Ys0K0R\/IRZrE+bFRcXRwyKfekycblcOn78uLZs2aLKysqIs447nc7A93Lw4EEdP348qmuGdqWF7iea3W4PC57Bc4TFavny5aqsrAz8nB0Oh8rLy+XxeBI2z1k6ijfghupLd3t7e7uhm9vhcKi4uNiMsgD0s4yurq4uq4tAaluyZIlhvJfD4Yg4CBgXtbe3G1rmBsr9KioqMgzW5lePUUVFhWHMYVVVFWPlgBRFyxX6LHT6hfvuu8+iSlJDTk6OYRJWj8eT9N2ZSKz29nbDfwMOh6PXlQ0AJC\/CFfrE7XaHjUuZPXu2RdWkjoceesjQZfu9730v7eeiamxsNCywjY899thjhv+PHn30UZa8AVIY4Qp9EmlNQ2YJ711OTo7+5V\/+JbDv8Xj0+OOPW1gRrNLS0mJo\/S0sLIz7gRAAyYExV4hb6NghSaqrq+MPQwxCx9lw\/wYWr9erESNGBFqtbDabjh07RqsVkOJouULcQmf9ttlsUS13g48tW7bM8LRlRUVFxIkkkX68Xq+KiooMwWrHjh0EKyANEK4Qt9CB7C6Xiz8MMbLb7WpsbAwELJ\/Pp\/z8fAa4pzn\/mo\/+qRf8wSrS9CoAUg\/hCnGJNJB9zpw5FlWT2vwBy\/8EoX\/ZH5fLlfaD3Aei1atXKz8\/PxCsnE4nwQpIM4QrmKK8vJw10PrAbrerurpatbW1gacI6+vrNWLECC1ZssS0SS5hDf8C47m5uVqwYEHgHyaVlZVqbGwkWAFphgHtQJLxer1au3atfvCDHwT+CA+UiUbTVehEuyUlJXr44YcJVUCaIlwBScrr9Wrbtm168cUX5XA4tHz5cqtLQpxaWlr04IMPqqCgQPPmzWO6EiDNEa4AAABMxJgrAAAAExGuAAAATES4AgAAMBHhCgAAwESEKwAAABMRrgAAAExEuAIAADAR4QoAAMBEhCsAAAATEa4AAABMRLgCAAAwEeEKAADARIQrAAAAExGuAAAATES4AgAAMBHhCgAAwESEKwAAABMRrgAAAExEuAIAADAR4QoAAMBEhCsAAAATEa4AAABMRLgCAAAwEeEKAADARIQrAAAAE11idQFmeO655\/Tf\/\/3fstls+uxnP2t1OQCQtI4cOSJJuvfeey2uBEhfGV1dXV1WF9FXGRkZVpcAACll7Nix2r9\/v9VlAGmJbkEAAAATpUW34Cc+8Qn96U9\/0qWXXabptxRbXU5cBl+Ruj+Kc+f\/1+oS+oR7b41Uvu9Sat779\/7wex08sFeSNGLECIurAdJXav92+7NLLrlEH330kQYPvlJP1K6zupy4jBgyyOoS4nbs9AdWl9An3HtrpPJ9l1Lz3jdu2aT7F9wlSfrMZz5jcTVA+qJbEAAApL2VK1cqIyPD8JUohCsAAJD2tm\/fbtgvLCxM2LUIVwCAtLZ69WplZ2cHWitWrlxpdUmwQFNTk2F\/+vTpCbsW4QoAkNY2bdokn88X2A9twUD6a2trC3tt0qRJCbse4QoAAKS1119\/Pey1UaNGJex6hCsAAJDWysrK1NXVZfiy2+0Jux7hCgAAwESEKwAAABMRrgAAAExEuAIAADAR4QoAAKQNr9cbNhN78JfL5Up4DYQrAAD6qK2tTStXrlRRUZFhwtKMjAyNHz9eLpdLbrdbXq+323N4vV7l5uYGPpednR1xfqberF692nD9aCZNbW9v77b+oqIiLVmyRC0tLTHV0dbWZjhXQ0OD4XoVFRWG77eoqEhutzvm7zeU3W5XV1eXtmzZIpvNFvZ+fn5+n6\/RG8IVAABxcrvdys3N1ZgxY7R48WI1NTUZJiyVpNbWVtXX16u0tFQjRowwhIxgdrtdZ8+eDez7fD6tXr065ppWrVpl2M\/MzOz2WH\/IcTgc3dbf1NSkFStWaOrUqSoqKlJ7e3tUdZw7d85wriNHjkiSGhoa5HQ6VVNTI4\/HY7hOaWlpVOeORnFxsb773e+GvX799debdo3uEK4AAIiR1+vV+PHjVVpaaggIvfH5fJo5c2a3LTShXVaxtuS0tbWF1TNjxoxuj\/WHnGg1NTXJ6XTG1aImXfx+Zs6cGRbg+tOUKVMSfo1LEn4FAADSzNq1a9Xa2hr2emFhoWHNuhMnTmjr1q1hgae0tFQTJkxQTk6O4fWysjJD2PH5fHK73VGPE3r++ecN+06nM+wa0sVglZ+fHxZyysvLdfPNN2vIkCGSpNOnT+u1114Lqyk\/P187duxQXl5eVHVJ0oEDB\/SDH\/zA8JrD4VBubq6OHz8eU0iN1okTJwz7iVysORjhCgCAPnA4HHr00Uc1Y8aMbmf9Xr16tRYsWGB4be3atVq+fLnhtby8PDkcDkPQePHFF6MOV6GtUN\/5znfCjvF6vbrjjjsMwcrpdGrNmjURw5LL5VJZWZkhjPl8Pi1evFiNjY1R1SVJ9fX1gW2bzabnnntOxcXFgdfa2tr0+9\/\/PurzRSO0C3PcuHGmnr87dAsCABCj0aNHy2azqba2VsePH5fL5epxOZWysjKVl5cbXtu6dWvEYxcuXGjYr6+v73EgvF9DQ0NYS1SkLsG1a9cawpvNZlNjY2OPrVB5eXnasWOH4bWmpqaYB7n7r7djxw5DsPJfI\/S1vmpqajLsT5482dTzd4dwBQBAjIqLi3X27FmVlZVF\/Zk5c+YY9iN1K0qRA9GGDRt6Pf\/LL79s2C8pKYkY+J566inDfnV1dVTr7OXl5YUFxPXr1\/f6uVCbN2+OqTsxXpHGhY0cOTLh15UIVwAA9ItRo0ZFdVxOTo5KSkoMr61Zs6bXz4UOfr\/rrrvCjmlpaQlrtYpl3qfQgLhv376oPytJVVVV\/TKgXJJef\/11w77D4Yg4\/iwRCFcAAPSDaFqH\/L785S8b9ltbW3ucAiG0S9Bms0XsYtu9e7dhv6CgIOqaJOkzn\/lMWF3RcjgcWrRoUUzX64vQlqtYv9e+YEA7AAAma2tr07lz5+L+vMvlUkVFhSEwbdiwodtwEtol2F1r1Pbt2w372dnZcY2bCub1eqMKjrm5uX26TqxCx7T1R1ekH+EKAIA+8Hq92rZtm1588UW1traaNqWAy+UyPP331FNPRQxXXq83rEsw2rFgNTU1Mc1zFcnbb7\/db1190fJ6vWE\/h5tuuqnfrk+3IAAAcVq5cqVGjBih0tJS1dfXmzpXU2hA8ng8EQdpb9u2zdDC5XA4um2l2bt3r2n1+YV2FSaDN954I+y1\/my5IlwBABAjr9eroqIiLV68OGGzjefl5cnpdBpeC50kVLo4D1aw++67r9tzml1reXl5vw0Sj4V\/qR2\/\/po81I9wBQBAjB5\/\/PGwOZQcDocqKyvV3NysM2fOqKurK+wrVvPnzzfsh3bheb1ew+SckjR79uxuzxcaMqqqqiLWGe1XdXV1zN9TfwgdWxY8a35\/IFwBABCD9vZ2rVixwvBaeXm53njjDS1fvlxTpkyJ6cnAnoQGJZ\/PZ1j4edu2bYb3u1vupjuhy8Oki9DgO3r06H69PuEKAIAYhE7oabPZtGzZMtMCVTC73R4251Xwk4GhXYKRlrsJFhq8epreIVVFGpc2ceLEfq2BcAUAQAxCW3smTJiQkGDlFzoZaE1Njbxeb8QuwUizuwe7+eabDftNTU1RLa2TSt5++23DvsPhSOjPJxLCFQAAMYi3tSdSi0o0iouLZbPZDK9t27YtrEuwvLy81xAxYcKEsNfWrl0bV13J6uTJk4b9\/p5fSyJcAQAQk9DB0cePH+\/1M21tbcrPz4\/7mqFr+r344ov64Q9\/aHjttttu6\/U8OTk5YedavHhxzMGvvb1dq1evjukz\/eXAgQOG\/XHjxoUd43a7VVFRkbAaCFcAAMQgdHC0x+PRkiVLuj2+oaFB+fn5fZoG4c477zTs19fXG5ae6W65m0geeuihsNfGjBmjlStX9tpF2NDQIJfLJYfDoQULFsTdGpdIHR0dPb7vdrtVWlqqmpqaPs9O3x1maAcADCh79+5VUVFRTJ+prq4ODAYvLi6Ww+EwTBi6YsUK7d+\/X3fffbeGDBkiSXrzzTe1Zs2amNbf645\/zqvuzhXaGtWTnJwc1dXVqbS01PD64sWLtXjxYhUWFoa1zm3fvj3sCTxJfVriJ1FCB+3X19frH\/7hHyRdnELD\/6RnXV1dwmaWJ1wBAAYUn88XMSj0ZO\/evYY\/2hs3bgxrjWpqaurxvE6nUwUFBWHTOERr\/vz53Yar0Jat3vjXHgwNWFLv34efzWZLumVvpIuD9oPnA\/N4PLrqqqsMx9TV1XW7\/qIZ6BYEAKS1SGNuYtXZ2WnYz8vL044dO+RwOKL6fGVlpRobG5WZmRl3Dd1NDtrTcjc9cblcam5ujnn2cpvNpsrKSh07dizma\/YHl8sVNrO9n81mU3Nzc0KDlUS4AgCkuTvvvDPsabtYRVr0Ny8vT8ePH1ddXZ1KSkrCglZhYaGqqqrk8Xi0fPly2e12DRs2LPB+tMHMz263q7KyMuz1hQsXxnSeYFOmTFFjY6MOHjyoqqoqFRYWht0rh8MR+F6am5t19uzZwPfTk8GDBxvO1Z\/L5DQ2NhrmB3M6naqqqtKxY8f6pbUtoyue+fiTzGWXXaaPPvpI9qv+Wnt+k5oToo0YMsjqEuJ27PQHVpfQJ9x7a6TyfZdS8943btmk+xdcnDPpwQcf1OOPP25xRYhVQ0ODZs6caXjN4\/Ek5fp+AxktVwAApIh169YZ9gsLCwlWSYhwBQBACmhvbw+bkf3uu++2phj0iHAFAEAKiLSmYW\/L3cAahCsAAFLAU089Zdh3uVz9vmYeokO4AgAgyTU0NBgmLZWkOXPmWFQNekO4AgAgyYUOZHc4HEk5gScuIlwBAJDEIg1kv++++yyqBtEgXAEAkMTOnTtnmIyzsLCw29nakRxYWxAAgCSWl5ens2fPWl0GYkDLFQAAgInSquXqLz6RkfJLaqQi7rl1uPfWScV7fzD7cqtLAAYEWq4AAABMRLgCAAAwEeEKAADARIQrAAAAExGuAAAATES4AgAAMBHhCgAAwESEKwAAABMRrgAAAExEuAIAADAR4QoAAMBEhCsAAAATEa4AAABMRLgCAAAwEeEKAADARIQrAAAAExGuAAAATES4AgAAMBHhCgAAwESEKwAAABMRrgAAAExEuAIAADAR4QoAAMBEhCsAAAATEa4AAABMRLgCAAAw0SVWF4D08Ytf\/EIPPPCAJCk3N1eDBg2yuCIAwf7rv\/4rsP3CCy\/oN7\/5jYXVDGzHjx\/XBx98IEny+XwWVwOzEa5gmhdeeEEdHR2SpNbWVourAdCT06dP6\/Tp01aXAUm\/+93vdO2111pdBkxEtyAAAICJaLmCaaZNm6YtW7ZIkiq+vVizZs+xuKLYXf2pK6wuoU9OvHfe6hLilsr3PpXvu8S9t8KCe0rkOXZUkpSdnW1xNTAb4QoJYcu2a\/jVOVaXEbPcIak9Tqzr8g+sLiFuqXzvU\/m+S9x7K1x66aVWl2CpiooK1dTUdPu+0+nUvn37oj5fbm6uPB5Pj8d0dXVFfb6+olsQAAD0q+rqah08eFBOpzPi+62trWpvb4\/6fMePH1ddXZ1sNlvYe1VVVTpz5kzctcaDcAUAAPpdXl6evv\/973f7\/rZt22I6n8vlksvlMrxWXl6uRYsWyW63x1VjvAhXAADAEldeeWW3761atSrm81199dWG\/Yceeijmc5iBcAUAAJKOx+NRW1tbTJ\/p7OwMbDscDuXkWDP2l3AFAAAs8eabbwa2q6qqwsZMPf\/88zGdb\/\/+\/YHtkpKSvhXXB4QrAABgiZMnTwa2MzMzw8ZM1dfXx3S+vXv3BrZvuOGGvhXXB4QrAABgia1btwa2r7\/+et18882G92PpGmxvbzcsJTRhwgRziowD4QpJaf++Pbplyo2aN3eWKed6fOX3NW\/uLF07dLC2bN5gQoUABpL9+\/bo2qGDde3Qwdq\/b4\/V5aSN4KXSRo0apRkzZoQdE23X4NGjRwPbNpvNsvFWEuEKSWb\/vj2aN3eWSr9SoJPvvtPnc90y5UaVfqVAtU8+puadr0qSfvn8v5lRKoABosN3VosfLLO6jLTT0tIS2HY4HLLb7bLb7WFzX0XbNbhr167AtpWtVhIztCNJ7N+3R9U\/WRkIQH43TZ4W1\/keX\/l91T75mCRp2PBrNH\/BAxpx3SiNG\/\/5PtcKYGB55ukn+\/yPPYQLHsyem5sb2J4\/f76hRcvfNZiXl9fj+YIHs0+fPt3ESmNHyxUsdfStw4GWKkl6+hcbNeeu+X0657y5swLBamHlMr3ackiuufcQrADEbP++Pap98jENG36N1aWknR07dgS2g8NQvF2DTU1Nge1Jkyb1sbq+IVzBUn81aLAkqe6XW7X2uU2aNr1AX\/zynXGf75ElDwZav37005\/p3vIHTKkTwMDj7w7MzOEhDTUAACAASURBVMzS0n9+zOpy0k5w69To0aMD2zk5OTF3DQZ3MUoXx29ZiXAFSw0ddrXWPrfJlFalLZs3aP26NZKkf\/6Xn2jm7bP7fE4AA9cL7nU6+e47+s53v69BgwdbXU5a8Xq9hoWWR44caXh\/\/nxjD0ZvTw0GdzE6nc5+X+4mFOEKaaHDd1aPVD4oSRp941i55t5jcUUAUtnRtw5r1YqlmjrtFn6fJMAbb7wR2I70ZF+sXYPBwWv8+PEmVNg3hCukhV3N29XZ2SFJmlf2LYurAZDqKhd+U5K0sPKfLa4kPR05ciSwHenJvli7BoPny+pt4Ht\/IFwhLaxd\/aQkKTMzSzNvn62jbx3WI0se1FdmTgvMTXPLlBv1+Mrvq8N31uJqASSzp2t+rCOHDmhh5TKN\/Jx1s3yns+3btwe2u3uyL9quwdAuxptuusmkKuOXVlMx\/PFPXTp2+gOry4jLiCGDrC4hbv57fqbzI0uu3+E7qyOHDkiShg7P0SNLHtT6dWtU\/MU7dOtts3TrbbP0ysubdOTQAdU++Zjq1q3Rs8838EsTQBh\/d+Cw4dfo71x3WV1O2gp+si94MHuw7roGQ1umgrsYJVquAFN4jv82sH3k0AGdeveEfr3rsH5c\/XPdW\/6A7i1\/QL\/cslP\/\/C8\/kSR1dnYEmvwBIJj\/d0PVj1Yry5ZtcTXpKbT1aeLEiRGPi7ZrMLiLsbCw0IQK+45whZR37LdvB7anTrtFa5\/bpKHDrg47zjX3Ho2+caykiyHs6FuH+6tEACnA3x1Y\/MU70nJevF1HT+iVA7\/t\/cAEe\/vtj39n+2dm7040XYPBXYzjxo0zqcq+IVwh5Z07935gu7cZ3W+97eO1Cpt3\/jphNQFILf7uwMzMLD2y\/PGoPnPq5ImUWmvwqW17tepXr1ldhl577eMaQlumQkXz1ODevXsD25MnT+5jdeZIqzFXQG\/GOj9ufn7z8EELKwGQTPzdgUOH5+gF97qw939\/+t3A9ksvPq8DrW+o\/t+e6bf6+iq41eqVA7\/VrWOvs6yWffv2Bbbz8\/N7PNbfNRg84Wh9fb2WL18uSWpvb5fP5wu8FzpfllUIV0h5gwdfGdfnzr3faXIlAFKV\/6GYI4cOBLa745+s2M+Ra11QidZT2z5u3Vn1q9csDVfBQen666\/v9fie1ho8evRo4HWHwxE2X5ZVCFdIeSOu+3iZg95aoz44dy6wHe+i0ADSz+9Onevx\/f379gTWQK375daUGpMVOtbqyMn3LGu9Cl2mZsqUKb1+pqenBoMHs\/fWxdifGHOFlBf8S663f3Hub\/14bMRnPjskYTUBQLIIbrXys2rs1e7duwPb0YahSE8N+icNDR7MPnbsWBMqNAfhCknt3Lnouu7m3HXxiZKT777T4+DShs0bJV2cbHTy1MgT1wFAuujuCUF\/61V\/O3Dg438AFxQURP250KcGW1tb1d7eruPHjwdemzRpUt8LNAnhCknnvT\/8PrDdsnN7D0d+bF7Zt5WZmSVJWv7IdyPOwv7Ikgd18t13Lm6v+BFz2ACIWvCUL8G\/o5JdpFYrPytar4LHTt1wQ\/QTOUfqGty7d69hZvZouhj7C+EKljt18oT279uj\/fv26PGV3w8swCxd7OZ7oOJu7dy+NXBMJEOHXa1nn29QZmaWjhw6oBlT8\/TIkgf1dM2P9fjK7+uWKTdq\/bo1yszM0o9++jPNvH12f317ANLA67s\/DiLbGl+2sJLo9TavVX+3XrW3txvC0KhRo3o42ihS12BFRUVgO5bxVu3t7aqoqFB2drYyMjKUm5urhoaGmI7vDQPaYbnGLS9q1Yql3b7f8NJGNby0MbDf3cDTkZ+7Qdua2\/SCe51eeXmT4Yme0TeO1YJv3aG\/c30t4gSjABDJ0bcO65v3\/X2g1Vu6+Dtp12u\/TvpltHpqtfLrzycHg5\/sk2Jfpib0qcHgKRii7WJsa2tTfn6+4bMej0dz585Va2tr2NOGkY632Wy9XodwBcv5l6gxQ5Yt29TzARjYRn7uBr3acsjqMmIW7Wzs\/fnk4K5duwLb8SxTE6lr0C+aLkZ\/UHI4HKqvr1dmZqYmTpwoj8cjn8+ntWvXBubPki4uCB0arKToghzdggAApJloWq38+mvslf8JP0nKysqK+fORugb9oulibGpqks\/n05o1a5STkyO73a5HH3008H7ouoVFRUXy+XxyOp3yeDyqra2VFN0s8IQrAADSSKxrCPbH2Kv29nZDl1680yaEPjUoXeymi6aLcdGiRfJ4PIZjg1vDPB6PvF6vpIvjuVpbW+V0OtXY2KicnByVlZWpq6tLxcXFvV6LcAUAQBqJpdXKL9GtVxs2bDDsnzhxIq7zROoanDBhQtSfDx1TZbfbDa1hb7zxhtxut2pqagLBqqeFpbtDuAIAIE3E2mrll6jWK6\/Xq9WrV2vx4sWG12tqauR2uwMtRdGK1DU4fXrf5iwcP358YPuJJ55QRUWFbDZb3MFKYkA7AABpI55WK79EPDk4YsSIsAHhfqWlpbLZbDp7Nnxewp6EPjU4evToPtUY3E3Y1NQkSTp48GDcwUqi5QoAgLQQb6uVXyJar86ePauurq5uv2INVpICY5\/8X9GMgepJ6OLRtbW1MU8TEYqWKwAA0sQT826P+PrSuq3qPH8hsL+stECZV1wedtyVV1yWsNqSVejM7qFhKx6EKwAA0sDkkVd3+97Suq2GfdeUvIjhaiBavXq1YX\/37t19XkqHbkEAADAgNTQ0aMGCBYbXgheXjhfhCgAADDhtbW2aO3eubDZbYIJQybi4dLwIVwAAYEAJXjOwurpas2fPDrwXPJlovAhXAABgwPB6vZo\/f758Pp\/Ky8vlcrlkt9vlcDgCx2zbts3wmfb29pgCF+EKAACkLbfbrYyMDLndbrW1tamoqCiwtE11dXXguOAFmV97zThjfUlJiYqKiqK+JuEKAACkLX9QKi0t1ZgxY9Ta2hqYgT3YnDlzAts1NTVqa2uT1+sNrDP4ne98J+prMhUDAABIW1u3GqehsNls2rFjR9gM7FOmTJHT6QwMaB8zZkzgvbq6OrlcrqivScsVAABIS8HjpGw2m8rLy3Xs2LFuZ2Bfs2aNCgsLA\/tOp1NbtmyJKVhJtFwhQf4681KNGDLI6jIGHO65Nbjv1knVe3\/ZJ2nb6A92u13Hjx+P+vi8vLyw7sJ48NMFAAAwEeEKAADARIQrAAAAExGuAAAATES4AgAAMBHhCgAAwESEKwAAABMRrgAAAExEuAIAADAR4QoAAMBEhCsAAAATEa4AAABMRLgCAAAwEeEKAADARIQrAAAAExGuAAAATES4AgAAMBHhCgAAwESEKwAAABNdYnUBZvqLT2RoxJBBVpcx4Pjv+VWZl1pcCQAA1qPlCgAAwESEKwAAABMRrgAAAExEuAIAADAR4QoAAMBEhCsAAAATEa4AAABMRLgCAAAwEeEKAIA0d+UVlxv23z9\/waJKBgbCFQAAgIkIVwAAACYiXAEAAJiIcAUAAGAiwhUAAICJCFcAAAAmIlwBAACYiHAFAABgIsIVAACAiQhXAAAAJiJcAQAAmIhwBQAAYCLCFQAAgIkIVwAAACYiXAEAAJiIcAUAAGAiwhUAAICJCFcAAAAmIlwBAACY6BKrC0B6+q\/Oj3Ts9AdWlxGzEUMGWV1Cn6TiPfdL5Xufyvdd4t5b4cP\/+ZPVJSCBaLkCAAAwEeEKQNrbdfSEXjnwW6vLADBA0C0IIO099qvX1Hn+Q9069jqrSwEwANByBSCt7Tp6QruOvqsjJ9+j9QpAvyBcAUhrj\/3qtcD2qqBtAEgUwhWAtOVvtfKj9QpAfyBcAUhbj0VoqaL1CkCiEa4ApKXQVis\/Wq8AJBrhCkBaitRq5UfrFYBEIlwBSDvdtVr50XoFIJEIVwDSTk+tVn60XgFIFCYRBZBWemu18vO3XjGxKEIdfeuwGl7+pd48dFDNO18NvD76xrG6ccxYzSv7toYOu9q6ApH0aLkCkFaiabXyo\/UKwXZu36qvzJym2wsnqfbJxyRJCyuXaWHlMo2+cayOHDqg9evW6AuTb9CWzRssrhbJjJYrAGkj2lYrP1qvEGzdMzU6cuiAJOlHP\/2ZZt4+O\/DeveUPaMvmDXrwG1+XJD34ja\/rxjFOWrAQES1XANJGLK1WfrReIdScu+YbgpXfzNtnq\/iLdwT2G7e82J9lIYUQrgCkhVhbrfx4chChvvjlO7t97\/obxvRjJUhVdAsCSAvxtFr5rfrVa3QNQmuf29TrMb8\/\/XGAv\/a6zyWyHKQwWq4ApLx4W638aL1CNE6dPKEtv7o4kH3qtFs0bXqBxRUhWaVVy9Uf\/9SlY6c\/sLqMuIwYMsjqEuLmv+dnOj+yuBIMZE\/Muz3i6\/ev3RzVcVdecZnpNSF9HH3rsCoXflOdnR2aOu0W\/fDJtVaXhCSWVuEKwMA0eeTV3b4XGq5cU\/ISXA3SxamTJ3ToYKv2vbFL69et0bDh14Q9RQhEQrgCACDEtUMHh702ZdoXNGjQlRZUg1RDuAIAIMTCymWB7dd37VTzzle1ft0arV+3JtAtmGXLtrBCJDPCFQAAIe4tf8Cw3eE7q0eW\/IMaXtqo5p2v6p65s\/TMc5sIWIiIpwUBAOhFli1bP67+uUbfOFaSdOTQAb3gXmdxVUhWhCsAAKI0r+xbge3Xd+20sBIkM8IVAABR+tSnPxvYbt75qoWVIJkRrgAAiIO\/ixAIRbgCACAOw4ZfY3UJSFKEKwAAdHFuq2uHDtapkye6PealF58PbM8ouq0fqkIqIlwBABDk2+VfU4fvbNjrWzZv0Pp1ayRdXFuQmdrRHea5AgBAUmZmljo7O3Tk0AHNmJqnmV+arc8OGS5JeuXlTTpy6IAkqfiLd+iR5Y9bWSqSHOEKAABJ+46c0s7tW7W\/dY\/ePHQw0EolXRxfNeeu+fril+\/UuPGft7BKpALCFQAAfzZteoGmTS+wugykOMZcAQAAmIhwBQBAmsu84nLDfuf5CxZVMjAQrgAAAExEuAIAADAR4QoAAMBEhCsAAAATEa4AAABMRLgCAAAwEeEKAADARIQrAAAAExGuAAAATES4AgAAMBHhCgAAwESEKwAAABNdYnUBSE9\/nXmpRgwZZHUZAw733Brcd+uk6r2\/7JO0baQzfroAAAAmIlwBAACYiHAFAABgIsIVAACAiQhXAAAAJiJcAQAAmIhwBQAAYCLCFQAAgIkIVwAAACYiXAEAAJiIcAUAAGAiwhUAAICJCFcAAAAmIlwBAACYiHAFAABgIsIVAACAiQhXAAAAJiJcAQAAmIhwBQAAYKJLrC7ATH\/xiQyNGDLI6jIGHP89vyrzUosrAQDAerRcAUAKcbe0WV0CgF4QrgAgRXSev6D7127W9H96mpAFJDHCFQCkmCMn3yNkAUmMcAUAKYqQBSQnwhUApDhCFpBcCFcAkCYIWUByIFwBQJohZAHWSqt5rgCkj7\/5+rKUOm8y8oesp7bt1X0zJsg1Jc\/qkoABgZYrAEhztGQB\/YtwBQADxKkzHeo8f0Gd5y9YXQqQ1ugWBIA0l3nF5XroSzfLNSVPmVdcbnU5QNojXAFISv\/5s6WmnCd0jJVZ57VC5\/kLGvGNVVEfT6gCrEG4AoA0Q6gCrEW4AoA0QagCkgPhCgBSHKEKSC6EKwBIUYQqIDkRrgAgxRCqgORGuAKAFLKstIBQBSQ5whUApIjMKy5XWcFEq8sA0AtmaAcAADARLVcA0tq+ld+0ugQAAwzhCkBaG\/7XNqtLADDA0C0IAECay7ziMsM+i3cnFuEKAADARIQrAAAAExGuAAAATES4AgAAMBHhCgAAwERMxYCEONP5kY6d\/sDqMmI2Ysggq0vok\/6+53+ddVnvB0Upa9AnTTtXf+v44H+6fe+\/Oj7sx0rik8r\/3afi7xlJ+vB\/\/mR1CUggWq4AAABMRLgC0G9e37NL878+V\/O\/PtfqUmJSWlqq0tJStbS0WF0KgBRAuAKQcO+eeEezZ92mWwvy1dHRoeKZt1tdUky+9KUvyefzaerUqSoqKlJ7e7vVJQFIYoQrAAn1+p5d+tupN+nXr27Vj37yU23Y9LK+MvvObo9vaWlRaWmpsrOzlZGRoezsbJWWlqqhocGUetrb21VRUaHc3FxlZGQoIyNDRUVFWr16dbefcblcamxsVG1trZqamuR0OmnFAtAtwhWAhDly+NCfW6t8at7dqrvvubfH4ysqKjR16lS53W65XC6tXLlSEyZMkNvt1syZM1VRUdGnetxutxwOh2pqapSbm6uVK1eqvLxcTU1NWrBggcaPHy+v19vt58vKynTw4EFJ0tSpU9XW1tanegCkJ8IVgITwnfXqi8UzJEk\/+slPNfqGG3s8ftWqVaqpqZHNZtPBgwdVXV2thQsXqrGxUeXl5ZKkmpoarVq1Kq56\/C1iklRXV6fGxkYtXLhQ1dXVqqurkyS1trbq7\/\/+73s8T15enqqrqyVJ+fn5PYYxAAMT4QpAQixf9og6Onz6wi0FvbZYvXviHS1atEiS9PDDDysvL8\/wfnV1tRwOhyRp0aJFcQWau+++W5JUWFgol8tleM\/lcgUCXFNTU69dkC6XS4WFhfL5fFq6dGnMtQBIb4QrAKZ798Q7Wrvm4himhxY93Ovxz\/7iZ4HtO+64I+IxJSUlge0NGzbEVE9DQ4M8Ho+kj0NWqNtuu+3jep59ttdzfu9735N0sTXt3RPvxFQPgPRGuAJguid\/8rgk6ZprcnTT5yf3evwvNz4vSbLZbMrJyYl4zOTJH59n06ZNMdXz8ssvB7ZHjRoV8ZiJEycGtt1ud6\/nnDJlSqA1zf\/9AoBEuAKQABs3XAxLX7mj+6cC\/d498Y7eeefi1AYTJkzo9riRI0cGtvfu3RtTPVu3bg1sh3Y5+tnt9kBYkhTVYHV\/a5r\/+wUAiXAFwGSv79mljg6fJOmWgsJej\/\/DH\/49qvMGt2j5fL6YavJ3CfYmNzc3sH3u3Llej7\/11lslSR0dPr2+Z1dMNQFIX4QrAKbat\/f1wPZ1143s4ciLjr79VmD7C1\/4QtTXiXaeqeAWqMLC3sOe3549e3o9JriLMfj7Ruo7dfKE3M89owcq7tYtU27UtUMH69qhgzV+9FA9UHG3dm7f2vtJMGCxcDMAU7Ud\/E1g25Zt7\/X4c+fej\/rcTqdTra2tMdUTTQuU37hx49TU1BT18Xb7x99f8PeN1HX0rcOq\/dcfquGljZKkYcOvUfHtd2jw4Ez9\/vS7Wr9ujRpe2qiGlzZqzl3z9cjyH1lcMZIR4QqAqTo6OiRJ\/2fsONPPHRxmEiErKyvmzxQWFqqpqSnwfSO1Ne\/8dSBYFX\/xDj2y\/HFl2bID77v+\/h599c5idXZ2aP26NRo\/cbJm3j7bqnKRpOgWBGCq\/a37JEnZUbRapRP\/9430kJmZFRasJGnk525Q6V3zA\/vbGl8O\/ShAuAJgLv9g9ni8807P80UdP348sN3dlArRfr63619\/\/fUxnbsv3zeST+ld88OCld+0\/ILA9rn3O\/urJKQQugUBWGrkqM8Fttvb23s8Nvipv2i7CINDWG9PDQZf\/8orr4zq\/Egv95Y\/oHvLH7C6DKQ4Wq4AmCoryxbT8YMHxx5inE5n1MfGO04r1paxWL9vpK4Pgh6SGHxlpoWVIFmlVcvVH\/\/UpWOnP7C6jLiMGDLI6hLi5r\/nZzo\/srgSJINxzvH69atbdfZsdOv\/Bc\/g3tOTesFTL4wfPz6mmvyDzv3nmTJlSsTj\/Mc4HI6oQ5l\/ncNxzthqQura8etXAttf\/XqZhZUgWdFyBSAhfnNgf9TH3jH745ncu5u\/6s033wxsB68DGI1Zs2YFtrubvyp4PqzgdQx7E+vUEEhtR986rPXr1ki6+DThuPGft7giJCPCFQBTTfvb\/MB2tAsaF8+8PbC9fv36iMesWXPxD5rD4VBxcXHY+0uWLFFGRoYyMjLClq6ZMWNGYHv16tURz\/\/88x8vYTNv3ryo6g4eoxX8fSN9VS78piRp9I1j9chy1pREZIQrAKYaP+GmwPbvfnc0qs98ZfadgXFUNTU1amhoMLy\/evXqQAvRz3\/+84jnCJ6jKnTi0JycHFVWVkq6OKh9yZIlhvfb2tq0YsUKSdLKlSu7XTw61NGjH39\/wd830tMjSx7UkUMHlJmZpRWr\/rXbpwmBtBpzBcB6N31+srKybOro8OmNPXs0o+DWqD7X2NiooqIitba2aubMmSovL9c111yjX\/\/614GxUHV1dd2Ol+rN8uXL5fP5VFNToxUrVmj\/\/v36whe+oHfeeUc1NTWSpPLyci1cuDDqc+7adXE9wawsm2HsGNKP+7lntH7dGmVmZunZ5xs08nM3WF0SkhgtVwBM5x9D9cuNz\/dy5MfsdrsaGxu1cuVKOZ1O1dTUaNGiRTp+\/LjKy8vl8Xjkcrm6\/XzwDOmf+cxnIh5TXV2turq6wAD3RYsWye12y+Vyqbm5WdXV1VHXK0n19fWSjGPGkH62bN6gf3z42wQrRI2WKwCm+9a3\/0Fr16zWO++0a9vWV6JuvbLb7Vq4cGFMrUd++\/dfHEBfWFjYY7eey+XqMaRFq6GhITBv1re+\/Q99Ph+S05bNG\/TgN75OsEJMaLkCYLrhV1+jefMvPqK+uuanCb9ee3u7mpqaZLPZVFVVlfDrSdITTzwh6WJX4vCrr+mXa6J\/HX3rsB6pfFCSCFaICeEKQEIsWfqIsrJs+vWrW\/XzZ55O2HW8Xq9KSkpks9m0Y8cO5eXlJexafqtXrw6EuWXLliX8euh\/p06eCCzQ\/KOf\/oxghZgQrgAkhC3brrrnfylJevDb39CRw4cSer3+ClZtbW1asGCBJGnz5s1xzwCP5Pbt8q+ps7NDCyuXaebtsyMec+rkCT1d82N1+M72c3VIdoQrAAlz0+cn6\/mNv1JWlk1TJzkT0oJlt9u1b9++fmuxGjNmjGw2m7Zs2RL3k4tIbk\/X\/FhHDh3Q1Gm39LjO4NrVP9GqFUuZkgFhGNAOIKFmFNyqlxq26ZF\/rNSD3\/6GXn7pVyor\/0bUg9yTQUNDg5544gk1NTWpsLBQVVVV\/RLm0P9OnTyhVSuWSpKGDr9aT9f8OOJxbx4+qIaXNvZnaUghhCsACTf6hhu1YdPLen3PLq15qkZNrzSkVLh6+eWXZbPZ1NzcTGtVmlu7+ieBbf8yN0CsCFcA+s1Nn5+ckpNtxjr\/FVIXgQpmIFwBiNt\/dXxo2rmyBn3StHP1NzPvA6z1u1Pnej8I6AUD2gEAAExEuAIAADAR4QoAAMBEhCsAAAATEa4AAABMRLgCAAAwEeEKAADARIQrAAAAExGuAAAATES4AgAAMBHhCgAAwESEKwAAABMRrgAAAExEuAIAADDRJVYXgPR0VealGjFkkNVlDDjcc2tw362Tqvf+sk\/StpHO+OkCAACYiHAFAECau\/KKyw3775\/\/0KJKBgbCVTfOXfhQx\/7wn1aXAQAAUgzhKoJzFz7U8l826e7qf9OBd05ZXQ4AAEghhKsQ\/mDV\/LZHkvStZzbovY5Oi6sCAACpgnAV5NyFD\/XEKzsDwUqSSiaN1aeyMi2sCgAApBLC1Z\/5g1XDgTcDr5VMGqv7b51mYVUAACDVEK7+jGAFAADMQLgSwQoAAJhnwIertdv3qH73gcB+yaSx+nr+TRZWBAAAUtmADlf1uw\/omR2vB\/aLx16vr+ffpMGXX2ZhVQAAIJUN2HBVv\/uAnnhlZ2C\/eOz1uv\/WaQQrAADQJwMyXIUGq7HXDCFYAQAAUwy4cBUarEZ8+m+0Ys7tBCsAAGCKARWu6ncf0M+CxliN+PTf6Ml7ZhOsAACAaQZMuGr4zVv62Y7Xde7CxZXACVYAACARBkS4avjNW3qi4f8nWAEAgIS7xOoCzPDHP\/5RkvTf58+r+idVhvf+438y9Nr7n9RHXRf3bZd06foLp\/Xs6h\/3d5k9yr7yUqtLiNvZ9z+SJO3ft8fiSgAAsF5ahavz5\/+vfvzYo4HXL8n+lLKK7lHGpZ+8eNwHHfI0PqPffdBhSZ0AACD9pW234MfB6nJJF4NVZ+Mz+iPBCgAAJFBatFzNmzdPH374oQYPHqwxY8boD\/\/3f\/TMWx367\/\/9kyTpLy\/5hO6ZlKNPz6jq5Uzoi61bt2rjxo1WlwEAgKXSIlytWbMmsH3k5HuaVfVsIFhlXnG5Ni3+qkYP+5RV5Q0YnZ2dhCsAwICXVt2CR06+p689+YI6z1+QdDFYPTHvdoIVAADoN2kTrk6d6dDXnnxBp85cHFPlD1a3jr3O4soAAMBAkhbh6tSZDn256lmCFQAAsFzKh6vO8xdosQIAAEkjpcNV5\/kLmlX1rI6cfE8SwQoAAFgvZcNV5\/kLun\/tZkOwWlZaQLACAACWSslw5Q9Wrxz4beC1ZaUFck3Js7AqAACAFAxXkYLVE\/NuJ1gBAICkkFLhihYrAACQ7FImXEUKVg996WaVFUy0sCoAAACjlAhXnecvaGndVkOwKiuYqEVfnmZhVQAAAOFSIlwtrdsqd0tbYL+sYKKWlRZYWBEAAEBkSR+uCFYAACCVJHW4WvniTq3e+kZgn2AFAACSXdKGq9Vb39Bjv3otsF9WMFEPfelmCysCAADoXVKGq9Vb39DSuq2BfdeUPD30pZuVecXlFlYFAADQu6QLV5GC1bLSAoIVAABICUkVrkKD1eSRwwlWAAAgpfRLuPrak8+r8\/yFHo8JHWM1etin9PNv3UmwAgAAKSXh4erIyff0yoHfavo\/Pa1TZzoiHuNuadNjv3otEMBGD\/uUNi3+KsEKAACknISHq4Y\/z6p+6kyHpv\/T0zpy8j3D++6WNi2t20qwAgAAaSHh4ap+16HAduf5C5pV9WwgYO06eoJgBQAA0kpCw9WRk++FdQX6A5a7pU13P\/mCIVj94lt\/R7ACAAAp7ZJEnnzX0Xcjvt55\/oLuX7s5sD\/0+LESgQAAHXVJREFUqiz94lt\/p6FXZSWyHAAAgIRLaMtV429+2+sxQ6\/K0ouLv0qwAgAAaSFh4arz\/IVuW66CnTrToZPdPEUIAACQahLWLehuaYv62FlVz2rT4q9q8sirE1UO+tmZzo907PQHVpcRsxFDBlldQp+k4j33S+V7n8r3XeLeW+HD\/\/mT1SUggRLWcrX7t723WgWbVfWsXjnQezciAABAMktIuOo8f0G7o+gSDPW1J5+PqcULAAAg2SQkXO0++m6vy9105\/61mwlYAAAgZSVkzJV7V+zhKPOKyzV62P+nksl5unXsdQmoCgAAIPFMD1ed5y\/oyMn\/iPr4ySOHq+j\/XCfXlDwmEAUAACnP9HC1++i73S7Q7Dd55HB9\/rrhKh57nUYP+5TZJQAAAFjG9HB1OGRhZr\/MKy7XvTMmEKgAAEBaMz1cBS\/UnHnF5bp17HW6b8YEDb0qi24\/AAAsEPr3N96HzhAdU8PVkZPv6f3zFzR55HDdN2OiJo0cTqACAAADiqnhauhVWWpd9S0CFQAAGLBMnecq84rLCVYA0E8eqLhb8+bOsroMACEStrYgACA+T9f8WKtWLI3q2KnTbklwNQPT\/n17tPjBMg2\/2qG1z22yuhykmIStLQgAQKrZv2+P5s2dpdKvFOjku+9YXQ5SFC1XAJCkpk67RTdNntbjMWOdE\/upmvS2f98eVf9kpZp3vmp4vbf7D0RCuAKAJHXT5Gm6t\/wBq8tIa0ffOqxVK\/5RzTtf1dRpt+jpX2zUjl+\/ovXr1lhdGlIY4QoAMGD91aDBkqS6X27VuPGflyQNGjyYcIU+IVwBAAasocOuZsA6TMeAdgAAABMRrgAAAExEtyAAJKnfn35Xjyx5UIcOHtCRQwckSZmZWZp88xf01a+XBcYIAUgutFwBQJIZPPhKSdL6dWt06t0TuvW2WVpYuUxz7povSWp4aaNKv1KgR5Y8aGWZALpByxUAJBnX3Hv06c8MUU7utRo67GrDew88tFT3zJ2lI4cOaP26NfrskOFM14BelUy+UZOuGx7YnzxyeA9Ho68IVwCQhKZNL4j4epYtWz+p+YW+MPkGSdKqFUv1d667lGXL7s\/ykGImj7za6hIGFLoFASDFDB12tYq\/eEdgf1fzdgurARCKcAUAKej6G8YEtv\/996ctrARAKMIVAABIuJaWFuXm5qqoqMjqUhIu7nDV3t6uoqIiNTQ0mFlPGK\/X2y\/XQf86d+FDHfvDf1pdBgAgwVpaWlRUVKSpU6fK4\/FYXU6\/iDtcPfbYY2pqatLcuXPV1tZmZk0B\/mDV1NSk+++\/PyHXQP87d+FDLf9lk+6u\/jcdeOeU1eUASeXpmh\/r2qGDe51m4c3DBwPb1173uUSXBcQsOFQ1NTUFXp8+fbqFVfWPuMPVsmXLZLPZ5PP5NGbMGLndbjPrUltbm4qKitTa2ipJ2rhxo6nnhzX8war57Yv\/evnWMxv0XkenxVUByWfLrzaow3c24nsdvrNqeOni78Rhw6\/p9slC9N25c\/x+ipX\/7\/fUqVMlSVu2bFF5ebnFVfWvuMOV3W7X5s2bA\/ulpaUqKipSS0tLnwpqb29XRUWFxowZEwhWdXV1ysvL69N5Yb1zFz7UE6\/sDAQrSSqZNFafysq0sCogOXV2dug735oXFrA6fGd1z9xZgf2l\/\/xYf5eW9t77w+8D2y07eRIzVoMHD5YkNTc3q7GxUcXFxZozZ47FVfWvPs1zNWXKFNXV1amiokI+n09NTU1qamqS0+lUQUGBJk+erCuvvFKjRo2S3W4P+7zX69Xbb7+t999\/X7t27dLWrVsDgUqSbDabNm\/erClTpvSlTCQBf7BqOPBm4LWSSWN1\/63TLKwKSD6f+eyQwHbzzlc1+4t\/q+Lb79DgwZn6\/el3teVXG9TZ2aHMzCw9suJHtFqZ4NTJE\/rP\/\/iDJGnnjq2qW7cm8N6RQwf0QMXdmjV7jgb9OTSw7FDPcnJy1NjYaHUZlurzJKIul0ujRo3S4sWLA32qra2thpAUj\/Lyci1btixiKEPqIVgB0Zl5+2zdOMapXa9t1+u7X9ORQwdU++THrVNTp92iGUVfVNHMLzNxqEkat7yoVSuWdvt+w0sbA92wkvS7U+f6oyykMFNmaM\/Ly1NjY6NaWlr0r\/\/6r6qvr4\/rPDabTS6XSw899JBycnLMKA1JgGAFxGbosKvlmnuPXHPvsbqUAeHe8gdYQgimMnX5mylTpmjKlCn66U9\/qm3btunw4cPav3+\/jh8\/HvHxy8LCQmVlZWns2LGaNGkS3X9paO32ParffSCwXzJprL6ef5OFFQEAkFgJWVvQbrfL5XLJ5XIl4vRIEfW7D+iZHa8H9ovHXq+v59+kwZdfZmFVAAAkFgs3IyEOd\/6vdr+yM7BfPPZ63X\/rNIIVACDtRTUVg9vtVlFRkTIyMpSRkaHx48f3OHFoW1ubKioqlJubG\/hMNLOsu91uuVwuZWdnBz7ncrn6PL0D+tdffu7z2n32fwP7Y68ZQrACAAwYPYYr\/wzppaWlhtlVW1tblZ+fr\/b29rDjlyxZojFjxqimpsYwzqqpqUkzZ87UkiVLwq7jX0qntLRU9fX18vl8gffq6+s1depUlr9JEX\/5uc9r0IRbA\/sjPv03WjHndoIVAGDA6DZctbW1aeLEiYZQFczn82nDhg2BfX8QW7FiRY8XXLFihSGUtbW1yel0dnsdP5a\/SX6t3v\/VX43JD+yP+PTf6Ml7ZhOsAAADSsRw1dbWpvz8fHk8HjmdTh08eFBdXV0qKSkxHLd9+8WZa\/3BqrW1VYWFhWpublZXV5c8Hk\/EKe\/9ocx\/HZ\/Pp5KSksB1\/NcN5vF46B5MYu6WNu0580dlXHq5JMl+6ScIVgCAASliuLrjjjvk8\/lUXl6uxsbGwNIz3\/zmNyOexB+s\/Mf7p1TIyckJrEEYbPv27Wpvb\/9\/7d1hbBTnncfx357aO4SUkLEtXU5tyGmXFkc4gjNrkwPDyZZgHdRrE7WNl0ptkNKG2hK8qDg7gqCqgpCYIlVqJK9RiXS5NPU6TVUaOcZ2JaziBQW8cOwBwpG6K+qkFZG8OG5OiKvS+l7Qne7Mrndmzdq7Y38\/UqR5ZufZffCrX57nmf9jBqu+vj5Fo1Hzd\/x+v06ePJnzO9evX8+5h\/KLxhI61Deiu3+elSR9evuW\/v2fPkuwAgAsS3nD1S9+8QuFQiH19PQUrJDe0tKijo4OM1j19PTkPFNdXa3Gxsac+21tbWawyleyId9ZgjMzHKBZac5N3NShvhHN3Lkr6V6w+uOZn+kf\/s5X5pEBAFAeeUsxZCqu29lnjm7evKlIJGIGMbcy+6sOHDhQVC2szZs3u34WC+\/a5C3tfvXnZrB68LM+3TzzM\/35fz8u88gAAJVqOUyUuCrFkDE5OWlpRyIRGYahN998s+gfDgaDeumll4rqkzlpG+V3bfKWnu5+wwxWj9Q8pF3\/\/PcEKwBAjg8\/\/NC8HhkZKeNIFkdRRUQvXbqUc++nP\/2p4+HKFy9ezLmXb09VNvvmdcMw8i4VYvHZg9WqlSv0+t6v63T\/G2UeGQCgEqRSKf3hD3+QJJ0+fVqRSMT8LB6PKxwO61vf+pYefPBBSVpyx98VFa7s5RJCoZB27txZsE8qlbLUrZKk9vZ2x6BkX4LcsWNHESPFQskXrH7Z9U3VrX5Yp7Oe+8tf\/qI\/f\/pp\/i+pYJ96cMzZvPg3z\/Dy397Lf3eJv305zM7OlnsIC+rtt99WV1fXnJ\/39\/erv7\/fbC+1v4frcJWvDIKb2lMTExM59\/bv3+\/Yb3R01NJubm6e40kslmuTt\/Rs1h6rVStX6MfPfVl1qx+WJP3mN3877uboD17Q0R+8UJZxAoCX3L59WzU1NeUeRkl1dnaqs7Oz3MMoG9d7ruwzSYZhOM5aSdK5c+cs7WAwKL\/f79jPvib7xBNPuBglFsoHUx\/r2Vd\/rg+m7u2pygSrJ+vXlnlkAABUFtczV\/aZJLdv+dn3aX3729927JNIJCxLiey3Kq8Ppj7WU91vOAar2tpac+m4urpaK1euXPSxAoAXfPTRR\/rTn\/4kSfriF79Y5tGg1FyHq3g8bmlv27bNVT\/7Pi03M1Dvvfeepc1+q\/KZuXPX9YzV8ePHdfz48cUeIgAAFcXVsmAqlbIcwiwpb2FQu3z7tNzMQNlnyerr6x37oPRm7tzV091v6NrkLUksBQIA4IarcGUvpWAYhqt9U\/Z9WqFQyNWg7PutKB66+Gbu3NW+196xBKvDu3YQrAAAcOAqXF29etXSdjNrJeXOQG3cuNGxj32\/lbT06l9UukywOn35ffPe4V07FG5i3xsAAE5chSv7TFJLS4urL7fv03r88ccd+9j3W7md7UJp5AtWP37uywQrAABccgxX6XQ6JyS5WabLt0\/rsccec+yXSCQsbbdBDvePGSsAAO6fY7i6ceNGzj03ISlf8VA3m9lTqZSlvWrVqpxnOjo6FI1GHb8L7uULVvu\/sk17dmwq46gAAPAex3B1\/vx5SzsYDDqeJSjlFg91u7xn3zxvD1cdHR2KRCLq6OhQOp129Z0obObOXR3qG7EEqz07NqnzqX8r46gAAPAmx3B15swZS7uhocHVF9uLh7pd3quqqrK0T506JenejFZra6sikYgMw9Do6KirkAdnh\/pGFI39bTl2z45NOryL2mIAAMyHY7iyzyS5rZRuLx5aV1fnqp+9YGh\/f798Pp8CgYCGh4fNYEXF9tIgWAEAUFoFw1W+sghuKqznKx5aW1vrakB79uyRYRh5PwsGg4rH4wSrEjl26jc6MXLBbBOsAAC4fwXDlb0sguRu5irfIc9uio5mvv+dd95RMBg074VCIfX19Wl8fNz196CwEyMXdPxXZ832nh2btP8r7o40AgAAc\/PNzs7OlnsQWFwnRi7oUN\/fapeFm9br8K4dWrVyRRlHBQDA0uCqiCiWDoIVAAALi3C1jNiD1ZbaRwlWAACUGOFqCXj21bc0c+duwWfse6zqVj+s\/9z7DMEKAIASI1x53LXJWzp9+X21fP8n+mDq47zPRGMJHf\/VWTOA1a1+WL\/s+ibBCgCABUC48rjBv1ZV\/2DqY7V8\/ye6NnnL8nk0ltChvhGCFQAAi4Rw5XH95\/7HvJ65c1dPd79hBqxzEzcJVgAALDLClYddm7yVsxSYCVjRWEK7X\/25JVi9vvfrBCsAABbYZ8o9AMzfuYnf5b0\/c+eu9r32jtl+pOYhvb7363qk5qHFGhoAAMsWM1ceNvTf7zs+80jNQzrV9U2CFQAAi4Rw5VEzd+7OOXOV7YOpjzU5x1uEAACg9AhXHhWNJVw\/+3T3Gzo3cXPhBgMAAEyEK486\/77zrFW2p7vf0OnLzsuIS8m5iZs5pSkAAFhohCsPmrlzV+ddLAnaPfvqW0XNeHnVuYmberr7v\/5aluKjcg8HALDM8LagB52f+J3jcTdzybxFGG5aX8ohVYRzEzd1\/FdnXe1FAwBgoRCuPCh6rvjZp1UrV6hu9T+qbct6PVm\/dgFGVT6EKgBAJSFceczMnbtFLXVtqX1Urf+yVuGm9UuugCihCgBQiQhXHnN+4ndzHtCcsaX2Uf3r2ke1s36t6lY\/vEgjWzyEKgBAJSNceczVOd5+W7Vyhb6zvXHJBiqJUAUA8Abf7OzsbLkHAfc2\/ser5szVqpUr9GT9Wj2\/vVGP1Dy05Jb9MghVAMqpbvXDOvOD75R7GPAQZq485NrkLf3xzl1tqX1Uz2\/fpM21jy7ZQCURqgAA3kS48pBHah5S\/Id7l3SgyvbHO\/+nyamZcg8DAICiUETUQ1atXLFsgpUkPVm\/Vpd+uFev732Gg6cBAJ7Bnit4xunL7+vFvhHHtyWz\/fi5Ly\/JgqkAgMrFzBU8g5ksAIAXEK7gOYQsAEAlI1zBswhZAIBKRLiC5xGyAACVhHCFJYOQBQCoBISrAtLptI4dO6aGhgb5fD75fD6Fw2Gl0+k5+wwODiocDquqqko+n09VVVUKh8NKJBIFf+fEiRNqbW01f6eqqkodHR1KpVIL8U9b0rJD1nIqXQEAqAyUYphDIpHQV7\/6VSWTyZzPgsGgxsfHLfdSqZQ6Ojo0PDw853e+++672rlzp+VeLBbT7t278\/6OJBmGoXg8Lr\/fP49\/BQAAWGzMXOURjUbV3Nw8Z+CJx+OKxWJmO5FIKBgMFgxWkrRv376c39m6deucvyNJ09PTOn78eBGjBwAA5US4solGo9q1a5emp6fV3t6uZDKpqakpBQIBy3Pnz5+XdC9YNTc3W56fnZ3VlStXFAwGLX2SyaQZyjK\/I0nd3d1mv7GxMRmGYekXiUQW6p8LAABKjHCVJZVKmYGnt7dXPT098vv9qq6u1vPPP5\/3+Uyw6uvrM5+XpPXr1+tHP\/pRTp\/z589rcHBQu3btkmEYunLlijo7O81+TU1Nevnll3P6FdqzBQAAKgcHN2fx+\/3q6+vT1atXtWfPnoLP1tXVqa2tzQxW4XA455mmpqacezdv3tQrr7wiwzA0Ojqq9etzj2ZZt25dzr1PPvmkiH8JAAAoF8KVTTgczhuULl++bGkPDAwoHo+ru7s77\/NzySzxvfvuu3mD1VzyBTUAAFB5WBZ0yb7pPBKJKBQKqbOzs+jvOnDgQM5bg4XY92ABAIDKRbhyKR6PW9qGYainp6dgn3z7pAKBgL73ve8V7JfZLJ+xY8cOl6MEAADlRrhyIbvsQsYLL7zgWHvqxo0bOfeOHDmi6urqgv3sS5DNzc0uRgkAACoB4coF+0ySJD333HOO\/a5evWppBwIBV\/uzRkZGLO0nnnjCsQ8AAKgMhCsX7DNJbW1tjrNPUm5Iamtrc+yTSCQ0PT1ttg3DKGrjOwAAKC\/ClQv2kPTUU0+56mffp\/XMM8849nnvvfcsbfZbAQDgLYQrB6lUyjKTJEmNjY2O\/ez7tNzOQI2Ojlra7LcCAMBbCFcOLl68aGkHAgFXhyjb92m5CWRS7ixZvoKiAACgchGuHJw9e9bStp8XOBf7Pq2WlhbHPvb9VhLFQwEA8BrClYPx8XFLu76+3lU\/+wxUXV2dYx\/7fqtQKOTqtwAAQOUgXBWQTqdzNqVv3rzZsV++fVq1tbWO\/ez7rdzMdgEAgMpCuCrgwoULOffcLNPZ92kZhuFqn9Z8ghwAAKgshKsCrl27Zmm73W9lLx7qdjO7\/fzCBx54wNJOp9NqbW3Ne6wOAACoDISrAs6cOWNpu605Zd9v5XYzu112uMoEq+HhYXV1dbkax3KQSqXk8\/kK\/jc4OOj6+6LRqOP3HTt2bAH\/RQAAryNcFTA8PGxpP\/744676zWd5zz5LJUm\/\/vWvJd2rmbVp0ybF43EFg0G9+eabrsaxHPj9fk1NTam3t3fOZwYGBlx\/XzgcVjKZzFtNPxgMamxsTJ2dnfMaKwBgefDNzs7OlnsQlSiRSGjDhg2We1euXHEsBBqLxbR161bLvampKVfH5axZsyZnaTBbMBjU0NCQq+9ajhoaGnKCrXRvz9vt27eL+q5UKqVAIGC5l0wmXe2dA+B9sVhMu3fv1po1azQ0NFTu4cBjmLmag70sgtsK6\/bioYFAwHUYOnLkyJyftbe3E6wczPW3mZ6eVjQaLeq77CGqvb2dYAUsA7FYTK2trdq6dWvB\/9kFCiFczcFeFsHtpnR78VC3m+Cle0tSvb295oyJYRhqa2vT2NiYenp6CFb34dSpU0U9n06nLe1t27aVcjgAKkx2qMreEkJJHMwHy4JYMqqqqsz6Yr29vfrud79r+dzt8qyUu7xbTF8A3pFIJNTV1aXh4WGFQiHt27dPAwMDikQikqTu7m72WaJozFxhSUin02awMgxD27dvz3km84KAG9evXzevi1naBeAtmZeJxsbGNDQ0pJ07d+ob3\/hGmUcFryNcYUnILvja2Ngov9+fsyG9mKXB7NIYbktwAPAev9+voaEhznFFSRGusCRkF3zduHGjJOWUU+jv78\/ZSzWX7DMl3bzIAABABuEKS0J2wddMPbItW7bkPOdmadB+puS6detKMEIAwHJBuMKSkH2e4+c\/\/3lJ0s6dO2UYhuU5N0uDN27csLRZLgAAFINwBc9LJBLmZnbJGobC4bDlWTdLg9m1ykKhUIlGCQBYLghX8LzsmSZ7XbEvfelLOc87LQ1mLzFm9m8BAOAW4Qqed\/XqVfO6oaHB8tl8lgazlxjz7dsCAKAQwhU8b2RkxLzO92ZfMUuDqVTKssRYW1tbolECAJYLwhU8z+nNvmKWBrNnrQzD4DxBAEDRCFfwtFgsZmnne7OvmKXB7CVGiocCAOaDcAVPy36zr9Ah2W6XBrOXGOvr60swQgDAckO4gqddvnzZvC400+R2aTB7iXHz5s33OToAwHJEuIKnZYehTGX2fNwsDdqXGB977LESjBCAl83MzJR7CPAgwhU8K5VKKZlMmm2nMOS0NHj9+nXzOhgMqrq6ukQjBeAlH374oXmdvVUAcItwBc+amJgwrw3DcDxg2WlpMJFImNf2elkAlq5UKqVYLKZYLKaDBw+qo6PD\/CwejyscDmtwcNB8BnDymXIPAJiva9eumdeNjY2Oz2eWBrPrWJ06dcqc0RofHzfvb9u2rYQjBVDJ3n77bXV1dc35eX9\/v\/r7+8327OzsYgwLHsbMFTwr+5ialpYWV30KLQ1m799ivxWwfHR2dmp2dtb1f4ATwhU8a3h42Lyuq6tz1Sff0uCFCxdypvqdlhgBAJgL4QqelL0\/SpI2bdrkql++twYHBgYsG1hDodD9DxAAsGwRruBJN27cMK8DgUBRb\/bZlwZHRkY0OTlptt0uMQIAkA\/hCp509uxZ87pQZfZ87EuDyWRSr7zyitl2s8SYTqfV0dGhNWvWyOfzqaqqSgcPHizYJxqNyufz5fw3ODhY1PgBAJWNcAVPyn6zr9hjavItDWa\/Qei0xJhIJPSFL3xBkUjErLM1PT2to0ePFgxY4XBYY2NjOb+d\/dYjAMD7CFfwnHQ6fd\/H1NiXBjOclhgTiYSam5sVCASUTCY1NTWlQCBgfn706FGlUqk5+zc1NeWUjfja175W5OgBAJWMcAXPuXDhgqX9wAMPFP0d+d4alJyXGN966y1NT0\/r5MmT8vv9qq6u1pEjRyzP5DuzcC7t7e3y+\/2unwcAVD7CFTxnYGDA0p5P2YR8S4OS1NzcXLDfSy+9pCtXrlh+c\/v27ZZnRkdHC37HxYsXzes9e\/a4GS4AwEMIV\/CUdDqtaDRquVdoGa6QfEuD69atc+xnD3PV1dWW8g3ZS5Z2iUTC3N8VCoWopwUASxDhCp6RSCTU2tpq2XwuSR0dHTl1r9zItzTY1NQ0r7Fll29IJpOWA6GznThxwrx+8cUX5\/VbAIDKRriCJ5w4cUIbNmzIOys0PDysDRs2WIKLG\/alwWJLOmSzl2\/IrsOVkT3rFgqF5h3kAACVzTfLQUnAfUun06qpqTHbvb29Ofupjh07Zh4Om0wm2cgOAEsUM1dACVRXV1tmwbIrvkv3wlemUClvCALA0ka4Akoku37VpUuXLJ+99tprmp6elmEYOnz48GIPDQCwiAhXQIlkb2r\/7W9\/a15nz1q9\/PLLRZ2DCADwHsIVUCLZm9ozx+JI0qFDhzQ9Pa1gMEhdKwBYBj5T7gEAS8XnPvc5SzuRSOiTTz5RJBKRJJ08ebIcwwIALDJmroASsRcE\/f3vf6\/du3dLkrq7uykYCgDLBKUYgBJqaGgwa3EZhmEuB46Pj5d5ZACAxcLMFVBCgUDAvM68Hdjf31\/GEQEAFhvhCiih+vp6S7unp4eaVgCwzBCugBJavXq1eX3gwIG8h0MDAJY29lwBJZJIJNTc3Kzp6Wm1tbWZ5wgCAJYXwhVQAtnBKhgMamhoiGKhALBMsSwIuBSLxeTz+XTw4EHLfYIVACAb4Qpw6fr165Kk\/v5+pdNppVIpHTt2TBs2bCBYAQBMVGgHXEokEpLuHW1TU1Nj+YxgBQDIYOYKcGmuQqAHDhzQ+Pg4wQoAIImZK8C17PAUDAbV0NCg\/fv3U8cKAGDB24IAAAAlxLIgAABACRGuAAAASohwBQAAUEL\/D8v81dhkgPlhAAAAAElFTkSuQmCC","width":235}
%---
%[text:image:3778]
%   data: {"align":"baseline","height":182,"src":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAcUAAAGNCAYAAABgymqFAAAA4WlDQ1BzUkdCAAAYlWNgYDzNAARMDgwMuXklRUHuTgoRkVEKDEggMbm4gAE3YGRg+HYNRDIwXNYNLGHlx6MWG+AsAloIpD8AsUg6mM3IAmInQdgSIHZ5SUEJkK0DYicXFIHYQBcz8BSFBDkD2T5AtkI6EjsJiZ2SWpwMZOcA2fEIv+XPZ2Cw+MLAwDwRIZY0jYFhezsDg8QdhJjKQgYG\/lYGhm2XEWKf\/cH+ZRQ7VJJaUQIS8dN3ZChILEoESzODAjQtjYHh03IGBt5IBgbhCwwMXNEQd4ABazEwoEkMJ0IAAHLYNoSjH0ezAAAACXBIWXMAAA7DAAAOwwHHb6hkAAAgAElEQVR4nO3de1xUdd4H8M+spUQSDLCmraAyKmgqiqgloGHJzdIsV4ayzSJDptUu3vLWzUtBuraaApnWlsbg6lq2IpfSVYbyMhFjlpgOKlj5uAwD2RJmOs8fwxwZ7peZOXP5vF8vXs9h5sw5X+bZVx9\/5\/x+3yMxGAwGEBEREf4gdgFERET2gqFIRERUh6FIRERUh6FIRERUh6FIRERUh6FIRERUh6Foh5RKJeRyOfr37w+JRCL8eHt7IyYmBkuXLoVKpRK7TCIip8NQtCM6nQ4xMTFISEiAVqvFrl27YDAYYDAYUFxcDJlMhtzcXKxevRr79u0Tu1wiIqfDULQjf\/vb35CbmwupVIqcnBwEBwcL7wUHByMnJwdSqRQAEBYWJlaZREROi6FoR9LS0gAAo0ePho+PT6P3fXx8kJycDAAYM2aMTWsjInIFErZ5sx8SiQQAIJPJcObMGZGrISJyPRwp2hHTpVGtVguFQiFyNUREroehaEdef\/11YTstLQ0xMTEoLS0VsSIiItfCULQjSUlJWLJkifB7bm4uQkNDkZqaKmJVRESug\/cU7ZBKpcLMmTOh1WqF16Kjo7F9+\/YmJ+AQEZFlcKRoh8LDw3HkyJFGo8ZHH31UxKqIiJwfR4p2TqlUIiEhQfi9uLjYbP0iERFZDkeKdk4ulwtrEwHjiLGh1NRUeHt7QyKRYOnSpU0ep2G7OCIiaoyh6ADGjRvX7HsKhQKLFi2CXq8HAKxevRpKpbLRflqtVljyIZPJrFMoEZGDYyiKTKPRQCKRICMjo9l9qqurhe0hQ4YI29nZ2VAqlSguLkZxcbHw+vvvv9\/oGAEBAcIIcd68eRaonIjI+TAURXb48GEAwO7du5vdx\/ReaGgo4uLihNeDgoKwbds2BAcHIzg4GPHx8QCavsSq0+mg1Wohk8kgl8st+ScQETkNhqLIDhw4AMAYZHK5HBqNRnhPo9FALpcL6xWzsrLMPhsQEGAWkg8++KDZZ+vLz88HAKxcudLifwMRkbPg7FORaTQa5ObmYv\/+\/dDpdFCr1cJ7UqkUo0ePxsyZM9s0utNoNBg+fDgAYO\/evWaB2b9\/fwBgT1UiohbcJHYBrs506XPhwoUWOZbJiRMnhFBUKpXQarXIzMzs9DmIiJwZL586mejoaABAUVGR8NqyZcsQHR3Ne4lERK1gKDqZkSNHAgCqqqoAABkZGdBqtUhJSRGzLCIih8BQdDJDhw4FABw9ehSlpaVYvHgxlixZwi44RERtwIk2TkalUiEiIgKAcQkHAOTk5LCROBFRGzAUnZBEIhG22SuViKjtePnUCZlGiOnp6QxEIqJ2YCg6mezsbKjVaiQnJyMpKUnscoiIHAovnzoRjUaDyMhIREVFNdkUnIiIWsaRogPz9vZGamoqdDodMjIyMHz4cMhkMmzcuFHs0oiIHBJHig6qtLS00SOgkpOTsWLFCs40JSLqILZ5c1A\/\/vijsB0dHY25c+ea9TolIqL240iRiIioDu8pEhER1WEoEhER1WEoEhER1WEoiqyw5Bympnwgdhkor6jC3C17UF5RJXYpRESi4exTkRSWnMOaTw6hsOS82KUIlCoNlCoN5OHBWDBlHPx8vcQuiYjIphiKNmaPYdgQw5GIXBVD0UYcIQwbYjgSkathKFqZI4ZhQwxHInIVXLxvJc4Qhs1hOBKRs+LsUyv5ueYKyiqqxS7DKsorqlBdUyt2GUREFseRopXtKzqFZZl5rS51uPTechtV1LTyiiqMXLChxX3CgvpgRUIUhvj3tFFVRES2xXuKVhYbEojYkMA2h6M9YhgSkatgKNqII4Yjw5CIXA1D0cYcIRwZhkTkqhiKImkYjvaAYUhEro4TbYiIiOpwSQYREVEdhiIREVEdhiIREVEdhiIREVEdhiIREVEdhiIREVEdhiIREVEdhiIREVEdhiIREVEdhiIREVEdhiIREVEdhiIREVEdhiIREVEdPjqqzqZNmzBs2DCxyyCiOsePH0fXrl3x1FNPiV0KuRA+OgpAYGAgvv\/+e7HLIKIm8D9RZEscKQLo0aMHvv\/+e\/TtJ4OPbw\/R6rilWxfRzm3y65VrYpfA7wH8DgwGA4rUhyGRSESrgVwTQxHAwIEDoVKpkPTMPDwc\/5hodQzo3V20c5ucvvCL2CXwewC\/g2u\/\/45B\/aTo0kX8fxyQa+FEGyIiojoMRSIiojoMRSIiojoMRSIiojoMRSIiojoMRSIiojoMRSIiojoMRSIiojoMRSIiojoMRSIiojoMRSIiojoMRSIiojpOE4qFJedwouyi2GUQEZEDc5pQXPPJIczd8qnYZRARkQNzilAsLDmHwpLzOFF2kaNFIiLqMKcIxTWfHBK2OVokIleTkZEBiUTS6Ifaz+FD0TRKNOFokYhcTVJSErRaLaKjo4XXQkNDRazIcTl8KNYfJZpwtEhEriYgIAATJkwQfo+KihKxGsfl0KHYcJRowtEiEbmi6upqYTssLEzEShyXQ4diU6NEE44WicjV5OXlCdtBQUEiVuK4HDYUmxslmnC0SESuRq1WAwBkMhkCAgJErsYxOWwotjRKNOFokYhchUqlErY5yabjHDIUWxslmnC0SESu4osvvhC2IyMjRazEsTlkKLZllGjC0SIROQulUomYmBiztYijRo2CQqHAO++8I+x31113iVilY7tJ7AI6YveivzR6rccTKwAAl95bbutyiIisSqPR4OGHH4ZWq0V8fDyKi4sRHBwMnU6Hv\/3tb1i9erXZ\/sHBwSJV6vgccqRIROQqNBoNIiMjodVqkZ6eDqVSKYSej48PVq1aBZlMJuxffwE\/tR9DkYjITpWWliIyMhJ6vR7x8fFISkpqcr\/+\/fsL2\/UX8FP7dSoUVSoVFAoFRo0a1ajnXkxMDBQKBZRKpaVqJSJyKQqFAnq9HgAaXSJtztixY61ZktPr8D3FpUuXYvXq1ZBKpdi2bRvi4uIAGP9ls2bNGqSlpQEAjh07BrlcbplqiYhchEqlQm5uLgAgOTm5xXWHR48eFbYHDRpk9dqcWYdGitnZ2cK\/Wvbs2SMEImDsv7dp0ybhujb77xERtd\/bb78tbDd32RQwDkRMo8nQ0FD4+PhYvTZn1qFQ\/OCDD4Tt8PDwJveZO3cuAPbfIyJqL51Oh6ysLACAVCptcTZpSUmJsM1BSOd16PJpVVWVsF1aWtrksD4uLg4Gg6HjlYmgh7QbBvTuLnYZonL1v9+E34O438Hvv\/8u2rntwZEjR4Tt0aNHt7jvv\/\/9b2F76NChVqvJVXRopOjl5SVsx8fHQ6fTWawgIiJXd+LECWG7tdmk9Sczthag1LoOheLixYuFbbVajTFjxpj13SMioo47d+5cm\/bLyMgQ7idKpVI2AbeADoVicHAwMjMzIZVKAQBarRYRERFQKBQcNRIRdVJpaWmr++h0OrMBSlvuJ6ampsLb2xsSiQRLly5tcp\/6S+u8vb3bXrST6PA6RblcDrVabdY9IS0tDWPGjIFGo7FIcURErqgtI75HH33U7HJpSEhIi\/srFAosWrTIbN1jU+vItVqtMOCp3ynHVXRq8X5AQABycnKQnp5uNmqMjIzkiJGIqIP69u0rbBcVFTV6X6FQ4MyZM8Is\/4aUSiUUCoXwe3Z2NpRKJYqLi1FcXCy8\/v777zf6bEBAgDBCnDdvXgf\/AsdlkTZvSUlJOHDggBCMer0eO3futMShiYhcTv0rcFlZWcKIrrS0FHK5HEqlErt27cJtt90m7FddXQ3AeJ8xISHB7HhBQUHYtm0bgoODERwcjPj4eAAQmgPUp9PpoNVqIZPJXLLxisV6nwYHB2PTpk3C77t3725yP6VSif79+wut4BrKzs4W3q\/\/Lx0iIlcRHByM5ORk4feEhARIJBLIZDKo1WocOHAAwcHB8PDwEPZZvXo1JBIJZs+ejZSUFLP\/HgcEBJg1WXnwwQeF7Ya3u\/Lz8wEAK1eutPjf5Qgs2hC8tenAKpUKCQkJ0Gq1AIz\/Sql\/TTs7OxuTJk0S3ufjT4jIVW3atAkpKSlm9\/dSUlJw5MgR4b+NwcHBZrev4uPjUVBQgIULF7Z47Pqt4H744Qez95YtW+ayo0SgnaHo7e3d4hd1+fJlYXvkyJFm7+l0OsycORNLliyBwWAQLg+YrmlrNBrMmDEDycnJQijyQZlE5MoWLlyIyspKGAwGnDlzBgsXLmzUxi0pKUnYR6lUNttlrL76A476ayKVSiW0Wq3LjhKBdoSiRqOBXq9HXl5es5NoTNenpVIpEhMTzd7z8fHBypUrsWrVKgA32sAdPXoUOp0OTz31FORyOTZt2oSAgAAYDAaOFImIrMQ0MKk\/kWfZsmWIjo522VEi0I5QPHz4MADjJJqYmBhkZ2cL75WWliI1NRWLFi2CVCrFnj17mpxSXP+LHjNmjHC8Z555BgDMroETEZH1mK7mmdp2ZmRkQKvVIiUlRcyyRNfm3qcTJ05Eeno6Dhw4gKqqKkyaNMns\/ejoaKSkpCAxMbFNXdp9fHwQGhoKtVqNvLw8nD59uv3VExFRh5j6pB49ehSlpaVYvHgxlixZ4vJX6NocigEBAUhKSmrxESbtNWrUKKjVakRFRfFxJ0RENtS7d28Axqt18fHxkMlkeOGFF0SuSnwdfshwZ2k0GmHmqVqtFqsMIiKXVH9CjlqtRnFxMQcnsPCSjLbS6XSIjIwU7jFqtdo29fojIiLLCQ0NBQCkp6e7\/GVTE5uHok6nQ0xMDEaPHo1NmzYJvfWOHj0q7KNUKtkmjojIirKzs6FWq5GcnGzR22KOziahaOpiYwpErVaL7du3A7jR2X3t2rXQ6XRCzz5TyyIiIrIs07rw+Ph4zvpvwCah+M0330Cr1cLX1xdarRYHDhwQrl3ff\/\/9AIzXtH19fZGQkCCsVSQios7z9vZGamoqdDodMjIyMHz4cMhkMmzcuFHs0uyOTULR1KEmNDRU6NlnEhcXJ\/T4k0qlyMzMdOmFo0REllRaWgq9Xo9FixbB19cXs2fPRnJyMnJycjixpgk2mX2qVCqbfG6XyaZNmziEJyKygh9\/\/FHYjo6Oxty5c82ag5M50ZZkEBGR9YWHh8NgMIhdhsMQZUkGERGRPWIoEhER1WEoEhER1XHIe4rlFVXtes\/P18ua5RARkZNwyFB885NDUKo0Tb43csEGs99jQwLxjznTbVEWERE5OIe8fLpgyrg27\/v0xNFWrISIiJyJQ4ain68X5OGtN6+NDQlEWFBf6xdEREROwSFDEWjbaJGjRCIiag+HDcXWRoscJRIRUXs5bCgCLY8WOUokIkdVXVOL1I8PorqmtsPHKCw5Z7mCXIhDh2Jzo0WOEonIEZnCMHTBBqz55BB+7kQoLs\/Mx4SXN2Nf0SkLVuj8HHJJRn0LpoxrtDyDo0QiciTVNbVYU7fUrDOjw4bHLK+owuMbdiAsqA\/WJ07mmu02cPhQNI0WTcHYmVHiJf0VnL7wiwWra58BvbuLdm4TMf9+E34P\/A6u\/f67aOe2pfKKKryTf7TZMLzN3c0i5yksOY+RCzYgNiQQKxOiGI4tcPhQBMxHixwlEpG9Kyw5h5yvv7foyLAt9hWdwr6iU5CHB2PBlHEMxyY4RSiaRovVNbW8l0hEdqu9YThzw45Go0XPFn73uKWbsN3S\/UilSgOlSoOkqDF4euJohmM9ThGKgHG0WNZCT1QiIrH9XHMF2UWn2jw6LCw5b9V6MvKOICPvCOZPGYeFD4636rkchdOEop+vF\/+1Q0R2LTYkELEhgVCqNHjzk0MtPtwAAHYvegw\/11wxe61hoNb\/vf725vyjrYavp7sb5k8Z16YOYa7CaULRWvx7uFv8mGWXaix+TCJyHPLwYMjDg5GRdwTv5B9tNhw7cztoc\/7RZt+rH4YNL8e6OoYiEZFIkqLGIClqTKvh2BFNjRL9fL3w9MTRDMMWMBRtRFepw5DBgfjpYoXYpRCRnakfjssz8wA0nlDTGWFBfRAzIpBh2AYO3dHG1gq\/UGHwoIEo\/ELV7D4PPBALt25dGv38qVcP6PV6G1ZLRI4mKWoMLr23HCsSoixyvLCgPvjHnOl4f850JEWNYSC2AUeKbaCr1OHVV17COxnpbdpfKpUidNSoRq97eXIiEBG1zjRy7Izdix7jErUOYCi2IicnG0\/M\/Avuu28ipFJpm0Z7oaNG4dNP99mgOiKipjEQO4aXT1tQXnYOGzduwBdfHsWH2zKbHP0REZHz4EixBX7+fTniIyJyIQxFsqqJEcEoLzuH2zy9cPNNN7fpM126SKxcVeuuXTOIen5X\/w5MZ\/7999\/Rq1cv0epwRgaDAf\/9739xyy234JdfxH8AgL1hKFqB9owWj81IwD\/\/uQMAEBAgQ+JTszBv3gKRK7O9n3+uxvXr11GlrxS7FHJQFy9eFLsEp3TlypXWd6pTWHIOnu5uGOLf04oV2QeGooXdc88EeHl64bUVq\/DhtkycPVuKl5YvxdIlL+L8+XNYv36j2CXa1NBhITj0n3z87e33cNfYCNHq6NfrVtHObXL2p\/+JXYLo3wO\/AyMxv4eLP\/2EhyZF4I477mjzZ9Z8cgjVNVew\/9VZVqzMPjAULWzevAX47+Vf8EcP4\/Pw+vULwFvr38ZXX32FdzLS8dRTT8Or5wCRq7Q9T08v+P7xdtHO37On+M8nvPy7+JeqxP4e+B0Yifk9\/Pbb1XbtX1hyTmhMfqLsomijRY1Gg+HDhzd6vaCgAOHh4RY7D2efWtjVa9egu\/w\/\/FB5o12Tj7cP7ps4EQCQn58nVmlERO225pNDwvbcLZ+KVkdwcDAqKiqwZMkSs9cHDRpk0fMwFC3smuE6AOBy7RWU6W7cR+vTp69IFRERdUz9USJgHCmeKBPvHq+Pjw9iY2OF30NDQ+Hj42PRczAULez69evCds2VqzhXUYlrBgPOnz8HAPD09BSpMiKi9qk\/SjQRc7QIAD\/\/\/LOwHRVlmXZ49TEULWjt2jfh7eGOC2U3\/mVV+9tVlOv0+Cw\/HwBw7733iVUeEVGbNRwlmog9WiwsLBS2hw4davHjMxTb6OzZUqiPHQMAnPzuuxb3XThnthCMF8rO49mkRJSWarHl\/Q\/Rr1+A1WslIuqspkaJJmKOFr\/66ithe\/To0RY\/PkOxFS+\/tAyDBw3EoKABQt\/Tvz6TjF49ffHAA7E4flwj7PvQQw\/jmWdfgJeXFPePvxvD+92B+8ffDQB4758fY3Tkfaj9vX0zv4iIbK25UaKJmKPF3NxcAMYHLwQEWH6QwSUZrXj1tZV49bWVbdq3X78AvPDiUlyubXpR7LXrBpy7VImS8iqE9POzZJlE5EI2p72FN1cvx4IlKzAr+TmLH7+lUaLJ3C2f2nzdokp147F91rifCHCkaHGm2actmbN1J7K\/bvkSLBFRU0q++wZvrl5uteO3Nko0sdZoUaVSQS6Xw9vbGxKJBBKJBP3794dcLsfKlTcGKJGRkRY\/N8BQtLirv7ccil3+IEFcyJ0Y0NPXRhURkTNZsuCvGDIsxGrHb8so0cSS9xZ1Oh1iYmIQEREBrVaLbdu2wWAwwGAwYOXKlcjKyhIunQLAnXfeabFz18fLpzbi3u1meLi54Tb3W7B0qnWG\/UTk3DanvQUAiL1\/Kk4cL7LKOXYv+kuj13o8sQIAcOk964xQTYGoVqsRHx8PpVJp9r5cLsehQ4eQlpYmvGbJLjb1caRoYVevXRO2u\/xBAreuxidDdLv5ZkhvdUcXifhPPyAix2O6bLr6zbfFLsXiTIEok8mwcWPT\/aH79u0rbEdHR1utFo4UW1F2qabN+16sqsbMjdsxoNcfETviTkQMkkH\/cy0eXrsVHm7dkLNUYcVKichZVekr8denH8WCJSsQNHgoCg5+LnZJFpOamgq1Wg0AWLlyZZs61EyYMMFq9TAULehWNzdseHIaBvTqIbzm4dYNPaWeuKivRvbX3yFuxGARKyQiR7R18wYAwJ\/ljS9tOjKdToc33ngDACCTySCXy5vdt6joxuXiIUOGWK0mXj61IA+3bmaBaJIYeRcA4NDJM7YuiYgc3FfHvkT6hjVIWZcBL6m32OVY1M6dO4X13wsWtPy8WdNoEgDGjBljtZoYijYQMUgGACg4qcXFqmqRqyEiR1Glr8Si55Mwe858jBx1d7P73Bc+zMaVWca7774rbN91113N7qfT6aDVagFYpwl4fQxFG\/Bw64a4EOP04YPfaUWuhogcRc7ej1F2\/izSN6zBQD8P4ce0TvHN1csxelgflJ0\/K3Kl7afT6cxGf8HBwc3um1\/XOxoARo0aZdW6GIo2Ejvc+Myv9w4cFrkSInIU8hlP4vvyy41+FiwxLpFYsGSF8JqjOXnypLDd2mzSjz\/+WNgeN26c1WoCONHGTA9pNwzobZ2ncg\/oPQipe7xQXlGFS79UICyor1XO01mW\/vvd3bpY9HhE5BwuXLjQpv00Gg2ysrKE3y39UOGGOFK0obiQQABAVuFxkSshIhJXWVlZm\/ZbtGiRsC2VSlu8zGqydOlSoU1campqo\/dTU1OF9xs2CuBI0YbmTxmHjLwj2Fd0CuVTquDn6yV2SUTkgC5frjb7v5Y0NeUDnCj7vybfG\/DMm2a\/jw3qg3\/Mmd6h8\/j7+7e6T2pqKs6cOQOZTAatVtumR0WlpqZi9erVwu+LFi3CtGnThCdqLF261Oz9hiNPjhRtyNPdDfLwYFTX1CK76JTY5RCRA9qc9hbSN6wBAGR+8C6U27Za9PgrEqJQXVNr9mPS8HV5WOujtub07t1b2D569Gij95VKJRYtWoRdu3YJM0\/r02g0iImJgU6nM3tt0aJF2Lt3LyoqKiCVSgEYl36Yjrl69Wqkp6dj7969TY48GYo2NjawDwDgnfzG\/yMgImrNrOTnhMk1x06UQz7jSYsef4h\/Twzx79nqfmFBfRBbd0uoI8LDw4XQ0uv1UCiMHb90Oh1SU1ORkJCAzMxMs9AyBaBKpUJkZKRZIALGoN27dy\/i4uLg4+OD5ORkAMaF\/xqNBgqFApmZmUhKSkJcXBwqKysb1cVQtDF5eDD8fI0TbsorqsQuh4iokfWJD7S6z9MTO7+AftOmTcJ2WloaJBIJfH198cYbbyAzM1PocGMKT7VaDYlEgoiICERFRSEnJ8dszaKPjw\/i4uKE38PCwoTPPfXUU0hOTm6xaw7AUBSFacLNm+14RAsRka20Nlrs7CjRRC6XY+\/evZDJjA1OpFIpkpOToVarzcJrz549wj6hoaHIzMyEUqlsdRG\/qfONVquFj48PVq1a1WpNnGgjgvoTbpAodjVERI2tT3wAE17e3OR7lhglmsTFxZmN7poSHh6OM2fa3ybTx8dHmKQzderUNn2GI0UReLq7wc\/XC9U1tVCqNGKXQ0TUSHOjRUuNEm1BqVQKk3Q0mrb9t5ahKJIFU4xdGTjhhojsVVP3Fi05SrQmlUqFhIQEYbJNXl5emz7HUBSJ6V9aJ8oucsINEdmlhqNFRxklajQaTJ48GSkpKZg\/fz4A433F0tJSAMZZrA0X7ZswFEViWrMIcLRIRPar\/mjRnkeJCoUCCoUCGo0GkZGRGD16NBYuXIiAgABhks6aNcb1ncuXL8fatWubPA5DUURPTzR2Z+B9RSKyV6bRor2PEo8dO4a0tDQMHz4cMpkM27dvF96Lj48HcGPZh1KpNHtsVX0MRREN8e\/JCTdEZPfWJz5g16NEAMKEmvj4+EbrF1944QWEhoYCAGQyGQ4cONBsD1UuyRDZginjMHfLHuz7+pRwOdUZ3eHrZrUnkDgKV\/\/7AX4HJmJ+D92uu7f7M23tciOmprrTmPj4+ODYsWNtOg5HiiIzXY7YV3SKE26IiETGUBRZ\/Qk3bBJORCQuhqIdiA8bBgBYw7ZvRESiYijagbCgvsKEm8KSc2KXQ0TkshiKdsLUJDyr8LjIlRARuS6Gop2YX9f2bV\/RKbOHehIRke0wFO2EacIN1ywSEYmHoWhHxgb2AcC2b0REYmEo2hF5eDD8fL1QXlHFNYtERCJgKNoZ04SbN7k8g4jI5hiKdqb+hBsiIrIthqKd8XR3Y5NwIiKRMBTt0IK60WJWIUORiMiWGIp2yNQkvLDkPCfcEBHZEEPRDtVvEs7lGUREtsNQtFNPTxwNALyvSERkQwxFOzXEvycn3BAR2RhD0Y6ZRov7vubyDCIiW7hJ7ALsySX9FZy+8Ito5x\/Qu7vZ7\/LwYCzPzMO+olMor6iCn6+X1Wuw9N9fU3vNoscjIrImhqIdM024Uao0yC46haSoMVY\/5x+9uln0eM89+1dMnzYFgwcPtuhxiYisgZdP7VzsCOPyjDVs+0ZEZHUMRTsXGxIoTLg5UXZR7HKIiJwaQ9EBmJqEc80iEZF1MRQdQP0m4dU1tSJXQ0TkvBiKDsDT3Q1hQX24ZpGIyMoYig4iPoxt34iIrI2h6CBME27KK6rYJJyIyEoYig7C091NmHDzpgjLM97fuhlSj66NfoiInAlD0YEIbd+KbN\/2beaTs1D8zSnce1+U8NqIkJE2r4OIyJoYig7Ez9dL1Cbhffr2w\/h7IoXfJ0yYaPMaiIisiaHoYBbULc\/IKhRnFmp1VbWwPebuu0WpgYjIWhiKDia27r5iYcl5USbc7N+fL2wPHBhk8\/MTEVkTG4I7mPpNwjNVGix8cLxNz\/910VcAgH79AtCnb782f+5\/v\/6Oql+uWqusVnl1v1m0c5uY\/v7\/Vl0RrYaGT2KxNTGfQmMi9ncAiPs9\/PhTjWjndgQcKTog04SbzTZes3j4y0JhO2RkqE3PTURkCwxFBzTEv6coE26OHT0sbIdH2HaESkRkCwxFByUsz\/jasssz\/rVzB6ZNvd9sLeKE8Xdj\/vNz8N6WzcJ+oaOs\/2xHIiJbYyg6KHm4se3bvqJTFplwo9Fo0L9\/fyQ+MQNeXl4o+EIN\/eXfUHr+J0yYMBFb3s3A2bOlwgf6IbYAACAASURBVP5Dhg7r9DmJiOwNQ9FBmSbcAEB2JxfzazQaREZGQqvVYt3fN+Ld97YJoSf19sGyl19Dv34Bwv71F\/ATETkThqIDix1hXJ6xphNt30pLSxEZGQm9Xg+5XI6ZT85qcr8AWX9hu\/4CfiIiZ8JQdGCmJuHVNbUdvoSqUCig1+sBAKtWrWrTZ0aNvqtD5yIisncMRQfXmSbhKpUKubm5AIDk5GQEBAQ0u+9X6mPCdmAgF+0TkXNiKDq4+XVt3\/YVnUJ1TW27Prtx40ZhOykpqdn9zp87i6oq42hyRMhISL19OlApEZH9Yyg6OE93N4QF9UF1TW27np6h0+mgVCoBAFKpFMHBwc3u+\/33JcI2m4AbqVQqSCQSscsgF3Fwfx5eWfo87gsfhoF+Hhjo54GHJo3Hwf15YpfmdBiKTiA+zBho7bmEeuTIEWF79OjRLe6buy9b2B5855A2n+P8ubP429o1ZiPSluTn7cP85+dgwvi7hTWS\/fxux1NPzEB+3r42n7c5paWlSEhIQExMTKv7mkKvuZ+IiIhO10PUFpvT3sKsxx8GALz30R58X34Zmf\/Kw8\/Vesx6\/GHs3bNT5AqdC0PRCcSGBMLT3Q3lFVVtnnDz7bffCtv33ntvi\/vu2rlD2B4ZOqrVY+srdZj\/\/BwMHxoIlaqgTfX087sd0x+egl07d2D1G2uENZIvzF+IXTt3YPrDU7Dy1ZfadKym6lEoFJDJZMLomMiReHp64ZVV6+Dn3xcAMHLU3Vj+2hoAwJaMDSJW5nwYik6g\/prFto4Wz54926b93t+6Wbif6OUlbbEJuL5Shw1\/X4uQ4MHY8m5Gm45vYjrHp9n5uOvuMADGNZJznp2HxKeM9zvXrnmjXcesX09aWlq7PmuSmpoKg8HQ7A+Rtc1Kfg7HTpQ3er27hwcA4Odqva1LcmoMRSchtH1r433F0tLSVvfRV+rw6svLhN\/vva\/l+4n33hOG97Zsxtp1G6C\/\/BtGhIxsUy0mD0+b3mSnnD59+7brOE3VYzAYEBrKJubkPC7+9AMAIHx8y1d6qH346Cgn4efrBT9fL5RXVEGp0ggjx+a0tPzCZFbi4xgZOgqff2a8mR88fESL+6esWYfQ0NHC7FTvdsxS1V\/+rdn3NMVfA4BZV522aFiPjw9nzZJzqNJXYkvGBnh6eiEx6Vmxy3EqHCk6kQV1yzOyClt\/cka\/fjcugxYVFTV6f\/7zc1CqPYOk5Gea\/Py\/du7A\/OfnmL02MSrW4ss1\/rVzB3bt3AEvLyk+2L6j9Q9YuR4iMZWXncPePTsx7YF7IJV641\/ZBcJ9RrIMhqITia1byF9Ycr7VCTdRUTf6lyqVSmECyvlzZ\/HUEzOwa+cOfLB9Bzw8bhP2q66qBmC8z5j4xAxLly84f+4s8vP2YdrU+5H4xAwkPpWE\/xQcFqUJ+eeff46YmBizmacJCQlQqVQ2r4Vc2+a0t3Bv2FA8\/8wTqK7Sw+M2T1z6v5\/ELsvpMBSdSHuahAcHByM5OVn4PSEhARKJBMOHBqLoKzU+zc7HkKHD0L27h7DP2jVvQOrRFc8\/+wxeW\/k61qyz\/Ky3aVPvx\/ChgZj+8BR8\/lke+vULwJChw3Dbbbe1\/mELCg8PR0FBAZYtW4acnBwYDAZUVFQgPT0dSqUSERERWLp0qU1rItc2K\/k5fF9+GUePn8crq9eh8NDnSHgoimsVLYyh6GRME27a0iR806ZNSE1NhVQqBQDIZDK8tvJ1fP6fQmFUNmToMKz7+0Z4eRn3eXjadOzLO4A5z86zSv07d\/9bWI6xY9cnCJD1x\/PPPoOQ4ME48c1xq5yzOeHh4QgPDxd+9\/HxQVJSEtLT0wEAq1ev5oiRbM5L6o1Jk6ch\/T3j7YT5cxNFrsi5cKJNPT2k3TCgd3exy+iUIf492zXhZsGCBViwYEGL+zw3V4Hn5iraXctNXTre8UXq7YOJUbGYGBWLaVPvx+ef5eGBuIko0nwn+n3CadOmYfbs2QCAjz76yCw4iWxl5Ki7AQDV1VUo+e4bBA0eKnJFzoEjRSdkGi1+ceq8yJVYhmmyT1WVHgf2fy5yNeazWNuytIXI2m6td5uDOoeh6IRMo0OlStPhR0rZk\/qTfX74ofEiZiJnljhjKh6aNL7R6+Vl5wAA\/n36cQaqBTEUnVB7JtzYA32lDoe\/LMT5c6132QkaNNgGFbVMp9MJ21OnThWxEnIVJ44X4ZWlz6NKXwkAKPnuGzyb\/Dg8Pb3w9jvbRa7OuTAUnVTsCOPyjLZMuBHbqVMliI2KxLwG6x5NPsszPvOxX78ATIyKNXvv8JeFQvNwS5JIJOjfv3+T7+Xn5wMwPl1k2rRpFj0vUUMLlryG2XPm43hxEUYP64OBfh54bHochg0Pwb+yC3gv0cIYik4qNiQQfr5eqK6ptfkl1NLSUiiVSuEBxidPnkRWVib0lboWP\/f5Z3mY\/\/wcYcRo6l26ds0bHVq8b3L+3FmzenJzc5GRkWE24muKVqtFQkKCcN\/Q9LgthUIBqVSKAwcOsEsOWV3Q4KF4YeHL+Nfeg\/i+\/DK+L7+MYyfKzRqEk+UwFJ1YXN1i\/vY8UqozUlNTIZFIIJPJkJCQYPbeqy8vQ0CfXpB6dMXq11Pw36orwk\/XW30xe858RIy\/D\/n5eRg+NBBSj64I6NML+fmfY8GSFcg7VIzb\/QLNPvffqiuo+uWqcI6G761+PQVSj64YPjSwUT2zZ8+Gr68vJBIJUlNTG\/0te\/fuRXJyMvR6PWQyGSQSCXx9fbF27VokJydDrVa3+AzKppjqIiL7xSUZTmz+lHHIyDtibBJug6VMCxcuxMKFC81ei42NRU5ODrZ8uBsR99zX5Of8\/PvihYUvd+icp0+dBADMnjO\/0Xuzkp\/DrOTnAKDdS23i4uIQFxfXoZqIyHFxpOjEPN3dEBbUB9U1tVCqWu+H6miq9JV4N\/0tDBkWgidnNX0\/koioPRiKTi4+zHiJ7538oyJXYnmFBfsRPv5ebN22G15Sb7HLISInwMunTi42JBCemW44UXYR5RVV8PP1Erski5k0eRomTebsTyKyHI4UnVz9NYu2mnBDROSoGIouwNT2bZ8DLOQnIhITQ9EF+Pl6CWsWnXHCDRGRpfCeootYMGUc5m7Zg6zC1p+cYQ13+Lo5\/BNIOsvV\/36A34GJmN9Dt+vuop3bEXCk6CJi6xbyF5acd4om4URE1sBQdBGO1iSciEgMDEUXEh82DIBjNAknIhIDQ9GFhAX1FSbccCYqEVFjDEUXYxot7vuaoUhE1BBD0cUkRY0BAChVGk64ISJqgKHoYjjhhoioeQxFFxQ7wrg8gxNuiIjMMRRdUGxIoDDhhpdQiYhuYCi6qLi6xfxsEk5EdAND0UXNnzIOAJuEExHVx1B0UZ7ubmwSTkTUAEPRhS2oGy2+k39U5EqIiOwDQ9GFxYYEwtPdDSfKLnLCDRERGIouzdPdTXh6BkeLREQMRZdnuoTK+4pERAxFl+fn68UJN0REdRiKJIwW2SSciFzdTWIXYE8u6a\/g9IVfRDv\/gN7dRTlvbEggsMW4ZrHgxA\/o6eVpsWPX1F6z2LGIiKyNI0UyaxJ+8DutyNUQEYmHoUgAbjxn8b0Dh0WuhIhIPAxFAgCEBfWFn68XLtdeQdHZcrHLISISBUORBKYm4fuKT4pcCRGROBiKJDA1CS\/47gwuVlWLXA0Rke0xFEng6e6GuJA7cbn2CifcEJFLYiiSmRF9ewMAdhwuFrkSIiLbYyiSmbgRg9FT6omL+mpeQiUil8NQpEbGD5IBALYcOCJyJUREtsVQpEaeiLwLgHHCDRGRK2EoUiMebt3QU+qJy7VXkP31d2KXQ0RkMwxFalJi3Whxx5dfi1wJEZHtMBSpSRF19xVP\/3SJE26IyGXwKRnUJA+3bogLuRPZRd9ix5fFmBs7vlPH+7Gi1iWfQFKfmH+\/idjfA78DIzG\/hx9\/qhHt3I6AI0Vq1vS7hgMAsou+FbkSIiLbYChSswb06sEJN0TkUhiK1CLThJtDJ7k8g4icH0ORWmSacFNwUssJN0Tk9BiK1CLThBsAbBJORE6PodhJq3bnYf2+g079YN7Y4YMAAO8dOCxyJURE1sUlGZ00om9vrPpXLrK+KAJgvNw4om9vxIXcCQ+3biJXZxkh\/fyEJuFFZ8sR0s9P7JKIiKyCodhJpntuJgUntSg4qcX6fQcR0q83BvTqgfCgAIcPkvGDZMj6ogj7ik86\/N9CRNQcXj7tpPr33BoqOnsBWV8UYc7WnQhbvg4vfrTHYZc21G8Sfrn2isjVEBFZB0PRAkz33FpTfPaCw15SNYX\/5dorXMxPZGNV+krs3bMTzylmYtQQPwz088BAPw88p5iJr459KXZ5ToWhaAGme24t8XDrhqUPRTe63OpIRvTtDQDYcbhY5EqIXMu8OYl4\/pkn4CWVIr9Ag+\/LL+O11\/+O7E93IeGhKAajBTEULWR8K2E3N+4ehw5EAIgbMViYcMM1i0S2FTH+Pryyah28pN4AAPmMJ\/HIX54CAHz68Q4xS3MqDEULMd1za8qAXj0QN2KwDauxHlP4bzlwRORKiFzHlm27sWXb7kav\/6l3HwBA+flzNq7IeTEULcTDrRtC+vVu8r3TP13CnK3\/tHFF1lF\/wg0R2Qe\/Pn3FLsFpMBQtKHaE+SzU+LEheF\/xKDzcuqHo7AXM3LRdpMrazr+He5M\/Jh5u3dgknMhO7Pu3cfT4wIPTRa7EeTAULShikEyYXRoXcieeiLwLA3r1wIYnp8HDrRtO\/3QJMzdtd\/glDaYm4fu+5ixUIrEc3J+HE8eLMHvOfIwcdbfY5TgNhqIFebh1Q8Tg\/hjQqwfmxo4XAnJArx7457xEDOjVQwjG0z9dErnajjNNGCo6e4ETbohEUF52DvPnJiJi\/H14YeHLYpfjVNjRpp6rV6+itvbXTh1jSsgg9PQcjZtx3exYNwNY88gkzNv2Kc78XwX+unUn1s54AP1v9xX2+fXXLp06NwB069a5dZDXDdebfL3h3xIdHIhczSl8VKCGYuLYZo937fq1TtVDROaq9JV4Nvlx+PUJwNoNW8Qux+kwFAGoVCoAwEuLn8VLi5+1+vm8Yp7ALz374elN21C9\/yNcvXjOYsdWq9UWO1Z9oaGhZr\/f5N0T0skK7FQdRbriQauck4jMVekr8eSMqQCArdt2C8szyHIYirgxuuratSu6dOn8aK01Vw4qIQl\/GDf9aSC8Yp7Erwe249qlMosc+w9\/aNsV8erqanz22WfYv38\/vv32W\/z888\/Ce3fffTcmTJiA++67D56exqYEt9xyi\/kBfq3G9f9V4w+3euK2wWNw9ezxJs\/z22+\/4do1jhaJLOGtNStQfr4U\/8ouYCBaCUMRwPHjTf8H3dqWZ+YhI+8Ibol8FOsTJ0MeHmyT82ZkZGDx4sXQ6\/VNvv\/ll1+ipKQEK1euFF6rqalptJ9SpcHcLXvw4OwX8I85Tc9+i42NRU5OjmUKJ3Jhym1b8dEH72JP7hfw8+8rdjlOixNtRLQiIQpJUWMAAHO37EFGnvUXxCsUCsyePRt6vR4ymQzp6enQarUwGAzCT3FxMQ4cONDqsWJDAgEA+4pOobyiytqlE7ms8rJzWPvGy1iwZAWCBg9t9N5zipniFOaEOFIU2YqEKHjc0g1rPjmE5Zl5qK6pxcIHx1vlXAqFAmlpaQCA5ORkbNq0qcn9goPbNmL1dHeDPDwYSpUG2UWnhIAnIsvakvF3VFdX4XDhQRwuPGj23vFiNYYND23mk9ReDEU7sPDB8QgL6oOpKR9izSeHcEFXjfWJky16joyMjDYFYnvFjgiEUqXBmk8OMRSJrOSjD94FABQc\/EzkSpwfQ9FOhAX1xe5Fj2FqyodQqjQor6jC7kV\/scixdTodFi9eDACQyWQWC0TAeAnVz9cL5RVVOFF2EUP8e1rs2ERk9H35ZbFLcBm8p2hHwoL6Yv+rswAAhSXnMTXlA4scd+fOncKkmvqTZywlru7e4jv5Ry1+bCIiW2Io2pkh\/j2x\/9VZ8HR3Q2HJeUx4eTOqa2o7dcx3331X2E5ISIBEImnxp73mTxkHwDjhprO1EhGJiaFoh4b494T6zTkY4t8TJ8ouYsLLmzs8u1On07VrQX90dHS7z+Hp7obYkEBU19RCqdK0+\/NERPaCoWinPN3dsHvRYxji3xPlFVWY8PJmnCi72O7jnDx5UtiOj483W3rR1E9H1xTGjuAlVCJyfAxFO2YKxrCgPqiuqcXUlA\/bHYwXLlwQtkNCQixdoqD+hBuuWSQiR8VQtHPGYPyLcHlywsubUVhyrs2fLyuzTPu41ni6uwkTbt785JBNzklEZGkMRQfxjznThTZwpmUbbTF27I0nWJw7d84apQnqT7ghInJEDEUHsj5xcrvbwt1xxx3CtlKphE6ns1p9nu5u8PP14oQbInJYXLzvYNrbFi4gIADR0dHIzc2FXq9HTEwMXn31VcTFxQn7aDQa\/PDDDygvL0dSUlKn6lswZRzmbtmDrEKNWYPzO3zdMKB3904d29G5+t8P8DswEfN76HbdXbRzOwKOFB3QwgfHC0+lWPPJIczdsqfF\/bdv3y48D1GtVmPSpElm6xKHDx+OSZMmCV1vOsPUJLyw5Dwn3BCRw2EoOqjYkEDsXvQYAOMjnFrqfuPj44OcnBykpKQ0elhwaGgokpOTkZmZidOnT3e6LlOTcADI5CVUInIwEoPBYBC7COo40+J+AAgL6mOxfqmdYarJ090N\/Uv3IycnB\/v27UNMTIzYpRG5vLKyMvTp0wf+\/v44f\/682OXYHd5TdHCmtnBTUz4U2sLtXvQYPN3dbHL+a9cN+M+3Whz4RovTP+nwf9W\/4KL+Mnp4dsel6l+gu6WHTeogIrIEhqITMAXj4xv+KYzSPl70GPx8vax2zi9PnUdG\/lHka07j6u\/XGr0\/Y9wIbDv0NS53vc1qNRARWRrvKToJP18vi7SFa03ByXN47O9ZmPLGB8j+qqTJQASAsUF9AAC6W3qgS3frhTMRkSVxpOhETG3hZm7YUffoqQ+FoLSENZ8cQurHBxu9PtjvdtxzZwDCB\/VFT6\/u6OnlAd\/bbsWh785CqdKgq\/8gi5yfiMjaGIpOxtQWbmrKB2b3GMOC+nb4mBU\/\/w\/Pbv0U+Rrz2amPRAxHUtQYDOrd9H3D2BGBUKo0uHV4ZIfPTURkS7x86qR2L\/pLh9rCNfS\/2t\/w+IYdZoEYNXwAVKuS8daTDzQbiIBx2UjXa1cg6eqGiv9d6dD5iYhsiaHoxDrSFq4+A4DHN+zAsTM3nrSxaOo92PasHAPv8G3TMaRXKgEAu7\/9sV3nJiISA0PRya1IiBIadS\/PzGvynmBznt\/6KQ59d1b4fd0TD2De5Ih2nb\/nZeNTOoou6Nv1OSIiMTAUXcDCB8djfeJkAM1PlmloX9EpfFRQLPy+6pFoPDpueLvPfZPhGq5ePIv\/Xb3GJuFEZPcYii5CHh4stIVb88mhFtvC\/X7tOl7d8Znw+6PjhmPWxNEdPnftGWO48jmLRGTvGIouJCyorxCMxiUbTQfj9kNfo\/T\/jPcCfW+7FS9Nv69T571SdhK33twF5RVVbBJORHbNoUOxtLQUS5cuRUxMjNlTHyQSCUaNGgW5XI6MjAyrPkPQ0YQF9cX+V2fB091NWLJRXVNrtk\/WF8eF7XmTIyC99ZZOndPwWy3C+xkn5nC0SET2zGFDUalUIjQ0FKtXr8bUqVNRUVEBg8GAiooKpKenQ61WIysrC7Nnzxa7VLtjagvn5+uFE2UXMTXlQyEYv9JegLputulNXf6A6WOHWeSc0QNvB2C8V0lEZK8cMhRLS0uhUCig1+uRkpKCpKQk+Pj4ADA+JikpKQkpKSkAjI9GMr1HN\/j5emH\/q7MwxL8nTpRdROiCDThRdhFfnioT9pk+dhg8bulmkfP53toNfr5eqK6p5YQbIrJbDhmK+fn50OuNU\/zHjh3b5D6JiYkAgKioKJvV5WhMbeGG+PdEdU0tpqZ8iC9O3XiUzEhZb4ueb0Hd0pCsQoYiEdknhwzF6upqYfvChQtN7uPj4wODwYBVq1bZqiyH5Onuhv2vzkJYUB9U19Tis+NnhPcG9\/6jRc8VGxIIwDjJhxNuiMgeOWQoenp6CtsKhQIaDUcenVW\/LRwAeHd3x2C\/2y16Dk93N+Ec2by3SER2yCFDcdq0aZDJZAAAvV6PyMhIZGRktPgZlUrVaIaqSqWyRbkOo35buMpfavDBf4osfo6n69Y7ruEsVCKyQw4Zij4+Pti1a5dZMM6ePRsxMTHNjhrDw8Oh1WoRGhpqy1IdTmfawrXFEP+enHBDRHbLIUMRAIKDg3HkyBEsWbJEeC03NxeRkZFQKpVNfiYgIAB\/\/vOfbVWiw+pIW7j2MI0W60\/qISKyBw4bioBxxLhq1SoUFBSYjRoTEhJ4abST2tMWriPHBgClSsMJN0RkVxw6FE3Cw8Nx5MgRs0ujb7\/9togVOYe2toVrL064ISJ75RShCBhHjVlZWcLv9bfbKzs7W2gf5+3tbdY6Ljs7u9H+DSfwSCQSxMTEtLpPU6NZlUoFuVxudl65XN7kvqmpqU2eU6PRQC6XN1tLe7SlLVxHxI4wLs\/ghBsisidOE4qA8Z6h6TJqR\/Xv3x+TJk1CXl4eli1bhsrKSlRUVCA5ORlqtRqTJk1qFFANJ\/DEx8dj+\/btze4jlUpRUFCA8PBws30UCgUiIozPK1Sr1TAYDCgoKEBeXh4iIiIa3StduHAhCgoKIJVKhdeUSiWGDx\/eqX8UNNRSW7iOig0JFCbc8BIqEdkLhwpF0wiqJZWVxqc7dHSWqVarBQCsW7dOCC0fHx+sWLFC2Oejjz4y+0xAQABeffVV4feQkJBGreUCAgKE7jqvv\/56o0DMyMhAWloaZDIZlEolAgICABgvDb\/44osAjKHZsLl5eHg4Ro82Tlw5evQoFAoFCgoKhEC1lObawnVGXN1ifjYJJyJ74VChmJeXB71e3+wkGpVKJbR\/qx9S7VFQUNDkKM7HxwfR0dEAjL1XG4qLixNGqUVFTa\/vy8rKglQqxbRp08xe1+l0WLx4MQBg5cqVjT5nOq9er0d+fn6ztev1euzZs0eoPTw8HAaDATk5Oc1+pj2aagvXmWA0Lf1gk3AishcOE4oajUYIvMmTJ0OpVAqjJp1OB6VSicmTjcsI0tPTERcX16HzhIeHNwrEtnr66acBGMOv4YguOzsbWq0WL774YqNR5JEjR4S\/rXfvxv1Gg4NvdJopKytr9L5JaGhoh2tvq4Zt4Sa8vBmFJec6fCzTcbhmkYjsgcOEYu\/evZGZmYnk5GSMHj0aCoUCvr6+kEgk8PX1xdq1ayGXy6HVapGUlCRKjfVHgA1HdOvXr2+0j8mJEyeE7YiIiCYn5Zjs37+\/2fPb8mkg9dvCTU35sMOjvfgw4zHeyT9qsdqIiDrqJrELaCsfHx\/I5XLI5XKbnC87OxuFhYX46quvcObMGeFeY0sCAgIQGhoKtVothDRgvNyam5uL5ORk4V5hcwwGg0Xqt4X1iZPh6e6GjLwjeHzDDqxPnGzWP7UtYkMC4ZnphhNlF1FeUQU\/Xy8rVUtE1DqHGSnaSnZ2Nvr3748ZM2bA398fy5Ytw5EjR2AwGIR7ey2ZN28eAOPsUdO9xzVr1gAA5s+f3+rnm7pfac\/qt4Wbu2UPMvKOtOvz9dcscsINEYnNYUaKtqBSqTBp0iRIpVKcPn26Q5cjJ06cKGzn5+dj2rRpUCqViI6ObnaUWP+pHyUlJa2OJu3NwgfHAzCuOVyemYfqmlrhNZMfK2px+sIvTX4+auidyMg7gr3qEsyJnmCVGgf07m6V47ZHc3+\/LYn9PfA7MBLze\/jxpxrRzu0IOFKsx9QFRyaTNQpEnU6Ho0dbv+\/l4+OD5ORkAMCbb76JnTt3Qq\/XY9myZc1+pn6Qmu49NqTT6ZpckmEvFj44vsNt4Xp6eaKn1BOXa68g++vvzN67XHsFF6uqm\/kkEZFlMRTrMa2BVKvVZo+iys7ObldXmPvvvx+Acc3j4sWLIZPJWpwVGhAQIFyazc3NhVwuFy6jmmbWjhkzBmlpaThypPnLk2IHZmfawiVG3gUA2Pf1twCAi1XV2LL\/S8zZuhNbDrTvkiwRUUe5VCiWlpaarSH84osvzILkkUceEbZnz54tzPycNGkSRo0aJSyiz83NbbLdm0n9NYt6vb7JtYcNbd++XWg4kJWVBZlMJsysTUhIQGVlJYqLixstNVGpVMIIVq1Wt1iXLTRsC3fSd3ibPhcxqG6N59kLmLlpO2Zu3I6tBw7j9E+X4OHWzZolExEJXCoUZTKZWfuzRYsW4dFHHxV+Dw8Px969exu1bNu7dy82bdpkdqxJkyYhNTW12XPFx8cL52zLjFkfHx\/k5OQgJSXF7PwymQxLlizB6dOnzdYrAsZAjIiIENY4tqUuWxji3xO7Fz0GT3c31NzkDulkBX79\/Vqz+xedLcd7Bw4L4Xf6p0u4XHtFeP\/Wbl2tXjMREQBIDI60BsCBKJVKJCQkICUlBQsXLhS7HFFU19QiOOlVYzDe0hXrnvgzBvTqAcB4r7DgpBY7vvwap3+61OJxlj4UjbgRgztVi6tPrjAR+3vgd2Ak6kSbHy7gnrsGwd\/fH+fP85mmDXH2qZUsW7YMUqkUiYmJYpciGk93NwzQfYOvbu4DvXdPzNm6ExuenCYEY8zIANwz1K\/V4\/zJ2xMebm5tPm\/ZJc6uI6KOcanLp7aSkZHRbEs3V3OT4Rr0ezYhQHorLtdewZytO1F0thwebt3Q19cbt7Th0ujNN3WxQaVERAxFi6j\/zEKVSiXMOHXlUWJDT4cEIGKQTAjGgpNaXDMYcFO9FnbN6SLh\/0ypscQZU\/HVsS\/FLoOcDP9rYyG5ubmQSCTC8xB37drl8qPEht54ZDLix4YAd\/P+cgAABcRJREFUAF78aA9+qNSbTaj5Wn0UD9wzFl+rzdeD3tzlxkjx+HENXn5pGcLGjoFbty5w69YFgwcNxNq1b9rmjyCb2pz2Fgb6eTT5U3DwM7HLsynltq0YNaT12w3UObynaAEymUzojRofH4\/Fixc3milKRnNjx+PWbl2x9cBh1Fy5CgCo1uux8W+p2LHtH432rx+IhV+ocG\/keIwcGYq09HcwbFgwzp4txdy5z2Dpkhdx\/vw5rF+\/0WZ\/C9lOxPj7mny9x+29bFyJ7ZV89w3eXP2Sy\/0jQCwMRQs4c+aM2CU4lO711h2qDnyOJc\/\/FXdHjMdtnp74udq8e02XLo0vZpgCEQD69QvA+\/\/Yhj\/16oF3MtLx8iuvAbjFqvWT7W3ZtlvsEkSxOe0tvLNxLcLG3St2KS6DoUg2lfVFEdbvOwgAuFB2Hh+9vwUf7clBb\/8+UDz+CL449B\/cdsuNmaZ\/qHfLMWxsOGqvNF7v6ON94zL1z9XV6HIrQ5Ec3949O3H5cjXyCzTwknoj+9NdYpfkEhiKZDP1AxEAevv3waZ\/fNRov+vXr6NvD2+UV+jNLp825+xZY0u8gAAZ+vUL4JIMcgqTJk\/DpMmNn79K1sVQJJvQ1\/6GHUXFbdr3lyu\/oYvkD\/DzleLXuvuOLVm3bi0A4G\/r3upUjUREDEWyCalbV+x64UkUnS3H6Z\/+C1VJyw9t\/qX2CqS3usPtppubfF9XqcOxo0ewceMGaM9o8fmBgwgb23zTdXJsryx9Hns\/2Ynq6ip4enph0pRpeG7+cnhJvcUujZwMQ5FsKqSfH0L6+SF+bAj8e7jjcm0tan67ip9rfjXbr\/KXGkhvdW\/yGKZZqCYTo6Lww4UfrFo3iWNg4GBEjL8PDzw4Ha+sWocqfSX+qfwAb65ejuPFRdi6bTeDkSyK6xRJVB5ubrj9Ng8M6NkDbl2N3W3cbr4ZV69dw9VrTTcRN024qb1yDR9\/8ikqdZX4y2OP4OWXmn9mJTmm8ROisGXbbowcdTcAwEvqjVnJzyHugYdx4ngRcvZ+LHKF5GwYimQ3TKsvenh2R9Adt7fpMzExcdjz72xIpVKkpLyO48c1VqyQ7MXEGOMzS\/NzPhW5EnI2DEWyW22ZeQoYl2SEjhoFADh6lA8kdgU9e\/1J7BLISTEUyancdpun2CWQDVz8yXgP2YP\/\/yYLYyiSw1i79k24desCXaWu0XvaM8bZrKPqRozk+L469iUG+nlg756djd47dqQQwI3LqESWwlAku3D2bCnUx44BAE5+912L+06+P05YsG\/qfVpaqsUHH36Efv0CrF4r2dYrS54XnoZRpa\/E5rS38NEH7+KRvzzlMovbD+7PE7ZLvvtGxEqcH0ORRPXyS8sweNBADAoaAL1eDwD46zPJ6NXTFw88EGs2ceahhx7G2xvT4O3jjUFBA+DWrYvxc5WV+PzAQUyfHi\/Wn0FWIOsfiNde\/zuGDQ\/F7CemY6CfB0YP64N9\/96NdRvfwyur1oldotUlzpiKUUP8MOvxh4XXJkePxX3hw5A4Y6qIlTkvicFgMIhdBDmv2NhY5OTkYMuHuxFxj\/mTDvx7NL0OsbOaavM2oHd3q5yrPU5f+EXsEkT\/HvgdGIn5Pfz4wwXcc9cg+Pv74\/z586LVYa84UiQiIqrDUCQiIqrDUCQiIqrDUCQiIqrDhuAkGj73kIjsDUeKREREdRiKREREdbhOkaxq8ODBOHnyJIYNHwlvb1\/R6rj1FvHvFPzv19\/FLkH070H070AC3OpmB\/9bqBXve6itrcXhwoPo3r07Ll++LFod9oqhSFb1xz\/+ERUVFWKXQUQN3HTTTbh69arYZdgd8f\/JRE7thx9+wOrVq9mom8iOHDt2DMOGDRO7DLvEkSIREVEdTrQhIiKqw1AkIiKqw1AkIiKqw1AkIiKqw1AkIiKqw1AkIiKqw1AkIiKqw1AkIiKqw1AkIiKqw1AkIiKq8\/8q1YIvHMVkZAAAAABJRU5ErkJggg==","width":208}
%---
%[output:1b7a169d]
%   data: {"dataType":"text","outputData":{"text":"Adding matlab path to: D:\\code\\SIM\\code\n","truncated":false}}
%---
%[output:73309dbf]
%   data: {"dataType":"textualVariable","outputData":{"name":"total_iteration","value":"1"}}
%---
%[output:44eeed17]
%   data: {"dataType":"text","outputData":{"text":"Wireless packet type: SC\n","truncated":false}}
%---
%[output:1c38dc4a]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_y","value":"4"}}
%---
%[output:6b8cd335]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_x","value":"15"}}
%---
%[output:0c285bac]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_y","value":"15"}}
%---
%[output:726ab75d]
%   data: {"dataType":"matrix","outputData":{"columns":20,"name":"zeta","rows":1,"type":"double","value":[["0.9800","0.9810","0.9820","0.9830","0.9840","0.9850","0.9860","0.9870","0.9880","0.9890","0.9900","0.9910","0.9920","0.9930","0.9940","0.9950","0.9960","0.9970","0.9980","0.9990"]]}}
%---
%[output:1bcb5afe]
%   data: {"dataType":"textualVariable","outputData":{"name":"T_coh","value":"0.0038"}}
%---
%[output:477a7ee4]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_packets_coh","value":"12"}}
%---
%[output:21114024]
%   data: {"dataType":"textualVariable","outputData":{"name":"T","value":"144"}}
%---
%[output:23397e9b]
%   data: {"dataType":"textualVariable","outputData":{"name":"SNR_dB","value":"36.5005"}}
%---
%[output:180eda8a]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAcUAAAERCAYAAAAKbry9AAAAAXNSR0IArs4c6QAAIABJREFUeF7tnQfYFcXVxwfsohIBQQTFBlhAI9bEGqPGAkYUa9REY29R1KCgaMBekAh2sUSssUVFY4mxxqhRYxT91Cj2ElFsWFDwe36j53VYtsze3b13794zz8MDvO\/ulP\/Mzn\/OmVPaffvtt98aLYqAIqAIKAKKgCJg2ikp6ipQBBQBRUARUAS+Q0BJUVeCIqAIKAKKgCLwPQJKiroUFAFFQBFQBBQBJUVdA4qAIqAIKAKKwOwIqKSoK0IRUAQUAUVAEVBJUdeAIqAIKAKKgCKgkqKuAUVAEVAEFAFFIBQBVZ\/qwlAEFAFFQBFQBFpBffriiy+aSy65xNxxxx3m008\/NZ07dzbbb7+92WWXXUzPnj1rWgTEOnj11VfNxIkTzV133WXeeustM\/fcc5t+\/fqZbbbZxmy33XamQ4cONdWtLykCioAioAg0FoFKSooQ10033WSOPvpo8\/XXX8+BMOR41llnmfXWWy8V+t9884259tprzahRo0LrpbLll1\/ejB8\/3vTp0ydV3fqwIqAIKAKKQOMRqCQpPvXUU+Y3v\/mN+eKLL8zBBx9s\/73QQguZN954w5xxxhnmtttus+R14YUXmqWXXtp7Fu69916z3377Gchx5513NgcccIBZYoklzMyZM81jjz1mTj\/9dPP000+btdZay5x99tmma9eu3nXrg4qAIqAIKAKNR6BypPjVV1+ZY4891lx\/\/fWWDIcPH27Vm1JQox5++OHmnnvuMYcddpg56KCDTLt27RJn4rPPPjNHHHGEVZmG1UsFqGv3339\/M2XKFHP88ceb3XffPbFefUARUAQUAUWgPAhUjhSff\/55S1qzZs0yl156qb3rC5b777\/f7L333mbFFVc0F1xwgVl88cUTZwTC22OPPcyMGTMi60Vte\/LJJ5uLL77YDB482Jx00klmvvnmS6xbH1AEFAFFQBEoBwKVI0XuEpEEuS\/kbm+RRRaZA2nUqBDc66+\/bq688kqz5pprJs4GKtljjjnGkhxEuthii4W+w50jd5mDBg0yp5xyillggQUS69YHFAFFQBFQBMqBQOVI8dRTT7WktcMOO1iDmHnnnXcOpD\/55BOrNn3ooYesZLfjjjvmMhsqKeYCo1aiCCgCikDDEKgUKaLaHDlypLnuuuvMvvvua4YNGxYKLAY4Rx11lLn11lvNgQceaCXLPAruGfvss49BhYtUueeee+ZRrdahCCgCioAiUCcEKkWKvmSH9ejo0aPNFVdcEUueaeaAOseNG2f\/LLPMMmbChAmpLFvTtKXPKgKKgCKgCBSDQGVJMUktKmrWOInSF3LUpli7jhgxwr5y4oknmiFDhnhZtfq2oc8pAoqAIqAIFI+AkmKMmtUHfggRv0fUpbh74BfJH9cNJK6eN9980\/BHiyKgCCgCVUSA6GG1RhBrBB6VJcW4u8K81Ke4fdxwww2WEImc8+tf\/9reTxIowKdAhueOGmWeffZZM33GDDPlww99XtNnFAFFICUC88w1l+kbYTGesip9PCUCuMUdMHJk0xBj05DiSy+9ZN0o3n777TmmhKgy+CT26tWrboY2BAmgzTFjxtgIN3vttZcNBpDGBeOf\/\/ynOWnoUPO73\/3OdO3Rw0x3ggykXHf6eACBRx991Pzxj3+0UYZ69Oih+OSIQFNiO2uW6fDVVzmiUExVHJCvvvrq7\/aECkTEkvEMHzPGrLPOOsWAlnOtlSLF3r17mzPPPNOcc845hbpkEN2Gdi6\/\/HKrJj3yyCMtYfuqTGUOhRS5g+w\/YIAxHTvmPL2tWx3YEvj9qquuapqPsVlmqymxnTnTmGnTSg\/xM888Y20T7J7Qv3\/p+5vUQRmPkmISUgX+ftKkSfZOL2\/nfekyPo6EcLv55pvNPPPMY0444QSbGaN9+\/apR6WkmBoy7xfIZMKhBZV2mvi23g208INNiW2TkCKasNtuvdUMHDTIxlVu9qKkWIIZLCrMG0NDQjzuuONsBo6FF17YquY23XTTmq1MlRSLWzBffvmleeedd0z37t3N\/PPPX1xDLVhzU2LbJKTItcwHU6eazl26VCJEpJJiCTaINAHBMcZBd++j9sTK9Pzzz7dECCGSBWODDTaomRCBSkmxuAXTlBt3cXDkWnNTYqukmOsa8K1MSdEXqYKfg2zwP8RFghRPqFO7detmU0eddtpp5vbbb49MHSX+i6uuuqp1wO\/UqZPtrUig77\/\/vlXJkTZqrrnmihwJmTc6duwYq1ZVUixuITTlxl0cHLnW3GzY3nfffeaRhx4ym625pll99dVzxSLvylRSzBvR9PU1jaFNmqFlSTIcRYoEF8fS1LcESTXsPSVFXzTTP9dsG3f6ETbujWbC9g9\/+IO1AZh37rlN\/x49LCkSG7msRUmx8TNTSVIEVoiRdE+4Tdxxxx1WauzcubPZfvvtrVVilDNpGCm64eN8p0xJ0RepYp5rpo27GASKq7VM2CIFkgpuww03NBtttNFsg+Z3P\/vZz+zPhBT5N6QYlBjJgoPfMYXvHdeqLz7\/3Jh27ey\/P\/\/8c7PGGmtEulzx+yeffLLt2QUXXHC2uvjPCiusYDp06BA6MSQnZ48iC8\/0zz6z7XJNQ71SV9++fSN9oCFTEp2HtSvjiXt\/8uTJhnmVZ9125d\/saVFXTR999JF5+eWXZ+svdZGAnRy0g\/bYo2mswCtLisVtCfnVrJJiflgGayrTxl3cKBtTc1mwFSlQUIAU\/\/73vxshSqxkL7vssjlIkaD9\/HELyQEwzIorXMVAVGEFQsO\/sNb3fdofOHBgpEWqT\/tx79N36ogrce9bq9nbbot8fZWNN1ZSbMzn2lytKikWN19l2biLG2Hjai4Dtq4U6CJBgnEhQvfnSZIihADhuX\/zvvxM\/p2EuhCLWxfvSL1R78vvsTyd9tFHZskll7QJzd0+RBGy1BnV92Bfokg9bLzueGoZ+yOPPGLGjh1rDjvxRCXFJAD192p9WuQaKMPGXeT4Gll3GbANSolJeAgpDho40Bx3\/PFJjzfs93qn2DDo2xpW9WkD50AlxeLAL8PGXdzoGltzI7FFJUowBqRBokj5FKTHpZdc0gxcd121PvUBLMdn1CUjRzBboSolxeJmuZEbd3GjKkfNjcQWCfFHP\/qR9S\/GiAY1qhTIEtIMFu4ZN1p\/\/aYI86aSYuPXuEqKDZwDJcXiwG\/kxl3cqMpRcyOwxbrxL3\/5iyU9N3SfhJwT69MgUSIlYoFu1Hm\/IYtHJcWGwN68jSopFjd3jdi4ixtNOWoWq851113XLLfccnUNoSfkJ1JiHCJBorTPKik2ZBEpKTYE9uZtVEmxuLlTUvTHNs7XT2oJGrasvfbaVnVZj7iytEMaMAiR2MM1BXhXUvRfEDk+qaSYI5itUJWSYnGzrKToh22Ur5\/7dpT7w91332022WQTv4ZqfMqnf15VKyl6wZT3Q0qKeSNa8fqUFIubYCXFZGyjyM4apjjRYaLcHwifhuRWVPHtn1f7SopeMOX9kJJi3ohWvD4lxeImWEkxGVtfsotyfwiSZ1yLPira4Pu+\/Useqd4pemFUwENKigWAWuUqlRSLm10lxWRs05Bd0KqTxNoTJ070ulOsVQWapn+Jo1VJMRGiIh5QUiwC1QrXqaRY3OS2Oin6SmaRLgwhUyNWnRjZ9OjRw\/Tu3buNFKPa81WBRr2fpn+xq0lJsbiPLaZmJcWGwN68jSopFjd3rUyKaSUzITv+\/vGPf2wd48PK5Zdf3uYcv9lmm5nVVlvNkmJcez4q0KT+hrpYpF06SoppEcvleSXFXGBsnUqUFIub61YlRV\/JDOT\/\/e9\/WxKUAjlRIEXcH4LFJUV8Fddbbz3DGpb0TO7zct+YpAJN099Mq0VJMRN8tb6spFgrci36npJicRPfqqToI5kFSVAsSIkYA1G6EWPcGQqTFMk\/ihVqsLiWqXEq0DT9zbRalBQzwVfry0qKtSLXou8pKRY38a1KikmSmSBO2DRxiBd1qfwM6fGXv\/xlrKRIwtnNN9\/cXHPNNaGBuYOWqUiEECCuHq4bh29\/M68UJcXMENZSgZJiLai18DtKisVNfquSIoj6GKcgESIZEh0GyZAid3fuz6IkRbLIQ5zcKaZpL4xwfd7PvFKUFDNDWEsFSoq1oNbC7ygpFjf5rUyKoArpHXbYYfbO8KyzzpoD6Jtvvtk8\/fTTc6hK5V4xzCnfVZ8S+3TIkCFt1qdJxjBCwlFSKP1BcoWQt9lmm1BJNdNqUVLMBF+tLysp1opci76npFjbxPu4G7Q6KYapR8OkvuD9IcTEu2HGNi4pduvWzZCBQmKfQqZREqaQNJJpFClGSapJ0qv3ClJS9IYqzweVFPNEswXqUlJMP8lJ5vtSo5LinHeGgk0cYcYZ29RCiq6EKNauYfeV0regpKqkmP4bKdMbSoplmo0m6IuSYrpJSmO+X0VS9JGQg8TH\/4MSXFaimTBhgplvvvlmU59GSYpCpOQ7vP\/++2MlRZEW3SwYWfvatsJUUkz3seX0tJJiTkC2SjVKiulmOo35ftVI0VdCDiPFYA5Ckd6WXXZZs9tuu6WbBGNMGLZRPo5CilG+j0mNKykmIVTu3ysplnt+Stc7JcV0U5LGfL9WUhRpjJ5JNvd0vcz\/6TQSsrQuZCL\/DxrOQFa9evWaLRuGb8\/TkGKc4Y5Pe0qKPiiV9xklxfLOTSl71uqkmEYdKBPoa77vQ4rB9sMkUfzq8LlrZEkjIfuSYpbx+JJikrGPTx+UFH1QKu8zSorlnZtS9qyVSTGtOtCdQDZKsZKMysSeRIrB9rnHot6wkiZFUhELLY2EHGw\/zpq01r76kmIehCbEGmfZ6jUOvVP0ginvh5QU80a04vW1KinWog4MLgXZcMNM\/Kn\/nnvuMSuvvLIZPHiwjc+JkYeoQ6Mkr6jllkcy3VqkYrc\/vhJyFCnmmQw4LSmSVYPoN3ElSs2a5N\/ovUUoKXpDleeDSop5otkCdbUqKdaiDgwuhygJIkkCjJMIo5ZcVkkxi1QsfeIOEIKQ2KT4CPoUMXSJimfqU0fwGV9SfPLJJ82tt97qdTerpPgdyl999ZX5YOpU07lLF2vh2+xFSbHZZ7DO\/W9VUsyiDnSnKLiRRkmgvtMaRpiQz6WXXupbhX3OlQr5f1wWCZ+K5QAgz6YxAKqFFN32wiTMKNU0qlrIN5hhA6nedbMIG3NWg5xEHFVSTISoiAcqR4rffPON+fTTT823335bBF5m7rnnNgsvvLBp165dIfWXvdJWJEUhDIjRvcOrhXyC92Vp1aLu+pD25b4SiSwYvNpHBerbh7QqWYiKP0nkElzz4oyPw7ybJirp2whzsRC85fDwq1\/9yiy\/\/PJtVUWRIg8k3S\/SHmRKPyHyzHeIwQEqKSZNeSG\/rxwpvvTSSzYC\/ttvv10IYIMGDTKnnHKKWWCBBQqp\/8UXXzSXXHKJueOOOyy5d+7c2Wy\/\/fZml112MT179sytzc8++8xG\/r\/pppvMySefbHbccUevuluNFMNUm8S5ZLP3iXcZtGaUTVpUg1ESqM9kuCrS4D0WZMhGTf1SIEz+uFJbGkk1q0rWZ0xhz4BZVBJh9\/kwA50gKaYh2jhSdOeVuQxm76h1rLO9p6SYC4xpK1FSTIlYUaSIZAtBHX300ebrr7+eo1eQI0GSSZKatdDWddddZ9uiVI0UfaQjHwyjCAPV5GuvvZYY6UTaEIkiamMPGqQE+4YEwh\/6IwVCZr1ICYYmC8sX6NYrbhu+UiLtM27eCyt5YR6sO42LhPTBvYsUUkTiBKM0alxfUmReC1GlKin6fKa5P1NJUhw9erQZNWpUarVNHLoQCR\/dpEmTDPXnLSk+9dRTNljxF198YQ4++GD774UWWsi88cYb5owzzjC33XabVftceOGFmcf1wgsvmN\/+9rdt0nSVSDEPAxGXzMLIRSQuH3WZZFKIyvzAmrr33nsNHyIEQHHJT\/qClAYRI\/mFZWUQUqSOsWPHem0UMo6wMdIeakzak35RaZgPZJ6YBzseli4qanBCYq7lqJDiVlttZb9dNC69e\/f2wkdJ0QsmNbTxg6nQp9p9G3NhiPoUtdGIESNyJ66i6sZ669hjjzXXX3+9JcPhw4fbu0spqFEPP\/xwa7JPap2DDjqo5jtN1KZIiGwQUqpCilGS3TnnnGM3w6AxRdIqjVJtMkcQUzAUWbC+JLII\/h4twOqrr25VccECcYm0w++CEo+QBxiEkWrSWN3fMz5U6\/TPVb+6BC0SYx6uKlJvGAmlCbkW5vIipOgbx9TFIY4Ug78rwrfSqKSYZtnm9mwlJcVmI8Xnn3\/ekuGsWbOsiqpfv35zTDA+a3vvvbdZccUVzQUXXGAWX3zx1IuAs8Sf\/vQnu9mhfqNdpMaqkGKUKlAkIgDr3r272WGHHbwJMqjadOuiPtfSMY0FZxSZbLfdduaGG26YY27DssKLHyMEzZqnQI5Ip2kL6042dtT0jBN\/yTCCdQ1u8nBVCSNFwrkxPrEC9XXPCKoxiyLFIL5KiuqSkfaby\/P5WElxypQpdlM44IADcveZKapu7oaQBJEUxo8fbxZZZJE58EKNigHR66+\/bq688kqz5pprpsb02WeftWrTpZZayqqXkUhJ2loVUoxzmwBT\/M+kiOTgc8cEMbDZQhQYaqBWlCJBo33v5oRQop6nvieeeMI89NBDbW0ErVyD7yJBcsgRdW7SHWXYwgm6djBWiIg1FywuQdfiqhJ1\/yjSV1DapW98ez4lSE7yf6xO+W7cwAngGKcCT7I+dfvjupHwnht4waffoc+opFgzdFlerJykmAWMRr176qmnWukPCQaymnfeeefoyieffGLVpmyWaUhMKpo2bZq14OPuEuLt37+\/JcgqkWIY0QQJ5ZVXXrHjdiWgpFihrrEHJBokRaSzML++sPUkhBJFJldddZWV0P76179ayQ2ycw10oiRMxilERru+JB235ukrpJjkhhIXuSZoFRunUuZZxpykso3rc9CdQ0hRDjMuCQZJMaiqTUOKbrusFyXFRu2m2dtVUsyOYaYaZsyYYUaOHGmtQffdd18zbNiw0PowwDnqqKOstHPggQdaydK3oDY9\/\/zzzemnn272228\/M3ToUAPJVokUfe+2fJ9zsY0jRaQpNtMka0\/qCxJ0kExwi+HAg4r3zjvvtKrQoAtBnIqY+ukPJe4+FAyiYqa64w5Kj9yhUm9Ywl3IBw0NBM4dOIcP+gJ2kr2+Y8eOsUEBGC935kkqW991z3NBUuT7mn\/++W0VomoV1WwwaEAaUuQb+\/jjj61aXg4Cadw\/VFJMM6vFPqukWCy+ibX7kh1BCbB6veKKK2LJM6xBsWzt06ePGTdunL2P\/PDDD5uGFH3M\/X3vtnyfC+IobhUiKW699dZmtdVWiyUgJC0hVFfiY+Nn44RAUOexGVPvOuusY955551YUowjO1dSpGNhEhz9oO1VV121zaCH93xInTqjHPhdQpG7QMhG\/CXFMCesHamTfoFz2J1oVh\/JuDBvQop5uVX4rNfEjYEHVH3qBVPeD7U0KXK6e+yxxwzWiTilU7D+4z5y3XXXNe3bt88b7znqc0kxSS0qatY4iTLYwPvvv28338mTJ9txip9js5BikgWnjNf3bsv3OalXJA3+jxQAmQRVY2mkz7jxuBt3lKQYRnZRLiKQjMQflUg3ouZDCgpTUyYt+CRSFDUlWHEg8CFbITyRsPKIHBQcR1LsU5EqkyyKk\/Dh90qKPiiV95nKkiKWnGxeY8aMsYQwzzzzmJ\/\/\/Of2Xm6llVayYeBwgcB1AynMLbhDHHnkkfY+xXWNKGIaiyRFxgURsrHj+8gfGU9WUtx5551N3379TKdllqnJEtYHS+5PN9100zkevfvuu0ODGPCsa6CCccXFF188x\/u+z+G+An5S8BuFFB9++GF7aJIDxgknnGBOPPHEOdphbR1zzDFtP08aDxv3u+++a7p162ZuueUWq+JEpRoWJm3ixInWcIQxQnj8m4hH\/F8K7dFXivyOn\/FsGCFedNFF5r\/\/\/a+t780337TWzsFCpKUwB37uxCE1IUPe8yFdd44wBMNViEwhWGDzftDa12fdhD3jYis+xhwyKWDMdxGGYS3tCe7uGqmlnnazZhkzbVotr9b1Ha6AbEDwzp3NvE0cEPyDDz4w\/Pnf\/\/5n98zhY8ZY7U0zlFjrUwYA4aFmZLMKEh5xS+Vj5Q6G+4+wwnN86EWDUqT6FOkXqRI3jrPPPtt07dq1bahZSZGKps+YYTYYNMjsvvvuhawbFmaYzx6Sb1h0GEiMEHls7Pglxs0dmz6uDzh6xz2HCwtkSN2UsLFSD4eoYMFoxq07aTwQ2Msvv2wJlzG89957ZrPNNms7dMjv3XZ4tkOHDuauu+6yZPqLX\/zC\/jr4rPu7pH5I\/Rx8Hn300bbmcBfhXjqsHHLIITbARFJh3sTthDs4DqqMgcLY6fdyyy3X9rOk+nx\/jy8wWpPFFluszSqduaWAMSWIoW\/dwecwXuMPByj+1FogxXm\/X3e11lGP9yDFaR99ZBb90Y9CjQTr0Yc82mD93ub4b1eKFMXvj48grHCvwmnx6quvtqcbNtABAwaYmTNnmgcffNCwmRFqLc4a1GcS4uKwLrHEEtYnEX+sIgxtOO2wUYGFqzaVfmclRTa3rj16mIV79ixMUoRosJQNlihJEXUdBxkkK99YrnHziKR27bXXetW3xRZbzGYgwoHrvPPOm616pLsw6Yt7RYynkLLQahB9hU0V4kaSkpi3fLT8nsLP+D2Ewr\/pJ0XG7j4rP5cA3ZBoGIm7uCI1Ub\/ruoK0hWowKL1xyNx\/\/\/19PgnjtiGSGge3tIEVvBpzHuLwySGD+3QxtJH2wXjq1KnWuIl5XGWVVRKrj5MG85IUuVNs932Uo8QONfCBGaSO+uADmzoqzHK+gV1L1bRIimgs4IZKkSIbKWpTTmkQjpzWnnvuOXuRz4Yz11xzWUIkbBrqVLewqNmkevToYSZMmGANH2opPqRIyKkzzzzTEleeLhlIiZB9mgJOjLdTp06Rr9UzILhEVElyCZDOprEW9MFF7rgIr+eqJaPelQDcEic07Lk4H0Leg0whIkK6MR7XaV3uA4Xk+L343QXHLoYv8qwErYYYOdDQD8YnJWgZG7TEjLoLTeP6EWxDxuPrmO8zZ1HPhN0pumNkT5C8jz5ZPeLuDfVOMctMNf7dyt0pTp8+3bouQICc1LG4dIsb95PTMi4KwTRQqF8hVggTFYtYGRY1XdyjcN+Xp\/N+s5OiGLgQMUY2\/CRH+zSbkU92dHkGNSJZV5J8z1zSissaTz\/FAjW4psStgo0ZAnNTJ7n1S8i3WklRCEGsYNGeuMXN5gH+YX6YGNHEGdJgQMM4uJsPOyiIZa7rUJ\/XNxa0JIUUOaSiPkVSRDINOtxDjBgK+ZBiWHqrvKxX2zBQ69O8lkOqeipHiqhM99prL6tWgvSChCd3eP\/4xz8iQ6qBIG4M3B9hfFD0vWK9wry5KyOr+hTDkv4DBhjTsWOqBefzcK0SX5o8fD6kiFEJ1snB0GlRzv4uaUmkm7DxMj6IIswfj7pZc27OP6kjjBTFGT1OUhTScaO9uHe1YX0NPutjReqO1TfXpG9aKJ91EyR15k3CxHFARhvD3bCMNymZcVyGjrC1lnuoNyXFtNOey\/OVI0VRWRIZZuONNw4FibsESBHLRE6OYUXqIUNF0aSYJiA4al0+6qxWsWUmxVo3lzTBo31IkfogrjArSkkf5UZICSPFsI2XzRZSDPPHE0kxzMLTrR+JBslGnPWDG3hYu1GkKFKtu9EjFYuqlX6GhXujr1HY8LtGlrBg66jAuVf0VdcqKfrNIPuXtT7t0iX30Jp+Pcj3qcqSYhyZQYqoF+Puz+pJikypWIpiDYu6DnUqFoPEPD3ttNPM7bffHpk6SvwXfe4EZfmUlRRF4gnzFwvLghCm8ouT0uR5X1Jk0w+T6A499FCrggv2M0jMcdJI8H5RYphGqYnjJNGoZMYuFqLe42eupBhGihLKTn4XFcotLsRbcKtKmr+8trYov1Hcsbp06ZILKYYdwJIkz9TjU0kxNWR5vKCkGGFUUm9SzJJkuEqkKPeCUXdycfc2ae50fEiRDyzK2Z8g7khqFLevQbVa0kYpxjkivcTdW7qkGJR20pCiu3G4ochcwxfJuuGODVLjLpSDAJk0xGLU985XMN98882tK0xYYRxZLVGTsqX4hl+TQPBh96Fh85q74ZCSYh4cl7qOypIi0WHWX3\/9SPUp5uannHKK9fEKFggK03FOlhjaFK0+lfZp98UXX7R3ndxnITViIbv99ttbS1IxzQ\/2t0qkmLSCo1SraTK004YvKfJsmETHHAWlKZ4l5yW+dlHxNIPjc0l0hRVWaAvzJm4D7vN5kmKYcYvUL24YYdJ6EsnHzZ9gHiUJpzGUimsn6iAjkqIPKSZFUoojRZ\/6k9a5\/b2SohdMeT9UWVLEWjCPEnTAzqPOZq6jni4ZYTjldRpPQ4r0g02STRsygSzcFFIiTUmcT56HFLkXTEMiYW4DPmtFxsKzLpFxgICAxIJVJGmeiyNFN1RbMEhCmIFJknRPezxDf+gfeIVZeOZFimEHGfxWBw4caAMEJJGWT9i+sHkViTmpfp85VVL0Rin3B5UUEyBVUpwdoEaTYl5m\/Gnvt0RCxacVdx82PtcYhd+7FppioVpPUgwSnZAiRCZkGXcY8LHeFdzC\/CejDFiSpC5ZYZJpI5guq9Zdz1Xrou3BuI7gHG4AcHDh4MDfQtS1Bo13Dyf0OTM5qqRY69Rnek9JUUkx1QIqCynS6ThfwKRBpXX7EFKUdEouKWLgRA7EYMFPL8wJP6pvrqSIStw3OHUU0dFn+uneD2YlxbD344jfR+oSSTzsUJE0j76\/B1tyVHIPLNKza2zkYp02aLz0QdaU\/N\/XyjVyDEqKvtOb63OVJEUsA4899ljl\/iptAAAgAElEQVR7mR\/0U\/RBj7s9wmmhOsIfr153ij59a\/QzQorbbrut+eyrr0y\/ddcNDRBdZD9dK8qsRhm+\/YwjxaiME2zyQqI+G6SQIiEICVfnQ4pgIXeAIimKGpK\/aRdilkNEnKSdRlJ0pdI4VxgfqcuXOH3nKuw5lxQ5TIEDRkPix+iTdot7ZJ+SRjsQW5+Sog\/cuT9TSVJk8ZKhIMyIxhdBiBFjHXwdlRR\/QA1SPHrvva0R0KdffmlefO89S4pIRfUqbrQVn+gjefRLNjrJAuGqxmqVLOiXa20ppIjfl8RcFT\/EsDFIn1xSRJILOtpL+DjIIKukKKTq+mfG+ZX6YONDnFnnMHhf66PS9bWqDfZNSTHrbDX2\/cqR4pQpU2z6mX322SdzcFo2JsLEFR3mrbFLIF3rnOqP+N4xW0iRGrIkgZUg07494W4I6+DMdzYxDUa5VoSRItWk8dfj+bCNMy0pSh9FGoWwxo4dGzoqiTCT1sDIZ06SXGGSsPEhTp9+xD3jkiIHu7CwdVnWsNu2kmLW2Wrs+5UjxbzgZONdcskl7R8tPyDABnjr92oklxSjks8mYZf2bi+pvjS\/j7N2DG5scZKitCmGIkhkYRFpkjbO4J2UK42FjStIiownLNAA74o0HzQG8VHRxmHq6wqTJHUlEWeaeQ17NnhfGxa2rtY1rJKiRrTJuj6zvp+YTzFrA7xPXj7KnnvumUd1lamDU\/3444+343FJkXtcyCDpji9IRElShi9w1JNmg09Sn0WRYtzdYBoJIcy1JEiKQWtSwQ5MuO9+8sknrTWlSK\/BGK0udm4sUiEyfp8Gs7C5yPNQk0ScvmshiRSvueaa0LB1eUmK4r7DWskU7k7vFLNMec3vVkpSJB8i4dBI0JqlfPLJJzZhKsl5yZaxyCKLZKmuUu+iehq6++6GxKJCihKejIHGSTdBIpK7LtTTW2+9dc04hd1zxVXmY9hRS\/zVWkhR+il3h2Ipys+TSBESdCPghKXaknokCwYkKGmkokgxjbtKnqRY8wLweDF4p5hFMo2TjpMOWx5d\/eERJcVUcOX1cKVIEVBuvfXW0KzstQCGoU49UkfV0rdGvSPWp0QHeXHKFGt9yubNRsGGHkWKUUTkK2HGjVc25r59+5qddtopEZokww5flWCwISFFcX6Pcxlxo9NQj0uKIvkFI78EJUXGTT1CniJ1854rOaM6dceURIq1ONHTD8Zd1hIkRfAAIzBE6k5Sd7vjIkA6hyZwJ9ydFJ\/DVip8lBRTwZXXw5UjRYlZinUkG3T79u3bsOLDIKIFbhqk5onKEk0WbqRF3ifxr0+C2bwmpOz11OqnmEREtY47LIpMUl1Jhh1ZSVHajwtMHkeK8n6tpCik7EqaRZNiEuaN\/n2t0YKC\/Y6TBHNf40qKDVk2lSNF8qUNHTrU5kIkaa8UXCzOP\/98G41k5MiRkSmjeJ46uHTHz3HIkCE1+To2ZDbr0GitpJhERLV0PYuqKqg+Q71IkG+Kr0owaMkpkmJWUoyTtlGXUsLcK0RSVFKcczUJKaL5EfzSrjnSZ4WlypK7yNzXuJJi2inK5fnKkSKosPCJMsIfKaRgGjZsmDnppJO8MmuTZHj\/\/fc3JCdVl4wf1lqtpEgNWe5xgqs9q6pKMj6gCnMJkXZ87yh9SBGilLyErhFS0AoUleZCCy1kE+FGGb+46lNxQHfrDJLidtttZ\/r16zcH0UOaEs3FbStozRoVuDvNnWMuu1QOleRBihymR48ePUdvXKvVPNe4BgTPYeJrqKKSpBiGA5v51VdfbTNjEDEkqYgaFutTtUDNhxRdKSxqw02aF\/l9VlVVkpQpkWLi7prist3TT4gOogkjRRmHa5yTlhSDWLl3isEUVG5fk0hR6o3yA61Vvew7t0U8lwcp+kqCuVnRqqRYxFJIrLNlSPHxxx83F110kRkzZow9kSeVJ554wppT77rrrlbC1PIdAlkkxTwx9N2gwtrMImW6JEbdYcZFURnsw9xV6k2KQTzCJMUkUuT3ebnS5Lkm4uoSUrzxxhvt9YhPIuqw+nKVBJMGr6SYhFAhv28ZUnz33XdtNnusJpFS4gofDeop7piOOeYYlRQdsMpCinSp1g0qi5TpkhiEIumQ3BRLuAVxCAtLLxVcd259iy++eGw+xaD6tBZJUd4Ju7cMM\/6JCqNH3FBUwIxb3D0K2aFyqjQvUqQ7uUmCSWNTUkxCqJDftwwpYmjDXSOS4vDhw61fXFCNOmvWLPPCCy+YM844w4YtwyUDiWT11VcvBPxmrLRMpFjrBlWLlOkSEu0KCYaRoisphiUiduc9DSlGrZdg34KuIK7PZZzrjC8pJqmey7iu8yTFuo1PSbFuULsNtQwpMmgkwBNOOMFcd911ZuGFF7Z+Rqussop123jzzTfNY489Zt566602fLBgPfroo818883XkMkpY6NlI8VaMUorZYYZxiBFhakR5VkfSdHtf61uA0l+hb6kmFQPfc2ieq51rvJ4T7C98847zXvvvdcmxWeN6JNH3yLrUFIsFN6oyluKFAGBDAS33HKLOe2008wHH3wQisvcc89tdtttN5taxuf+sSEz16BGq0KKqMAgC1\/n7aA05iaqZSqiHPXT3L3VagySRGZ5kmIW1XODlqxtVkmxOPTZUz+YOtV07tKlEgJEy5GiLI0vvvjC\/Otf\/7LSIX\/z0XTr1s389Kc\/NT\/\/+c\/NEkssof6JId9RVUgxOLQk8lJS\/A6xWlTPxW3H\/jULKT766KM26lFTWNCqpOg\/wTk+2bKkmCOGLVVVVUkxKdZplPo0afKjyDYsQAAbN65AWEj6qvXcgAFRbi4yNqTZuMAESRKnjDWt6jkJo3r8PizMW9h9cD364t2GkqI3VHk+WDlS\/Oabb8znn39u7wwJ55ZnKbLuPPtZZF1VJUWMTHDkl6S\/qEddH0UfUkwTEDwsQAAb98SJE82VV15pA0z4ZHp3jWOSSDHOOCjtmqmbBWbajkU8r6SYE5Ah1aj6tDhsfWuOTR3FSZvNacSIEV5O+r6N8lyRdafpRyOfrRIpupIRzu5ujj3JPyhYB0mRaDEYbrmFOiAL3u3Vq1fsNH388ceWhJEIJZB2WB+Cd5XBwNvSJo1hOOZGcZIOBPvlZuLgGQlG3sh1VXTbHGiJaQze888\/v21OcCj1+D\/+uGhoMtdPdqKPpk0zCy+ySObE7pk7U0MFPZdYwvTs3r3tzcpJikUSV5F11zCXDXmliqQIkGFJZ938em5IN1x0jjzySMP9lBZFQBFobgSIcX368OFtxFhJUkTthKSIn2Ge5cUXX7SnSxz6fULF5dl2WeqqIilGZayPysQOBrvssos5\/fTTTY8ePcoyNdoPRUARSIkAB1vulq867zyzzvf+6JUkxQMPPNBamPXp0yclRNGPk1SXjN2QoW\/81NwaL1FFVSRFVJ5x2Q+C8AspXnXVVWadddYp0exoVxQBRSANAm3fctVJkRQvb7\/9dhpsvJ8dNGiQkuLQoebEE080\/QcMMKZjR2\/syvSgGIrQJ+708ElFYpRC3NsoQxclxTLNpPZFEagdgZYgxSlTptj7IZIEF1EGDBhgfv\/731fCSbUWfKoiKQZJESMVX4tKJcVaVo6+owiUD4GWIMXywV6tHlWFFMUlQiRFSNG3KCn6IqXPKQLlRkBJsdzz0xS9U1L8Ln0WhjZ6p9gUS1Y7qQhEIqCkqIsjMwKtTopYqv3f\/\/2fIbC0kmLm5VT6CvBFRa3eu3fvlr0yKf0kZeigkmIG8Or1Km4fl1xyibnjjjvMp59+ajp37my23357K5n07Nmz5m7gvPzwww+bCRMmWP+6r7\/+2roTkAuPbCCLLbaYV91VI0XfcGqScom\/yc1ZBVJkwydjDOviwQcfNKwRSt++fW0AAtbdsssu67UuqvQQgRXAhW\/w2WefNSuvvLL9bjp16lSlYepYvk+abrU+VbY+bdaZJt8jSY1JVQVhBQvkeNZZZ5n11lsv9RCnTZtmU2ZRf1gh+PmZZ55pcGJNKq1MikiJlKqQosz19OnTzVFHHWUmTZpkunfvbq1us7ozQbCs6XnmmSdpSZXy9\/Sd6EFDhw61UYLqSYq0jQuYpqwrfmmopFg8xjW38NRTTxncAMjecfDBB9t\/k7bqjTfesEmPb7vtNrP88subCy+80MbF9C3Ud+yxx9og05AflrkbbLCBIT3W008\/bUaNGmX\/XnPNNc348eMTJUYlxeqRIgQ2evRoc8UVV+RGALfffruNQTxkyBDfpVq652TDrDcpoq4lITpRk1o1SEi9FoOSYr2QTtkOAXUhruuvv96S4fDhwy1pSUGNevjhh5t77rnH+tMddNBB3sHOiXu59957m0UXXdQSajA2JqHrxK8TSQg\/zLiipFg9UmS+Tz31VHPBBRfkQopcAey\/\/\/5mn332MTvuuGPKr6E8jzeCFFFpE++Wg0orBwmp1ypQUqwX0inbef755y0Zzpo1y6qu+vXrN0cNQm4rrrii3bwWX3zxxFZcso0iU5ESONkTGPmAAw4w7du3j6y7KqSYCF7IA5IGqmrq0zxJ8a233rIHOPKUnnzyyUqKKRYa3yvamnPOOcceTpUUU4BX46NKijUCV\/Rr3PWxkXBfyEexyCKLzNEkalQkutdff92mFkLdmVTkHQwHosg2qY7g75UUW0tS5D4awyzU76Rj22+\/\/ez9GkYopGPCSAt1P2uWwwLqeAx30G5gyNWlSxfjBrzg\/pJwiVjuEmiDO8fBgwfbOtw4su+\/\/769A6cuDiOoE0mrtdlmm1myJS0cVw6ofDfddFN7F4omhL5ikParX\/3K7LzzzrOpH7mrQ4rlHQ6DCy64oHnggQfMaqutZg+EGNS46ebiJEWJg8z9P9cc1MO1BONAK0PhXpA4mhw4H3nkESuNP\/744\/b7ZeyECOQ6g\/tb6uGwe+211xoOFmDNNQmGYDyzzDLLpP1U9XkPBJQUPUBqxCOiutphhx3spjLvvPPO0Q0i9KA2feihh7xP4DLhZHVgw5CPNcsYlRRbhxSRXFCvs1ljhNO1a1drmYo6HmtlAuM\/8cQT9u4LsoRQuMPGaOfWW2+dY53+73\/\/M4ceeqhZf\/31rWYEQxKseIkQRd3nnXeevfeGMCFd7rr5PxawEM\/f\/vY3qyEhxOC9995rbrjhBku+EAb9QVX7wgsv2AMgJAMpjhw50rYjhmwQqmuwRp8OOeQQS7D8DoIWYowiRbn\/x3Kbaw+5n99rr70saUubEB+ZVpD46MNSSy1l+0SSZhILXHzxxfYgzLWFfJuQIsZ2Kilm2aX831VS9Meqbk9ymuQjwgR83333NcOGDQtt291sCHiOZJlUgh8Yp+\/zzz\/f3H333eaDDz6wmwkbCcYQvubmVSJFSdBL7FOfiDZx6lPqSiqSNzHsuaLfj2s7SX0q64gNHCtlcd+BKJGMNtlkEzNmzBhLXFGkiGTG4Q+i4FnRhnCHhoUn9+XuoZDDH1IohMOahUgwQEHSXGmllQyELeQLIVOH3MNDmPyMgoSJRCZE9otf\/MIaFbmWnbhd\/Pa3v7WSL4HhkRwpUaQoh1hyamLVTV0ffvihrQOShpSXXHJJW4fc2dPfcePGtQWQj9LiKCkmfUX5\/l5JMV88c6nNl+xcC8E48nQ7hSqWDYgTLPeUnGr5aIMFq1ae9THDV1I0dmPDkMR13hfCjFsUwYTB7rNFvx\/Xti8pBqWXMNKIIkWXBFx\/W+7RITvWpWvlmWTkEieRukTLARLCJM0X5HrSSSfZDDpucZ+H2DiYQrBRfYDYORBsvPHGlqApQoocPCFFnP1dUuTf7s\/leaRhdx0pKeayrXpXUnlS5DT2n\/\/8x54iiyzo+7l\/cC1Ea20v7uMO1iknVF9SlOdRQbHprLHGGuaII46wKjA2Iz5ITrr8vdZaa5mzzz7bqrHiipAiaqC+\/fqZTsss42X0Uys+Rb6Hm8vkyZPNVlttFWrcFNX2v\/71LyvFuJsZSYiTSpw0WvT7SZJwnPVp1EadhhTlWd+1m4UUmQckWjFYQR2Jijbu6kGed+\/1k\/pAO5AgKuD33nuvzQ9YSTHpSyjP72WOL\/vjH83yvXpZDRoqdVTaw8eMaZrUcO2+5YIgpLinryJhz1PfXw9SBIso0hPzeckuwmbvQ4o8M33GDLPBoEGWIJqxYMTx8ssvm3XXXdcst9xy3kPgEIH0UaUwb\/UiRe7skNaSHNOTCCnpu3GJHELkPp55C5MUmXh53ldaxRiGjRNDIrQG3GEiZaqk6P0ZleJBWWfjRo82b736qrlt0qS2flWCFCVsFdaWRRbuC7beeutQg5i07RapPpWNjj5xqY+6J1g4X6A6xfjAvRuKGodIir\/73e9M1x49zMI9ezalpEi4NgxIKGkkReKe8u6ee+6ppPh9YHSXSKLISjYfJFaMviCTYOF+nYKhWV6kiPoU6VTuLblKQHJ0rUxdUnS\/gag+IHHi4kQYPLnLVPVp2p2vHM\/LHF8xfrxZpmdPKylyx3z11VdXQ1IsB8w\/9MJ1jA\/2DZUmapZevXoVZmiD6TuBAKQtuecI9kWMGlZYYQVrbo95exIpNnuSYU0dNfsMFy0poonA4vSdd96xd3vBAxp35vz8Jz\/5icFaOgspctDDkhSjmYsuusi6SsidIvfBqFWDltiiPoXkkCopYX1wrcBdTYGSYtl2X7\/+VP5O0Q+G+j3lQ4oQlXyQebtkiHVgEinKwkh6TjaKk4YOtWbx\/QcMMKZjx\/oBmmNLSoo\/gJkU5q3WO0VXVekGksDqGe0EgSgotA\/B4E4h7gxZSBEVJsSGilbcHeRbRIuE64cbQ1gMbVClI8WKKj2sD+43DYnjgkGh76hPKXncKaK9wJWjQ4cOOa56rSqIgJKix5rgrgCHZazK4iK7eFTl\/YiQV97O+5ihc983\/\/zzxzrvY8KOWsknxmNVrE+VFH9Ynq71JYHng4EeyNqCQVZwo5Z1079\/f6ueF1cNsXrmHhurWnwWN9xwQ9OxY0d7\/yZO+5tvvrnVYqCpYPN3Db2StBeumhbJ8rTTTrMuRhIVBod5yE8sqpEeMazCtxIylrb4OeEVeR9CHjhwYJtqNWx87A2oZPkO0PLg40iMV8aIsc1rr71mf0ZgAKxcCbZB0A1Uwm6gdSRmSBR1PG4jQtLyzbL3cKXBONmT+D7rtR95b1wVeFBJMWES+aD4aDA+cT9yXuNDJnQVH03ep7eiwrxNnTrVxp\/EspGoGGEGMWwKolryMSBqZVIUIq1KmLeo1FFs9hAWzun\/\/e9\/rSry7bfftlFWcNxHo8E3goYD4xWssIkIQwoeiIlNHLU96aiIvITFM5bP3ONBEueee25bajSi2FAf7+O7yDfIIZHvD8KgQMb40qL6FAMdlxTxF4SQkBAhKAgG\/8mwUIjSPhbEqFXBgKgxu+66q3Wup6Aiveuuu2wKN\/rA+LAhwOKagyPfK5jwLXCIgPzoP0TM4QFJkyAcBPOG2HDgp4And9HsH9xn45tMob+EVwSjmTNnWila7l05RHB3mWSYVAF+asgQlBQ9YCeSBidESCRIfs8995wN0cRHnkd0GOlOmoDgnFIxcvFxB4HwUPFAemxW7slZ2ha1D2bIY8eONVtuuWUsSlUjRd98ioBSNVL0+BxK+0iS9WlpO64dKxUCSooe0wGREE4K6YpTHZFeuPN45ZVX7OmRi3tIid\/lWZgcrOTwJ+REykm3W7duNnUUqh1OoVGpo+KMJCSMFVIup1xCcqHWotAmJ1qkAU7jnH6JShJXWpkUUQXed999VgIhxmaVXDLyXMv1qEtJsR4oV78NJUWPOYYUkQhQeaD6QGXEBbwk\/sWkHCMTMRLwqNLrkSxJhpPS\/kCsROlgAYQV1DcQohuQOarTrUqKECLaAwp3tFjoKil6Le1CHlJSLATWlqtUSTFhypEISeiLPp+CipJI9ejzIRUu9ItM+ilR\/LmQR42L1MidBXcK3Ne44bHcoSSRIs8SMxJpE\/8b\/HAohH5DKkVl6ntPWhVSZPxYJlKQ\/OMK0iGxN6UoKTZ+72Q9E\/uU+8coh\/zG91J7UHYElBQ9SJFgwRgEYFCA+hJixMgG3yYu41FBBh1\/yz7xefavSqToi4srJaqk6Itacc\/xPeKDyAGPgywWrBi7\/PznP89dg1PcKLTmMiCgpOgxC5w8yfEm\/kfyCuoajFb4ALEia1VrsFYkRTZgzOpVUvT4gPQRRaCJEFBS9JgszLGxBhWfK\/cV8S3CTBr3hVYsrUiKzDPqU9SoKim24qrXMVcVASVFj5lFHcNdHndswWS\/ZM0mozcqVvIQtmKpEimmzadIiiOSw3KniJ+ZGtq04hegY64SAkqKCbNJQGKcbydOnGjvJ3DCJSkp94hEk8AQhgt+olW0anSJKpIijtcYVPmWtg\/pqquaJrWM79j0OUWglRBQUkyYbUgPVwziI2KRyd0i\/n3kSyOihZBjKy2a4FiVFH8IFK2SYit\/CTr2KiCgpFjjLHLHeMstt9jwTjjtR7lG1Fh9U71WJVJEFYpKVCXFplqC2llFIDcElBRTQonRDaHdAO6RRx6xUW64T0LFinVqK1qgVoUU3YDgSoopPwx9XBGoCAJKih4TiZHNX\/\/6Vxu5fvLkyfYNotoQjJgEpKhQCSqMWhVfxjArVY9mmvYRJUVVnzbt4m2ijmPfQPhFtFIEStdSDAJKigm4oiYluv9NN91k+vbta9Vq66+\/fpsTv\/s6odOI5E8WCp\/waMVMaf1rVVKsNim++eabNpUScX4lSwUBLPgeMD7DNQVfXozQXAtsMlQQCermm2+22ct5h2eIAkWYRHx7eT4ux2jcao7LNVr\/r6CYFtl\/2HuIPEVsXQ7cbm7GYlpt7VqVFD3mn5BpLEayhCdZmJIiB9P8ww47LNesGR7dbNgjVSFFACRSDUXVp8b65nLHSjooJBNC3xEknmwws2bNskl0iQV8zz33WMyIlSukKEHnIUaC1w8YMMB8+eWX1q+TOMGknSK7y+GHH95GihiyEfQeLQxFEhm7ia4xfCN6DSEW+TlJd4sMs9iwjyrQMFooci2Sf1JJsdhZUVL0wJeTLidcEqcmFQAls8XRRx9tw0y1QlFSrJ6kiG\/uuHHj7J+oFGOsbUngS8hD1j1kReGqgWDp5CINZo8h7yLBLpAy+U5QCXInz+HTzXcYRoryPZHAFzJuFVIUaZrxKykWu6sqKeaMLyoONgjSMfmQaM7NN6S6KpLicccdlwrLqvkp3njjjTa4NoVQhhtvvHEkHiTlhdy464IU3SuHqMDcLqlx7fDnP\/\/ZfjNucIw4UkRahBxotxUkRSXFVJ9jpodbnhTJichJlY\/eJ0lvJrQr+rKSYrUkxXfffddKfc8884y1qCZLTFJOTZJwE90pSIrcHyJtBjPeEx4RtezIkSMjjUbiSLGin1LksJQU6zfjLUmKnGRxwud+EACIrN8qapgilpaSYvGkyF3c\/fffbzbccEOz0UYbFTGNbXViNMPdHiVM\/RnWOOpW7gyFPEV9yrNYZ6NKXWWVVVJlk8lKivQJX2KuP1ZeeWV7j8ldKFchqG\/FfQo3q2uuuca6V\/Xq1cs89dRTVmJ1tT3UhcER6ds4AHAviv3AmDFjrEU696Dcue6+++5m2rRpVn1MuzyDlTrq4WWXXdZChxUpBw6k4y5dutigIOeee6558MEHbbtcu6By5u5WShwpMqaHH37Y1vHEE0\/YLCFrrrmmncN11123DXNfPApdXE1QeUuSIqohTqm4U7DwUMXEkSLqnQ8\/\/DD1R90E859LF5UUiyXFYJoqSJEoS0UVycVJ\/WzuJJxOW1CpooLGcpKCFoY8nQcddJC3y1JWUuQ7hyiwCCdkH0Y6jOfJJ59s+94hMMhs6tSpZsKECTZRNCQ5dOhQ62aF9SzvYlSEJAw2+CWvtNJKNiXVL3\/5SxsXGTUu+wQGWvyNUQyGQNRJRhWRuMmsAwEzNkiaZzDiGzJkiCVTDJuw1AVzcnsKMUaRotz9EkQEvLF6Zz+j\/5A15E3dpLbzwSPtPFfx+ZYkRT6OBx54wBxyyCF2QfPBklwXyzrcLTjRuapUFt51111nUPkccMABLXGHkWaxV4kU04zbfbaoO8VgMmNpE1IsQmJ0s9fTVpawdZDL2LFjbdxgviEKCbJ\/\/\/vf2xCJSYEuspCi3GtCFkJ2tM83fMkll1jS4C5S5g1jIsiTvylyMED1y75A4WAM2T399NOWbImBDNmwn5x++un27jWoLiY6Eu8wfsgxWD91Y7UrEjbJvnkeq13XmjeKFNEegCfkjUQuhTGKdMrda9euXa1rWRIeta7\/Kr3XkqTIBLKQ7777bntylQ9WJhbVKmqqTTfd1Ko+OH2x+G+77TarwkIVlHTHUqVFkjQWJcXiJMWglChzwRpMawyUNI\/8Pk9SlO8MYmCDF9cNfj548GCrmnVVhMH+ZSFFvmky11xxxRVWpQkJ4lbCz5HkllxySXvwRaJFgkJa43uXw7CQoktMLikGDwvSV9LHuVqnqHekftdil\/G7\/QYjDJU4PISRImpY7mSRxpdffvnZjJSQfFHdysFmjTXW8MLDZ41U\/ZmWJUVZgCy61157zZ4G0csjQaK+cAv3BZiPc8rjRMY7rZo7MeyDUFIsjhSDyYyLlhQ5LEIESE0U\/o6zPPXdILn3ghRHjBjR9n1tu+22dqOOsh7NQor0S1w\/UCdyV3jssceaDTbYINKgjgMBoRu5U0QafOihh2aT1upBivQbksN\/E8kPKbdTp06hpCj94R15Lm4+0uLhO7dVe66lSZHJ5AO48sor2z5OPl6s7wCG0G5Ihl9\/\/fVs8x483VVtUaQdT5VIsYwBwd1kxswNQSRQiRVVhIyo39fQxrcvqO\/QzkBUSGXc8aGNCStZSZE6MWhBmoLkKLTFPZuoMfkZZIjkhzsVbij8joADBOGot6RIf2RT9iVFyP4VVHwAACAASURBVNFVzcbNhQ8evnNZ1edanhRRV3ChvtRSS4WeIMXqDIszrMNQt2C5lsfpuSqLSkmxOElR1ogkM66H9akbds3XJcNdy1iv8k3tt99+odam+Chi3EJxSSf4PeRBitQpGW0gOrRAbjACIu8cccQRlhhd15FGqU9dUkxSn4qkiEFN1OGCQz5jI0G6lDg8qrInZRlHy5NiFvD03e8QUFIsnhTrudY4CBKAAutHpLkk5336xjtYU3JX969\/\/csaeWCMEhYDWDYd3otTz2YhRTZ+jE0IO4f6kYJ1J1IjpM0dIxLr+PHjrVtFUPvTSFLEOhWjGO6NuQ+lhN0pukES0B7wTtDXGhUwxjt77LGHFx71XGdlbUtJsawz00T9qhIp\/uUvf7H+aq0e+1QkKDZVJCsc+FdbbbXQVYmVKcRCXE5841CNsgmvvfbaNpZs0ChNJEUiPqGiDDr2SyNiQenGPvX9LJCOuEPEDQSLUCmMRwxvsPJEXUp\/dt11V0uYYnyDiwXkWW\/1qRg6oWYWd5AoUuTngiX95n52++23b4vPzNUQrmccULp165aIB4cELc4B97zzzDrfrx3UztyHDx8zJlLdXzbs2n2LhYCWhiBQJVLUgOA\/LCEkK1wqCG6B9TVEh18ezu9YaKOKxDUEYuN3SGVszq5BCr7AEA++fhScy\/k\/1pFIo1E+kK4VJu+l9ZcUckF9yB0ilqfip0hdEA6uV0iKBDVnPGS3wckfh39cN7iHpP\/cQ6611lqWWPbaay97Txm0PhXpjuc5IMhBANcKeccdg0iicn9JwHT6h0M\/BxD65GIj0h7O\/dwn9+nTx+IZ9AfFaZ++4lcJKUo9Pngwr1pagBT5KD7++GO74IosbAZYqbJ5tFpRUqyW+tRdv3w3uDDgtsCdOiER8e3F3xDfXqKxYJktKkrenT59unUX4R7rvffes6o\/pEcK7+H6wH0j9\/jBgpsBpISB25133tnmLoVTPX6BW2yxhY3WklRQLULo\/E3\/e\/fubQ3oiLqDRITzPQXi5y4RYzsKY+H37Bv0kbFy\/wnRoEXAEAvChrCQNCFRiXdMWi32AcgV63QM9DCAYTzuO5AsfeMwAYnyDrjOnDnTpuPC8lQIiv6DBURIOxQMr5B2qQd3DfCGlCFqcGYf4u6ZcQh5+uKRhGsr\/L7y6lP31FrkhAb9k4psq2x1KylWlxTLttaq0p8oP8WqjK+Zx1F5UpTLdeIWFllQfxBZIilKR5F9aFTdSopKio1ae83arpJieWeu8qRYXuir0zMlRSXF6qzm+oxESbE+ONfSipJiLajpO7MhUEVSTBtCrajYp7rUqoeAGzUIIxwsXVvRFqGsM6ukWNaZaaJ+KSmqpNhEy7WhXcUaFStU\/mDEg1EMVrsYERHBRnO6NnR6bONKio2fg8QeELMQny1yuYnlH\/5Iu+yyi812XmvhnhWLNXydCB6MWTq+ZVi2kS3E935USVFJsdY1qO8pAmVDQEmxbDPi9Ac1C8GBUa8E46\/yGObtQX8m3+HgxHzYYYfNEfxc3t9hhx1s3EufbCBKikqKvutOn1MEyo6AkmKJZwjnW8I34XhLpBD+DUnhd4WDL6msSBnjRr7wGQ5xNPGlwjcKnyfID+dqfLjwqSILiPhnHXjggYkqnSqRog9+Yc\/onWKtyOl7ikC5EFBSLNd8tPUGZ1vCVF1\/\/fWWDINxDSEtnHxJx4PEh6Ou72W9hM8KJkSlcaRTEiojnRLxA6dh4lnGFSVFlRRL+hlptxSB1AgoKaaGrD4vEPsQMiSyBsRE9JBgIdLF3nvvbVZcccXYGJLue5KYFOKLSoFFVAzaRiL1ybyupKikWJ+vQltRBIpHQEmxeIxrakESjRJOiviMxG4MFkiLmJOk6SFMlU\/4K5cUo8zBpd5XXnmlJUmR+Ke5BASfObOmua\/bS3PNVbemtCFFoFkQUFIs6UyJcy8GL6NGjTIEAg4WshOgNg1mCE8aEubgpKUJU5\/yrkTeb1X1aW6kOHVq0lQ09vddujS2fW1dESghAkqKJZwUHxUn3ZbI95AYBjHcMfoU0gIdcsgh5rHHHpvN0AYL17\/97W82hQ5B1Em+ino26a6yaupTJUWfVaTPKALVREBJsYTz6kt2bnqdqPvBqOG52QGC7h69evWycVxxKG7fvn0iQlUjRZLrkiJJ0h0lAuA6\/F511Q\/51lRS9IFOn6kDAuwp+CWTqqqsAQIw8sMynqwfZDHx2XuKgE5JsQhUM9bpkqKb5DSs2lpiKLL4Hn74YZuAlJxxwSK55ZASw+4yg88LKZLQtW+\/fqbTMstEJo7NCE3hr4MnhSgjaUiRbPMEPZjNMKkJSZH0SuT0QyX\/+OOPt+HNfTVplQgYIb6r5PJjvASVYB2x2RL0gcASm2++eeFzpQ3EI8D1Ct859gkY5TEnp5xyillggQVKCZ0YF3Jgv+iii2z6q0YUIcXL\/vhHs3yvXtaXG+0ah2VNMtyIGQmoRfMmRTcgAP+GyPBZJLs5EiM528jgjQ8jRj74Q3bt2jUWCSFFHpo+Y4bZYNAgSxDNWMgbyGa\/2WabpSJ2SGHo0KFNT4oyZyTZJV8gOQLJ20dC4EUXXTR0SomGhNEWxCnZ65tx7qvYZ7RJGOqdffbZNsdjmUmREHhc2bD2yDWJVNuIIqQ4bvRo89arr5rbJk1q64aSYiNmJMVdYS3qU1QokCCbHQEB+BNUpxBWbv\/997cJS4888kibbDXuXlFIkWSsXXv0MAv37JmKUBoEc2izWPGC0bBhw1J1Cwz23HPPypCim4c0aTMVzcZPf\/pTK2FrKRcC1157rfU7TprHcvW6cb0RUrxi\/HizTM+eVlJ89tlnzdVXX62SYhHTQhZy3CjefvvtOapHWsMnkTs9TtxxvoS87Hv36DY0adIkS4ScwiZMmBCpInQtVImYEyUlUHfV7hRrmffQiDZNqD6VsSsp1rIKyvmOkmK6edE7xXR4ZX7ahxR79+5t7\/tQWeXtkiEfyFZbbWVVKR06dAgdkywMIWr6FFWUFCOc95UUM38vWkF2BJqNFAlYwlWOb0KC7AjNXoOSYt6I5lSfSHR5O+\/LB7LJJpuYMWPGRAb8VlJMN5EqKR5lRH3KXfV7771nHnzwQcMd7WqrrWYGDhxo77Zw+Zl\/\/vmtS9Cuu+5qNz6eR0XFffbEiRPNdtttZ9Zaay1z+umnm7vuusv+HktotCcYft15553msssusyqtxRZbzIZA3HLLLWdT8WPBeM0111h1NtcAGI8NHjzYakl69OjRNrloW\/jWqJOD3wsvvGCQkg899FCzwQYbzFbnM888Y8aNG2fv2KkPS0kOjdynkgKKuMEYYxCCUWwBuCfjm6MvaIdcFSabP+1dfvnlpkuXLmbrrbe2oR2JeUw84t122832c\/Lkyebcc8+12EEWffv2NQcccIDZYost5rj6IJDH+eefbwgT2alTJ4PhFHjfcMMNXupTrmQwGiMU5Kabbmq1VlwlgAkHdfYNX2zpO2PEyAfNU\/fu3e2\/ic7F2FgvroUpWBHPGQ0ZEbWCqniuNcCRO+zFF1\/cPPDAA3YusV9w68JQh7XE2mNeuP5BK4ZBGPGdeZ51EGVEqKSYbu+r29NFhXm799577UecpD4V8sTBv5XUp\/fdd5\/9cLF822ijjbznW0nxB1LE6vHll1+2geWfeOIJSxwQI25Dn3\/+ub3fImrS2LFjLZlNnTrVPPnkk+a8886zVqxYsKLBwAiM1Ggc3iAuIT6C2GP8g58tyaDZ9CFJwh1SsBSE1KiHzRXihfRwM4LQaKdPnz6WOAiMAfFKKEVIgZ\/95S9\/sXXSbwrj4S6eeiE2CqQMYTAOCBWyOOqoo2xfgwZybOaQt5AiBAlR3H777fZA8OMf\/9hu4JAeawkCP\/HEE20gDe6y+PcKK6xgsYI46XPQJgCLYdrnWSF0CIQ+Mw9Jd4oQCZaq\/KEf4MeBBJJnXvCFhsx8sAUf3iV2M4ch9hDInG+LgCNgJYWDC+sCMuZO3z1UyDOMDSMcfLGHDBliDysYxDH\/4qdN\/2bOnGnQxmGowzww37SLJT0HKA4bYBFnK6Gk6L3t1ffBNAHBWQwYufj4H3FyZHPi1MsJCovJ4Hs8wwfH4vEJNl4V9SlO+0T6kQIp\/v3vf\/eaeCXFH0hRABN3ISQ\/iEZcAdgkITo2WDZ4WX\/yPGSIVCjqM7DldM\/\/eVdM9d1vBCkGIoHUqAfpkDZEGmADZa0TQF+uJF577TV7v4+EwsbMAZAiB0KeZwOnSNjFiy++2Gy88cb2Z7SFZIiEBynGuVKFqTAhDciTOiF0iAOJ8bnnnrObOYcLSB1rcKyhpcjBloMDxApxS+YbyJXnXdWjJABIIkXqd4kdqRBckebo01JLLWX76oMtEbjkqghcxerVnTMICuM0KVH4ydjAJmgFDelzWKEtOWS5c4imjTFAiBTRwMVpypQUvba8xjzE5EBgnJzYKCCqbt262VP2aaedZk+ZUamjZIMhmzeqA1QpUlArcKKk4FNGG2TC4JTFifCEE06wf8uH6qqbwpCoAikiISKBBAuk6CMxKilGk2IwsETUHVeUz61srswNEp17vx18R+L2EpHJTcCNGo\/NlW9JvgkIEymDCFIQm5C29M\/ttxAR79ImkiaFTRkJj7rSkiLvx\/kZiyvFcsstZ6VhKZAlxEQRiVQM4+Rw4K7jNHeKcWNIgy37jRwkgmQs\/eEQgzZBCDyqbXk+LFaza4HPwQEXMvxoo8Ys32nYviiYKSk2hu+8Ws2SZDiOFFlILBpO72HJi+kcZMtHKR9\/XIerQIpBKVHGi+SIiiapKCmWgxRlHtJGeJK7NFSIkCfqU7cOUdWx0XOfyAGVoPEQYpKk40ouQYKIIkUhCLQ1wYNAcC0ifaGapW9hWW3yIsW02EbZL0g9QSPCMFL0CXkp5OvGalZSNKbdtzBIBQvDwm+QD4OLYk66nTt3thIe0UXc07A7\/DhS5DkJqYRhA\/cTnHhRY3EJjlTK\/U2UZWoQ5iqQIvdHqNJUUvwOAVfdmKR2i\/JTjNrwi5QUZcMNSiFRWwNkyHfFvR1XBaj6iOjDvWeQWDHgQBLhWQrfIfd3qOJQMeYpKUpd9C0pA47rPlMPUvTFFm0Tau\/gnS\/3g\/ycA6cb5CMMP\/dnUVl9wowClRQrTIrNwPNVIEVwRn2KGlUK9zkcRnxK1SRFV\/pIslSWeyisBbnDkdJIUuRuTe7ogvOH9EHhb64K2KS5YsAgw5XqwqRNDpP\/+c9\/rNqPMHgcJCFGjD+wbORaApVs0NAm7UHAJ9QjfcFAhbki+hAkVKT6VNa4D7bcKboaKVSbI0aMsKpS7jz5PXeKEjIQ3MPG7KpHo9zIpF8YIjGPWLkqKSop+uzbhT1TFVIEIFRnmMi3uvUpWIhBjLvZhC0iMMPQC3Jwk2E3ghQlQTahwnBNEKMY6TebLD\/\/yU9+Yt0VUING3Xm5pMj9PcYakrOUepDKIFUsYcEKtWpepEh\/xVdZDF5cVS2\/R4OEdSfGQKj\/USMGjZdckk+S+KOISbBLg60YLYETB0usbblnRGrEMpZ74WDA76Q7xSiLeVGfuupYJUUlxcIIz6fiKpGiz3jDnqmapMgYxS0I60wkHzadYLg\/Nj1891C\/IwG4waYbQYqudSObKGQlrhpCZPgGYt2KtSKm+xCnWKryDP+GOF1SZJPF1QMCEgzABZVex44dbTtIQXK351ptUyeSJSp63ztF8Oc+EW0FkjgW5rQtRikYvaDixUUB61PJgQpxIy25h5M01qdxEmoabOkn48ZaFJ\/CsLUT\/I6i2gZnxo5kHpTARZK8+eab7VxyQIk7CKihTa07nL6XCgElxepFtGEBBIPHYz6PJIJ6is0ISYVA09w\/QoBYMLsSGSR5xRVXWOJgAxcykU3aVYe57glBtSXtcN+LulP8CaV\/4tLgtuHG7kV6IzsEvpKoSbknp8+4PAiRoALljh71OffrEA4GN1go4k7Cu2zMkD8qWdIaUfBTpK8Q07bbbmt\/JnUi1eEbh\/M7P0NCwjGdnyNVY+3KnaT0P0zCkwMH7VJQ72IJTWAErKIh4KDPHlIT\/UVVibTGc9yD4gtJ24yHtGiMLay4LhlBlwmRTiUuchy2PCuuNAReYK5RlYI1BjG8S39cdzD3HjvYNvMBzpCt+JnSBnPKAQS3DNaI1Be2xnherIj79+9v3UvEVcPFQq1PU23\/+nCUlHTS0KFWfdZ\/wABjOnZsOaCqKCnKJBJJBFUhGyxO7GzWuOlgoYxRFk7eroSIywAnd9R\/GIaxSXPSRw3I5gYRoE4T4uC+CUd8pDN+zvP40PJzCApDE+qjQFzcn0E2EACqbjZ8eQcCxJ+NyC5EgRHjNPqLtILFqPguIpkgvdEu\/4ZEkMhIGQZxQXo8T\/Qdoqf84x\/\/MB999JGtX3z3fvWrX80WWQZpiv6yceMWguTCZk79Z511Vlv0FTDEjYI\/YAQWGJ1g4CaSLeOlPgIPkEqJyDaQCfe2kABRf1zJPRhpBtLlIMNBAByIIITqMqiGlXlGPQoeGBLRPzClT0TOcQ88PthSJ\/VxaCDjTlhhDnEtQyWNVgJXMeZT2mYNyHzyPgcT8CJyEBiABfgwByuvvLLFgv\/jixpcY8w9hyVJlwd5MrcYKwazcSgpttz2nf+AVVKspqSY\/0rRGlsJAQgK63bIlYQCkB13vRyyKISzQzqMMohqFFZKio1CvkLtKikqKVZoOetQckAAyZjYtRTUvGHRtpBs0S6h9oxLOJBDd1JVoaSYCi59WNWn4WsgVH365ZflXjBOlJRyd1R712wIEOOUO1UChOy0006h3cc4CymROKSua0ajx6qk2OgZqED7KilGSIoVmFsdgiJQCwJCihg0YfTDvTPEJxlRHnnkERs1CCkSw5syFSXFMs1Gk\/ZFSVFJsUmXrna7IAS4PyQFFFbJGBK5BetlDICQIH2jZhXUzdBqlRTriXZF21JSVFKs6NLWYWVEADcagjvg48m9IsHNsYoNOu5nbCbX15UUc4WzNStTUvyBFDHlX3vttVtzIeioFYEKIMBdJ\/ecV513nlnn+3RipNpDDTx8zJi2AAFlH2plA4KXHXj6p6RorJk5HxJ+eFoUAUWguRHgYHv68OGmZ\/fudiBKis09n3XvvZLid5BDjOKDldckQLIkssW8PSmvZV5ttko9TYntrFnGBO7oyjhfBD4gCACaE4xtmq30XGKJNkJUUmy22StBf4UUiW7St18\/M2OBBUrQq2p0QVQ5qpbNfz6bEdt555rLzPPZZ\/mDkXON3CFymGNPcOOw5txM3aqT8aj6tG6QN3dDSEfnjhplw2JNnzHDTPnww+YekPZeESgpAvPMNZfpu9hiJe1dtbsFuR8wcmRkDtuyjV7vFBs8I0WoDhs8JG1eEVAEFIE2BEjoHpXUvYwwKSmWcVa0T4qAIqAIKAINQUBJsSGwa6OKgCKgCCgCZURASbGMs6J9UgQUAUVAEWgIAkqKDYFdG1UEFAFFQBEoIwJKimWcFe2TIqAIKAKKQEMQUFJsCOzaqCKgCCgCikAZEVBSLOOsaJ8UAUVAEVAEGoKAkmJDYNdGFQFFQBFQBMqIgJJiGWdF+6QIKAKKgCLQEASUFBsCuzaqCCgCioAiUEYElBTLOCvaJ0VAEVAEFIGGIKCkWGfYv\/jiCzNp0iRz+eWXm8mTJ9us2quvvrr57W9\/a9Zff30z33zz1blH5W7u008\/NYcffrhZYIEFzCmnnGL\/jipZsM3ybrkR\/KF3jPGee+4xf\/7zn23+yq+\/\/tpmc\/\/Zz35mdt99d7Pyyiubdu3ahQ4nCz5Z3m0WbOnn9OnTzTXXXGNuvvlm88ILL9iuEwybjBdbbrml6dChg67dJphQJcU6ThJpVI444gjz0EMPhba6ww47mGOOOcYstNBCdexVeZv65ptvzLhx4+yfQYMGxZJiFmyzvFte9Gbv2RtvvGGGDRtmE1uHFQ5nBx54oP3Dv92SBZ8s7zYLtvTzxRdfNAcddJD573\/\/G9rt5Zdf3owfP9706dNnjt9nwSjLu82Ebz37qqRYJ7S\/+uorM2rUKJtAdNVVVzUjR460f\/NzTu\/87oMPPrBZ6Pfbb7\/IE3udutvwZiDEq666ypxwwgmGf8eRYhZss7zbcJA8OyDSNutsiSWWMMcff7zZYIMNzLzzzmvef\/99M3HiRHPeeefZ2k488UQzZMiQtvWXBZ8s73oOrRSPgSF5Ozlw8E1zsOXv9u3bm+eee86cdNJJ9nfrrLOOzZW4mJPCKgtGWd4tBXAl7YSSYp0m5oknnjC\/+c1vzIILLmgmTJgwRwLRe++915LhkksuaX+\/9NJL16ln5Wvmk08+MWPHjrWbNYRIiSPFLNhmebd8yIX36P777zd77723WXTRRUPXniuR9+\/f31xwwQVm8cUXt5VlwSfLu82CLf289dZbLSkus8wy5sILLzTLLbfcbN1\/9dVX7fXIlClTrNZjq622avt9FoyyvNtM+Na7r0qKdUCcTefUU0+1G9Juu+1mjj322DlUVJ999plVrd511132JM8dT6uVWbNmGTZwpObXXnvN3ndxSPj3v\/8dSYpZsM3ybrPMzbfffmtOPvlkc\/HFF0euPcaC+m+PPfawkuOVV15p1lxzTXsgqXXdZnm3WbClnzNnzjTXX3+9vUtcbbXVzPDhw+f4trlTPeqooyx57rvvvlaNTcmCUZZ3mwnfRvRVSbEOqE+bNs3ss88+9tQdPCm6zXPnMGbMGDN48GCrcmk1o5uXXnrJbsxvv\/222WSTTawa6pFHHjFHH310JClmwTbLu3VYNrk0gYrttNNOs4ct7gt32mmn0Ho\/\/PBDK808\/fTTVm2Nqi8LPlnezWXgJarEJUXmAMMxShaMsrxbImhK2RUlxTpMi2z2H3\/8sfnTn\/5kT5RhBavUgw8+2N5HIFV26tSpDr0rTxOol1CZYnCEQQKWkNdee20sKWbBNsu75UEtn56ESYpZ8Mnybj4jKk8tWKJy4MAo5qKLLjIbbrih7VwWjLK8Wx5kytkTJcU6zMtTTz1l1aEdO3Y0l156qendu3doq1zG77LLLlZleNlll9k7ilYvSaSYBdss71ZpXlCxclj7wx\/+YFZYYQV7IOvevbvJgk+Wd6uCLdIcEjraIbQf2267rRk9enSbW1EWjLK8WxV8ixqHkmJRyDr1+pLds88+a9WHqE3jyLMOXS5NE0mkmAXbLO+WBqAcOoKUuP\/++1tDkMMOO8y6FiClZ8Eny7s5DKmhVbjXANKRQw45xN4nun62WTDK8m5DwWmCxpUU6zBJsoCT1KLyMdElJcXvJsaXFGvBVufFWJUeG\/Zjjz1m1lprLXP22Webrl27Wuyz4JPl3Tp8koU2IVIczvxSFl54YYvzrrvu2mYrkAWjLO8WOvgKVK6kWIdJ1AVcO8hKirVjl\/QmlqYjRoywfrKo6vFVdJ3Ls6zbLO8m9bvsv58xY4btYtAPFItRbAb4Q4CELBhlebfs+DW6f0qKdZgBVXXUDrIvKSbdw4applt5Xt566y1rwER0JRz6zzzzTLP22mvPNlFZ8Mnybu2rpZxvun6gOO5jL7Diiiuqerqc02WUFOswMXopXjvISaSYBdss79Y+osa\/yR0ivnK4XyAhnnHGGaEW0VnwyfJu4xHKvweuAz9+ozvuuKMaMuUPcy41KinmAmN8JRgwEM0GXzB1yUgHeBIpZsE2y7vpRlGepwkEjp8c1pDcwyIhLrvssqEdzIJPlnfLg1Z+PSE4x9ChQ62qWhz4s2CU5d38RlXNmpQU6zCvhC3Dog9VlTrvpwM8iRSzYJvl3XSjaPzTuF088MAD1tiDWKjrrbeejXTTo0ePyM5lwSfLu41Hy78HuF0Qx5ioS0StcUO4ubW4zvaQI\/tBFoyyvOs\/utZ8UkmxDvOuIZlqBzmJFLNgm+Xd2kfUmDc5kBFNBUJk4yaUHrFQ40oWfLK82xiEamuViEGEdrvppptsIHX8EMMiUT3\/\/PNWW4RxEyH3Nt54Yw3zVhvkhb+lpFg4xN814Bu8F3N4dcf4YVKSSDErtq0wL9xnEWaQtEbbbLONja27yCKLeK38LPhkedercyV5SAKC43aBEU0wYhWq0+OOO84SJzFlCecomTKyYJTl3ZJAV8puKCnWaVqIf0gg8BtvvNFa+3FSJ30PiV7vvPNO+9FwitfUUbNPiA8pZsE2y7t1WjqZmnElNjZiNuRgFodgAziYi5N5FnyyvJtp0HV+OZiaS75tUkdxICE5NneJkOY555xjVddSsmCU5d06Q9RUzSkp1nG6NCFoerB9SJFas2Cb5d30I6rvGyQXJkrSK6+84t2wWEfKC1nwyfKud4dL8KDr4hLWHTK+cPc4cODAOXKlZsEoy7slgK2UXVBSrPO0EOXi9ttvt1aokydPtk68q6++ug0YvP7667dcZowk+H1JkXqyYJvl3aQxNPL34i+Ypg9BUlRs\/dDjfhGtDypU\/GKR0oklu+WWW9r7RslRGVZblvWX5V2\/kbXWU0qKrTXfOlpFQBFQBBSBGASUFHV5KAKKgCKgCCgC3yOgpKhLQRFQBBQBRUARUFLUNaAIKAKKgCKgCMyOgEqKuiIUAUVAEVAEFAGVFHUNKAKKgCKgCCgCKinqGlAEFAFFQBFQBEIRUPWpLgxFQBFQBBQBRUDVp7oGFAFFQBFQBBQBVZ\/qGlAEFAFFQBFQBFR9qmtAEVAEFAFFQBGIQ0DvFHV9KAKKgCKgCCgCeqeoa0ARUAQUAUVAEdA7RV0DioAioAgoAoqA3inqGlAEmhUBSaFF\/0lSK9ndSU20+eabz5GjL2qcZIF\/5513zDLLLGPTlrVymTFjhrnlllsMOSdnzZplHnroIfP0009bSEjt5iYDbmWcWm3seqfYajOu421KBIQUf\/azn5kzzjjDLLroot7jeP\/9982DDz5o9aOtdAAAC+xJREFU\/vrXv5r777\/fbLbZZjYbfIcOHbzraIUHSRS81157mRdeeEFJsRUmPGKMSootPPk69OZBQEhxq622Sk1oSIcvv\/yyOf74460kVEsdzYNU7T398MMPbbJvMFJJsXYcm\/1NJcVmn0Htf0sgkIUUBaBTTz3VXHDBBUqKEStGSbElPqXEQSopJkKkD1QZAe6Sbr75ZjPPPPNYsmjfvn3kcKdPn27efvtts\/zyy3vf4eWFnZJiXkhG16OkWDzGzdCCkmIzzJL2sRAEIMQ\/\/\/nP5thjjzUdO3Y05557rllzzTVD2\/rmm2\/MuHHjLHkedNBBhfQnrlIlxeIhV1IsHuNmaEFJsRlmSfuYOwJBQhwzZoy1NmzXrt0cbX3xxRfmrLPOMpdddpm9a1pnnXVy709ShUqKSQhl\/72SYnYMq1CDkmIVZlHHkAqBNIT4+uuvmxNOOMHcc889BveHCRMmmO7du6dqL4+H05AifT7\/\/PPNo48+auaff37b\/H777Wf+\/e9\/m0svvTTyThErVdq56aabzJQpU6zLxkYbbWQOOeQQs\/LKK4ceGMDy4YcfNpdccon56KOPzP\/+9z+z3HLLmUGDBpmf\/vSnZoEFFrDtU9fCCy9s6+Cd1157zVx33XVmrrnmsvXfeOON5swzzzS4SZxzzjlt7hBfffWVueuuu8xVV11lnnjiCYPE3rdvX7P\/\/vubX\/ziF2a++eabA95a3qESJcU8Vmrz16Gk2PxzqCNIgYAvIWKxifTIRvzqq6+aTz\/91HTu3Nn07NnTbu5\/+MMfrK9fvYoPKX777bfm7rvvNiNGjLAq3h122MGSEnehf\/zjH83FF19suxtmffrUU0+ZI444wmyzzTZmxx13tOR17733mpNOOskgKZ944olmyJAhsxEjGEFkDzzwgBk\/frxZccUVDT877rjjLLG6hTap484777TS9uTJk+2vd9ttN7PIIotYUoSUIT3cIo4++mhLsiNHjjSff\/65Oeqooyz+kDWHFCxEBw8ebOdhoYUWamtq2rRpqd+Rl5UU67Way92OkmK550d7lyMCvoToNomjO2b6\/\/d\/\/2fvFNncG1F8SBFi+81vfmN22WUXc+ihh84mRX3yySdm6NChluiCpIh\/3j777GM23nhjS6YifUGykB2qYwgJUl111VXbhi994t3f\/\/73bUZKHCLADAKDVHfaaafZIIO4DjzwQPPPf\/7T1gsZQ3CPPPKIVVEjBQ4YMMC++9xzz1ni7dGjR1sdHFQYJ2TPAWDPPfe0ZA2hpn3H7ZiSYiNWdvnaVFIs35xojwpAwCVEpKezzz7bbLDBBolWpGzcu+++u+natatVPfbu3buA3iVXmUSKSGiQC8SIirdfv35zVBrmkiHEh\/QWJD0qIMoL46eIBAcBIT0ivd16661zEB8qUCQ81KNIl6NHj56NoN13t912W\/t7UbNKp4X4IHchPfkdEiV9eeaZZ0z\/\/v1tvxdbbDEr1UOWad5RUkxee632hJJiq814C46Xjf\/666+3UkUaQgQqJCXUqJtsson921XV1RPKJFKEDCEvSPvCCy80Xbp08SLFqVOnWinx2WeftXeBcgcpL3\/55ZfW8R8pzJUwXanKJUt5jztG1JxhqlqXFMPeZb5OP\/10ey+KhBgcCwccUWlzvyuHlbTv9OnTZzaMVFKs54oub1tKiuWdG+1ZDghkIUSkL1SOGNnwdyNcMQSCJFJM+j31hEmKL730ktljjz3a1KNIXD7FxSaM+KJUq9SdRIru730jy9TyTnCcSoo+M1\/9Z5QUqz\/HLTvCLIQIaEIYWFQ2yhXDlxR9otWEPYN6mDvIJZZYIrV6WIgveN8okt6VV145myWpjCWJFF1yOvnkk63hT1Kp5R0lxSRUW\/P3SoqtOe+VH3VWQgSgSZMmmYMPPrihrhhpSRFDGO4UO3Xq5KU+FeInUg9uD2l8MHF9uPzyy60hzEorrWQDlS+99NLWYGbUqFHmgAMOMAMHDpwjG0cSKbq\/33fffc2wYcMS12st7ygpJsLakg8oKbbktFd70HkQIndoGIBcccUV1rWBTX7eeedtGHBJ6lFXauOOzdfQxrWuhYAOP\/zwyJRS+AsS+WfttdduwwGcuPvDuhVXFf6PawbPRGXySCJFF3tSZJ133nnW0CmskPbp73\/\/u9l5553tHSbz5fvOrrvuOltYP1WfNmx5l6phJcVSTYd2JisCeRAifRADFBzeUeFBjDjD4zjOZg8J4LT+n\/\/8x7pr4FeH4QYbOkYmGK4QPg73httvv93+efzxx62Kcq211rJ5\/PDb23LLLa0BUBLhJpGiGNrgpnDYYYfZ+89gdJ4w9SnSHv3EEIlUUhgWbbjhhnNMA\/XTd\/wChey4V8SK9yc\/+Yl18g+LBhQ2n0mkyDtYtf7ud7+zrxN4gDvdYP5HXDvAEPUq4flqecftn5Ji1q+vGu8rKVZjHnUUxpi8CBEwhWSwxoQMcCTHBWDvvfdu25zd9vDLQ80HKXL\/iG+fSzDiLrD99ttbSebdd9+1zv9Ys66xxhqJCX+TSBGigdxwgofcICuXqLgXJXLMY489ZtZff337e6Q+CveKSIkEKEDaI8UUxjMQNfViaITlLT8XNxbGjoQ4ceJEO05cI3yTFvuQIm4XkCJ9o2BZC9FjiYr1Kf6L+CSuvvrqbb6VtbyjpKhbRxABJUVdE5VAwCUoiKmWIub9SHxEZEGViBEJ0hwS1THHHDOHS4Y4otMeaj6IRiLA\/PrXv27zaxTfuueff95KneIe4tvPJFKknldeecX2mWgvEBTERii0Dz74wEaSQbJD8qUsueSS1sEetSMSHveJqB\/DsJMIPltvvXWbuhE8hg8fPkfkGuoWN4of\/ehHNtwbfXATGruBBPBTRNoLC9cmUXYIAhBWIEokSCLiSKnlHXlXJUXf1Vjt55QUqz2\/LTE61zG\/VkIEKJcUkVAwssERHekQt4WwTPViaYk0CbEgBeJDh+M6ko5s9uLQ\/re\/\/S3UST5ponxIkTpQn15zzTU2HRYZ5Lt162Y222wzA0HfcMMNVhqELJHsXJUt4yD0GplC6OPXX39tDwQYyjB2SDSoHiUSzimnnGINkuIKPp4Y4+AjSkQdpEuCAkjZfPPNLXmuu+66sxEcv+cwgeTNmCB3spQQuB1CRw0dJp3W8g5tKSkmrcLW+L2SYmvMs46yQATY6HFCR71ILE\/IsVevXm1BrWka0iH+KGrLtFaevO9LigUOc7aqGQ93rJAvEuOCCy5oCZefU2bOnGnefPNNS7SoWYlwAzmXuSgplnl26tc3JcX6Ya0tVRQBLCCRpsgKQZgxpBru79zoN0hthB8jOgzh0QhdlqaUjRSR9Aj8zV2jGw81bEzggZTp42+YBpO8n1VSzBvR5qxPSbE55017XSIEJLoL6j0Igjs0122Be0cMb7ibxLWDUGwYieAbiMrWJ3RcmUhR7lGRDLG0jYuCgzp77NixBhVpmJtIiaZR1adlmowG9kVJsYHga9PVQYA7M\/IAcg\/JHwouAhjePPnkk2aVVVaxd2aoGnHXIEvEe++9ZyXGMCOTIDJCikSeIbYpzvKNKhi+IBEjIWPowv1eMKA3fcPCFuMj+orVbfv27RvVZa92MVAiDqzcSXJ3qaX1EFBSbL051xEXgACkha8fUhHWl0hRWIJiDYmrBnkKMQrBZQL1KYYlWLNGObgHu4iPo2ucIr8n8TFSmK+PYB5DR\/ojkg1+j\/wb4xd8G3HapyAxE9UGQx7cRIiSU0ZCxPgJf1HIPVgwRgoGDM8DO62j\/AgoKZZ\/jrSHJUcA45ILLrjALLvssmbTTTetK0E1ChrG\/OKLL1or2\/vuu8\/mTqTge4mEtcUWW9iciElBCRrVf21XEYhCQElR14YikBEBJDgsLckd6OvAnrFJfV0RUAQKQkBJsSBgtdrqIoDKkBBvc801l8FB\/YEHHrDWp2H3atVFQUemCFQTASXFas6rjqpABIjIQsgxHPwxOOHfblSVApvWqhUBRaBgBJQUCwZYq68mAoRyI+oLIdDqaeRSTTR1VIpAeRBQUizPXGhPFAFFQBFQBBqMwP8DSTBMua6FROAAAAAASUVORK5CYII=","height":273,"width":453}}
%---
