%[text] # Parameters
%[text] This file list the parameters used to evaluate the performance of the SIM. The parameters follows those listed in \[1, Sec. VI\]
%[text] ## Simulation parameters
total_iteration=10^3;%referred to the number of transmitted packets, see [1, Sec. VI.D]
%fixing parameters
total_iteration=1 %[output:1d0373a0]
%[text] ## Scenario parameters
%[text] ![SIM_reference_system.png](text:image:813f)
%[text] Fig. 1: Scenario
%room dimensions as follows from [10, Table 6.1-1]
x_max=10;%maximum mobility area in the ground plane
y_max=10;%maximum mobility area in the ground plane
z_max=10;%height of hall the BS in the room as given in 
h_SIM=4; %height of the SIM
h_MU=1.5; %height of the MU
pos_SIM = [x_max/2 y_max/2 h_SIM]; % SIM position in meters (x0, y0, z0)
pos_MU_init = [2 4 1.5]; % MU position in meters (x0, y0, z0), the height of the MU is given in [10, Table 6.1-1]
pos_MU_max=[x_max y_max 0];%max position of the MU in the positive ground plane
d_MU_SIM_max=norm(pos_SIM-pos_MU_init);%maximum distance from the MU to the SIM
theta_max=atan2(norm(pos_MU_max),d_MU_SIM_max)+pi;%maximum elevation angle between the MU and SIM, see in Fig. 1
theta_min=atan2(-norm(pos_MU_max),d_MU_SIM_max)+pi;%minimum elevation angle between the MU and SIM, see in Fig. 1

%[text] #### MU mobility parameters
%time step
delta_time=0.1;
%total simulation time
time_simul=40;
%% Parameters for pause time
pause_min = 0.1;   % minimum pause duration [s]
pause_max = 0.1;   % maximum pause duration [s]

% Create a configuration object for 802.11
cfgDMG_loc = wlanDMGConfig;
fc=28*10^9;%transmission frequency


lambda=physconst('LightSpeed')/fc;
% SNR_dB=[2 4 6 8 10];
idleTime=20e-6;% Idle time between packets [6]

fs = wlanSampleRate(cfgDMG_loc);%sampling time of the receiver node [6]
BW = 400*10^6;%bandwidth, see [10, Tab. 6-1]

%Configuring the transmitted waveform for localization applications
%modulation format [3]
cfgDMG_loc.MCS="1"; % Single-carrier modulation
%packet format
%cfgDMG_loc.TrainingLength = 4;   % Use 4 training subfields (TRN) [3]
%cfgDMG_loc.PacketType = 'TRN-T'; % Transmitter training
cfgDMG_loc.PSDULength = 1; % PSDULength in bytes
ind_loc=wlanFieldIndices(cfgDMG_loc);
T_PPDU_loc=(double(ind_loc.DMGData(2))/fs)+idleTime;
disp(['Wireless packet type: ',phyType(cfgDMG_loc)]) %[output:00def506]

waveform_k= 2*pi/lambda;
kappa=waveform_k;
%[text] #### SIM paramaters
%Number of atoms in the first layer
%N=16; %see [1, Sec. VI.C.]
N_x_vector=2:5;%SIM dimension of the zero Layer
N_y_vector=N_x_vector;
N_vector=N_x_vector.*N_y_vector;
N_x=4 %as follows from [1, Sec. IV A] for the (4x4 grid) %[output:4cd315df]
% N_x=2 %as follows from [1, Sec. IV A] for the (2x2 grid)
N_y=N_x; %to account for a balanced error in the x an y axes
N=N_x.*N_y %[output:18b88e21]

%number of meta-atoms in the intermediate layers
M_x=15 %[output:6088ac05]
M_y=M_x;
M=M_x*M_y %[output:38d6f3e6]
%M=225;%as follows from [1, Sec. IV A] for the (4x4 grid)
% M=121;%as follows from [1, Sec. IV A] for the (4x4 grid)
% M_x=sqrt(M) 
% M_y=M_x
%number of intermediate SIM layers
% L = 13; %as follows from [1, Sec. IV A] for the (4x4 grid)
L = 7;
% L=7;%as follows from [1, Sec. IV A] for the (2x2 grid)
%distance between atoms
d_x=lambda/2;
d_y=d_x;

%Thickness of the SIM
T_SIM=12*lambda;%as follows from [1, Sec. IV A] for the (4x4 grid)
% T_SIM=9*lambda;%as follows from [1, Sec. IV A] for the (2x2 grid)

%spacing between atoms on each layer
s_x=4*lambda/9;%as follows from [1, Sec. IV A] for the (4x4 grid)
% s_x=lambda/2;%as follows from [1, Sec. IV A] for the (4x4 grid)
s_y=s_x;

% spacing between adjacent layers
s_layer = T_SIM/L;   %as follows from [1, Sec. II B]

A_atom  = d_x*d_y; % meta-atom area (paper uses A_atom-atom)

%SIM Position
%defining the hall dimensions in accordance with Table 7.2-4 pp. 21 in [8]
L_hall= x_max; % length of factory hall
W_hall= y_max; %width of hall
H_hall=z_max; %height (ceiling)
SIM_Pos = [L_hall/2 W_hall/2 4];       % SIM center on ceiling
simSize = 1.0;          % just for plotting SIM frame size (meters)
% Rx_element='38.901';   % good surrogate fkr directional, patch-like element; Peak gain ≈ 8 dBi; realistic azimuth / elevation beamwidths (not a sphere).
% Rx_array_normal='z';
Rx_orientation=[0; 0; 0];% [azimuth; downtilt; slant], % boresight -z
rxElem = phased.CosineAntennaElement( ...
    'FrequencyRange', [fc-5e9 fc+5e9], ...
    'CosinePower',    [2 2]);  % ~65° HPBW in az/el
rxArray = phased.URA( ...
    'Size',           [N_y N_x], ...
    'Element',        rxElem, ...
    'ElementSpacing', [lambda/2 lambda/2]);  % 0.5 λ spacing

Grx_dBi = 8;                          % desired antenna peak gain
Grx = 10^(Grx_dBi/20);                  % convert to amplitude factor

%[text] #### Gradient Descent Algorithm 
maxIter = 600; %number of gradient descent iterations
eta0 = 1; % initial learning rate, as follows from the value in first paragraph, Section VI.B
% zeta_ini=0.985;
% zeta_delta=0.001;
% zeta_end=0.99;
zeta_ini=0.980;
zeta_delta=0.001;
zeta_end=0.980;
% zeta = 0.1:0.1:(1-0.1); %decay parameter in [Eq. (20), 1], it controls the maximum phase rotation per iteration
zeta = zeta_ini:zeta_delta:zeta_end %decay parameter in [Eq. (20), 1], it controls the maximum phase rotation per iteration %[output:1d0ad6ac]
tol = 1e-8; %tolerance of the approximation
seed=1;

%[text] #### Mobile User parameters
total_MUpositions=1;
%MU speed of velocity
MU_speed=3*1000/60/60;%corresponding to 3km/h meter per second as follows from [10, Tab. 6.1-1]
Ptx_dBm = 23; % transmit power in dBm
Ptx = db2pow(Ptx_dBm-30);
Gtx_elem_dBi = 8;                          % desired antenna peak gain
                 % convert to amplitude factor
Gtx_array_dB = 6;     % 4-element array gain
Gtx_dBi   = Gtx_elem_dBi + Gtx_array_dB;   % ≈14 dBi
Gtx = 10^(Gtx_dBi/20); 
%[text] Antenna: Configure the transmit array size as a vector of the form \[*M* *N* *P* *M*g *N*g\], which specifies a rectangular panel of dimension (*M*g *N*g) of an antenna array of dimension (*M* *N*) and number of polarizations (*P*). The total number of polarized elements in the array is *M*×*N*×*P*×*M*g×*N*g.
% tx_Antenna_Size = [1 4 1 1 1]; %as follows from the mmWave Antenna chip from [7], except for the number of polarizations
% Tx_element='38.901';   % good surrogate fkr directional, patch-like element; Peak gain ≈ 8 dBi; realistic azimuth / elevation beamwidths (not a sphere).
Tx_array_normal='z';
Tx_orientation=[0; 0; 0];% [azimuth; downtilt; slant], % boresight +z

% Define a valid phased-array element (pick one):
txElem  = phased.CosineAntennaElement('FrequencyRange',[fc-5e9 fc+5e9], ...
                                 'CosinePower',[2 2]);
% figure; pattern(txElem, fc);
% title('Tuned Cosine Element (~38.901-like)');

% Build URA in x–y plane, boresight +z
txArray = phased.ULA('NumElements',4,'Element',txElem,'ElementSpacing',lambda/2);

%[text] #### Localization protocol parameters
T_coh=sqrt(9/(16*pi))*physconst('LightSpeed')*MU_speed/fc %[output:7c3c3c33]
N_packets_coh=floor(sqrt(T_coh/T_PPDU_loc)) %[output:23c1d2de]
%T=1024;%see [1, Sec. VI.C.]
%T_x=ceil(sqrt(T_coh/T_loc))
% T_x=1:ceil(sqrt(T_coh/T_PPDU_loc));
% T_x=64; %as follows from [Sec. VI.C (2x2) grid,1]
% T_y=T_x; %accounting for a balanced error in the x an y axes of the Fourier transform
% T=T_x.*T_y
%Fixing parameters
T_x=60 %[output:5580c95d]
T_y=T_x; %accounting for a balanced error in the x an y axes of the Fourier transform
T=T_x.*T_y;
T_x_vector=35:40;
T_y_vector=T_x_vector;
T_vector=T_x_vector.*T_y_vector;
%[text] #### Channel
maxDoppler = (MU_speed/physconst('LightSpeed'))*fc;
%Evaluating the delay spread according to Table 7.5-6 Table 7.5-6 Part-3: Channel model parameters for InF in page 48
V_hall= L_hall*W_hall*H_hall;%hall volume (m³)
S_hall= 2*(L_hall*H_hall+W_hall*H_hall)+2*(L_hall*W_hall); %total surface of 4 walls + floor + ceiling (m²)
%LOS
u_lgDS=log10(26*V_hall/S_hall+14)-9.35;%mean
sigma_lgDS=0.15;%variance
%NLOS
% u_lgDS=log10(30*V/S+32)-9.44;%mean
% sigma_lgDS=0.19;%variance
%Evaluating the wanted delay spread, as follows from Eq. (7.7-1) in page 79
DS_ns = 10^(u_lgDS+sigma_lgDS);
%Delay profile
% CDL-A → NLOS, low delay spread, small angular spreads
% CDL-B → NLOS, slightly larger angular spreads
% CDL-C → rich NLOS scattering, larger delay spread
% CDL-D → LOS model (moderate K-factor)
% CDL-E → strong LOS (larger first-tap power)
profile   = 'CDL-D';      % NLOS: 'CDL-A/B/C'; LOS: 'CDL-D/E'

% %% --- CDL channel object (TR 38.901) --------------------------------------
% cdl = nrCDLChannel;
% cdl.DelayProfile         = profile;          % NLOS: CDL-A/B/C, LOS: CDL-D/E LOS (§7.7.1)
% % Draw one delay-spread realization
% lgDS = EnvPars.mu_lgDS + ...
%        EnvPars.sigma_lgDS*randn;
% 
% DS_s  = 10.^lgDS;       % seconds
% DS_ns = DS_s*1e9;       % nanoseconds, only for display
% 
% fprintf('Generated InF DS = %.2f ns\n', DS_ns);
% 
% cdl.DelaySpread = DS_s;
% 
% % cdl.DelaySpread          = DS_ns*1e-9;       % set RMS-DS via scaling (Eq. 7.7-1 in §7.7.3)
% cdl.CarrierFrequency     = fc;
% cdl.SampleRate           = fs;
% cdl.MaximumDopplerShift  = maxDoppler;
% 
% % Attach the antenna elements to the CDL
% cdl.TransmitAntennaArray = txArray;
% cdl.ReceiveAntennaArray = rxArray;
% 
% % Orient the transmit and receive antenna arrays to point at each other by using the LOS path angles returned in the characteristic information.
% cdlInfo = cdl.info;
% cdl.TransmitArrayOrientation = [cdlInfo.AnglesAoD(1) cdlInfo.AnglesZoD(1)-90 0]';
% cdl.ReceiveArrayOrientation = [cdlInfo.AnglesAoA(1) cdlInfo.AnglesZoA(1)-90 0]';

%% Channel model used during evaluation

% Options:
%   'LoS' : LoS steering vector plus AWGN
%   'Rician_LoS_NLoS' : Rician LoS + clustered NLoS channel
EnvPars.channelModel = 'rician_los';
% EnvPars.channelModel = 'rician_los_nlos';

EnvPars.fc_GHz = fc/1e9;

% Hall geometry required by the InF delay-spread formula
EnvPars.V_hall = L_hall * W_hall * H_hall;

EnvPars.S_hall = 2 * ( ...
    L_hall*W_hall + ...
    L_hall*H_hall + ...
    W_hall*H_hall);

% Delay spread: lgDS = log10(DS/1 second)
EnvPars.mu_lgDS = ...
    log10(26*(EnvPars.V_hall/EnvPars.S_hall) + 14) ...
    - 9.35;

EnvPars.sigma_lgDS = 0.15;

% Large-scale angular spreads
% lgAS = log10(AS/1 degree)
EnvPars.mu_lgASD    = 1.56;
EnvPars.sigma_lgASD = 0.25;

EnvPars.mu_lgASA = ...
    -0.18*log10(1 + EnvPars.fc_GHz) + 1.78;

EnvPars.sigma_lgASA = ...
    0.12*log10(1 + EnvPars.fc_GHz) + 0.20;

EnvPars.mu_lgZSA = ...
    -0.20*log10(1 + EnvPars.fc_GHz) + 1.50;

EnvPars.sigma_lgZSA = 0.35;

% Table 7.5-11
EnvPars.mu_lgZSD    = 1.35;
EnvPars.sigma_lgZSD = 0.35;

% Rician K-factor in dB
% EnvPars.mu_K_dB    = 30;
% EnvPars.sigma_K_dB = 0;
EnvPars.mu_K_dB    = 7;
EnvPars.sigma_K_dB = 8;

KR_dB = EnvPars.mu_K_dB + ...
            EnvPars.sigma_K_dB*randn;

EnvPars.KR = 10.^(KR_dB/10);

% Delay-distribution proportionality factor
EnvPars.rTau = 2.7;

% Cross-polarization ratio
EnvPars.mu_XPR_dB    = 12;
EnvPars.sigma_XPR_dB = 6;

% Per-cluster shadowing standard deviation
EnvPars.clusterShadowingStd_dB = 4;

EnvPars.Nc   = 25;
EnvPars.Mray = 20;



% Intra-cluster angular spreads from Table 7.5-6 Part-3
%for arrival angles
EnvPars.clusterASD_deg = 5;
EnvPars.clusterASA_deg = 8;
EnvPars.clusterZSA_deg = 9;

EnvPars.rayOffsetAlpha = [ ...
   -0.0447,  0.0447, ...
   -0.1413,  0.1413, ...
   -0.2492,  0.2492, ...
   -0.3715,  0.3715, ...
   -0.5129,  0.5129, ...
   -0.6797,  0.6797, ...
   -0.8844,  0.8844, ...
   -1.1481,  1.1481, ...
   -1.5195,  1.5195, ...
   -2.1551,  2.1551];

% Table 7.5-6 gives cluster DS as N/A for InF
EnvPars.clusterDS_ns = NaN;

% ZOD offset from Table 7.5-11
EnvPars.ZODoffset_deg = 0;

% Horizontal correlation distances of all listed InF LSPs
EnvPars.corrDistance_DS_m  = 10;
EnvPars.corrDistance_ASD_m = 10;
EnvPars.corrDistance_ASA_m = 10;
EnvPars.corrDistance_SF_m  = 10;
EnvPars.corrDistance_K_m   = 10;  % LOS only
EnvPars.corrDistance_ZSA_m = 10;
EnvPars.corrDistance_ZSD_m = 10;

% Controlled-SNR experiment
EnvPars.normalizeH = true;

% A fixed value is useful initially for debugging.
% EnvPars.KR_dB = 25;%Rician Factor
% 
% %25 NLoS clusters and 20 rays per cluster
% EnvPars.Nc   = 25;
% EnvPars.Mray = 20;
% 
% % Cluster powers.
% % Leave empty to use the temporary exponentially decaying profile below.
% % For the final paper, replace this with Pc from your 38.901 realization.
% EnvPars.Pc = [];
% 
% % Temporary cluster-power decay factor
% EnvPars.clusterPowerDecay = 1.5;
EnvPars.rTau = 2.7;
EnvPars.clusterShadowingStd_dB = 4;
% 
% % Temporary angular-spread parameters
% % These are Monte-Carlo sensitivity parameters, not yet exact 38.901 values.
% EnvPars.clusterThetaSpread_deg = 8;
% EnvPars.clusterPhiSpread_deg   = 8;
% EnvPars.rayThetaSpread_deg     = 1;
% EnvPars.rayPhiSpread_deg       = 1;
% 
% Approximate cosine element-pattern amplitude
EnvPars.elementCosinePower = 0;

% % A fixed value is useful initially for debugging.
% EnvPars.KR_dB = 25;%Rician Factor
% 
% %25 NLoS clusters and 20 rays per cluster
% EnvPars.Nc   = 25;
% EnvPars.Mray = 20;
% 
% % Cluster powers.
% % Leave empty to use the temporary exponentially decaying profile below.
% % For the final paper, replace this with Pc from your 38.901 realization.
% EnvPars.Pc = [];
% 
% % Temporary cluster-power decay factor
% EnvPars.clusterPowerDecay = 1.5;
% 
% % Temporary angular-spread parameters
% % These are Monte-Carlo sensitivity parameters, not yet exact 38.901 values.
% EnvPars.clusterThetaSpread_deg = 8;
% EnvPars.clusterPhiSpread_deg   = 8;
% EnvPars.rayThetaSpread_deg     = 1;
% EnvPars.rayPhiSpread_deg       = 1;
% 
% % Approximate cosine element-pattern amplitude
% EnvPars.elementCosinePower = 0;

% For controlled-SNR Fig. 11, normalize every channel realization
% so mean(abs(h).^2) = 1.
% EnvPars.normalizeH = true;

%% Additional NLoS power multiplier for diagnostic tests
%
% 1     : use the complete nominal NLoS component
% 0.1   : use 10% of its nominal power
% 0.01  : use 1% of its nominal power
% 0     : completely remove the NLoS component

EnvPars.nlosPowerScale = 0;

% figure; displayChannel(cdl,'LinkEnd','Tx');
% view(0,90)
% figure; displayChannel(cdl,'LinkEnd','Rx'); title('SIM Rx array');
% view(0,90)
%[text] #### Noise parameters; see \[3\]
% Noise and interference parameters
noiseFigure = 7;                             % dB see [10, Tab. 6-1]
thermalNoiseDensity = -174;                  % dBm/Hz [11, Tab. 5, entry (14)]
%Interference noise
rxInterfDensity = -165.7;                    % dBm/Hz
rxInterfDensity = -Inf;                    % no interference noise

% Calculate the corresponding noise power
% Receiver thermal noise density including noise figure
rxNoiseDensity = thermalNoiseDensity + noiseFigure;   % dBm/Hz

% Sum noise and interference in linear mW/Hz
totalNoiseDensity_dBHz = 10*log10(10^((rxNoiseDensity-30)/10) + 10^((rxInterfDensity-30)/10));

% Total noise power over bandwidth BW
noisePower_dB = totalNoiseDensity_dBHz + 10*log10(BW);

%[text] #### SNR evaluation
%Free path loss in the direct link IRS-MU
Lr = (lambda/(2*pi*norm(d_MU_SIM_max)))^2;
SNR_dB=Ptx_dBm-30+pow2db(Lr)-noisePower_dB %[output:36e595f1]

%[text] #### Agent parameters
%Environment related parameters
EnvPars.N = evalin('base','N');
EnvPars.N_x = evalin('base','N_x');
EnvPars.N_y = evalin('base','N_y');
EnvPars.T = evalin('base','T');
EnvPars.T_x = evalin('base','T_x');
EnvPars.T_y = evalin('base','T_y');
EnvPars.SNR_dB = evalin('base','SNR_dB');
EnvPars.theta_min = evalin('base','theta_min');
EnvPars.theta_max = evalin('base','theta_max');

EnvPars.fc = evalin('base','fc');
EnvPars.lambda = evalin('base','lambda');
EnvPars.Ptx_dBm = evalin('base','Ptx_dBm');
EnvPars.Gtx_dBi = evalin('base','Gtx_dBi');
EnvPars.Grx_dBi = evalin('base','Grx_dBi');
EnvPars.txArray.NumElements = evalin('base','txArray.NumElements');
% EnvPars.cdl = evalin('base','cdl');
EnvPars.var_noise_dB = evalin('base','noisePower_dB');
EnvPars.r = 0;

EnvPars.d_x = evalin('base','d_x');
EnvPars.d_y = evalin('base','d_y');
EnvPars.pos_SIM = evalin('base','pos_SIM');
EnvPars.pos_MU  = [L_hall*rand(1,1)...
                   W_hall*rand(1,1)...
                   1.5];                 % Mobile user coordinates


% Meta-atom indexing — store inside EnvPars so closures don't depend on workspace
n = 1:EnvPars.N;
EnvPars.n_y = ceil(n ./ EnvPars.N_x);
EnvPars.n_x = n - (EnvPars.n_y - 1) .* EnvPars.N_x;

% Environment variables
n = (1:EnvPars.N);
n_y = ceil(n ./ EnvPars.N_x);
n_x = n - (n_y - 1) .* EnvPars.N_x;
n_psi = n;
n_psi_y = ceil(n_psi ./ EnvPars.N_x);
n_psi_x = n_psi - (n_y - 1) .* EnvPars.N_x;

% Snapshots indexes for indexing psi_y and psi_x components of the electric
% angle psi
t_psi_idx = (1:EnvPars.T);
EnvPars.t_y = ceil(t_psi_idx ./ EnvPars.T_x) ;             % now in {0, ..., T_y - 1}
EnvPars.t_x = t_psi_idx - (EnvPars.t_y-1) .* EnvPars.T_x;   % now in {0, ..., T_x - 1}

% MU height — taken from Parameters
EnvPars.h_MU = pos_MU_init(3);

% Room dimensions — taken from Parameters
EnvPars.L_hall = L_hall;
EnvPars.W_hall = W_hall;
EnvPars.N_cal=L_hall*W_hall; %approximante number of calibration measurements to compute the maximum of the DFT per MU position, it is approximate due to the margins between the MU and walls

EnvPars.MU_margin = 0.5; %margin of the MU from the walls.

%Training Parameters
% Episode parameters (will be initialized on reset)
EnvPars.MaxEpisodes=T*25; %number of posible combination of phases, and 10 times to assure the random reset makes at least once per position
EnvPars.MaxEpisodes=5000;


EnvPars.psi_x = 0;
EnvPars.psi_y = 0;
%EnvPars.MaxStepsPerEpisode = EnvPars.T;%number of steps per episode equals the number of packets along the channel coherence time
% Max steps: longest possible Manhattan path across the grid
EnvPars.MaxStepsPerEpisode = 1.5*(EnvPars.T_x + EnvPars.T_y);
% EnvPars.MaxStepsPerEpisode = EnvPars.T_x*EnvPars.T_y;

EnvPars.tolerance = pi/80; % error tolerance reward C
EnvPars.tolerance = 2*pi / (EnvPars.N_x * EnvPars.T_x);  % ≈ 0.13 rad reward B
EnvPars.StopTrainingValue=1000.0; % reward C
EnvPars.StopTrainingValue=11 * EnvPars.MaxStepsPerEpisode; % asking to stop early only when the whole episode has been perfect for reward B
% Stop when average episode reward reaches ~95% of maximum possible
% Maximum episode reward = 1.0 × MaxStepsPerEpisode (peak found, stay there)
EnvPars.StopTrainingValue = 0.95 * EnvPars.MaxStepsPerEpisode;

EnvPars.episode_counter=0; %counts the number of episodes during training

% 9 relative moves in (t_x, t_y) grid
% Action 5 = stay, others = move in each diagonal/cardinal direction
EnvPars.delta_moves = [-1,-1; -1, 0; -1,+1;
                        0,-1;  0, 0;  0,+1;
                       +1,-1; +1, 0; +1,+1];
EnvPars.n_actions   = size(EnvPars.delta_moves, 1);   % 9

EnvPars.DiscountFactor=0.95 %[output:2177d08d]
%[text] Calculation of the EpsilonDecay factor
%[text] We calculate the number of random visits to states, which is given by
%[text] random steps needed = number of actions × visits per action = 144×150=21,600 steps
%[text] Then decay of the probality of exploring should follow the relation:
%[text] EpsilonDecay=(ε\_start − ε\_min)  / random steps  = 0.95 / 21,600,
%[text] where ε\_start =1 is the starting point, always exploring, and ε\_min = 0.05 is when the system is exploiting and ε\_new=ε\_old(1−EpsilonDecay)

% EnvPars.EpsilonDecay=0.95/(EnvPars.T*150);%ε_new�=ε_old�−EpsilonDecay, steps to minimum=EpsilonDecay/(ε_start�−ε_min��)
% 
% % Epsilon decay: explore each of the 9 actions ~100 times per position
% random_steps = 100 * EnvPars.n_actions * EnvPars.MaxStepsPerEpisode;
% EnvPars.EpsilonDecay = 0.95 / random_steps;

% Explore for ~3000 episodes regardless of episode length.
% 3000 covers the 1600-position grid ~twice with random starts,
% giving the replay buffer a representative sample of the state space
% before exploitation starts dominating.
exploration_episodes = 3000;
%EnvPars.EpsilonDecay = 0.95 / (exploration_episodes * EnvPars.MaxStepsPerEpisode);

EnvPars.EpsilonDecay = 3.0 / (exploration_episodes * EnvPars.MaxStepsPerEpisode);

EnvPars.ExperienceBufferLength=1e5;%set the lenght of the agent's circular buffer, too small (e.g. 1,000): the agent only remembers recent experience, forgets early exploration, Too large (e.g. 10^7): memory cost is high, and very old experiences (from when the policy was much worse) pollute the minibatch
EnvPars.MiniBatchSize=128;
EnvPars.TargetSmoothFactor=1e-3;% this factor weights the amount of the NN coefficients used to update the target estimation of the Q function as wtarget←(1−τ)⋅wtarget+τ⋅wmain


%Reward
EnvPars.threshold = 0.8; %defines a reward for those actions that makes the optimum closer to the right one, only positions with >80% of peak get positive reward
% In Parameters.mlx — Agent parameters section
EnvPars.reward_threshold = 0.8;   % Reward E threshold
% ---- ACQUISITION reward constants + episode-budget fix (local overrides,
EnvPars.step_cost  = 0.01;  % per-step penalty: wandering/hovering strictly negative
EnvPars.peak_bonus = 10;    % terminal bonus, paid once on reaching the peak

% Definition of Observations and Actions
% Observation: [|r|² (N values), t_x_current, t_y_current] → N+2 dimensional
ObsInfo             = rlNumericSpec([EnvPars.N + 2, 1]);
ObsInfo.Name        = 'observations';
ObsInfo.Description = 'Power pattern |r|^2 plus agent self recorded (t_x, t_y)';
% Lower and upper bounds — helps training stability
ObsInfo.LowerLimit  = zeros(EnvPars.N + 2, 1);
ObsInfo.UpperLimit  = [inf(EnvPars.N, 1); EnvPars.T_x; EnvPars.T_y];

% ActInfo = rlFiniteSetSpec(1:EnvPars.T);
% ActInfo.Name = 'actions';
% ActInfo.Description = 'Phase snapshot index t_psi in the range 1 to T';

% Action: one of 9 navigation moves
ActInfo             = rlFiniteSetSpec(1:EnvPars.n_actions);
ActInfo.Name        = 'actions';
ActInfo.Description = 'Navigation move index into delta moves';

% Actions given by the Upsilon coefficient
EnvPars.U_func = @(n_, t_n_) exp(1i * ( ...
    -2*pi*(EnvPars.n_x(n_)-1) .* (EnvPars.t_x(t_n_)-1) / (EnvPars.N_x*EnvPars.T_x) ...
    -2*pi*(EnvPars.n_y(n_)-1) .* (EnvPars.t_y(t_n_)-1) / (EnvPars.N_y*EnvPars.T_y) ));

% ---- CST amplitude interpolant for the input layer (layer 0 / v0) ----
% Built the same way as in SIM_Training_CST_SingleZeta_Parallel.m
% (pchip + 'nearest' flat extrapolation, NOT periodic).
load t_y_x.mat
[EnvPars.F_amp, EnvPars.phase_min_meas, EnvPars.phase_max_meas] = ...
    build_amplitude_interpolant(t_y_x_amp_dB, t_y_x_phase_deg);

% CST-coupled input layer, SEPARATE from the analytic EnvPars.U_func above
% so existing code keeps using U_func untouched. SIM-2 uses U_func_CST.
EnvPars.U_func_CST = @(n_, t_n_) U_func_cst(n_, t_n_, EnvPars);

EnvPars.episode_counter = EpisodeCounter();

%Analytic FFT
% Kernel 2D-DFT == TO BE REPLACED BY SIM 1
G_func = @(n,n_psi) exp(-1i*2*pi*(n_psi_x(n_psi)-1)/EnvPars.N_x.*(n_x(n)-1)).* ...
    exp(-1i*2*pi*(n_psi_y(n_psi)-1)/EnvPars.N_y.*(n_y(n)-1));
[n_psi_grid, n_s_grid] = ndgrid(1:EnvPars.N, 1:EnvPars.N);
EnvPars.G = G_func(n_s_grid, n_psi_grid);

% ---- Trained CST-realistic SIM-1 G (separate from analytic EnvPars.G) ----

% sim1_file = fullfile('..', 'Dataset', 'SIM_training_CST_single_zeta_Nx_4_28_GHz.mat');
% sim1_file = fullfile('..', 'Dataset',
% 'G_Nx_4_L_2_Mx_5_Zeta_0_988_CST.mat');%Relative Loss -0.9 dB, reward<10
% sim1_file = fullfile('..', 'Dataset', 'G_Nx_4_L_7_Mx_12_Zeta_0_988_CST.mat'); %Relative Loss -3.3 dB, reward <12
% sim1_file = fullfile('..', 'Dataset', 'G_Nx_4_L_9_Mx_12_Zeta_0_988_CST.mat');
% sim1_file = fullfile('..', 'Dataset', 'G_Nx_4_L_11_Mx_12_Zeta_0_988_CST.mat'); %Relative Loss -18.3 dB, 
% sim1_file = fullfile('..', 'Dataset','G_Nx_4_L_13_Mx_12_Zeta_0_988_CST.mat'); %Relative Loss -20.8 dB, reward<22
sim1_file = fullfile('..', 'Dataset', 'G_Nx_4_L_13_Mx_15_Zeta_0_988_CST.mat');%Relative Loss -38.5 dB, 

% sim1_file = fullfile('..', 'Dataset', 'SIM_training_CST_zeta_0.988_Nx_6.mat');
S_sim1 = load(sim1_file, 'G_vs_eta', 'beta_final');
assert(isequal(size(S_sim1.G_vs_eta{1}), [EnvPars.N, EnvPars.N]), ...
    'Loaded G is %dx%d but expected %dx%d.', ...
    size(S_sim1.G_vs_eta{1},1), size(S_sim1.G_vs_eta{1},2), EnvPars.N, EnvPars.N);
EnvPars.G_CST = S_sim1.beta_final * S_sim1.G_vs_eta{1};
dft_residual = norm(EnvPars.G_CST - EnvPars.G, 'fro') / norm(EnvPars.G, 'fro');
fprintf('Loaded CST SIM-1 G_CST. Deviation from analytic G: %.3f%%n', 100*dft_residual); %[output:700bead8]
%[text] #### Plotting parameters
font=20;

%%
%[text] ## Helper functions
function v = U_func_cst(n_, t_n_, EnvPars)
    phase = -2*pi*(EnvPars.n_x(n_)-1) .* (EnvPars.t_x(t_n_)-1) / (EnvPars.N_x*EnvPars.T_x) ...
            -2*pi*(EnvPars.n_y(n_)-1) .* (EnvPars.t_y(t_n_)-1) / (EnvPars.N_y*EnvPars.T_y);
    amp = EnvPars.F_amp(mod(phase, 2*pi));
    v = amp .* exp(1i*phase);
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
%[text] ## References
%[text] \[1\] J. An et al., "Two-Dimensional Direction-of-Arrival Estimation Using Stacked Intelligent Metasurfaces," in IEEE Journal on Selected Areas in Communications, vol. 42, no. 10, pp. 2786-2802, Oct. 2024. DOI [10.1109/JSAC.2024.3414613](https://doi.org/10.1109/JSAC.2024.3414613)
%[text] \[2\] J. An *et al*., "Stacked Intelligent Metasurface Performs a 2D DFT in the Wave Domain for DOA Estimation," *ICC 2024 - IEEE International Conference on Communications*, Denver, CO, USA, 2024, pp. 3445-3451, doi: [10.1109/ICC51166.2024.10622963](https://ieeexplore.ieee.org/document/10622963)
%[text] \[3\] WLAN PPDU Structure, [https://de.mathworks.com/help/wlan/gs/dmg-ppdu-structure.html](https://de.mathworks.com/help/wlan/gs/dmg-ppdu-structure.html)
%[text] \[4\] Model Reconfigurable Intelligent Surfaces with CDL Channels [https://ch.mathworks.com/help/5g/ug/model-reconfigurable-intelligent-surfaces-with-cdl-channels.html](https://ch.mathworks.com/help/5g/ug/model-reconfigurable-intelligent-surfaces-with-cdl-channels.html)
%[text] \[5\] [https://ch.mathworks.com/help/wlan/ug/802-11ad-waveform-generation-with-beamforming.html](https://ch.mathworks.com/help/wlan/ug/802-11ad-waveform-generation-with-beamforming.html)
%[text] \[6\] [https://ch.mathworks.com/help/phased/ug/signal-collection.html](https://ch.mathworks.com/help/phased/ug/signal-collection.html)
%[text] \[7\] [https://ch.mathworks.com/help/wlan/ref/wlanwaveformgenerator.html\#bvdi39e-4](https://ch.mathworks.com/help/wlan/ref/wlanwaveformgenerator.html#bvdi39e-4)
%[text] \[8\] J. Oh, K. Kim, J. Choi, and J. Oh, "Tightly Embedded Modular Antenna-in-Display (MAiD) Into the Panel Edge of Display With Dual-Polarization for 5G Smartphones," IEEE Transactions on Antennas and Propagation, vol. 73, no. 2, pp. 1209–1214, Feb. 2025, doi: 10.1109/TAP.2024.3501415.
%[text] \[9\] "Study on channel model for frequencies from 0.5 to 100 GHz (Release 16)," 3GPP, TR 38.901 V16.1.0, Nov. 2020.  
%[text] \[10\] F. Munier, Y. Guo, and R. Da, "Study on NR Positioning Enhancements (Release 17)," 3rd Generation Partnership Project, Sophia Antipolis, France, Technical Report (TR) TR 38.857 V17.0.0, Mar. 2021
%[text] \[11\] ETSI, “Reconfigurable Intelligent Surfaces (RIS); Communication Models, Channel Models, Channel Estimation and Evaluation Methodology,” European Telecommunications Standards Institute, GR RIS 003 V1.1.1,Jun. 2023 

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":31.6}
%---
%[text:image:813f]
%   data: {"align":"baseline","height":289,"src":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAUEAAAF6CAIAAAAI0X0IAABGVklEQVR42u2dCXxTxdr\/0auo4ALqVa4Lopd7\/1XkvQgqtYDUiwqKSBW4goiy+CqySC0uFJfCFSiIgiCvFQUqawHBIgiUpZQ2XdJ9b7qkaZt9aZauaZq2\/J8yOBxPktOTNE2T9vl9Bj5pOnNOenK+53memWdm+lxCoVC+rD54CVAoZBiFQiHDKBQKGUahkGEUCoUMo1AoZNjnVFBR64EmnWyIQoZRDnU8SeWBJp1siPwjwyifZ7gzbVHIMDKMDKOQYW\/V0XiFU\/Wl2savD4obmlo8cC5kGBlGcQk4PJ2q\/nKPqKbeyh\/g85nan05Vwv9OYQyVY9LU6\/eX8T8XUxpD09q9JSXSOvzWkGHUVaiAQ6Di5zPSU0I1H7QIwNAQTCJ9zf9c0GT\/OdnZdK2zGAPA0OrgecXFHB1ijAyj\/gQw8VGBrg4xZkJL3FqeGFOASUMCJH+MaX1oa21pQ4yRYdSfAGYCyYExaUJxpaEptCpX1PMEmDbkjzGzJmmLGCPDqHYwyuRXwaNAqvRmDqPa2tpm24TPuSjAzIY6k4WPHy4sMlDUaVvAGBlGhlFX5QIPLiPUGfawXxoZdoOsVmtoaOjAgQP7XNb48eOjo6Ppb81m8+DBg\/v376\/Vauk7U6ZMgZp9+\/bds2cPx2FDQkLIMcPCwvCrdTv\/KGT4KqKAmZ+f3zvvvDNv3jxCXVRUFAfDpAmI+T5LAoGgzx\/yBoYxqxHVMxneuXMnMBYUFETfkUql06dP75BheOf1119n0s4SGGEw1OSh4HaGXQDSZa+1M\/DjgwMZ7nIvmnjFjjjkZvjo0aMs\/qkkEgk45yNHjty6dWtXMIyplChk+Kq15GaMg2GFQgGPADC2qampds07PBrIC2QYhQx3lUjUytE7xcEwvGMX0Q4reJ7h1ta2A+dlyDCqBzIMCg0NJT1PhOSGhgb+DBOfmdWzRZ4LhNuuYLhMVv\/zGSn\/+g1NLSeF6v\/uKWYO6vI\/146TVS6GKi1tu2OkzCFoFDLcVYqNjQ0MDGSSzJNh6o3TiJrE2NTBdjvD2WWm+NzqDQdKeAJJc6oOX1CcEqqdwpiey4URIJJ6FX6gVFhkQIyRYQ9JJBJRkil1HTJMrO7IkSOJASeWmXZ0uZdhgIogcSRewcx27BBg4tbW1Fv5Y8w6l1MY09xJOCk9Tk+6TY1GI7LqjQwzqaOIdsgwy\/CyzLIbGWbCAGywkpa5AaahKcFYpTe7cC6eGDOTn8lJex7Gc+fOhcd0dna2U61iYmLg+c7sBKVJBBwjI8iw0yKI0gvdIcOXGIPMdEiJBtXuYriospaJAWGDNWnBlmGFzmzbvQQYA2Yc5wL87J7rQpaurtHaYecZc\/YCPSlgDKUnMUzY408y3Bvz588nNw9zPBIe+hyZQsiwhximPVvr1q1jEesuhs2WFrvzE\/gbN\/7OMBDu2lwIIp3JYrct\/Ak9j2FnSSbQMp\/y8GMPS8X1KMPAHis7miY506vMh2HqQttmX3bR2JK70oyPHj0a15EORF+ESxTmkpZ\/9k1Y1wsQCvSsBg0a1MdG8DHg3uATrDEjtUceeaQnGeFuYJhOdfDz8wsICKBd0zRi4ckwDWxYaVtdxLDbLnefPv7+\/h3esvBHhXmx4BET51lNmjSJBTBY5oqKCj7XHG4VeoPB6543H8bTvrRUKl26dCklGS4uBC3MIWKeDLM8cJ9geMHCxX2uuXbovx7HrtTO+NL86WVajqioKHD6Zs2a1cOM8CWcP+wx7fh577XX973m5tuvGzrqk7A1eEFcYBjcE6foZT7u4bEORrgndUcjwx5Vbn7Brbfe+uX6jdfefs+\/XpgxcHiAucmCl4W\/IiMjnR1YYva5TJkyZeLEicuWLeuRFwcZdk4uTO4zmmr6D7j9P3PeqqmpAYY\/Xrfp8ImYbXui8GJ6TD1vPAkZdh1IFyYV3Pf3fw7+h19qgQoYvuamW6JPn6uSyu56\/Jmth3cWarO4i6WliZT6JjN9bVs6PI4bS74mHf5PrIrxoS86NDS0R3rRyLAnZhR+smLF3\/\/5\/6Qy+eHYipoaE2FYrVZv33f4tmH+pSqRukHGUVrarKTUNFjg\/1OnT5LuwHnz59JfQeE+iBuLqkFaVVeWrU46J\/7VV75lCIO\/\/vrrHnwbI8Nd2GT5yi+uv+HG0tJSsMDAsExbRXxp9WW9u2LV9Ddm11trOEpre\/ZHe6ltbBaXiwcNGpQiTGlobBg8ePCBAwfob7kP4sZisGhKDXnlBlGa\/KL3297U1NSYmJj58+f37NsYGXZCgMuhWDnPynsOHr72uusi9+6vuax9Z0UybcUt9\/+DMgx6afb88G3f8WR4x44dU4OmwmtLs+WlKS99EfaFhxk2WnSVdaVghNsutXo5wzSpvmdb4N7OsMbQtP1EJf\/cSagpLDL89+cSPnnIarXmb\/fcG7JiJQG4uFK58WC2Vq+9+4G\/E1+aCDzqAcMDqut13Aw3W1tqGprjE+JHjhyZl5\/XLQzXNBuU9ZWpitgaix7+QO+3w71HfXotwGfTtZt\/EfOc30MAhnJMoID\/uTFWa7T9B\/71tVmvE4BLKzX7z+eF780\/l15515B\/xAqS1Qw9M+vtTZE\/czBsrKkb9\/T4R4Y9qlSrdu\/eTdNjPOlL1zUbdU2KVMWF6kYV\/IgMI8NeATDZfIjPND0KMLyAJuRHDowfHPavwUP\/SQAul+oOnC8olymizpUlZFfd+ei47yL3MRmuksrWbfvx94Q4u\/CYLdb\/i9iem5d7992DLgqSra0tQCwNjD3DcF2zSd+kzlIJFHUS8g4yjAx7BcCX+M22ZQJMm5A3c8V2MP5m0+bbBt5eWia+DHB1VGxRmVSqN+qAYfj\/jmFjWAwTj\/rWR0bbetQpxcqGJiuB9r777lOoVPVmK2AMgTE41XX1dZ5h2GDRiKqzivU59B1kGBnuNsENyNxnjDnb1tGuZYBIibSONbmXvG+oa2ZVDvlsVb\/+N6enpwPAVUr94QsloioJoEsZ\/usDQ1m+NNGLby0MeHk6k5zUEsWZTClBFKD914jHANrmlhZTnZkVDHcpwwaLVlJTmKtOYb6JDCPD3SnmDH4XxpY4JiEejD7+l743\/PDTDuJF6wyaYlkZAZgyfM\/Qh+0ynFcoWhm2+uSFs5STqupqU5OJIPpByAcvTXmZvI5PiO\/fv79ao\/YAwzXNekW9JF0ZX2PRI8PIsDfKjZsPmc3m+wYPeXvhewRgg1EvUZVU\/wEwlPRCGfz\/t+FP2vrSRKdiL97+r3Esj5oy\/PLLU4lTHRAQwIyEu47humajplGWqjivN2tZv0KGkeGeJmNN7YBB901+aQoB2GQySlSlWoNGz2CYFA6GQZPnL3510Qe2DAO6pDt6\/Pjxefl5LIC7gmHSj5WujFM3yGx\/iwwjw94rFzKozU2Wv4944m\/3P6DRaAjCVRqJWq+0BRjKw6P87frS1KN+J3R1dmmebY4Hd3F\/GNykKdClSUxFdn+LDCPD3gukCxEyBMD9br7leGwOMcIKnVSuq7ILMJR\/jhydkZOrdizA+L5RY6lH3S0MGyzaUmNegTbDUQVkGBn2kDwwpWHpytW33Hrb+fPnL09pqFHrVVVaiSOAodz+yBPMPC27euGthaNeeKW7GK5p1lfVlmWqBHVWIzKMDPdwhncePPqXG25aEx5OpjRUG3UV6rJqxwDzZBhM8eDRE46eOel5hmubDepGaaoi1mjRcVRDhpFhL2VYX9O876wT+yrd88CD0\/7zGnGhD54rB4B1Ro2ek+ERY5\/h8KWrpLLPv\/7Ob0LQ4uCQca+2T2lqbKnnU9zVj1XdpExTXtQ0yLlrIsPIcJfL2tImyNP\/9Hsl\/yZSbeNJoXrDgVI+W6tUyhVDh48cO3Ys0Ks3mM4JpV8fzNboVdwAQxn1zMSSMrEtvQC2VCb\/1wszpry9VCQuVWqUIybP+PGXQx6bVFhnNRksmiyVQFZb3mFlZBgZ7nKAL+bocsU1X0WV8hwBBm7JBg589lUyN1kGD3984B13SqVSAPh0ivRcRsmqn7PJCDB3uW2YP7XDADO4zZGHo0e\/8tbdT0w4fTEB0KU141NTX5z7bnWdxlMTg7XF+mxRdQ6fynARUuUXZDXlyA8y3FUAE3R5bj5GAb7Eb1+lzd9vJwmVBODkfIlMW7H7dOkxQXmHGN\/s98Q3P\/4c\/N8N1w0d1f8R\/1X\/d2jPr7+HfvsLNPw9SaKp1jIrL1z83siAsR4A2NRcXVEjylEn8ancYG3v7T9ZckDfqEV+kOEuBPgSv83HFDozcwsl5r5KzD1QqD5Z89XA22\/fv38\/ATi1QFqhKSOplHKNpkOMr7mxf05RAbwoqVT+Gl9eqVDTNExbjItLi6+\/4cblq8K6FOBaxsRgPvVb2lqUtdJKYynCgwy7X9llJiauFMikfL2jXcvgt3YzqO1uVvRT1C833jpwRWgohMFJucr0QsXlfiwt5RAwjhFWsMwps\/Tp0wf+l6rUFGDalmCcnC9l1v9izZrrbuqvqJZ33bxCbaM8TRGnN6v51Le2NZvM+pLqPCQHGe4SseYPundKA4TBd9435PlJL1zJiDYYmAmVlEPuQhiGotRqWNMhHJV1GzacSbzYpf1YdGIwd2lqaWyymgs06S2tViQHGfaE3DilwVhTO+rpCUP\/8Y8\/EiprqjQSVbWCNaWBP8N6m+kQjopSoxz9yhtd0UdtsGgLq9PFxkI+lc0t9YBukTbLbG1AbJBh39Pp06chDB702NivftjVYUKlswx3WFJzsu4aPrqootit\/Vg6sakgT5vKp3JjS13bpbay6gKjWYc3AzLsexKJRO1r36SkFJdXbN65Nzk7o0QmcgFFlxmG8j8jR74fssyN\/VjyuvIMZTxHQiWrH0teUwEFbwZk2PdkNBqHDBkSERFBfqyz1Pzv6pV+zwaBbfQYwzlFBeNnvHl93xvOJpx3SxisbpSmK+NsJwbb78dqbQbzC0YYbwZk2Cc1adKk4OBg8rq51ZKtSjZadBCdjpj8H72pWiQu7VKGIRhWalT3+j\/75fc\/rNmw5snRT3QeYL1FlaFKsDsx2LZYWs0QABdo0tsuteLNgAz7noDewMBA8hpu4kJtJjOX+HRiwh2PjQ9Z85VEWtEVDP9y6vQ9\/s\/GpafQqYinz5\/s\/AJ3udqUClMJ\/34sALjJasabARn2PUVFRYEXDb40+VGsL6qqKWPd5eXKirc++nTD9p\/i09L42GQ+DMNx4Girt\/7fw8+9cuzCOff2Y5Ua8gp1mfz7sUS6nFqLCW8GZNj3lJKSMmDAAJFIRH5U1UnLDHkcd\/xHG74Gmzx+5nyIXe2aZZ1BK5FVAMPlVRJH5VyiYOJbC\/s\/4r9sTXhGcY6jsu6b9Ry\/dVSyynJTxEnHs4+mF2fxqZ+Sk3si+eRp4dnkzBxmySkqxtsDGfZ2qVSqQYMGRUdHkx9NZn0+vzEYMJvyagWEyn7PBq346luIZs8KEoh9LpdWKDQKYFilU9FyMi6uRCJ+PfiT\/3lx+v+bMLVEKt66b2+lRqoyqRyVSnXl0H8O\/fb7bznq2BaFSVFuKLkoOS03yPjU15kMYnV5ZmVqtdHELMBwuVSOdwgy7NUym83+\/v7h4eHkRwgFs1VJPHOJry7vnp+1cedOiGMDZ82\/1\/\/Z64aOqlJJx8+cBwz7v\/pGXlkRvANl9CuzBTnpoZu+\/eHQwdSibFLyyq8kXcAL+iaz7Izac8stt5xKOM+szFE\/VZSdKcmMzjokyE\/hc\/yk3KwzQkFUwuGLaRlZBSJ6ZRQabWZ+Ed4hyLC3a+bMmUFBQeR1S6s1X5PGM5fYUZFWS3PFBfBi92\/RwPCe30+J5FJHJT4ns0zRPoO3sLI4o7TIUbU3317wyn9m0Mrc9YtV5edLzqZX5PI6vlSeVy4\/nRubWyY9FZ9eVikjl8JgqgEj3NqKvdPIsHdr8+bNI0aMAFNMfizR5fHMJXZU1DUqiC3rmk2V2iphUR4wLDPoHJX8yvKs0nxoRSpz1BSr5HHpQlKZu77UqMqQp6XL0vkcX27UK\/T1yZIMsVqTUyJLzbkyJtxksSRmZDc0Yu80MuzdOn36NITBFRUV5McqU1m5UdSpDXvN+tTCTPgfSBbkZkr1Wg6GSxRSYWE20E4rczBMK5PHhMP6Rk2+Ji+xQsDz+Ep9Q3plXqG8orhCe1GYRawu\/A8WGOww3iHIsFcL0AWA4+LiyI\/6Rm2hNp3MtuNZbKf1gQUGYPQNuqT8zAqtCiBxxDD8FupATWZlR4VWLpeVrt2wxnF9bUl16UXJ+apqFZ\/jKw31QC8wXKWqOZeUQa1uTlGxVKnGOwQZ9moZjUY\/Pz+aUNnYXJ+jTia5xDxXlrRdXDJfUgQuK5AM1g9sIOHELsNgEsEwAu2syvZ94z8qwynk2qo77rxj75FD9lHXV10ojynXyfkcX2GoKVdrE8vT5Lq6M4J0anXLpfLCUjHeIciwt4uVUJmnTqW5xK4xLFZJRFXtuVAQf0IUSlGxy3CqKL9cVWFb2W6hlUn9r77bMvQf\/4Dw2CYMVlysOF+sKedzfLnBWKU1JoqFMl1NQkaBRKogl4IMJuHtgQx7u1asWBEYGEj6sdoutYp02cxcYhcYVhjkWZf3WymRizNKipi02DKcJS4prCy2W9m20MrM+i+\/+sq6bzYyq1UZlKmy5Bx5Lp\/jk34ssMBgh1Pzy\/OLy8hlAV86PjWj2YrT\/ZFh7xYrobLcIKoyldnuWsafYQg4Uwsza5oMQHJSfjYLGBbDIrk0XZQLLq7dyqxCK5PHBK1fWCH+kwU2qHNUWcmVKXyOLzdUK\/UNqRXZIqWsSKJJzMil\/VgAcF3Dlen+8IAbPHhw\/\/79tVpvXwfPhz4qMuwGkYTK7Oxs8qPdhEqnGAZ000XZ1fVaKIK8zMpqDQfD5RqlIDcdmjiqzCy0MpyFs762SCuKr4iTGjR8jq801OfLxFlVhRKF8YwglVrd9LwClbbaq8CwWq1Tpkzp27dvamoqMowMXyZWpQILTBMqay2mfG2a7Zx4pxjOEefLquWADcADCNkyQxkm\/UzaOg1HZVY\/lvbyWtOO6v929vSmbVvL9RWx5WcqqpV8jq8w1IrVGvCiFbqGM4L0mrp6cilEYklZpdTbwECGkWH2N+3v7x8WFkZ+5Eio5M9wsayUpEBBMHwuPSUhN9O2AMPkxcWsNNnl1So5KtNCK3PU\/+X0iVtuvWXL0U1nsxL4HD8uI+P3+KTI8\/tOJQh\/v5BCrS68ACPshWAgw8jwn7Rw4UKaUNl2qbVAk17dqLKbp0ERtTRbVoSu2L17N3kBNML9tHHjRvJba4uVJFQSX9fRXIL2OQ+XX2j\/2MCBozItWsZuD47qy03Sj9YGPxnwBJ\/jq01qndGQXH5RUa2qNpqoBYYXiRnZtgmVtmBIJJKBAwfCQ1AqlS5durTPZQ0fPpwFGK0GBNJqcN327NlD6wgEAniTfh3MhiNHjmy4HJOHhIT0sVFUVJRrDCuVytDQULItO9mZnX7snTt3wjv04U7Fej82NjYwMJD+OevXr29oaLD9q+EFqQZ\/SNc9U3ojw04lVBJEGxobxgeO\/yLsC3j9QcgH8JXU1dcBz3CvqDXqltYWs6WJ9DZxF\/g6u2KV2ZpmPdlp5cnRT1xIPN\/hRg3wV9gucAfBcHxqht2ESkcMz5o1C\/6Hm\/idd96ZOHEiuaGZGLOqwYt58+aRW5\/ywIdhYAZOQaiDg8DrxYsXFxQUuMCwUCgkxwF0bT8267x2XQDCM\/lz6KdiNmH+1fCrUaNGdalf0OsYjouLGzBgAE2olNVIxIYCjjueWOCXprw0NWgqvBaXi9vXxxOmwOv4hPjLDKsA4Ja2Fj6wdQXDdc1G\/jutXP6L2jdqYO2WBLY3NScfbDJPB5XcpoQ9eu+Sm5tJo91qBFp6ND4Mu9GXhtPNmTOHyT\/zY5OzsIw88xPa\/WDETaBN6F\/NehYgw24QK6ESDBFJqORmmMntjh07iBGmrw1Gg7XVCmBkluTZn\/rHKJThmiZDh\/WZ8wod1hdlp5alHs2ISinMoE04Dp6YnX06Je5wwtH4tCzmvMLCUjHHxGBHDLPuUds3Ocwavek9zLCtyLnoB7B9EjERZeFq9wjkR4\/F5L2IYVZCpdnakK1K7HBxVhbD4EgTjxq8a7hXPvv8s+aWZvgRTHGeRMwxqZDM+6MMpxfnctdnzivkqF+kKD0tOplbVcJsEhL6UUTkLjuVpbIcceXp3NiCcgVzXmGlXJlXXOoUGKy7llnNlmFWNUoCcae7hWEIiY8fP\/7OZRH3nn4A1hGYP9KnD\/GiqcgR6Kd19Fcjw50VXNOFCxeS1y2t1hxVMp\/FWSmuk1+aXFhUCC8OHDiQm5cL4fFjjz1mrDESZ7up2cKdoUHm\/RGGCyuLs8QlfCYh0rnBdutXGZWJFXFF6hJmE6gcdeLYX++6i5X7Idfr5bq6RLGwUqtnzisE\/xm8aO6JwW5nmNlF5GGGmb1rTDkyvMyPRxm2K2S4a8WdUMnNMJTKqspp06fRbwvseeTuSPCi4VfNLVYwwtaWFj5TBaFth3ODmfMKOeb6So3qNHlyliyX2YRWXhy8bM68uax8rGRJRqlKyZxX2NBoTszIbrJYOhyK6zF2mJwajkxDYtsPyfxITJ55fgZk2P2Kjo4eMmSISnVlC7VKY2mFqZh3D9DVEeD4hHgSDLe0tXdEQwwM9ryxqR1gDobpvD\/Sp8U9N5hZmWtusFGTq85NrkxkNmFWFqvkT40dQ+dCqPQN2dKiXFkZc14hYAwA04EljzHMiod5htZuYdjuQWw\/JD1IVVUV6y+yGw8jw12r7OxsZkKlpl5RrM\/m3+vLZBiCYdI73d4R3doCJAPAYIc5GGZOFQTM4OvnmBvMrEzqO5jrqy3VlcZLYqUGDW3CMTFYoa8VKWWpFdkkH4vOK8zML1JoeHW6dJJhVr90TEwM0\/MkrZhcSaVSeMe2X5cPP84yTHuVWX8L8fYDAgJYw8XERNt2OMfGxnIYdmTYdbESKussNbkaIc9NhmxzPF6a8hIEw03NYHithGSL9QrAjhim8\/7I3F3utXiY8wo55vqWGyouSs6V6+S0CefEYFOFphrCYPClmfMKyyqlIrGE52XsJMMQerCGkVkoEoqYA8ikCQsVOjBLjsOR40GOEMDQ\/PnzyaGYo7twLprpYdeZtx3xvsRIOCEjzHTEmzW2hAy7QfB1QgwMkTD5sclqzlULjRadU6OvrIRKi7UZCrxosoAJbqYA22WYOVWQzN3lYJhZmWOub6VRHl9xgUwMpk0cVVYYDP+ZOTvilx+rtEbmvEJNtT41J9+pK9lJXxpMK6V3+PDhrPQMeBJGRERQZtavX19dXc06FFFoaCg5CHwYu041ZdhRhxNoz549zCStjIwMaOLI4bc7xguuxKhRo2ieFhwkJycHfWn3i39CJU+GwfyCESYkmy0WJsC2DDOnCtK5u44YZlbmmOsrNaqSpYJcRQGzicOJwe39WPX\/3bLu4UcfzRZJ6bzCuoYGCIM9MzHYw3eze8XHdfcG9ViGN2\/eDN4UTags1efLastdyIJqbKknxdRoKJAUN1rrdLXaXHGJptbIKo6mCjLn7tplmFmZVZ\/Vj5WjzEqpEjKbcEw8Buc5q6owXyaeMvXVN+a9TaDlSKhEhl0YZ0aGu0pxcXHMFSoVtVVlhvxOJSQ3GVIL2\/t++cz1ZU4VZNW3ZZhZmXNusLZQW5RQEU\/7saAJ58TguiJFZXpFnkLXcPC3mPkL3iaXAlxocKQ99kX4LsO22VrIsOfESqjkv9OK44RkU1Zpnsqk4jPXF0pyQTaZ92db35ZhWpljbjCUMn35BfHZqmoVbcJRWa43kgXuwBTHpmTTeYXcCZXIMPcQFDLsORdoxIgR4Ehf+dHakKtOcXanFVYRVZWIVRKec30TcjJEsjJHc33p\/GHbyhzHTy5OP5x+8Hx2ErOJo8oXszJOCZIjz+87KUg+nZBWJP7DGcGdVnqoehrDriVUchSJurKwUsR\/ri8d3bVbn84ftq3s6PiKGmlyVWyJppDVxPHEYL2wIrFKK6s2muhQMNlpxc\/Pj8YXKGTYGxUWFubv70\/7sS5Ifv+teG+M+LDL5Vjhvp9Tvj8tPtSZgzALMOxU\/fOS6FTFhRJ9Hs8nTktbS4WxRFuvZF4WutMKuCd0S3QUMux1YiVUguIrfy83FblcinS5Z3PPFOvzOnMQVmnff9iZ+rEVv2ka5LwBtqrrZFLTn5Z0Z+20As+4yMhIvO+RYa8TK6GSKFl2rjP9WOmibNJd7HBPUHtzfbkrA8N8Nxy9XHYLdjGPz1GZbDh6MOEIa8NR1k4rcIno7EsUMuwtMhqNYIFtx+I7w3CuuEBaLe1wD1GnNhyFAgzz3HCUlKPZv9Ljc1WWyXLLpCdzzrImBuNOK8iwb3REMxMq3cIwAFMsK+Wzh6hTG46SsSU+G47SElN8ihyfo\/KVjRrEwgpNNWtiMMdOK3RZfBQy3M0KDg52NPzoGsN0pxWn9hDlWRkY7njDUUaJzomG46uMSo7KSsOVjRpYE4M5dlpRqVSDBg1i9h2gkOHuEYR2fn5+jkyKCwxX12shDK5pMji1hyj\/ymQdDz71SZOdF3cqDUqOykpDfa6sLFtaxJoYzNxpxdGzb+7cuQgAMtydYiVUdp5hmlDp1B6iTlUmdrjD+rTJuZLfOCorDKZSlTJZksGaGMzaacVRDAJX7\/Tp08gAMtw9YiVUuoVhcKHBkXZ2D1GnKpN4uMP6tMmBlD2OKpMNRxPKkuS6OubEYNudVjgegikpKcgAMtw9\/VjMhEq3MEwTKp3aQ9TZDUeB4Q7r0yZw8H3J+xz1Y8l0tWSBO+bEYLs7raCQYa9TUFAQn1iOP8PSainZacWpPURd2HCUex0PZhNZtQwOHlN8ytGGo0JJVolKztxw1NFOKx0+EDEBExn2qMLDw5kJlZ1nGMLUjOIcwMypPURd23C0Q4ZJk\/Ze68sHt8uw0lCfIy2Gwtxw1OWJwVFRUcyJ1ihkuGtlm1DZSYaNZn1qYSb879Qeoi5vOMrNMGmiMqnowW0ZVhhqwPyCEWZuOMq900qHmjRpkt0BdhQy7GaJRKL2\/RZ4d8N0yDDYXrDAZCaQU3uIurzhKHvuob0mzIPviNvB2nD0RIIg8vy+GEEqc8PRTk4MhmeibaYqChl2sxwlVHaG4XxJUaW2yoU9RF3ecJQ199BuE+bBz5UcZ84rVOt1yeUXldUa5oajHe60wkdxcXGYuYUMd60cJVS6zLBYJRFVlXTFTqJu3PeQ\/glkw1GRLqfW8ieHmc9OKyhkuPsVHBwMMZuzrTgYpgmVfPYf5LOfoN05TLb1ybwl\/vXpvCVBVtaviSeOJ51h7lfIc6cVp6IVEFKBDLtZ3AmVLjCsb9ClFmaS7uIO9x9kTktyqrLd+sCwU\/XJvCVRlVJYXHQuX1Ak0dBpSfx3WuGv8PBwXCQAGXazUlJSuBMqnWUY0E0XZUPMyWf\/QdZ+gvwrO6rP6pfusH5M8Sm54eoCd8xpSfx3WnFKI0aMwEUCkGG3iSRUupzTK5RdyNUIWeV41q8J4vPwIl587lDKkQSJgKOcLT5\/OOVQtirZqcocB2\/vl3am\/hnRWZmuJlEsrNIaS6XVF1IySejr1E4rTik7O5v\/6B0KGeaS2Wz29\/fvMKGS6wjWhlqLiVkyS3JzxYXwQqqT\/RZ\/sVSpEqs1jkqRTHZCEK+p0ThVmfvgwLBT9dtTKSuy4QVzWpKzO624gDGCgQy7QTNnznTv5Di6OGuTxXI+KUOiMKr0ZkdFoWs4n5xVbTQ5VbnDg7ePLTlTHwq40Mz6ntxpBYUMd6pzhWdCJU\/RXGLQRWFWcYWWg0koiZlFZZUypyqTfibu+oRh\/vVZx\/fkTitRUVHoUSPDLgoCYPeuMkEXZ73UvkZcSUZhBTczUAGqOVuZT\/32eUvO1Gcd35M7rSxcuBBcISQEGXZaziZUdihmLjFYM7Bp3MzQFW2cqszz4MCwU\/WZx\/fwTivgBA0ZMgQXCUCGnZMLCZUdKq+4tFLevmY6YBwjSOdmBuJSiE7BbjtVmf\/BgWGn6tPjd8tOKwCwn58fQoIMO6FJkyYFBwe78YB0cVZwpE\/FC6WaOu6uKbKijVOVnTo4MOzsh7n0x04r3ZJQiSExMuyEXEuo5BDNJQadTUwTy\/Xcdu9iWp5UqXaqMvHV+dcHhp09PjOYR6G8l2EyE92NU2eYi7Om5xb+elYAvitHOZWQRla0caqys\/WBYafqs3Za6RZBYBweHo6oIMNcIgmVbsy2Z+US1zU0gE3usLhQ2dn67etaOlOftdNKd2nmzJmIMTLMFXG5fZHUzPyis4IUsGDeVoBh\/pXhMeQlO62Q7winNCHD9v20TiZU2hVPW+f5Agw7Vd97bpTIyEgcLkaG7TtpverOIP3SPipcNw8ZZgvMr3sTKpFhFDLsOZGEyt62srGvMwzfF66AiQxfuRXcm1CJDHtM4DrhIgG9nWGj0ejn59c774MewDBJaMf8rV7NsNsTKpFhDyssLMztQwkon2EYoin3JlQiwyhk2HOKiooaMmRIb16LHBlG+TDDbk+oRIa7V3FxcdhH3YsYVqlUOKG8hzFMFgng3sYd1UMYJgmVmDTf83xpskgA5m\/1fIZxZaYeHA\/Dl9sLx\/l7F8ObN28eMWIEPqp7KsOoHs4wBEu9MKESGUb1EIZ7bUJlL2QYvK2wsDD8fnsUwyShMiIiAi9xb2AYFwnogQwHBQX12oTK3ulLw\/Pa398fv+IewvCKFStwG9teGA\/DUxv7PnoCw9HR0b08obLXMozqCQxnZ2djXNTLGcZxRB9mmCRUgh3Gy9qbGZ40aRIuEuCTDMPTF2JgTKhEhlNSUgYMGICLBPgewwsXLgwKCsILigxfuty5hdm1PsYwJlQiwyynDDDG+8FnGI6Li4MwGAcVkGGUTzJMEipxEikybFcYFXs7w+AsgQuNCZXIsCPhwg\/ezjAmVCLD3CIJPxgYeynDYWFhgYGB+PUgw9yaOXPmwoUL8av3CoYBWhr3YkIlMsw\/JI6KisKv3isY9vf3v\/HGGyH6zc7OHjBgACZUIsMoX2IYfGYAuM9lAcD4ZEWGnRU8+vEG6E6GwYvuwxBEwuhII8P8VVFRAY9+xLg7GQ4PD+\/zZ0E8jF8JMsxfuEhANzMcFBTEBBj86rlz52JIjAw7JWAYswm6jWFwhAi9gwYNCgsLw\/wbZNgFwUMf75zuYRh8Zrj5yKbBOCCMDKN8kmHMmEOG3SUwxZhg3w2+NKozt2zgHyJd+kS99rEIJmHIkCHoVHc\/w1arNTQ0FG7KkSNHNjQ0wDsCgWDgwIG4EoCtWD2CpFuhN0clK1aswEUCup\/hPXv2REVFAbd9+\/ZNTU2VSCTPPvssgRllt0OBqc2bN\/fmCwLPLz8\/P\/SovcKXhi9j8ODBu3btmjVrllarxQvKxxT3ciNMhItGeAvD4FFPmTIFTHHvzL4kfz7LxtLggimIftEIo7yRYVBISIjdu7Y3CCKIgQMHQlhBfty5c2f\/\/v0d+SMjRoxAI2z7aEOPupsZJjcxx43bsyUQCGgfHgBMugYcVY6OjkYjbMswLhLQnQyDJ7l8+XKhUAgY9\/KZTKRvr8OLMGnSJLxfWZo7d+6KFSvwOnQPw6GhoXDXkm4tMEfwYvHixb3QqSbOCJ+nGAJsK7LvKQ4Xe5ph4jcy40DwEocPH15QUNDbrqBUKoVHGO6g3RnhPNbujId7uYgPMn\/+fNvwGIVChr1dZGCJQksMMq5w4rIiIiJw0BgZ9qhCQkJYI8Pc\/dIobkVGRuLW88gwyrcFDOMiAcgwyocFvrSfnx9eB2QYhUKGUSgUMoxCuSaz2bxw4ULM+kCGUT6ssLCwuXPn4nVAhlE+bIr9\/PxwFTdkGOXDiouLw\/V6kGEUChlGoVDIMArlssjKvjhnExlG+bAmTZqEiwQgwyjfNsW47ykyjPJtRUVF4VwIZBiFQoZRKBQyjEK5rLi4OH9\/f7wOyDDKhwUMY2CMDKN8WCKRaMCAAbjsFjKM8mFFRkampKTgdUCGUShkGIXqVuHK8sgwyocVHh6OMxORYZQPy2w2DxkyBBcJQIZRPizc9xQZRvm8Nm\/ejFExMoxCIcMoVLdKpVL1Zo8aGUb5vObOnRscHIwMo1A+bId78yIByDCqJygyMnLEiBHIMArlw+q1u7cjwygUMoxCeY3i4uKQYRTKh9ULFwlAhlE9SiKRaNCgQb1q31NkGNXTFBYWFhQUhAyjUL4qs9ncq8aKkWEUChlGoVDIMAqFQoZRKGQYhUIhwygUChlGoVDIMArlkpqaWxubWqzW1rpGq0JnrlQ3KqvNtQ3W5pY2+K3F2mq9\/AIZRqG8SC2tbYBuqawuIVf\/w2+S3TFVkaeqln+fP3991rLvclf\/XPzNIfHmw+KPfyiMFiiAZ2QYhfIWdHVGS7miPiZds35\/6azV6a+tyggMTvVflDh2SULgssSnFiWOXpQ8Zkny2CWCZ4IF495P+v6YpMHcggyjUN2qtkttbW01Dc0xqerPdhQt2ZIXuEz49PsJY5ckPrs85c11ma+tTp\/2WWrw1vxvj5Sv2Vv84ff5b2\/MmrM2a8mW\/BNJXjebAhlG9TppTRZhkWHN7uKXVwrHLE0Z\/37SCx8lz1ydEb6v5DeBMr3YWFhRI1Y06IxNNQ1WQ11ztanJUGsx1VsV1U3VNRZkGIXqNjVaWvLKa74\/Vv5yqHDc0oSAxUlvhWet318SI1SLKmt1NZbWNg7j7aVChlG9RTJN4ymh+p2vc8YtTXx6aULQp2nf\/SrJEdfUNVrBtfZmSpFhVG+XtaVNoqzfe1Y6PSwtYLEAjPDqn0XnMzR1Dc28wuc2wjgyjEJ1h1raAW7Y9Ev5hA8Sxy9LWrQ593iSskxWbxfL1rY2U32zsa65pKquQFKbLjKeTVeniQx5YlOJrE6qaWxDhlEoD1vgzBLjxxGFgcuS\/h2cGLq9UJCnN9TZMb9mS0uJtPZ4omrz4dJPd4qmf5H2ymfC2WsyXvgkZfLKrOdCkqd\/nnYuXYN2GIXypNrAlr79dW5gcNLTy5LX7BZllxqbmluZNZqtbTpTU3yu7vOdojlrM6Z+mj52acq4pYLLPV6JEz\/OmBaW+dInKWOXJPxnVXpCbjUyjEJ5ThJF\/do9xYHvC8YuEez4vVKqaWj5c79zY1NLbKZu0+Gyl1cKn35f8MTCxJdWCOesyVy8OW\/br+UHzskOX5AfT1aB730oTv5rgqJK3YAMo1CeUGtrm9rQFHmqavz7yRAGr91XYrTxnytV9V8fLHv9y4yxSxKfXiqYszZzY1Tp78nqoqo6qNzc0mZuamllMG9p9rpMaWQY1WNVU98cdV726uepgcsSg7\/LK5bW\/Sn0bW5JLtB\/vqMoMDhlzJJEiHu\/\/UWcKzbV8uumRoZRqC42wm1tcdm6BRuynlqcvGxLXkaJyWK9GgPXN1pjM7ULv8kZu1Tw7PLk0B8LM0qMcp0Pb1+MDKN6mqpUjUu\/zR2\/LOn1\/2acTFZB0Et\/1dTcekqonhue6b9I8Mrn6d9Hl8t1jbZHIB5zS0tbvbmltrGF1Q2GDKNQXam2tgvZ2okfJj\/1XsK+szImwKD4HN0rn6UFLE58KzxzT4xUa2hitba2ttXUW0ukdefSNRBObzpcFnFMsutU5Zk0jcamMjKMQrmd30tSTePnO4vGtQ8FpeZLapm\/LZXWLvgq6+mlCc9\/mHxaqDbUNrPSPNR6S3SCcu3e4hlfpL34cfKLK1LHLE158r3EJxYKvtglsmuxkWEUyp2qN1ujLshf+CTj5VDh\/rPSusZmyra82vzlz8X+7yU8vzwpOl5Z3\/gn+2xpbo3P1YVuL5oQIgxYLADOA99Pmv5F5qLNuW+syXgmOGnTobKa+mZkGIXqWpUr6z+KyA9YIly6Jbe4qo7Bdsv+87LnP0x7NiQpdHshyys21jXvPSv9T1ja2KVJgcFJQZ8Kv9gpOpehTRMZyuR1EmV9mawejozzllCoLldMmmbm6vSnlwl\/jgEjfHXFnDJ5ffB3eY8vTPoookAsr6fvt7ZeUujM234tf255yjgwvGHpGw6UFktrLda2lj8PBbe2eu+kB2QY1UPU3NK2bm\/xv4MT56\/PLFdcBRXc5j1npVNChRM\/TDl0Qc5cSUdRbd56tHzqypQxS5KWfpt3PlPL6gPzCSHDqJ4giHgrNY1vhmcFLhNsOiy2MEaD8sQ1r61Ke3pp4tq9pVWaRmpQGy0tUbGyKSuETy1KXPlTYZ7Y5J1pWMgwqncYYWvbuXTtCysyJn+S8htjyatma+uRi\/KxSxOfW54Ul6VjNsksMU77In3c0oTZa9KzS400ldpU25xdZopJ1QjyqpV6c0trGzKMQnW5ahqsW49KxixNXbAhK1dsou8bapu3\/CIe\/V7izNVpEuVVBxt85shTVf6LBC9+nBKXoyPhLvxT6Zs2HRJP+jD5lc+Ec9ZmrN1bItM2trUhwyhUF0uhM288WPbCx8K1e4tV+quJkyXSuuDv8gJD0tbsuTrtoa2tTZBb\/cbazGeCUz76Pl+tv9JNXddo3XWqElzrcUsSoAQsFvz7g5SfTki83MdGhlE9QYBfqawuMb86vdjY2HQ1GD6brnn1s9SXPs2NTlDSILmusQWAD\/wgfV54VmLe1SnBEC2H\/lgYsDhx\/PsCUgIWJ8z+Mr1C3YgMo1DdoNbWSznlpqMJyh9PVFSoGqhLXKlunLM264mFyZsOi5lrzVaqGhZtyglYlMBgWLB0S26prB4ZRqG6keS22gYr0x8urKjZGFW6+Nuc4wIVc0pTtanpvU05Y5dcAfjpy0t5bNhfojU2IcMolBepydIKWJYp6itVbCc5r7xmwgeJTy1qD4bHvZ\/0Wlh6XLau2YrxMArlI2ppaUvOr444Jln5Y+F\/fy7OKjV5\/2dGhlGoP6mtra2usUWhMxvrfGNZD2QYhfJtIcMoFDKMQqGQYZRTkkgkAwcODAoK6rCmUqlsaGhwthUKGe7VMpvNU6ZM6dOnT9++fffs2eOomtVqDQkJ6XNZYWFhXcGwQCCAg\/fv31+r1SLDyDDKCYYHDx5M4KT8OALMAwyPHDmSmGJkGBlGOcEw0Pv6668DQlFRUXargREGQz1v3ryuY9gtrVDIcO9l+OjRo8CnXWYITmAht27digyjkGEvZVihUEBgDMY2NTWVVWfnzp3ERJMXLIZjY2MnTpxIPe3x48cXFBTYpRGC6qVLl5Jqfn5+rPCbBa0jhuF0gYGB5CDwadevX0+7wVDIcK9mGCJhu4hyVyDvgGbNmvXOO+9Qupg+OaERoIX\/gTqoSXxy1qH4MExORw4Cp4MKzBAahQz3doYJNqyeLdLVRGBjMUx7kpmGNyYmhtU9Rg5LHHUKG6sXmg\/D1KWnB6G95Y7CeBQy3LsYvnS574qJBEDCdLBZDDvih\/W+LXv0yLbVOBi2ezoMm5FhZPhPDHMP8DAZZuFt6\/FS1B1hRph0VI31I2WeeNFUxC1HdxoZRoavMMwik2X6mHCyGtoy3GHvFDfqjhi2K2QYGUaGr6JICbT1gbvdDts9HQoZRoYH2\/YtwTvr1q2z2wvNHQ93GOi6Nx5GIcPIsB2XmKZGs97n3y\/NtN52+6UdVeNgmBWrU8XGxrJGpFHIcG9nmGZH2x2eZVpmSjtzfJjl8RIaAwICyPgwVKM5IbbDyNxRND3d+PHjaYcWGmdkGBlmM0zetA0+7SaBgEUdNWoUTZyaM2eO3TwtaCWVSmkSyPDhw1kH55mnxTodwJyTk4PfIzKMQqGQYRQKhQyjUMgwCoVChlEoFDKMQqGQYRQKGUahuk8NDQ3btm2bM2fOjBkz1qxZU1lZidcEGUb5jORy+ZtvvkmSTEH9+vUbP358XFwcMoxC+YYFXrlype0cyQkTJhQVFSHDKJS368yZM4MGDbJl+Prrr1+1alVzc5fvURgdHS2RSJBhFMoVAaIrVqxwtFwBmOL6+vou\/QBarZaca\/78+SaTCRlGoZxTY2PjtGnTHDH84IMPSqXSLv0Azz\/\/PPOMYPmRYRTKCdXW1gYEBDhi+NZbby0pKem6sxcUFNiedMCAAbt27UKGe684ls665GBCYm8WuMpBQUHXXHONXYYfeughR5tauUV+fn6OHh9Dhw49e\/YsMowMI8MdyGKxfPjhh9dee61dkMaNG1dXV9dFpz558qQjgO+66y76Abpx2RNkGBn2AbW1tZ04cWLIkCG2IIFxXr16ddf1S4PPTE5Ex6WJHn74YaVSeeDAgTvuuIOuu9Kl7gAyjAz7fEj8ySef2DI8atSoioqKLjrptm3byFngBfy4atUqet6pU6eCd0CqrVu3jvoIH3\/8sdVqRYaRYTbDrG3N5s+fb7uKHce+Z3TtHnhBqo0cObJbjEZnpFAo6AZxRM8995xQKOxCPP7ovqLvmEwmuPjUMm\/fvv3S5RVFJ02aRD9Vv379CPPIMDJ8hWGynh7d1mzixImslSg73PeMMAy\/Jb8C28Wxubk3S6\/XHzly5Isvvvjoo48iIyPLysq67lxgUQmTv\/32G+tXYrGYjjaBhw\/OvK2DcO+9954+fRoZRoY7Xsmdz75ndC3bHrN7A\/Vju0iNjY2059lRHXi2Pvroo30c66GHHkKGkeGOGeaz75nd7Rd9rlsrOjo6ODh49uzZGzduzMjI6NLTTZ8+nXDY4RKfcOVpBzVLxNNGhtGXvupLQyTG6r\/hue+ZT29l2Nraum\/fvldffRWuGF1n\/7HHHuu6LWYqKyvJiZ555hmeTb766qvrr7+eCfAdd9yB8TAybL9Pa\/jw4XQ0kue+Zz7KMKX3b3\/7m+1fBxeki3qkH3\/8cXIKp9wWcL+XL19OP57Hcj+Q4e5k2JGH7GgbJKVSGRoaytz2gee+Zz7HMDe9VDNnzgRy3HtquiMHeD3OtjWZTKQtxMkeu1bIcLeJtUchT7yplaaE89n3zIcYrq2t3bNnz7Rp07jpJbrlllvcPh\/w\/vvvJwd3YZh38uTJpK0n07aQ4W4TjXLhlqVvSqVS4htT3gDpl156iXlPsKDls++ZrzAMvvH06dPtzhO2qxtuuGH\/\/v1gtN31AeCqujwzqbi4mLR9\/vnnPXnRkOHuFLGoRAEBAXTTIyaQxCzTPc1ohgaT2A73PfMJhi0Wy4IFC\/o4o2uuuWbMmDFunDwMDwWyroALbUeMGEE+lV6vR4Z7kcBa0i0LQX5+fmCWWRYVjDPNT7KbpHWpo33PfILhoqKi22+\/vY+Tuvnmm93lTq9du5Ycc+\/evc62jYuLI23fffddD183ZBjlFWppafn+++9ZwzM83ekff\/yx8ynK4JCTA0Ic7kJzOkTs+UuHDKO8Qo2Njc8++yxzhvB111333HPPTZs2DdwKbnf6ySef7Lw7TbOgBQKBs2137dpF2n711VfIMKqXiuVIA8DffvttaWnptm3bbrrppg57pzu50DRdLuvxxx93oflf\/vIXMtWhWy4dMozyCkc6IiKCaW+ffvppEvMvWrSIEMIh8MDDw8M7k0H9zDPPkEOJxWJn23766aesHkRkGNXbHWmANiQk5NLlnmo64sqt4cOHu+xO5+Tk0CnBzraFT0ja3n\/\/\/d119ZBhlNc50uA8nzlzhrA9evRoPgyDO52bm+va2elyWS4s6DNnzhzSNiUlBRlGoSN9dZkbg8EAv8rPz\/\/rX\/\/qiNvbbruNrp4BL5588sldu3Y5m0FNl8v64IMPnP3kKpWKtH3qqae68QIiw6huFsS9LEd64cKFZKzo3Llz\/fr1s9sX\/dBDD7344oskJYO5SN0TTzwBNCYlJfE8O10uy4VPPm7cONJWJpMhw6jeq7KyMrqsHBnv\/e2339oua8eOHXY7pcHqvvHGG+BvUwJZhIMlf\/nll5lJrHbFWi7LKWVkZNCl8Lr3AiLDqO5n+Oabb2Zmqmk0mkuXu4uCg4Ovu+46W0rhzeXLl+t0ugcffJAjSB40aNDcuXM5lry0XS6Lv+gim26fOIUMo3xMUqn0vvvuo+C9++67hDrwsR2tCw+2evv27VBt06ZNHXZ3HTp0CEJu2\/OCy+1ouawOdeTIEdL2008\/7fYLiAyjullgx+iar4MHD05NTQUv+tLlvR2efPJJR73QZDnLurq6rVu3cmzjAtF1aGio7dAxNOxwuSwOwQcgB\/eGC4gMo7pfBoPh4MGDQGNiYiIB+NLleRp33323o3kONBmjtbW1qKjop59+AqNt1+uOiIiwtcP8l8uy1ebNmz25XBYyjPJVXbx4sX\/\/\/nYZfvTRR23n91VVVYFXDAEwc4U6qGk7Hd+F5bJso2iPLZeFDKN8UmA5wbTa7ZSGCHnq1KmOVthVq9UCgeDLL7985ZVXZsyYcezYMdspTa4tl0W0ZMkSDy+XhQyjfFJ5opLpr82ymyndns4xNvDzr7et3rL92Nm4i8IMZskpat\/EFAJgMMsymYx65lR0uSzQsGHD4uLi+H+qblkuCxlGeZ2MNbUAG+AHEK7c+N2E2e9AuW7oqD+Vvz92zS23O5pteO2d97HrOy6jpsyC44es+QZOd\/DEmTvuvJN1PPCoi4uL+XxymrzNsz4yjPJtmZsswOruo8cBngWfrAKQhj0\/jSd4DwVMGHDnXx1NGB7y4AMffBS8em3YtHkLJsyYxSyjJk3lOOxf7nY4nvz8Cy+KSrh2fqHLZQHJXnWdkWGU2wR+LBg6IBZwHRo4xRFI941+BmCbPPtNgBDKxZP7oZgrUy4p02kpTzl2151c6\/IEBowqij\/MbOKokONv2bIeznX3PfdwDyb3G\/TAv1\/\/X\/AOth84IszOY\/513bVcFjKM6lqBpQVoJ89feufIQBarNz8yGlgNCf0Y4Dm4NwJAqsyK4UNdO3hHt\/fv18HU\/9EjH43\/9UeeByTFWHxh9quTOkgKufYv1971AP0r4HkEf+BXW7Z113JZyDDK\/QIDBbf1mOlzWdAOHff87HffJaaVP662pUWW+tPXn9504w1\/5Glc+\/ch98H\/dgaZ\/P5+cu+3zh6\/WHDkmTGPd7DU3q23Dh3lT\/+0Pn+5kvIJcXVMfBIyjPI9qbTVENnODg5l2dsxU2eApT126CdVfqzL0LJKU2XSsrdnXnfdlU7pG\/r2XbFk7vavVj5wn51V4\/\/50ODD28NdOEvckR\/8hg7hJvmJx4bt\/H5D0KtTr1joO670pd087ClwPbZE7i8ur0CGUd6OLkSGrH5jsLeLln8A3LKCWHeVRkkieLw0U\/rGG\/ru2\/YlvH9q35bB99pZPv7Bwfcc+iG8VZ7qwrn2fvffOwbe1uEq1u2h8k03gn+xMuxTVrfZsOengVfSjTAjwyg7MjdZbNGdNm\/B9ohvioUnuoJbZqkXJ4x54l9X3dr+\/ZKO76Rx8sP\/sNO3\/P\/+\/kBl2nGXz7h2xSK7vjpTR37aQOuD07F751YIHCDmZ8EMTz1kGNWdqpQrFn2+jukwA7pwvxpL47saXUed0oPuukOTd4b+Vvh75NP+j9kmfhz8YZ1rppiUurL4d+e86gjgIff\/zW4r8EQO7o1gwQwRB6tPGxlGeaiHGWI8ZqDrYXQddUqPGPZPAIxZIf\/CwRcnjGEyBv729+GftMhSO3lqMOaTnx1ry3DKiV3cDQFmCC4AZmZuye6jx5FhlIfopW4z2JMF7y\/xgMPsqIAt3fHNZ7RTGmLR115+rqFcwKpWePHwqP95mMnwjxs\/7TzDpACx8OCgB39q1HD+bcHNhpj5vtHPXOk7CJzS1SQjw71a4PJReu8cMW712rBuMbysgaX33ppOo9Prr79u9YfvNlfZ6Tyb99oUuiaeexkmBQLge+7+KxxZnnXKheZbtqxnknzwxBlkGOVOGWtqF3yyyqvopeEus\/MZEPr52zDbQNdckQThK0W9KxgmJf30HpfbgoPNJBlCla7ovkaGe6PAuyO9VuA5g+PnJfRCqSm5+Oy4J1mBbuJvO+zgUZH0v7ODPMBw5wuQvD3iG3hQkoHl1Vu2m5ssyDDKRcHdMzs4lPZadWPca7ds\/2ola4wn4PH\/qS4879MM0ziZ9niNmT7XjUNQyHAvUqVcMWrKLGJ+d+\/c6m13ecqJXQ89cO+fEh779zvw\/Zo2RVoPYJiUmOifiWt931PPXxRmIMMoJ5RTVEL852H\/npyTEO1tN3ddWfysoImsEZ03pr0A3rV9B9U3GW6fd1EaP2HGLGKQ3dJljQz3LoAnz37Te6JfZgECb7m5H2s+A0d\/ku8yTEpI6MfuwhgZ7l0Ad1GGc1d40bu3rLLrRfcMhqGsXhtGMO7ksBMy3PM7sch0fK8F2JEXXcvpL\/QAhinGNw97qlKuQIZR9rXo83UkBvZOgElfNMuLHv7w0A5HZXsGw1CmzVtAeqqRYZQdxcQnkV5o4blD3nkH58VGDR1yPxPg\/v1u+j78Ew4vuocxrMqPJT3VWyL3I8MotsgcBnDYvPP2BS\/6zRmTWV707Fcn2R0Q7qkMQzl26Ccy2oQMo\/4klbaaGGHv7Ihulafu2bqa5UU\/9MC9PHMbexLDUMi6AsfOxiHDqKva8EMk3BYL3l\/inXctGNsnHxvG8qK3fLm8Qy+6RzK8Zct6MvEYGUZdFcmpPLg3wgtvWQA15sBWOsGQetH6Ir6LcgHDb7\/+J4a3f7XSdxnOSYgmK4Egw6irIutOemdvFjAMjvQNffu64EXTdfMWzb06RREOtXvLqs6s49Ht8yLIIBMyjLoqMizcmTViu7Rkn9t\/2603X1mZ\/aYb+XvRNJw+vD2cMgyHomtu+WghE5tcGChGhnusyPQGrx1VslQlfxHy9s39+11zzTXB\/zvLVBzndOJx8YX\/vPwsYfiDd16v9cquO7TDqJ4ZD5MC3MYd+eFY5Nfq3DMuL391Yvem6F0bmYvm+WIpFp7AeBjF1uot2+G2WLT8A5++uXtJ2R7xTfsSou8tR4ZRVwWRFRkf9tosSyw4PozqQGS9uw0bv0RIvLlcPLmf5Gm5tkYPMtyTBc91Yoq9cNI\/FrokAMmX3vBDpGvfMjLcw0UWrwRXDT1q7yxkka1RU2a5vFAeMtzDZaypBSfNm5Mue3Oh84c7s2YtMtzzRdfxQIy9cx2PTi7Hgwz3LozBc0On2hvKho1f4npaKBcxhtjYaxMwe0knFlm7A9e1RDktur70nSPGHTv0E+LUDTvRnDs07N+T27+CkYG4vjTKxS6uae8tv7L9z+w3VfmxyJXHzC9djxaepJ1ZBA8ZRrUvD0D3W4LADCPkri67d26lO6et3Pgd7reEcoNU2mq68RLcXlu2rEeSu4jeoeOeJ9d5wux3cN9DlJsFIRmJkJFkt5eDeyMovbj\/MKprdexsHCX5zhHjVoZ9ih3XnYl7ITxh0uuWzmdkGMWLZLJ8D+3xAkuCZtmpqQsL3l9y8yOjPUYvMoyyIwjYFn2+jvR4kU4vuC9xIIp7ObtFyz+gXVZkecqY+CSPfWXIMMqOjDW12w8coQ428bEJzGiZ6UhvSOjH1Gcmhnflxu\/cuDk4MoxygyrlitVbtg97fhq9U4mbvWXL+mLhid7GbWVWzPaIb6bNW0AdZjLvFzyXnKKS7vqOkGEUL8E9CkaGaZlJV\/bsd98Fnr125T23rHQF3IIPQvKraIHnWsiab9yVa4UMozzqZh88cWbBJ6vIlEZawDSBfV69Nuzg3gifXnIAjG1M9M\/wh4C9JevFXg0oRgZCrAtRhhuzrJBhVDd3gO0+ehw8SZazTcqYqTPAfAEMEEVfPLnfC\/M6jaXx8MGgwIeE4HbCjFlMJ5lGuYTbbvSWkWGUh+xzTHwSBM9wx7Ncbpb7DaiAiQNsNmz8kiDU1f1k4OrDWcDnh5PCYwU+AMsxZkE7ef5S+EOOnY3zfAcVMozyIgmz88DrBhjAUE+Y\/Q4H2CyfHBhjFTCSgJ\/dsjLsU9v6zJEejgK4wgeDhw58SLC0ENy6N5MZGUb1QAEkgAqUDT9EAjnT3lsOFLFCa\/cWCGLhFFDgdFDAWYCz+4SBRYZRvuqTE8iZZUvkfkKgbYFngW39noQoMoxC9XD9f35V\/5SNi46kAAAAAElFTkSuQmCC","width":245}
%---
%[output:1d0373a0]
%   data: {"dataType":"textualVariable","outputData":{"name":"total_iteration","value":"1"}}
%---
%[output:00def506]
%   data: {"dataType":"text","outputData":{"text":"Wireless packet type: SC\n","truncated":false}}
%---
%[output:4cd315df]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_x","value":"4"}}
%---
%[output:18b88e21]
%   data: {"dataType":"textualVariable","outputData":{"name":"N","value":"16"}}
%---
%[output:6088ac05]
%   data: {"dataType":"textualVariable","outputData":{"name":"M_x","value":"12"}}
%---
%[output:38d6f3e6]
%   data: {"dataType":"textualVariable","outputData":{"name":"M","value":"144"}}
%---
%[output:1d0ad6ac]
%   data: {"dataType":"textualVariable","outputData":{"name":"zeta","value":"0.9800"}}
%---
%[output:7c3c3c33]
%   data: {"dataType":"textualVariable","outputData":{"name":"T_coh","value":"0.0038"}}
%---
%[output:23c1d2de]
%   data: {"dataType":"textualVariable","outputData":{"name":"N_packets_coh","value":"12"}}
%---
%[output:5580c95d]
%   data: {"dataType":"textualVariable","outputData":{"name":"T_x","value":"60"}}
%---
%[output:36e595f1]
%   data: {"dataType":"textualVariable","outputData":{"name":"SNR_dB","value":"36.5005"}}
%---
%[output:2177d08d]
%   data: {"dataType":"textualVariable","outputData":{"header":"struct with fields:","name":"EnvPars","value":"              channelModel: 'rician_los'\n                    fc_GHz: 28\n                    V_hall: 1000\n                    S_hall: 600\n                   mu_lgDS: -7.5916\n                sigma_lgDS: 0.1500\n                  mu_lgASD: 1.5600\n               sigma_lgASD: 0.2500\n                  mu_lgASA: 1.5168\n               sigma_lgASA: 0.3755\n                  mu_lgZSA: 1.2075\n               sigma_lgZSA: 0.3500\n                  mu_lgZSD: 1.3500\n               sigma_lgZSD: 0.3500\n                   mu_K_dB: 7\n                sigma_K_dB: 8\n                        KR: 3.4358\n                      rTau: 2.7000\n                 mu_XPR_dB: 12\n              sigma_XPR_dB: 6\n    clusterShadowingStd_dB: 4\n                        Nc: 25\n                      Mray: 20\n            clusterASD_deg: 5\n            clusterASA_deg: 8\n            clusterZSA_deg: 9\n            rayOffsetAlpha: [-0.0447 0.0447 -0.1413 0.1413 -0.2492 0.2492 -0.3715 0.3715 -0.5129 0.5129 -0.6797 0.6797 -0.8844 0.8844 -1.1481 1.1481 -1.5195 1.5195 -2.1551 2.1551]\n              clusterDS_ns: NaN\n             ZODoffset_deg: 0\n         corrDistance_DS_m: 10\n        corrDistance_ASD_m: 10\n        corrDistance_ASA_m: 10\n         corrDistance_SF_m: 10\n          corrDistance_K_m: 10\n        corrDistance_ZSA_m: 10\n        corrDistance_ZSD_m: 10\n                normalizeH: 1\n        elementCosinePower: 0\n            nlosPowerScale: 0\n                         N: 16\n                       N_x: 4\n                       N_y: 4\n                         T: 3600\n                       T_x: 60\n                       T_y: 60\n                    SNR_dB: 36.5005\n                 theta_min: 1.8485\n                 theta_max: 4.4347\n                        fc: 2.8000e+10\n                    lambda: 0.0107\n                   Ptx_dBm: 23\n                   Gtx_dBi: 14\n                   Grx_dBi: 8\n                   txArray: [1×1 struct]\n              var_noise_dB: -110.9794\n                         r: 0\n                       d_x: 0.0054\n                       d_y: 0.0054\n                   pos_SIM: [5 5 4]\n                    pos_MU: [4.2176 9.1574 1.5000]\n                       n_y: [1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4]\n                       n_x: [1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4]\n                       t_y: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 … ] (1×3600 double)\n                       t_x: [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 … ] (1×3600 double)\n                      h_MU: 1.5000\n                    L_hall: 10\n                    W_hall: 10\n                     N_cal: 100\n                 MU_margin: 0.5000\n               MaxEpisodes: 5000\n                     psi_x: 0\n                     psi_y: 0\n        MaxStepsPerEpisode: 180\n                 tolerance: 0.0262\n         StopTrainingValue: 171\n           episode_counter: 0\n               delta_moves: [9×2 double]\n                 n_actions: 9\n            DiscountFactor: 0.9500\n              EpsilonDecay: 5.5556e-06\n    ExperienceBufferLength: 100000\n             MiniBatchSize: 128\n        TargetSmoothFactor: 1.0000e-03\n                 threshold: 0.8000\n          reward_threshold: 0.8000\n                 step_cost: 0.0100\n                peak_bonus: 10\n                    U_func: @(n_,t_n_)exp(1i*(-2*pi*(EnvPars.n_x(n_)-1).*(EnvPars.t_x(t_n_)-1)\/(EnvPars.N_x*EnvPars.T_x)-2*pi*(EnvPars.n_y(n_)-1).*(EnvPars.t_y(t_n_)-1)\/(EnvPars.N_y*EnvPars.T_y)))\n                     F_amp: [1×1 griddedInterpolant]\n            phase_min_meas: 0.1501\n            phase_max_meas: 6.1982\n                U_func_CST: @(n_,t_n_)U_func_cst(n_,t_n_,EnvPars)\n                         G: [16×16 double]\n                     G_CST: [16×16 double]\n                  peak_map: [100×60×60 double]\n                 psi_x_cal: [100×1 double]\n                 psi_y_cal: [100×1 double]\n                   pos_cal: [100×3 double]\n            global_max_cal: [100×1 double]\n               best_tx_cal: [100×1 double]\n               best_ty_cal: [100×1 double]\n                     h_cal: [16×100 double]\n"}}
%---
%[output:700bead8]
%   data: {"dataType":"text","outputData":{"text":"Loaded CST SIM-1 G_CST. Deviation from analytic G: 11.630%n","truncated":false}}
%---
