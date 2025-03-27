%% DoA performance evaluation
%% Code

clc;
clear all;
close all;
Parameters %loading the system parameters

%%Uncomment the line below for recalculating the parameters
% Relative_errors
%%Comment the line below if the line above is uncommented
load ./results/simulation_results.mat
% Plotting

i_itr=1; %select the iteration to plot between 1 and total_iteration
i_T=3;%select the number of snapshots
for i=1:length(N)
    figure;
    plot_stream=strcat('./results/plot_N_',num2str(i));
    load(plot_stream)
    % transform a vector to the matrix with the reshape function
    % the fliplr, flipud, and transpose operations are for the convenience of
    % representing the x and y axes in the plot
    R_psi_x_y = flipud(fliplr(reshape(r_psi_x_y_t{max_t_psi(i,i_itr),i_T}, [N_x(i), N_y(i)])))';
    %coordinates with the electric angle indexes k_x and k_y as defined in
    %the Parameters.mlx file
    [X,Y] = meshgrid(linspace(1,N_x(i),N_x(i))+(max_t_psi_x(i,i_itr)-1)/T_x(i_T),linspace(1,N_y(i),N_y(i))+(max_t_psi_y(i,i_itr)-1)/T_y(i_T));
    %coordinates with the electric angle
    [X,Y] = meshgrid(2*pi/N_x(i)*(linspace(1,N_x(i),N_x(i))-(max_t_psi_x(i,i_itr))/T_x(i_T)),...
        2*pi/N_y(i_N)*(linspace(0,N_y(i),N_y(i))-(max_t_psi_y(i,i_itr))/T_y(i_T)));

    contourf(X,Y,abs(R_psi_x_y));
    colorbar
    grid on;
    xlabel('$\psi_\mathrm{x}$ [rad]','Interpreter','latex');
    ylabel('$\psi_\mathrm{y}$ [rad]','Interpreter','latex');
    %Tick angles
    X=2*pi/N_x(i)*(1:2:N_x(i));
    set(gca, 'XTick', X);
    X_Tick_text=cell(1,length(X));
    for j=1:length(X)
        X_Tick_text{j}=strcat('$',num2str(X(j)/pi+(max_t_psi_x(i,i_itr))/T_x(i_T),2),'\pi$');
    end
    set(gca, 'XTickLabel', [X_Tick_text], 'TickLabelInterpreter', 'latex');

    Y=2*pi/N_y(i)*(1:2:N_y(i));
    set(gca, 'YTick', Y);
    Y_Tick_text=cell(1,length(Y));
    for j=1:length(Y)
        Y_Tick_text{j}=strcat('$',num2str(Y(j)/pi+(max_t_psi_y(i,i_itr))/T_y(i_T),2),'\pi$');
    end
    set(gca, 'YTickLabel', [Y_Tick_text], 'TickLabelInterpreter', 'latex');

    % set(gca, 'YTick', [0:0.25:2]*pi);
    % set(gca, 'YTickLabel', [{"0","$0.25\pi$","$0.5\pi$","$0.75\pi$","\pi","$1.25\pi$","1.5\pi","$1.75\pi$","2\pi"}], 'TickLabelInterpreter', 'latex');
    % axis([0 2*pi 0 2*pi])
    set(gca,'FontSize',font,'GridColor','white');
    legend_text={strcat('SNR$=',num2str(SNR_dB,3),'$ dB,',' $N=$',num2str(N_x(i)),'$\times$',num2str(N_y(i)),', $T=$',num2str(T_x(i_T)),'$\times$',num2str(T_y(i_T))),...
        strcat('$\bar{\psi}_x=',num2str(psi_x(i,i_T,i_itr)/pi,2),'\pi$ rad, ',' $\hat{\psi}_x=',num2str(psi_x_est(i,i_T,i_itr)/pi,3),'\pi$ rad')...
        strcat('$\bar{\psi}_y=',num2str(psi_y(i,i_T,i_itr)/pi,2),'\pi$ rad, ',' $\hat{\psi}_y=',num2str(psi_y_est(i,i_T,i_itr)/pi,3),'\pi$ rad')};
    % Get current axis limits
    x_limits = xlim;
    y_limits = ylim;
    % Define position (upper-right corner)
    x_pos = x_limits(2) - 0.89 * diff(x_limits); % Slightly inside the right edge
    y_pos = y_limits(1) + 0.85 * diff(y_limits); % Slightly inside the top edge
    text(x_pos,y_pos,legend_text,'Interpreter','latex','BackgroundColor','w','FontSize',font);
    set(gca,'FontSize',font);
    %saving results
    plot_stream=strcat('./results/plot_N_',num2str(N_x(i)),'_',num2str(N_x(i)),'.svg');
    saveas(gcf, plot_stream);
    plot_stream=strcat('./results/plot_N_',num2str(N_x(i)),'_',num2str(N_x(i)),'.fig');
    saveas(gcf, plot_stream);

end
%% Evaluating the relative errors

figure;
legend_text=cell(1,length(N));
mean_relative_errors=zeros(1,length(N));
X_Tick_text=cell(1,length(N));
i_T=15;
for i=1:length(N)
    total_bins=unique(psi_x_relative_errors(i,i_T,:));
    h=histogram(psi_x_relative_errors(i,:),length(total_bins),'Normalization','probability'); grid on; hold on
    bin_centers = (h.BinEdges(1:end-1) + h.BinEdges(2:end)) / 2; % Compute bin centers
    disp(['N=',num2str(N_x(i)),'x',num2str(N_x(i)),'-> Mean=',num2str(sum(h.Values.*bin_centers)),'%']);
    legend_text{i}=strcat('N$=',num2str(N_x(i)),'\times ', num2str(N_y(i)),'$');
    mean_relative_errors(i)=mean(psi_x_relative_errors(i,i_T,:));
    %x-axis text
    X_Tick_text{i}=strcat('$',num2str(N_x(i)),'\times',num2str(N_y(i)),'$');
end

%\frac{|\bar {\psi }_{\text {x}}- \hat {\psi }_{\text {x},n,t}|}{|\bar {\psi }_{\text {x}}|}
xlabel('Relative errors$=\frac{|\bar{\psi }_{\mathrm{x}}- \hat{\psi}_{\mathrm{x},n,t}|}{|\bar{\psi }_{\mathrm{x}}|}\ \%$','Interpreter','latex');
ylabel('PDF','Interpreter','latex');
legend({legend_text{:}},...
    'Location','northwest','NumColumns',2,'Interpreter',"latex",'FontSize',font);% more properties on the legend at https://www.mathworks.com/help/matlab/ref/legend.html
set(gca,'FontSize',font);

%% Evaluating the mean relative error
close all


mean_relative_errors=zeros(length(N),length(T));
figure;
i_T_init=3;
legend_text=cell(1,length(T)-i_T_init+1);

for i_N=1:length(N)
        
        mean_relative_errors(i_N,i_T_init)=mean(psi_x_relative_errors(i_N,i_T_init,:));
        
end

p_1=plot(mean_relative_errors(:,i_T_init),'--o','LineWidth',2,'MarkerSize',8); grid on; hold on;
legend_text{1}=strcat('$T=',num2str(T_x(i_T_init)),'\times ', num2str(T_y(i_T_init)),'$');

for i_T=i_T_init+1:length(T)-1
    for i_N=1:length(N)
        mean_relative_errors(i_N,i_T)=mean(psi_x_relative_errors(i_N,i_T,:));     
    end
    plot(mean_relative_errors(:,i_T),'-','LineWidth',2,'MarkerSize',8); 
    
end

for i_N=1:length(N)
    mean_relative_errors(i_N,i_T+1)=mean(psi_x_relative_errors(i_N,i_T+1,:));
end
p_2=plot(mean_relative_errors(:,i_T+1),'-d','LineWidth',2,'MarkerSize',8);
legend_text{2}=strcat('$T=',num2str(T_x(i_T+1)),'\times ', num2str(T_y(i_T+1)),'$');

annotation("arrow",[0.3 0.5],[0.6 0.5],'Color','black','LineWidth',2,'LineStyle','--');
text(2,3,{strcat('$T$ increases to $',num2str(T_x(end)),'\times ', num2str(T_y(end)),'$'),strcat('SNR$=',num2str(SNR_dB,3),'$ dB')},'Interpreter','latex','FontSize',font,'BackgroundColor','white');

legend([p_1 p_2],{legend_text{1:2}},...
    'Location','northwest','NumColumns',2,'Interpreter',"latex",'FontSize',font,'NumColumns',1);% more properties on the legend at https://www.mathworks.com/help/matlab/ref/legend.html
set(gca, 'XTick', [1:length(mean_relative_errors)]);
set(gca, 'XTickLabel', [X_Tick_text], 'TickLabelInterpreter', 'latex');
xtickangle(90);
xlabel('$N_x \times N_y$','Interpreter','latex');
ylabel('Relative errors$=\frac{|\bar{\psi }_{\mathrm{x}}- \hat{\psi}_{\mathrm{x},n,t}|}{|\bar{\psi }_{\mathrm{x}}|}\ \%$','Interpreter','latex');
set(gca,'FontSize',font);
saveas(gcf, './results/mean_vs_N.svg');

%% exponent of the relative errors

figure;
i_T_init=3;
      
p_1=plot(ceil((-floor(log10(mean_relative_errors(:,i_T_init)/100)))),'--o','LineWidth',2,'MarkerSize',8); grid on; hold on;
legend_text{1}=strcat('$T=',num2str(T_x(i_T_init)),'\times ', num2str(T_y(i_T_init)),'$');



for i_T=i_T_init+1:length(T)-1
    plot(ceil((-floor(log10(mean_relative_errors(:,i_T)/100)))),'-','LineWidth',2,'MarkerSize',8);
end

p_2=plot(ceil((-floor(log10(mean_relative_errors(:,i_T+1)/100)))),'-d','LineWidth',2,'MarkerSize',8);
legend_text{2}=strcat('$T=',num2str(T_x(i_T+1)),'\times ', num2str(T_y(i_T+1)),'$');

lgd=legend([p_1 p_2],{legend_text{1:2}},...
    'Location','northwest','NumColumns',2,'Interpreter',"latex",'FontSize',font,'NumColumns',1);% more properties on the legend at https://www.mathworks.com/help/matlab/ref/legend.html


set(gca, 'XTick', [1:length(mean_relative_errors)]);
set(gca, 'XTickLabel', [X_Tick_text], 'TickLabelInterpreter', 'latex');
xtickangle(90);
xlabel('$N_x \times N_y$','Interpreter','latex');
ylabel({'Exponent or the relative errors $n$'},'Interpreter','latex');
set(gca,'FontSize',font);
saveas(gcf, './results/plot_exp_vs_N.svg');
saveas(gcf, './results/plot_exp_vs_N.fig');

%% bits to represent the exponent of the relative errors
close all;
figure;
i_T_init=3;
      
p_1=plot((log2((-floor(log10(mean_relative_errors(:,i_T_init)/100))))),'--o','LineWidth',2,'MarkerSize',8); grid on; hold on;
legend_text{1}=strcat('$T=',num2str(T_x(i_T_init)),'\times ', num2str(T_y(i_T_init)),'$');



for i_T=i_T_init+1:length(T)-1
    plot((log2((-floor(log10(mean_relative_errors(:,i_T)/100))))),'-','LineWidth',2,'MarkerSize',8);
end

p_2=plot((log2((-floor(log10(mean_relative_errors(:,i_T+1)/100))))),'-d','LineWidth',2,'MarkerSize',8);
legend_text{2}=strcat('$T=',num2str(T_x(i_T+1)),'\times ', num2str(T_y(i_T+1)),'$');

lgd=legend([p_1 p_2],{legend_text{1:2}},...
    'Location','northwest','NumColumns',2,'Interpreter',"latex",'FontSize',font,'NumColumns',1);% more properties on the legend at https://www.mathworks.com/help/matlab/ref/legend.html

set(gca, 'XTick', [1:length(mean_relative_errors)]);
set(gca, 'XTickLabel', [X_Tick_text], 'TickLabelInterpreter', 'latex');
xtickangle(90);

% set(gca, 'YTick', [0 1]);
% set(gca, 'YTickLabel', {'1','2'}, 'TickLabelInterpreter', 'latex');

xlabel('$N_x \times N_y$','Interpreter','latex');
ylabel({'Number of bits$=\log_2n$'},'Interpreter','latex');
set(gca,'FontSize',font);
saveas(gcf, './results/plot_bits_vs_N.svg');
saveas(gcf, './results/plot_bits_vs_N.fig');


%% bits to represent the inverse of the relative errors
close all;
figure;
i_T_init=3;

I=log2(1./(mean_relative_errors/100));

p_1=plot(I(:,i_T_init),'--o','LineWidth',2,'MarkerSize',8); grid on; hold on;
legend_text{1}=strcat('$T=',num2str(T_x(i_T_init)),'\times ', num2str(T_y(i_T_init)),'$');



for i_T=i_T_init+1:length(T)-1
    plot(I(:,i_T),'-','LineWidth',2,'MarkerSize',8);
end

p_2=plot(I(:,i_T+1),'-d','LineWidth',2,'MarkerSize',8);
legend_text{2}=strcat('$T=',num2str(T_x(i_T+1)),'\times ', num2str(T_y(i_T+1)),'$');

lgd=legend([p_1 p_2],{legend_text{1:2}},...
    'Location','northwest','NumColumns',2,'Interpreter',"latex",'FontSize',font,'NumColumns',1);% more properties on the legend at https://www.mathworks.com/help/matlab/ref/legend.html

set(gca, 'XTick', [1:length(mean_relative_errors)]);
set(gca, 'XTickLabel', [X_Tick_text], 'TickLabelInterpreter', 'latex');
xtickangle(90);

% set(gca, 'YTick', [0 1]);
% set(gca, 'YTickLabel', {'1','2'}, 'TickLabelInterpreter', 'latex');

xlabel('$N_x \times N_y$','Interpreter','latex');
% ylabel({'Number of bits','$-\log_2$ (Relative error)'},'Interpreter','latex');
ylabel({'Number of bits $I$'},'Interpreter','latex');
set(gca,'FontSize',font);
saveas(gcf, './results/plot_bits_vs_N.svg');
saveas(gcf, './results/plot_bits_vs_N.fig');

annotation("arrow",[0.3 0.3],[0.6 0.5],'Color','black','LineWidth',2,'LineStyle','--');
text(0.3,4,{strcat('$T$ increases to $',num2str(T_x(end)),'\times ', num2str(T_y(end)),'$'),strcat('SNR$=',num2str(SNR_dB,3),'$ dB')},'Interpreter','latex','FontSize',font,'BackgroundColor','white');

%% Plotting precision as (1-relative_errors)

precision=1-mean_relative_errors/100;

close all;
figure;
i_T_init=3;

p_1=plot(precision(:,i_T_init),'--o','LineWidth',2,'MarkerSize',8); grid on; hold on;
legend_text{1}=strcat('$T_\mathrm{upd}=',num2str(T_x(i_T_init)),'\times ', num2str(T_y(i_T_init)),'$');

for i_T=i_T_init+1:length(T)-1
    plot(precision(:,i_T),'-','LineWidth',2,'MarkerSize',8);
end

p_2=plot(precision(:,i_T),'-d','LineWidth',2,'MarkerSize',8);
legend_text{2}=strcat('$T_\mathrm{upd}=',num2str(T_x(i_T+1)),'\times ', num2str(T_y(i_T+1)),'$');

annotation("arrow",[0.3 0.3],[0.6 0.5],'Color','black','LineWidth',2,'LineStyle','--');
text(0.3,0.5,{strcat('$T_\mathrm{upd}$ increases to $',num2str(T_x(end)),'\times ', num2str(T_y(end)),'$'),strcat('SNR$=',num2str(SNR_dB,3),'$ dB')},'Interpreter','latex','FontSize',font,'BackgroundColor','white');


lgd=legend([p_1 p_2],{legend_text{1:2}},...
    'Location','northwest','NumColumns',2,'Interpreter',"latex",'FontSize',font,'NumColumns',1);% more properties on the legend at https://www.mathworks.com/help/matlab/ref/legend.html

for i_N=1:length(N)
    X_Tick_text{i_N}=strcat('$',num2str(N_x(i_N)),'\times',num2str(N_y(i_N)),'$');
end

set(gca, 'XTick', [1:length(mean_relative_errors)]);
set(gca, 'XTickLabel', [X_Tick_text], 'TickLabelInterpreter', 'latex');
xtickangle(90);
xlabel('$N_x \times N_y$','Interpreter','latex');
ylabel({'Precision ($1-$Relative errors)'},'Interpreter','latex');
set(gca,'FontSize',font);
saveas(gcf, './results/plot_precision_vs_N.svg');
saveas(gcf, './results/plot_precision_vs_N.fig');

%% Calculation of energy consumption, see [3]
% Description: The power is evaluated with respect to the 
P_cnt_board=1.5;%power of the control board ZYNQ7100 in Watts 
P_unit_circuit=(250+180)*1e-3;%power of the DAC3484 and AD8021
P_atom_disip=0;%disipated power of the atom unit
P_p=(250+180)*1e-3; %power of the RF-chain per probe

%evaluating the total energy
E_total=zeros(length(N),length(T));
for i_T=1:length(T)
    for i_N=1:length(N)
        E_total(i_N,i_T)=(P_cnt_board+N(i_N)*(P_unit_circuit+P_p))*T(i_T)*T_PPDU_loc+Ptx*T(i_T)*T_PPDU_loc;
    end
end

close all
figure;
i_T_init=3;
      
p_1=plot(E_total(:,i_T_init)*1e3,'--o','LineWidth',2,'MarkerSize',8); grid on; hold on;
legend_text{1}=strcat('$T_\mathrm{upd}=',num2str(T_x(i_T_init)),'\times ', num2str(T_y(i_T_init)),'$');

for i_T=i_T_init+1:length(T)-1
    plot(E_total(:,i_T)*1e3,'-','LineWidth',2,'MarkerSize',8);
end

p_2=plot(E_total(:,i_T)*1e3,'-d','LineWidth',2,'MarkerSize',8);
legend_text{2}=strcat('$T_\mathrm{upd}=',num2str(T_x(i_T+1)),'\times ', num2str(T_y(i_T+1)),'$');

lgd=legend([p_1 p_2],{legend_text{1:2}},...
    'Location','northwest','NumColumns',2,'Interpreter',"latex",'FontSize',font,'NumColumns',1);% more properties on the legend at https://www.mathworks.com/help/matlab/ref/legend.html

annotation("arrow",[0.3 0.3],[0.6 0.5],'Color','black','LineWidth',2,'LineStyle','--');
text(0.3,0.3,{'$T_\mathrm{upd}$ increases'},'Interpreter','latex','FontSize',font,'BackgroundColor','white');

for i_N=1:length(N)
    X_Tick_text{i_N}=strcat('$',num2str(N_x(i_N)),'\times',num2str(N_y(i_N)),'$');
end

set(gca, 'XTick', [1:length(mean_relative_errors)]);
set(gca, 'XTickLabel', [X_Tick_text], 'TickLabelInterpreter', 'latex');
xtickangle(90);
xlabel('$N_x \times N_y$','Interpreter','latex');
ylabel({'Energy consumption [mJ]'},'Interpreter','latex');
set(gca,'FontSize',font);
saveas(gcf, './results/plot_Energy_vs_N.svg');
saveas(gcf, './results/plot_Energy_vs_N.fig');

%% Precision vs energy
close all;

figure;
i_T_init=3;
      
p_1=loglog(E_total(:,i_T_init),precision(:,i_T_init),'--o','LineWidth',2,'MarkerSize',8); grid on; hold on;
legend_text{1}=strcat('$T_\mathrm{upd}=',num2str(T_x(i_T_init)),'\times ', num2str(T_y(i_T_init)),'$');

for i_T=i_T_init+1:length(T)-1
    loglog(E_total(:,i_T),precision(:,i_T),'-','LineWidth',2,'MarkerSize',8);
end

p_2=semilogy(E_total(:,i_T),precision(:,i_T),'-d','LineWidth',2,'MarkerSize',8);
legend_text{2}=strcat('$T_\mathrm{upd}=',num2str(T_x(i_T+1)),'\times ', num2str(T_y(i_T+1)),'$');

lgd=legend([p_1 p_2],{legend_text{1:2}},...
    'Location','northwest','NumColumns',2,'Interpreter',"latex",'FontSize',font,'NumColumns',1);% more properties on the legend at https://www.mathworks.com/help/matlab/ref/legend.html

annotation("arrow",[0.3 0.3],[0.6 0.5],'Color','black','LineWidth',2,'LineStyle','--');
text(0.3,0.5,{strcat('$T_\mathrm{upd}$ increases to $',num2str(T_x(end)),'\times ', num2str(T_y(end)),'$'),strcat('SNR$=',num2str(SNR_dB,3),'$ dB')},'Interpreter','latex','FontSize',font,'BackgroundColor','white');

x_data=p_2.XData;
for i_E=1:length(x_data)
    X_Tick_text{i_E}=strcat('$',num2str(x_data(i_E)*1e3,4),'$');
end

set(gca, 'XTick', x_data(1:2:end));
set(gca, 'XTickLabel', [X_Tick_text(1:2:end)], 'TickLabelInterpreter', 'latex');
xtickangle(90);
xlabel('Energy [mJ]','Interpreter','latex');
ylabel({'Precision as ($1-$Relative errors)'},'Interpreter','latex');
set(gca,'FontSize',font);
saveas(gcf, './results/plot_precision_vs_Energy.svg');
saveas(gcf, './results/plot_precision_vs_Energy.fig');

%% Calculation of energy efficiency

%evaluating the energy efficiency
EE=I./E_total;
EE=E_total/max(max(E_total));
EE=precision./E_total/max(max(E_total));



figure;
i_T_init=3;
      
p_1=plot(EE(:,i_T_init),'--o','LineWidth',2,'MarkerSize',8); grid on; hold on;
legend_text{1}=strcat('$T=',num2str(T_x(i_T_init)),'\times ', num2str(T_y(i_T_init)),'$');

for i_T=i_T_init+1:length(T)-1
    plot(EE(:,i_T),'-','LineWidth',2,'MarkerSize',8);
end

p_2=plot(EE(:,i_T),'-d','LineWidth',2,'MarkerSize',8);
legend_text{2}=strcat('$T=',num2str(T_x(i_T+1)),'\times ', num2str(T_y(i_T+1)),'$');

lgd=legend([p_1 p_2],{legend_text{1:2}},...
    'Location','northwest','NumColumns',2,'Interpreter',"latex",'FontSize',font,'NumColumns',1);% more properties on the legend at https://www.mathworks.com/help/matlab/ref/legend.html

annotation("arrow",[0.3 0.3],[0.6 0.5],'Color','black','LineWidth',2,'LineStyle','--');
text(0.3,0.3,{strcat('$T_\mathrm{upd}$ increases to $',num2str(T_x(end)),'\times ', num2str(T_y(end)),'$'),strcat('SNR$=',num2str(SNR_dB,3),'$ dB')},'Interpreter','latex','FontSize',font,'BackgroundColor','white');

for i_T=i_T_init:length(T)
    X_Tick_text{i}=strcat('$',num2str(N_x(i)),'\times',num2str(N_y(i)),'$');
end
set(gca, 'XTick', [1:length(mean_relative_errors)]);
set(gca, 'XTickLabel', [X_Tick_text], 'TickLabelInterpreter', 'latex');
xtickangle(90);
xlabel('$N_x \times N_y$','Interpreter','latex');
ylabel({'Energy efficiency'},'Interpreter','latex');
set(gca,'FontSize',font);
saveas(gcf, './results/plot_efficiency_vs_N.svg');
saveas(gcf, './results/plot_efficiency_vs_N.fig');
%% References
% [1] J. An et al., "Two-Dimensional Direction-of-Arrival Estimation Using Stacked 
% Intelligent Metasurfaces," in IEEE Journal on Selected Areas in Communications, 
% vol. 42, no. 10, pp. 2786-2802, Oct. 2024. doi: <https://doi.org/10.1109/JSAC.2024.3414613 
% 10.1109/JSAC.2024.3414613>
% 
% [2] P. Heidenreich, A. M. Zoubir and M. Rubsamen, "Joint 2-D DOA Estimation 
% and Phase Calibration for Uniform Rectangular Arrays," in IEEE Transactions 
% on Signal Processing, vol. 60, no. 9, pp. 4683-4693, Sept. 2012, doi: <https://doi.org/10.1109/TSP.2012.2203125 
% 10.1109/TSP.2012.2203125>

% [3] J. Wang et al., "Reconfigurable Intelligent Surface: Power Consumption Modeling and Practical Measurement Validation," IEEE Transactions on Communications, vol. 72, no. 9, pp. 5720–5734, Sep. 2024

