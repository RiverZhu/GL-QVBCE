% Performance versus Iteration for gridless channel estimation 
% Quantization given by B
clear;
close all;
clc;  
rng(1)
% Generate a noisy superposition of K complex sinusoids with angular frequencies in [-pi,pi)
% The channel
N = 64;            % size of the full data
M = 64;           % number of measurements with indices randomly chosen from 0,...,N-1
K = 2;              % number of sinusoidal components
Iter_max = 1000;
SNR = 0;
T = 1;        % Traning T durations
MC = 10;       % Set MC=10 to view the results quickly
B_all = [1;2;3;inf];
P = 1;
r = nan(K,1);
MSE_h_VALSE = zeros(Iter_max,MC,length(B_all));
MSE_h_VALSE_AQNM = zeros(Iter_max,MC,length(B_all)-1);
MSE_h_LS = zeros(Iter_max,MC);

tau = [];
yy_min = [];
alpha = [];
method_EP =  'diag_EP';
tmp  = randperm(N-1)';
Mcal = [sort(tmp(1:M-1),'ascend'); N]-1;    % indices of the measurements
for mc = 1:MC
        waitbar(mc/MC)
        tmp  = randperm(N-1)';
        Mcal = [sort(tmp(1:M-1),'ascend'); N]-1;    % indices of the measurements
        omega = pi *sin(pi/2 * (2*rand(K,1) - 1));
        A   = exp(1i*(0:1:Mcal(end)).'*omega.');         % matrix with columns a(omega(1)),..., a(omega(K))
        r(1)   = sqrt(0.45*P)+sqrt(0.5*P*0.1)*randn(1,1);
        r(2)   = sqrt(0.45*P)+sqrt(0.5*P*0.1)*randn(1,1);
        phase_varphi = 2* pi * (rand(K,1) - 0.5);
        al  = r.*exp(1i*phase_varphi);                 % complex amplitudes   
        h_orig   = A*al;                                      % original signal
        h = h_orig(Mcal+1);
        x = (sign(rand(T,1)-0.5)+1j*sign(rand(T,1)-0.5))/sqrt(2); % QPSK symbol 
        z = h*(x.');
        Pn  = norm(z,'fro')^2*10^(-SNR/10)/T/M;
        Y_unq_mat = z+sqrt(Pn/2)*(randn(M,T)+1j*randn(M,T));
        h_LS = sum(bsxfun(@times,Y_unq_mat,x'),2)/(x'*x);
        MSE_h_LS(:,mc) = 20*log10(norm(h_LS-h)/norm(h))*ones(Iter_max,1);
        Y_unq = Y_unq_mat(:);
            for B_index = 1:length(B_all)
                B = B_all(B_index);
                if B<inf
                    y_real = [real(Y_unq);imag(Y_unq)];
                    nbins = 2^B;   
                    yy_max = 3*sqrt(P/2);
                    yy_min = -yy_max; 
                    alpha = (yy_max - yy_min)/(nbins);    
                    tau_o = 1:nbins-1;
                    tau = yy_min +tau_o.*alpha;  
                    sigma_aqnm = 2*alpha^2/12; % (real part)
                    yy = floor((y_real-yy_min)/alpha);
                    index1 = find(y_real>=yy_max);
                    yy(index1) = nbins-1;
                    index2 = find(y_real<yy_min);
                    yy(index2) = 0;
                    y_q = yy;
                    y_pre = yy_min + (yy+0.5)* alpha;
                    y_pre_c = y_pre(1:end/2)+1j*y_pre(end/2+1:end);
                    
                    
                    EST_quant_VALSE = CE_VALSE_EP_all(y_q,Mcal,2,T,x,h_orig, yy_min,B,alpha,Iter_max,method_EP);
                    MSE_h_VALSE(:,mc,B_index) = EST_quant_VALSE.MSE;
                    
                    EST_quant_VALSE_aqnm = CE_VALSE_EP_all(y_pre_c,Mcal,2,T,x,h_orig, yy_min,inf,alpha,Iter_max,method_EP);
                    MSE_h_VALSE_AQNM(:,mc,B_index) = EST_quant_VALSE_aqnm.MSE;
                    
                else
                    y_q = Y_unq;
                    EST_unq_VALSE = CE_VALSE_EP_all(y_q,Mcal,2,T,x,h_orig, yy_min,B,alpha,Iter_max,method_EP);
                    MSE_h_VALSE(:,mc,B_index) = EST_unq_VALSE.MSE;
                end  
       
            end
end



MSE_h_VALSE_mean = mean(MSE_h_VALSE,2);
MSE_h_VALSE_AQNM_mean = mean(MSE_h_VALSE_AQNM,2);
MSE_h_LS_mean = mean(MSE_h_LS,2);


Iter_max0 = 15;
Iter_index = 1:1:Iter_max0;
%% Figures
alw = 0.75;    % AxesLineWidth
fsz = 18;      % Fontsize
lw = 1.6;      % LineWidth
msz = 8;       % MarkerSize

h1 = figure(1);
set(h1,'position',[100,100,750,500])
set(gca, 'FontSize', fsz, 'LineWidth', alw);
plot(Iter_index,MSE_h_VALSE_AQNM_mean(Iter_index,1),'-r^','LineWidth',lw, 'MarkerSize', msz);
hold on
plot(Iter_index,MSE_h_VALSE_AQNM_mean(Iter_index,2),'-rv','LineWidth',lw, 'MarkerSize', msz);
hold on
plot(Iter_index,MSE_h_VALSE_AQNM_mean(Iter_index,3),'-r<','LineWidth',lw, 'MarkerSize', msz);
hold on
plot(Iter_index,MSE_h_VALSE_mean(Iter_index,4),'-r>','LineWidth',lw, 'MarkerSize', msz);
hold on
plot(Iter_index,MSE_h_VALSE_mean(Iter_index,1),'-.b+','LineWidth',lw, 'MarkerSize', msz);
hold on
plot(Iter_index,MSE_h_VALSE_mean(Iter_index,2),'-.bx','LineWidth',lw, 'MarkerSize', msz);
hold on
plot(Iter_index,MSE_h_VALSE_mean(Iter_index,3),'-.b.','LineWidth',lw, 'MarkerSize', msz+18);
hold on
plot(Iter_index,MSE_h_LS_mean(Iter_index),'-k*','LineWidth',lw, 'MarkerSize', msz);
hold on
hhh = legend('1 bit, GL-VBCE','2 bit, GL-VBCE','3 bit, GL-VBCE','\infty bit GL-VBCE','1 bit, GL-QVBCE','2 bit GL-QVBCE','3 bit GL-QVBCE','LS');
set(hhh,'Fontsize',fsz-2);
xlabel('Iteration','Fontsize',fsz);
ylabel('${\rm NMSE}(\hat{\mathbf h})$ (dB)','interpreter','latex','Fontsize',fsz);
xlim([1,max(Iter_max0)]);
set(gca,'xtick',Iter_index);
set(gca,'Fontsize',fsz);

if SNR==0
    pos = [12,-8];
    ylim([-15,10])
    save('MSEh_Itersnr0.mat','MSE_h_VALSE_AQNM_mean','MSE_h_VALSE_mean','MSE_h_LS_mean','Iter_max')
elseif SNR==20
    pos = [12,-28];
    ylim([-40,20])
    save('MSEh_Itersnr20.mat','MSE_h_VALSE_AQNM_mean','MSE_h_VALSE_mean','MSE_h_LS_mean','Iter_max')
elseif SNR==40
    pos = [10,-12];
    save('MSEh_Itersnr40.mat','MSE_h_VALSE_AQNM_mean','MSE_h_VALSE_mean','MSE_h_LS_mean','Iter_max')
end
text(pos(1),pos(2),sprintf('SNR=%d dB',SNR),'Fontsize',fsz)   % 10 dB, pos = (1.5,-24) % 20 dB, pos = (8,-30) % 30 dB, pos = (3,-45)

