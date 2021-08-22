clc
clear all;
close all; 

imageName = 'cameraman.bmp';

Img = double(imread(imageName)); %Your Image goes here

N = numel(Img);

[row, col] = size(Img);

row = int2str(row);
col = int2str(col);

imageSize = [row 'x' col];

%K     =   fspecial('average',1); % For denoising. (K is the identity operator)
K     =   fspecial('average',9); % For denoising
f = imfilter(Img,K,'circular');


%sigma = 30; % Noise level ( in paper, 10, 20 and 30 are tested)

BSNR = 15;  % BSNR... The higher the BSNR the lower the noise 
sigma = BSNR2WGNsigma(f, BSNR);

fprintf('The noise std of the observed image: %g.\n', sigma); 

%rng(666);

f = f +  sigma * randn(size(Img)); %Add a little noise

%**************Initialize parameters for denoising*****************

%%%% For Alternating minimization (AM) %%%%%%
%%%% some parameters of AM are shared with Accalerated AM i.e, tol and Nit
opts.Nit           = 5000;
opts.tol           = 1.0e-8;
opts.mu           = 1.01;  %smaller the more noise filtering
opts.beta         = 0.075;
%opts.rho          = 1;
%opts.p            = 0.3;

%%%% For Accelerated AM %%%
optsAcc.mu =  1.01;
optsAcc.beta = 0.075;

%% for ADMM %%
optsADM.lam =17.9;
optsADM.rho = 1.3;


%%%% Test algorithms %%%%%
out = AMTV2D(f, Img, K, opts);
out2 =  AMTV2D_ACC(f, Img, K, opts, optsAcc);
out3 = ALMTV(f,Img,K,opts,optsADM);

figure;
imshow(out.sol,[]);
title(sprintf('AM Denoised (PSNR = %3.3f dB,SSIM = %3.3f, cputime %.3f s) ',...
                       psnr_fun(out.sol,Img),ssim_index(out.sol,Img), out.cpuTime));
  
figure;
imshow(out2.sol,[]);
title(sprintf('Acc AM Denoised (PSNR = %3.3f dB,SSIM = %3.3f, cputime %.3f s) ',...
                       psnr_fun(out2.sol,Img),ssim_index(out2.sol,Img), out2.cpuTime));
       
figure;
imshow(out3.sol,[]);
title(sprintf('ADMMM Denoised (PSNR = %3.3f dB,SSIM = %3.3f, cputime %.3f s) ',...
                       psnr_fun(out3.sol,Img),ssim_index(out3.sol,Img), out3.cpuTime));
                        
                   
figure;
imshow(uint8(f));
title(sprintf('Noisy (PSNR = %3.3f dB, SSIM = %3.3f)',psnr_fun(f,Img), ssim_index(f,Img)));

figure;
imshow(Img,[]);
title('Original');

figure;
 
semilogy(out2.relativeError,'--','Linewidth',3.0,'Color','red');hold on
semilogy(out.relativeError,'-.','Linewidth',3.0,'Color','black');
semilogy(out3.relativeError,'Linewidth',3.0,'Color','blue');

xlabel('Iterations (k)','FontSize',25,'interpreter','latex');
ylabel('Relative Error','FontSize',25,'interpreter','latex');
axis tight;
grid on;

l = legend('AM','Acc AM','ADMMM');
set(l,'interpreter','latex','FontSize', 25);
set(gca, 'FontSize',20)