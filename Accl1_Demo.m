% Created by Tarmizi Adam,15/10/2020


clc;
clear all;
close all;

imageName = 'peppers.bmp';    

Img = imread(imageName);

if size(Img,3) > 1
    Img = rgb2gray(Img);
end

[row, col] = size(Img);

row = int2str(row);
col = int2str(col);

imageSize = [row 'x' col];


A     =   fspecial('average',9); % Blur kernel
b = imfilter(Img,A,'circular');


b  = impulsenoise(b,0.4,0);
b = double(b);


optsADM.lam = 18.5; % Regularization parameter, play with this !
optsADM.tol = 1e-8;
optsADM.Nit = 5000;



opts.lam        = 0.17; % Regularization parameter, 1st order 
opts.rho_v      = 0.045;  % smaller, faster ?
opts.rho_w      = 0.045;   % smaller, faster ?
opts.res        = 0;

opts.tol = 1e-8;
opts.Nit = 5000;


%%% For Accelerated  AM %%%%
optsAcc.lam = 0.041;
optsAcc.rho_v = 0.0025;
optsAcc.rho_w  = 0.0025;


%******** Main denoising function call********

out = Aml1(b,Img,A,opts);
out1 =  Acc_Aml1(b,Img,A,opts, optsAcc);
out2 = ADMM1_SnP(b,Img,A,optsADM); 


figure;
imshow(out.sol,[]);   
title(sprintf('Restored (PSNR = %3.3f dB,SSIM = %3.3f) ',...
                       psnr_fun(out.sol,double(Img)),ssim_index(out.sol,double(Img))));

figure;
imshow(out1.sol,[]);
title(sprintf('Restored (PSNR = %3.3f dB,SSIM = %3.3f) ',...
                       psnr_fun(out1.sol,double(Img)),ssim_index(out1.sol,double(Img))));
   
 %                  
figure;
imshow(out2.sol,[]);
title(sprintf('Restored (PSNR = %3.3f dB,SSIM = %3.3f) ',...
                       psnr_fun(out2.sol,double(Img)),ssim_index(out2.sol,double(Img))));
                       
%}
figure;
 %
semilogy(out.relativeError,'--','Linewidth',3,'Color','red');hold

semilogy(out1.relativeError,'Linewidth',3,'Color','blue');
%}
semilogy(out2.relativeError,'-.','Linewidth',3,'Color','black');

xlabel('Iterations (k)','FontSize',25,'interpreter','latex');
ylabel('Relative Error','FontSize',25,'interpreter','latex');
axis tight;
grid on;
l = legend('AM','Acc AM','ADMM');
set(l,'interpreter','latex','FontSize', 25);
set(gca, 'FontSize',20)
                       
