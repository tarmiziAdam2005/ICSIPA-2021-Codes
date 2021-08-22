function out = ALMTV(g,Img,H,opts,optsADM)
%%========== ALMTV FOR DEBLURRING with gaussian noise =================================
%%
%==========Isotropic Total variation using Augmented Lagrangian====
% ALMTV: Augmented Lagrangian Method Total Variation/ADMM


%inputs:
%       g       : Observed blurred(and possibly noisy) image
%       Img     : Original image(clean/unblurred)
%       H       : Point spread function/Blurring kernel (A linear operator)
%       lam     : regularization parameter
%       rho     : regularization parameter of the Augmented Lagrangian form
%                 of the main problem.
%       Nit     : Number of iterations
%       tol     : Error tolerance for stopping criteria


% The program solves the following core objective function

%   min_f   lam/2*||Hf - g||^2 + ||Df||_1

%%

lam = optsADM.lam;
rho = optsADM.rho;

Nit = opts.Nit;           
tol = opts.tol ;          

[row,col] = size(g);
f         = g;

u1        = zeros(row,col); %Initialize intermediat variables for u subproblem
u2        = zeros(row,col); %     "       "           "       for u subproblem

y1        = zeros(row,col); %Initialize Lagrange Multipliers
y2        = zeros(row,col); %   "       Lagrange Multipliers

relError     = zeros(Nit,1); % Compute error relative to the previous iteration.


eigHtH  = abs(fft2(H,row,col)).^2; %eigen value for HtH
eigDtD  = abs(fft2([1 -1], row, col)).^2 + abs(fft2([1 -1]', row, col)).^2; % eigen value ofDtD
Htg     = imfilter(g, H, 'circular');

[D,Dt]      = defDDt(); %Declare forward finite difference operators
[Df1, Df2] = D(f);

%=================== Main algorithm starts here ======================
tg = tic;
    for k=1:Nit
    
        %Solving the f subproblem
        f_old   = f;
        rhs     = lam*Htg + Dt(u1 - (1/rho)*y1, u2 - (1/rho)*y2);
        eigA    = lam*eigHtH + rho*eigDtD;
        f       = fft2(rhs)./eigA;
        f       = real(ifft2(f));
    
        %Solving the u subproblem
        [Df1, Df2]  = D(f);
        v1          = Df1 + (1/rho)*y1;
        v2          = Df2 + (1/rho)*y2;
    
       
        
        [u1, u2] = isoShrink(v1,v2,1/lam);
    
        %Update y, the Lagrange multipliers
        y1          = y1 - rho*(u1 - Df1);
        y2          = y2 - rho*(u2 - Df2);
        
        relError(k)    = norm(f - f_old,'fro')/norm(f, 'fro');
        
        
          
        if relError(k) < tol
            break
        end 
    end
tg = toc(tg);
%======================= Results ==============================

out.sol                 = f;                %Deblurred image
out.relativeError       = relError(1:k);
out.cpuTime             = tg;

end

function [D,Dt] = defDDt()
D  = @(U) ForwardDiff(U);
Dt = @(X,Y) Dive(X,Y);
end

function [Dux,Duy] = ForwardDiff(U)
 Dux = [diff(U,1,2), U(:,1,:) - U(:,end,:)];
 Duy = [diff(U,1,1); U(1,:,:) - U(end,:,:)];
end

function DtXY = Dive(X,Y)
  % Transpose of the forward finite difference operator
  % is the divergence fo the forward finite difference operator
  DtXY = [X(:,end) - X(:, 1), -diff(X,1,2)];
  DtXY = DtXY + [Y(end,:) - Y(1, :); -diff(Y,1,1)];   
end


function [v1, v2] = isoShrink(u1,u2,r)
    u = sqrt(u1.^2 + u2.^2);
    u(u==0) = 1;
    u = max(u - r,0)./u;
    
    v1 = u1.*u;
    v2 = u2.*u;
end