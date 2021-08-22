function out = Acc_Aml1(b,Img,A,opts, optsAcc)


lam = optsAcc.lam; % Regularization parameter
rho_v = optsAcc.rho_v;
rho_w = optsAcc.rho_w;

res  = opts.res;
tol = opts.tol;
Nit = opts.Nit;

relError        = zeros(Nit,1);
psnrGain        = relError;     % PSNR improvement every iteration
ssimGain        = relError;
objVal          = relError;
rhoVal          = relError;


[row, col]  = size(b);

x = b;

t = 1;      
y = x;

%*************** v sub-problem variable initialization ******************

v_1         = zeros(row, col); % v solution of the v sub-problem for Dx_h
v_2         = v_1; % v solutiono for the v sub-problem for Dx_v

%************** u sub-problem variable initialization *******************


eigA        = psf2otf(A,[row col]); %In the fourier domain
eigAtA      = abs(eigA).^2;
eigDtD      = abs(fft2([1 -1], row, col)).^2 + abs(fft2([1 -1]', row, col)).^2;

[D,Dt]      = defDDt(); %Declare forward finite difference operators


[Dx1, Dx2] = D(x);

lhs     = rho_v*eigDtD + rho_w*eigAtA;
q       = imfilter (x,A,'circular') -b;

curNorm  = norm(Dx1 -v_1,'fro') + norm(Dx2 - v_2,'fro');

tg = tic;
for k = 1:Nit
    
  
    %%%%% Isotropic TV %%%%%%
    [v_1, v_2] = isoShrink(Dx1,Dx2,lam/rho_v);
    
    
    %%%%%% w-subproblem %%%%%%
    w  = shrink(q,1/rho_w);
    
    %%%%%% x-subproblem %%%%%%
    x_old = x;
    temp  = w + b;
    rhs   = rho_v*Dt(v_1,v_2)+ rho_w*imfilter(temp,A,'circular');
    x     = fft2(rhs)./lhs;
    x     = real(ifft2(x));
    
    q = imfilter (y,A,'circular') - b;
    
    %Acceleration part
    
   
    %%% Option 3 %%%
    t      =  (k-1)/(k + 2);
    
    y     = x + t*(x -x_old);
    
    [Dx1, Dx2]            = D(y);
   
    
    relError(k)    = norm(x - x_old,'fro')/norm(x, 'fro');
   
    
    if relError(k) < tol
         break;
    end
    
   
end

tg = toc(tg);
    
    out.sol                 = x;
    out.relativeError       = relError(1:k);
    out.psnrGain            = psnrGain(1:k);
    out.ssimGain            = ssimGain(1:k);
    out.rhoValue            = rhoVal(1:k);
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
