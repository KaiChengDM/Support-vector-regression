function model= SVRmodel(X,Y,Hyperparameters,lb,ub,Covariance)   
%Training Bayesian least square SVR model
[m n] = size(X);            % number of design sites and their dimension

MInput = mean(X);   SInput= std(X); 
X= (X-repmat( MInput,m,1))./repmat(SInput,m,1);%Normalization of input data
MOutput=mean(Y); SOutput=std(Y);
Y=(Y-repmat(MOutput,m,1))./repmat(SOutput,m,1);%Normalization of output data

Inputmoment=[MInput; SInput];
Outputmoment=[MOutput; SOutput];

mzmax = m*(m-1) / 2;        % number of non-zero distances
ij = zeros(mzmax, 2);       % initialize matrix with indices
d = zeros(mzmax, n);        % initialize matrix with distances
ll = 0;
for k = 1 : m-1
  ll = ll(end) + (1 : m-k);
  ij(ll,:) = [repmat(k, m-k, 1) (k+1 : m)'];        % indices for sparse matrix
  d(ll,:) = repmat(X(k,:), m-k, 1) - X(k+1:m,:);    % differences between points
end

par = struct('D', d, 'ij',ij,'Y',Y,'X',X,'Inputmoment',Inputmoment,'Outputmoment',Outputmoment);

[t, f, perf] = boxmin(Hyperparameters, lb, ub, par); %Hyper-parameter Optimization 
 
model=SVRTrain(par,t,Covariance);      %Training Bayesian SVR model
 
 
function  [t, f, perf] = boxmin(t0, lo, up, par)
%Hooke & Jeeves Algorithm for hyper-parameters optimization (This part comes from Dace toolbox)

% Initialize
[t, f, itpar] = start(t0, lo, up, par);
if  ~isinf(f)
  % Iterate
  p = length(t);
  if  p <= 2,  kmax = 2; else,  kmax = min(p,4); end
  for  k = 1 : kmax
    th = t;
    [t, f,  itpar] = explore(t, f,  itpar, par);
    [t, f,  itpar] = move(th, t, f, itpar, par);
  end
end
perf = struct('nv',itpar.nv, 'perf',itpar.perf(:,1:itpar.nv));
end


 function Loglikelihood=SVRLikelihood(Hyperparameters) 
 % Likelihood function of Bayesian SVR mdoel
  model= SVRTrain(par,Hyperparameters,Covariance);
  bias=model.bias;             %SVR bias
  Kernel=model.Kernelmatrix;   %Kernel matrix
  parameter=model.parameter;   %SVR coefficients
  Fmp=Kernel*parameter+bias;   %Prediction at training points
  Deta=model.Output-Fmp;       %Loss
  C=model.C;                   %Trade-off parameter
  Loss=0.5*Deta.^2;
  R1=model.Uppermatrix;
  detR=prod(full(diag(R1)).^2);
  Zd=0.5*(m*log(2.*C)+log(detR));
  Loglikelihood=(0.5.*parameter'*Kernel*parameter+C*sum(Loss)+m*log((2.*pi./C).^0.5)+Zd);%Likelihood function
 end


function  [t, f, itpar] = start(t0, lo, up, par)
% Get starting point and iteration parameters

% Initialize
t = t0(:);  lo = lo(:);   up = up(:);   p = length(t);
D = 2 .^ ([1:p]'/(p+2));
ee = find(up == lo);  % Equality constraints
if  ~isempty(ee)
  D(ee) = ones(length(ee),1);   t(ee) = up(ee); 
end
ng = find(t < lo | up < t);  % Free starting values
if  ~isempty(ng)
  t(ng) = (lo(ng) .* up(ng).^7).^(1/8);  % Starting point
end
ne = find(D ~= 1);

% Check starting point and initialize performance info
[f] = SVRLikelihood(t);   nv = 1;
itpar = struct('D',D, 'ne',ne, 'lo',lo, 'up',up, ...
  'perf',zeros(p+2,200*p), 'nv',1);
itpar.perf(:,1) = [t; f; 1];
if  isinf(f)    % Bad parameter region
  return
end

if  length(ng) > 1  % Try to improve starting guess
  d0 = 16;  d1 = 2;   q = length(ng);
  th = t;   fh = f;   jdom = ng(1);  
  for  k = 1 : q
    j = ng(k);    fk = fh;  tk = th;
    DD = ones(p,1);  DD(ng) = repmat(1/d1,q,1);  DD(j) = 1/d0;
    alpha = min(log(lo(ng) ./ th(ng)) ./ log(DD(ng))) / 5;
    v = DD .^ alpha;   tk = th;
    for  rept = 1 : 4
      tt = tk .* v; 
      [ff ] = SVRLikelihood(tt,par);  nv = nv+1;
      itpar.perf(:,nv) = [tt; ff; 1];
      if  ff <= fk 
        tk = tt;  fk = ff;
        if  ff <= f
          t = tt;  f = ff;  jdom = j;
        end
      else
        itpar.perf(end,nv) = -1;   break
      end
    end
  end % improve
  
  % Update Delta  
  if  jdom > 1
    D([1 jdom]) = D([jdom 1]); 
    itpar.D = D;
  end
end % free variables

itpar.nv = nv;
end
% --------------------------------------------------------

function  [t, f, itpar] = explore(t, f, itpar, par)
% Explore step

nv = itpar.nv;   ne = itpar.ne;
for  k = 1 : length(ne)
  j = ne(k);   tt = t;   DD = itpar.D(j);
  if  t(j) == itpar.up(j)
    atbd = 1;   tt(j) = t(j) / sqrt(DD);
  elseif  t(j) == itpar.lo(j)
    atbd = 1;  tt(j) = t(j) * sqrt(DD);
  else
    atbd = 0;  tt(j) = min(itpar.up(j), t(j)*DD);
  end
%   [ff  fitt] = objfunc(tt,par);  nv = nv+1;
[ff] = SVRLikelihood(tt);  nv = nv+1;
  itpar.perf(:,nv) = [tt; ff; 2];
  if  ff < f
    t = tt;  f = ff; 
  else
    itpar.perf(end,nv) = -2;
    if  ~atbd  % try decrease
      tt(j) = max(itpar.lo(j), t(j)/DD);
%       [ff  fitt] = objfunc(tt,par);  nv = nv+1;
        [ff ] =SVRLikelihood(tt);  nv = nv+1;   
      itpar.perf(:,nv) = [tt; ff; 2];
      if  ff < f
        t = tt;  f = ff; 
      else
        itpar.perf(end,nv) = -2;
      end
    end
  end
end % k

itpar.nv = nv;
end
% --------------------------------------------------------

function  [t, f, itpar] = move(th, t, f, itpar, par)
% Pattern move

nv = itpar.nv;   ne = itpar.ne;   p = length(t);
v = t ./ th;
if  all(v == 1)
  itpar.D = itpar.D([2:p 1]).^.2;
  return
end

% Proper move
rept = 1;
while  rept
  tt = min(itpar.up, max(itpar.lo, t .* v));  
%   [ff  fitt] = objfunc(tt,par);  nv = nv+1;
  [ff ] = SVRLikelihood(tt);  nv = nv+1;
  itpar.perf(:,nv) = [tt; ff; 3];
  if  ff < f
    t = tt;  f = ff;  
    v = v .^ 2;
  else
    itpar.perf(end,nv) = -3;
    rept = 0;
  end
  if  any(tt == itpar.lo | tt == itpar.up), rept = 0; end
end

itpar.nv = nv;
itpar.D = itpar.D([2:p 1]).^.25;
end  
end   
     
     
     