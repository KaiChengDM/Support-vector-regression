function  R= SVRcorrgauss(theta, par)
%Gaussian covariance function 
d=par.D;X=par.X;
[m n] = size(d);  % number of differences and dimension of data
if  length(theta) == 1
  theta = repmat(theta,1,n);
elseif  length(theta) ~= n
  error(sprintf('Length of theta must be 1 or %d',n))
end

td = d.^2 .* repmat(-theta(:).',m,1);
r = exp(sum(td, 2));

ij=par.ij;

m=size(X,1);
idx = find(r > 0);   o = (1 : m)';   
mu = (10+m)*eps;
R = sparse([ij(idx,1); o], [ij(idx,2); o], ...
[r(idx); ones(m,1)+mu]);  
end