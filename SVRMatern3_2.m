function  R= SVRMatern3_2(theta, X)
% Normalize data

[m n] = size(X);  % number of design sites and their dimension

mzmax = m*(m-1) / 2;        % number of non-zero distances
ij = zeros(mzmax, 2);         % initialize matrix with indices
d = zeros(mzmax, n);         % initialize matrix with distances
 ll = 0;
% 
for k = 1 : m-1
  ll = ll(end) + (1 : m-k);
  ij(ll,:) = [repmat(k, m-k, 1) (k+1 : m)']; % indices for sparse matrix
  d(ll,:) = repmat(X(k,:), m-k, 1) - X(k+1:m,:); % differences between points
end
% 
% if  min(sum(abs(D),2) ) == 0
%   error('Multiple design sites are not allowed'), end

% Set up  R
% r=corrgauss(D,Bw);

[m n] = size(d);  % number of differences and dimension of data
if  length(theta) == 1
  theta = repmat(theta,1,n);
elseif  length(theta) ~= n
  error(sprintf('Length of theta must be 1 or %d',n))
end

td =sqrt(3).*abs(d).* repmat(-theta(:).',m,1);
r = exp(sum(td, 2)).*prod((1+sqrt(3).*abs(d).* repmat(-theta(:).',m,1))')';

m=size(X,1);
idx = find(r > 0);   o = (1 : m)';   
mu = (10+m)*eps;
R = sparse([ij(idx,1); o], [ij(idx,2); o], ...
[r(idx); ones(m,1)+mu]);   

% if  nargout > 1
%   dr = repmat(-2*theta(:).',m,1) .* d .* repmat(r,1,n);
end