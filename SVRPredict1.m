function [Y,Variance]=SVRPredict1(X1,model)   
 %SVR prediction
[m n] =size(model.Input);  n1=size(X1,1); 

beta=model.parameter;

MInput=model.Inputmoment(1,:);
SInput=model.Inputmoment(2,:);
MOutput=model.Outputmoment(1,:);
SOutput=model.Outputmoment(2,:);

X1=(X1-repmat(MInput,n1,1))./repmat(SInput,n1,1);

X=model.Input;   theta=model.theta;

 mx=n1; 

 dx = zeros(mx*m,n);  kk = 1:m;
 for  k = 1 : mx
      dx(kk,:) = repmat(X1(k,:),m,1) - X;
      kk = kk + m;
  end
    
[m1 n] = size(dx);  % number of differences and dimension of data
if  length(theta) == 1
  theta = repmat(theta,1,n);
elseif  length(theta) ~= n
  error(sprintf('Length of theta must be 1 or %d',n))
end

Covariance=model.Covariance;
  
 switch Covariance 
        case 'Gaussian'
         td = dx.^2 .* repmat(-theta(:).',m1,1);
         r = exp(sum(td, 2));
        case 'Matern5_2'
         td =sqrt(3).*abs(dx).* repmat(-theta(:).',m1,1);
         r = exp(sum(td, 2)).*prod((1+sqrt(3).*abs(dx).* repmat(-theta(:).',m1,1))')';
        case 'Matern3_2'
         td =sqrt(5).*abs(dx).*repmat(-theta(:).',m1,1);
         r = exp(sum(td, 2)).*prod((1+sqrt(5).*abs(dx).* repmat(-theta(:).',m1,1)+5./3.*(abs(dx).* repmat(-theta(:).',m1,1)).^2)')';
        otherwise
        error('Error: Unknown kernel/correlation function family!')
 end       

 H= reshape(r, m, mx)';
  
 Num=model.SV;
 Y=H*beta+model.bias; %SVR Prediction mean
     
 [R1 rd]=chol(model.Kernelmatrix1(Num,Num));  %Cholskey decomposition
 for i=1:n1
 Variance(i)=1-diag(H(i,Num)*(R1\(R1'\H(i,Num)'))); %SVR Prediction variance
 end

 Y=Y.*SOutput+MOutput;    %Prediction
 Variance=Variance'.*SOutput.^2; %Prediction variance
end 
     
     
     
     
     
     
     