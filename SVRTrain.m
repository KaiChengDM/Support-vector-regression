function model= SVRTrain(par,Hyperparameters,Covariance)   
%Training Bayesian least square SVR model 

  C=Hyperparameters(1);                 %Trade-off parameter
  theta=Hyperparameters(2:end);         %Covariance function hyper-parameters  
  X=par.X; Y=par.Y;
  [m n] = size(X); 
  
  switch Covariance 
        case 'Gaussian'
                         H=SVRcorrgauss(theta,par);
        case 'Matern5_2'
                         H=SVRMatern5_2(theta,par);  
        case 'Matern3_2'
                         H=SVRMatern3_2(theta,par);  
        otherwise
        error('Error: Unknown kernel/correlation function family!')
  end       

   H1=full(H);  Kernel=H1+H1'-diag(ones(1,m)); %Full Covariance matrix
   
   Kernel1=H+diag(1./C.*ones(m,1));        % Regularized Covariance matrix
   
   [R1 rd]=chol(Kernel1);  %Cholskey decomposition
    
    NKernel=R1\(R1'\diag(ones(1,m)));  %Inversion of Covariance matrix

    B=ones(1,m); 
    NZ=[-(B*NKernel*B')\1 ((B*NKernel*B')\1)*B*NKernel; NKernel*B'*((B*NKernel*B')\1) NKernel-NKernel*B'*((B*NKernel*B')\1)*B*NKernel];
  
    beta=NZ*[0;Y];  %SVR coefficients

    model.Input=par.X;
    model.Output=par.Y;
    model.parameter=beta(2:end);
    model.bias=beta(1);
    model.Covariance=Covariance;
    model.C=C; 
    model.InverseMatrix=NKernel; 
    model.Kernelmatrix1=Kernel1;
    model.Kernelmatrix=Kernel;
    model.theta=theta;
    model.Uppermatrix=R1;
    model.Inputmoment=par.Inputmoment;
    model.Outputmoment=par.Outputmoment;

end 
     
     
     
     
     
     
     