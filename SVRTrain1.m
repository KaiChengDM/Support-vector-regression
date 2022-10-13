function model= SVRTrain1(par,Hyperparameters,Covariance)   
 %Training Bayesian epsilon-SVR model including bias b
  C=Hyperparameters(1);  %Trade-off parameter
  e=Hyperparameters(2);  %Tube width parameter
  theta=Hyperparameters(3:end);         %Covariance function hyper-parameters  
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
  
   H1=full(H);  Kernel=H1+H1'-diag(ones(1,m));
   
   Kernel1=Kernel+diag(1./C.*ones(m,1));      %Regularized Covariance matrix
   
   Hb =[Kernel1 -Kernel1; -Kernel1 Kernel1]; 

   f= [(e.*ones(m,1) - Y); (e.*ones(m,1) + Y)];  
   Aeq=([ones(m,1); -ones(m,1)])'; beq=0; 
   x0 = zeros(2*m,1);        % The starting point
   lb = zeros(2*m,1);        % Set the bounds: alphas >= 0
   options=optimoptions('quadprog', 'Algorithm','interior-point-convex','Display','off','StepTolerance',10^(-15));
   alpha=quadprog(Hb,f,[],[],Aeq,beq,lb,[],x0,options); %quadratic programming
   Beta= alpha(1:m) - alpha(m+1:2*m);   %Support value
   
   svi=find(10^(-10)<abs(Beta));  %Find support vector
   
   Index=find(alpha(1:m)>10^(-10));
   Index1=find(alpha(m+1:2*m)>10^(-10));

   Slack=alpha(1:m)./C;        %Slack factors
   Slack1=alpha(m+1:2*m)./C;   %Slack factors
   
   b=Y-Kernel*Beta-e-Slack;
   b1=Y-Kernel*Beta+e+Slack1;

   bias=mean([b;b1]);      %SVR model bias
     
   model.Input=X;
   model.Output=Y;
   model.parameter=Beta;
   model.Covariance=Covariance;
   model.C=C; 
   model.epsilon=e; 
   model.SV=svi; 
   model.bias=bias;
   model.Kernelmatrix1=Kernel1;
   model.Kernelmatrix=Kernel;
   model.Inputmoment=par.Inputmoment;
   model.Outputmoment=par.Outputmoment;
   model.theta=theta;
end 
     
     
     
     
     
     
     