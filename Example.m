
clear all;  clc;
format long;

g=@(x)x(:,1).^4+sin(2.*pi.*x(:,2));

d=2; n=20;
pp = sobolset(d,'Skip',10,'Leap',10); X=net(pp,n);  
X=2.*X-1; Y=g(X);

xx    = -1:0.02:1;
nnp   = length(xx);
[XX,YY] = meshgrid(xx);
xnod  = cat(2,reshape(XX',nnp^2,1),reshape(YY',nnp^2,1));
Z = g(xnod); 
Z = reshape(Z,nnp,nnp);

%Bayesian least square SVR model
Hyperparameters=[10^5   1.*ones(1,d)];             %Initial values of Hyper-parameters
lb=[10    10^-5.*ones(1,d)];                       %Lower bound of Hyper-parameters
ub=[10^10  10^5.*ones(1,d)];                       %Upper bound of Hyper-parameters
model1= SVRmodel(X,Y,Hyperparameters,lb,ub,'Gaussian');

[Y1,V1]=SVRPredict(xnod,model1) ; 
 Y1 =reshape(Y1,nnp,nnp);   
 V1= reshape(V1,nnp,nnp);
  
   mesh(XX,YY,Z)
   xlabel('x_1','LineWidth',3)
   ylabel('x_2','LineWidth',3)
   zlabel('y','LineWidth',3)

   figure
   subplot(1,2,1)
   mesh(XX,YY,Y1)
   xlabel('x_1','LineWidth',3)
   ylabel('x_2','LineWidth',3)
   zlabel('Mean','LineWidth',3)
   subplot(1,2,2)
   mesh(XX,YY,V1)
   xlabel('x_1','LineWidth',3)
   ylabel('x_2','LineWidth',3)
   zlabel('Variance','LineWidth',3)
   

 