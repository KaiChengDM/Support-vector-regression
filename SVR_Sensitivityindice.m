function [E,V,VU,VT]= SVR_Sensitivityindice(model,X,Y)
[m n] = size(X);  % number of design sites and their dimension
tht=model.theta;  
MInput=model.Inputmoment(1,:);
SInput=model.Inputmoment(2,:);
Moutput=model.Outputmoment(1,:);
Soutput=model.Outputmoment(2,:);
J=MInput;S=SInput;
N=size(X,1); d=size(X,2);
A= (X-repmat( MInput,m,1)) ./ repmat(SInput,m,1);
%gm=dmodel.gamma;
 gm=model.parameter;
%INV=inv(dmodel.C')*inv(dmodel.C);
%gm=INV*(Y-E1);
%--------------------------------------------------
M=model.bias;
for i=1:N
  m=1;
for k=1:d
  m1(i,k)=(pi/tht(k)).^0.5.*S(k)*(normcdf((2*tht(k))^0.5*((1-J(k))/S(k)-A(i,k)))+normcdf((2*tht(k))^0.5*(J(k)/S(k)+A(i,k)))-1);
  m=m.*m1(i,k);
end
  M=M+m*gm(i);
end
  E=M*Soutput+Moutput;
%--------------------------------------------------
 V1=0;
for i=1:N
for j=1:N
  m=1;  
for k=1:d
  m2(i,j,k)=exp(-tht(k)./2*(A(i,k)-A(j,k))^2)*(pi/2/tht(k)).^0.5*S(k)*(normcdf(tht(k).^0.5*(2*(1-J(k))./S(k)-A(i,k)-A(j,k)))+normcdf(tht(k).^0.5*(2*(J(k))./S(k)+A(i,k)+A(j,k)))-1);
  m=m*m2(i,j,k);
 end 
  V1=V1+m*gm(i)*gm(j);
end
end
 V=(V1-M.^2)*Soutput^2;
 %--------------------------------------------------
for kk=1:d
 VU(kk)=0;VT(kk)=0;
for i=1:N
for j=1:N
     m=1; mm=1;
for k=1:d
    m=m*m1(i,k).*m1(j,k);
end
  M=m*(m2(i,j,kk)./m1(i,kk)./m1(j,kk)-1);
  VU(kk)=VU(kk)+gm(i)*gm(j)*M;
  for k=1:d
  if k~=kk
     mm=mm.*(m2(i,j,k)./m1(i,k)./m1(j,k));
  end
  end
   VT(kk)=VT(kk)+gm(i)*gm(j)*m*(mm-1);
end 
end
end
VU=VU.*Soutput^2;
VT=VT.*Soutput^2;
VT=V-VT;
end
