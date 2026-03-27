clear all
%hypobolic equation
%tau与h选取法则：选好h再选tau使得cfl条件化简之后的值=0.8.本问题中tau*beta*j>=0
%1-tau*beta*j>=0.即1-beta*ej*tau/h>=0 beta*max_j{ej}*tau/h<=1
%选好h再选tau使得cfl条件化简之后的值=0.8 本例子中
tic

n=1;
alpha0=0.1;
alpha=alpha0;
beta=0.11;
T=600;
p=@(lambda) 1-lambda/0.392; %p(lambda)
f=@(lambda,x) alpha0*n*lambda+(alpha0*lambda-beta)*x;
H=@(lambda,x,mu) lambda*(n+x)*p(lambda)+mu*f(lambda,x);

test=-1;
h=0.1; %step size
N=T/h;


lambda=ones(1,N+1)*0.392; %initialize lambda_t=0.392
x=zeros(1,N+1);
x(1)=0;
mu=zeros(1,N+1);

while test<0
    oldlambda=lambda;
    oldx=x;
    oldmu=mu;
    for i=1:N
        x(i+1)=x(i)+h*f(lambda(i),x(i)); %solve x from ode about x_t. 
    end
    mu(N+1)=.02*x(N+1)/n;  %mu_T= phi'(x_T/n). 
    for i=1:N
        j=N+2-i;
        mu(j-1)=mu(j)+h*(lambda(j)*p(lambda(j))+(alpha*lambda(j)-beta)*mu(j)); % solve mu backwards from ode about mu_t. 
    end

    lambda1=min(0.392, (1+alpha*mu)*0.392/2); %r'(lambda)+alpha*mu=0. capped at 0.392. 
    lambda=0.5*(lambda1+oldlambda);
    
    test1=0.001*sum(abs(lambda))-sum(abs(lambda-oldlambda));
    test2=0.001*sum(abs(x))-sum(abs(x-oldx));
    test3=0.001*sum(abs(mu))-sum(abs(mu-oldmu));
    test=min(min(test1,test2),test3); %stop when lambda, mu and x converge. 
end

J=5000;
h=0.01;
tau=0.1;
N=6001;
Lambda=zeros(J,N);
%lambda(p)=a-b*p
a=0.392;b=0.392;
%p(lambda)=(a-lambda)/b
lambda0=0.196;
ej_max=h*J;
%tau=0.04;
U=zeros(J,N);
% for j=1:J
%     U(j,N)=0.000001*j^2;
% end



for n=N-1:-1:1
    for j=1:J
        bb1=j*(1-beta*tau);bb2=floor(bb1);
        cc1=j*(1-beta*tau)+alpha0/h*(1-beta*tau);cc2=floor(cc1);
        if bb2==0
        deltabb=U(2,n+1)-U(1,n+1);
        Vbb=U(1,n+1)-(1-bb1)*deltabb;
        else
        deltabb=U(bb2+1,n+1)-U(bb2,n+1);
        Vbb=U(bb2,n+1)+(bb1-bb2)*deltabb;
        end
        if cc1>=J
            deltacc=U(J,n+1)-U(J-1,n+1); Vcc=U(J,n+1)+(cc1-J)*deltacc;
            
        else
            deltacc=U(cc2+1,n+1)-U(cc2,n+1); Vcc=U(cc2,n+1)+(cc1-cc2)*deltacc;
        end
        A=zeros(197,1);
        for pp=1:197
            lambdah=lambda0+(pp-1)*0.001;
             A(pp)=lambdah*((a-lambdah)/b+Vcc-Vbb);
        end
            lambda_max=lambda0+(min(find(A==max(A)))-1)*0.001;
            Lambda(j,n)=lambda_max;
     
             U(j,n)=Vbb+(j*h+0.001*j*h*j*h+1)*tau*max(A);
        
%          if j<J
%          U(j,n)=tau*beta*j*U(j-1,n+1)+(1-beta*j*tau)*U(j,n+1)-tau*max(A);
%          else
%               U(j,n)=tau*beta*j*(U(j,n+1)+U(j,n+1)-U(j-1,n+1))+(1-beta*j*tau)*U(j,n+1)-tau*max(A);
           %   U(j,n)=tau*beta*j*(U(j,n+1))+(1-beta*j*tau)*U(j,n+1)+tau*max(A);
%         end
    end
    
end



toc

U_Hd=zeros(J,N);
for j=1:J
    U_Hd(j,N)=U(j,N);
end

for n=N-1:-1:1
    for j=1:J
        bb1=j*(1-beta*tau);bb2=floor(bb1);
        cc1=j*(1-beta*tau)+alpha0/h*(1-beta*tau);cc2=floor(cc1);
        if bb2==0
        deltabb=U_Hd(2,n+1)-U_Hd(1,n+1);
        Vbb=U_Hd(1,n+1)-(1-bb1)*deltabb;
        else
        deltabb=U_Hd(bb2+1,n+1)-U_Hd(bb2,n+1);
        Vbb=U_Hd(bb2,n+1)+(bb1-bb2)*deltabb;
        end
        if cc1>=J
            deltacc=U_Hd(J,n+1)-U_Hd(J-1,n+1); Vcc=U_Hd(J,n+1)+(cc1-J)*deltacc;
            
        else
            deltacc=U_Hd(cc2+1,n+1)-U_Hd(cc2,n+1); Vcc=U_Hd(cc2,n+1)+(cc1-cc2)*deltacc;
        end

            lambdahdd=lambda(n);
             A_Hd=lambdahdd*((a-lambdahdd)/b+Vcc-Vbb);

            
     
             U_Hd(j,n)=Vbb+(j*h+0.001*j*h*j*h+1)*tau*A_Hd;
        
%          if j<J
%          U(j,n)=tau*beta*j*U(j-1,n+1)+(1-beta*j*tau)*U(j,n+1)-tau*max(A);
%          else
%               U(j,n)=tau*beta*j*(U(j,n+1)+U(j,n+1)-U(j-1,n+1))+(1-beta*j*tau)*U(j,n+1)-tau*max(A);
           %   U(j,n)=tau*beta*j*(U(j,n+1))+(1-beta*j*tau)*U(j,n+1)+tau*max(A);
%         end
    end
    
end
eta_Hd=(U-U_Hd)./U*100;
toc
csvwrite('result_D.csv',eta_Hd);