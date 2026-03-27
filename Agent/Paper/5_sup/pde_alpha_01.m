%hypobolic equation
% Rule for choosing tau and h: first choose h, then choose tau such that
% the simplified CFL condition equals 0.8.
% In this problem, tau * beta * j >= 0 and 1 - tau * beta * j >= 0,
% i.e., 1 - beta * e_j * tau / h >= 0, which implies
% beta * max_j{e_j} * tau / h <= 1.
% After fixing h, choose tau so that the CFL condition (after simplification)
% equals 0.8 in this example.
tic
J=5000;
h=0.01;
tau=0.1;
alpha0=0.1;
beta=0.11;
N=6001;
Lambda=zeros(J,N);
a=0.392;b=0.392;
lambda0=0.196;
ej_max=h*J;



U=zeros(J,N);

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
            lambda=lambda0+(pp-1)*0.001;
             A(pp)=lambda*((a-lambda)/b+Vcc-Vbb);
        end
            lambda_max=lambda0+(min(find(A==max(A)))-1)*0.001;
            Lambda(j,n)=lambda_max;
     
             U(j,n)=Vbb+(j*h+0.001*j*h*j*h+1)*tau*max(A);
        
    end
    
end

toc
%csvwrite('result_U.csv',U);
%csvwrite('result_lambda.csv',Lambda);
U_H0=zeros(J,N);
for j=1:J
    U_H0(j,N)= U(j,N);
end

for n=N-1:-1:1
    for j=1:J
        bb1=j*(1-beta*tau);bb2=floor(bb1);
        cc1=j*(1-beta*tau)+alpha0/h*(1-beta*tau);cc2=floor(cc1);
        if bb2==0
        deltabb=U_H0(2,n+1)-U_H0(1,n+1);
        Vbb=U_H0(1,n+1)-(1-bb1)*deltabb;
        else
        deltabb=U_H0(bb2+1,n+1)-U_H0(bb2,n+1);
        Vbb=U_H0(bb2,n+1)+(bb1-bb2)*deltabb;
        end
        if cc1>=J
            deltacc=U_H0(J,n+1)-U_H0(J-1,n+1); Vcc=U_H0(J,n+1)+(cc1-J)*deltacc;
            
        else
            deltacc=U_H0(cc2+1,n+1)-U_H0(cc2,n+1); Vcc=U_H0(cc2,n+1)+(cc1-cc2)*deltacc;
        end

            lambda=lambda0;
             A_H0=lambda*((a-lambda)/b+Vcc-Vbb);

            
     
             U_H0(j,n)=Vbb+(j*h+0.001*j*h*j*h+1)*tau*A_H0;

    end
    
end
eta_H0=(U-U_H0)./U*100;
toc

lambda1=0.455/2;

U_H1=zeros(J,N);
for j=1:J
    U_H1(j,N)= U(j,N);
end

for n=N-1:-1:1
    for j=1:J
        bb1=j*(1-beta*tau);bb2=floor(bb1);
        cc1=j*(1-beta*tau)+alpha0/h*(1-beta*tau);cc2=floor(cc1);
        if bb2==0
        deltabb=U_H1(2,n+1)-U_H1(1,n+1);
        Vbb=U_H1(1,n+1)-(1-bb1)*deltabb;
        else
        deltabb=U_H1(bb2+1,n+1)-U_H1(bb2,n+1);
        Vbb=U_H1(bb2,n+1)+(bb1-bb2)*deltabb;
        end
        if cc1>=J
            deltacc=U_H1(J,n+1)-U_H1(J-1,n+1); Vcc=U_H1(J,n+1)+(cc1-J)*deltacc;
            
        else
            deltacc=U_H1(cc2+1,n+1)-U_H1(cc2,n+1); Vcc=U_H1(cc2,n+1)+(cc1-cc2)*deltacc;
        end

            lambda=lambda1;
             A_H1=lambda*((a-lambda)/b+Vcc-Vbb);

            
     
             U_H1(j,n)=Vbb+(j*h+0.001*j*h*j*h+1)*tau*A_H1;
        
    end
    
end
eta_H1=(U-U_H1)./U*100;
toc


toc
csvwrite('result_H0.csv',eta_H0);
csvwrite('result_H1.csv',eta_H1);
