clear;
clc;

%Inputs
m=11;
n=7;

x=0;
y=0;

X=zeros(m+1,1);
Y=zeros(n+1,1);

D=zeros(((m+1)*(n+1)),1);

D1=zeros((m+1)*(n+1),1);
D2=zeros((m+1)*(n+1),1);

for i=1:m
    for j=1:n
      y=y+j;
      Y(j+1,1)=y;
      y=0;
    end
    x=x+i;
     X(i+1,1)=x;
    x=0;
end

%D calculation

for i=1:m+1
    for j=1:n+1
     I=1+(j-1)+(i-1)*(n+1);
     D1(I,1)=((X(i)/m)-(Y(j)/n));
     D2(I,1)=((Y(j)/n)-(X(i)/m));
D(I,1)=max(D1(I,1),D2(I,1)); 

 end
end

% Mean of D
x_bar=mean(D);

% SD of D
sd=std(D);
    
% Probability 
count1=0;
count2=0;
for i=1:m+1
    for j=1:n+1
        I=1+(j-1)+(i-1)*(n+1);
    if D(I,1)>0.2
        count1=count1+1;
        if D(I,1)>0.6
     count2=count2+1;
        end
    end
    end
end

% prob of D>0.2
P1=count1/((m+1)*(n+1));

% prob of D>0.6 given D>0.2
P2=count2/count1;

