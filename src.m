%ECE539 Midterm Project
%src.m
%Created by Rong Zhang
p=10; %change this value to test different p
train=zeros(38*p,1024);
trainLabel=zeros(38*p,0);
trainIndex=zeros(38*p,0);

n=zeros(38,1);
for i=1:2414
    n(gnd(i))=n(gnd(i))+1;
end

index=ones(38,1);
for i=1:37
    index(i+1)=index(i)+n(i);
end

for i=1:38
    for j=1:p
        num=floor(index(i)+n(i)*rand());
        trainIndex((i-1)*p+j,1)=num;
        trainLabel((i-1)*p+j,1)=gnd(num);
        train((i-1)*p+j,:)=fea(num,:);
    end
end

for i=1:p*38
    train(i,:)=train(i,:)/norm(train(i,:));
end

train=train';

cnt=0;
all=0;

for k=1:2414
    if(~ismember(trainIndex,k))
        y=fea(k,:)';
        sol=SolveBP(train,y,p*38);%SparseLab Toolbox needed
        minRes=inf;
        sigma=ones(p*38,1);
        result=-1;
        for i=1:38
            sigma=zeros(38*p,1);
            for j=(i-1)*p+1:i*p
                sigma(j,1)=sol(j,1);
            end
            curRes=norm(y-train*sigma);
            if(curRes<minRes)
                minRes=curRes;
                result=i;
            end
        end
        if(result==gnd(k,1))
            cnt=cnt+1;
        end
        all=all+1;
    end
end

score=cnt/all;
            





        

