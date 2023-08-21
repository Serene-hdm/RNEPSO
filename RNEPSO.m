function [gbest,gbestval,fitcount,suc,suc_fes,Convergence_curve]=RNEPSO(jingdu,func_num,fhd,Dimension,Particle_Number,me,Max_FES,VRmin,VRmax)

fbias=[100, 200, 300, 400, 500,...
    600, 700, 800, 900, 1000,...
    1100,1200,1300,1400,1500,...
    1600,1700,1800,1900,2000,...
    2100,2200,2300,2400,2500,...
    2600,2700,2800,2900,3000 ];

rand('state',sum(100*clock));
ps=Particle_Number;
D=Dimension;

range=[-100,100];
recorded=0;  %达到精度时记录相关信息
suc=0;
suc_fes=0;

mv=0.2*(VRmax-VRmin);
VRmin=repmat(VRmin,ps,1);
VRmax=repmat(VRmax,ps,1);
Vmin=repmat(-mv,ps,1);
Vmax=-Vmin;
pos=VRmin+(VRmax-VRmin).*rand(ps,D);
fitcount=0;
e=feval(fhd,pos',func_num)-fbias(func_num);
fitcount=fitcount+ps;
vel=Vmin+2.*Vmax.*rand(ps,D);%initialize the velocity of the particles
pbest=pos;
pbestval=e; %initialize the pbest and the pbest's fitness value
[gbestval,gbestid]=min(pbestval);
gbest=pbest(gbestid,:);%initialize the gbest and the gbest's fitness value
gbestrep=repmat(gbest,ps,1);

flag=zeros(1,30);
Convergence_curve=zeros(1,30);
tt=1;

num_nei=3;
neighbor=zeros(ps,num_nei);
for k=1:ps
    perm=randperm(ps);
    neighbor(k,1)=k;
    for i=2:num_nei
    neighbor(k,i)=perm(i);
    end
end



K=0.2;% elite ratio
stag_pbest=zeros(ps,1);cc=2;stag=0;
p1=pbest;p2=pbest;p=pbest;
while fitcount<=Max_FES
    
    if floor(fitcount/10000)==tt && flag(tt)==0
        Convergence_curve(tt)=gbestval;
        flag(tt)=1;
        tt=tt+1;
    end

    w=0.35*(1+cos(pi*fitcount/Max_FES))+0.2;
    Pc1=0.15*(1/(1+exp(0.0035*(fitcount-(Max_FES/2))/D)))+0.1;
    [valBest, indBest] = sort(pbestval, 'ascend');
    [R1, R2, R3] = sort_fit(indBest,K);
%     R01=1:ps;
%     ind1 = pbestval(R01)>pbestval(R2);
    cc(1)=0.5*(1+cos(pi*fitcount/Max_FES))+0.4;
    cc(2)=1.4-0.5*(1+cos(pi*fitcount/Max_FES));

    if fitcount==ps
        p1=pbest;p2=pbest(R1,:);
    end
    
    for k=1:ps
        %ind1(k)>0,选择精英粒子学习
        if stag_pbest(k)>4
            [~,r2]=min(pbestval(neighbor(k,:)));
            for j=1:D
                r1=rand;
                if r1<Pc1
                    p1(k,j)=pbest(k,j);
                else
                    p1(k,j)=pbest(neighbor(k,r2),j);
                end
            end
            for j=1:D
                r1=rand;
                if r1<Pc1
                    p2(k,j)=pbest(k,j);
                else
                    p2(k,j)=pbest(R1(k),j);
                end
            end
        end
    end
    
    
    stag_pbest=stag_pbest+1;stag=stag+1;
    %PSO循环
    for k=1:ps
        aa(k,:)=cc(1).*rand(1,D).*(p1(k,:) - pos(k,:))+cc(2).*rand(1,D).*(p2(k,:) - pos(k,:));
%         aa(k,:)=cc(1).*rand(1,D).*(pbest(k,:) - pos(k,:))+cc(2).*rand(1,D).*(gbest - pos(k,:));
        vel(k,:)=w.*vel(k,:)+aa(k,:);
        vel(k,:)=(vel(k,:)>mv).*mv+(vel(k,:)<=mv).*vel(k,:);
        vel(k,:)=(vel(k,:)<(-mv)).*(-mv)+(vel(k,:)>=(-mv)).*vel(k,:);
        pos(k,:)=pos(k,:)+vel(k,:);
        
        if rand<0.5
            pos(k,:)=(pos(k,:)>VRmax(k,:)).*VRmax(k,:)+(pos(k,:)<=VRmax(k,:)).*pos(k,:);
            pos(k,:)=(pos(k,:)<VRmin(k,:)).*VRmin(k,:)+(pos(k,:)>=VRmin(k,:)).*pos(k,:);
        else
            pos(k,:)=((pos(k,:)>=VRmin(k,:))&(pos(k,:)<=VRmax(k,:))).*pos(k,:)...
                +(pos(k,:)<VRmin(k,:)).*(VRmin(k,:)+0.25.*(VRmax(k,:)-VRmin(k,:)).*rand(1,D))...
                +(pos(k,:)>VRmax(k,:)).*(VRmax(k,:)-0.25.*(VRmax(k,:)-VRmin(k,:)).*rand(1,D));
        end
        
        %更新适应度
        e(k,1)=feval(fhd,pos(k,:)',func_num)-fbias(func_num);
        fitcount=fitcount+1;
        if e(k)<pbestval(k)
            pbest(k,:)=pos(k,:);
            pbestval(k)=e(k);
            stag_pbest(k)=0;
        end
        if pbestval(k)<gbestval
            gbestval=pbestval(k);
            gbest=pbest(k,:);stag=0;
        end
    end
    
    
    %global stagnation
    if stag>4
        num_pr=0.05+0.15*(1-cos(pi*fitcount/Max_FES));
        num_nei=ceil(num_pr*ps);
        neighbor=zeros(ps,num_nei);
        for k=1:ps
            perm=randperm(ps);
            neighbor(k,1)=k;
            for i=2:num_nei
                neighbor(k,i)=perm(i);
            end
        end
    end
    
    %local stagnation
    for k=1:ps
        if stag_pbest(k)>1
            for j=1:D
                r2=rand;
                r1=rand;
                if r1<Pc1
                    p(k,j)=pbest(k,j);
                else
                    p(k,j)=pbest(k,j)+r2*(pbest(R1(k),j)-pbest(k,j))+(1-r2)*(pbest(R2(k),j)-pbest(R3(k),j));
                end
            end
        end
        
        if rand<0.5
            p(k,:)=(p(k,:)>VRmax(k,:)).*VRmax(k,:)+(p(k,:)<=VRmax(k,:)).*p(k,:);
            p(k,:)=(p(k,:)<VRmin(k,:)).*VRmin(k,:)+(p(k,:)>=VRmin(k,:)).*p(k,:);
        else
            p(k,:)=((p(k,:)>=VRmin(k,:))&(p(k,:)<=VRmax(k,:))).*p(k,:)...
                +(p(k,:)<VRmin(k,:)).*(VRmin(k,:)+0.25.*(VRmax(k,:)-VRmin(k,:)).*rand(1,D))...
                +(p(k,:)>VRmax(k,:)).*(VRmax(k,:)-0.25.*(VRmax(k,:)-VRmin(k,:)).*rand(1,D));
        end
        %更新适应度
        re=feval(fhd,p(k,:)',func_num)-fbias(func_num);
        fitcount=fitcount+1;
        if re<pbestval(k)
            pbest(k,:)=p(k,:);
            pbestval(k)=re;
            stag_pbest(k)=0;
        end
        if pbestval(k)<gbestval
            gbestval=pbestval(k);
            gbest=pbest(k,:);stag=0;
        end
    end
    
    
    
    
    if gbestval <= jingdu && recorded == 0
        recorded = 1;
        suc = 1;
        suc_fes = fitcount;
    end
    if fitcount>=Max_FES
        break;
    end
    
end
if flag(30)==0
    Convergence_curve(30)=gbestval;
end
end



function [R1, R2, R3] = sort_fit(indBest,K)


pop_size = length(indBest);

R1=indBest(1:round(pop_size*K));
R1rand = ceil(length(R1) * rand(pop_size, 1));
R1 = R1(R1rand);

R2=indBest(round(pop_size*K)+1:round(pop_size*0.9));
R2rand = ceil(length(R2) * rand(pop_size, 1));
R2 = R2(R2rand);

R3=indBest(round(pop_size*0.9)+1:end);
R3rand = ceil(length(R3) * rand(pop_size, 1));
R3 = R3(R3rand);

end





