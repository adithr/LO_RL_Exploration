% Friston's DKL minimization (first run EXPLR_publish3
epsiAll=0.01:.01:.8; % prior parameterization
na=4; ns=na+2; 

h1=zeros(ns,1); h1(1:3)=[1/4 1/2 1/4]; h2=zeros(na,1); h2(1)=1/4;
w0=toeplitz(h1,h2); 
figure(1);clf; imagesc(w0)
Hs=zeros(na,length(epsiAll));
MaxH=zeros(1,length(epsiAll));
for i=1:length(epsiAll)
    epsi=epsiAll(i);
    pLO_s=[0.99*ones(3,1); 0.01*ones(3,1)]*ones(1,4);
    pL_s=(1-pLO_s);
    p_LO=sum(w0.*pLO_s)';
    p_L=sum(w0.*pL_s)';

    pLO_se=[(1-epsi)*ones(3,1); 0.005*ones(3,1)]*ones(1,4);
    pL_se=(1-pLO_s);
    p_LOe=mean(sum(w0.*pLO_se));
    p_Le=mean(sum(w0.*pL_se));
    
    H=-p_LO.*log(p_LO)-p_L.*log(p_L)+p_LO.*log(p_LOe)+p_L.*log(p_Le);
  [dummy,MaxH(i)]=max(H);
    Hs(:,i)=H;
end
figure(2);clf; plot(epsiAll,MaxH);
figure(3);clf; plot(p_LO);
figure(4);clf;imagesc(epsiAll,1:na,Hs);colormap(hot); colorbar


