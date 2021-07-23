% model: predicted information gain (PIG)
% Friedrich Sommer Model
% author: RH

% the total reward is the sum intrinsic (information) and extrinsic
% rewards. We assume logarithmic discounding of the PIG

n=20; % states 1 to ncut are pitch states that trigger LO in the LO-contingent time window,
% states ncut+1 to 40 are sound features from another window (same syllable)

clear leg_str;
figure(3);clf; colstr={'k','b','g','m','c'}; %
figure(4);clf;
N=10000; % number of motor trials
leg_str={'deaf','hearing'};
for k=1:2
    if k==1
        ncut=n;
        sig=1./(1+exp(((1:n)-15)));
        %  sig(ncut+1:n)=0; % deaf birds, features are gone
    elseif k==2
        n=30;
        ncut=30;
        sig=1./(1+exp(((1:n)-15)));
        %    sig(ncut+1:n)=1;%0*rand(1,n-ncut); % hearing birds, some random features
    end
    tit_str='(black=deaf)';
    
    figure(1);clf; plot(sig,'k','linewidth',2);
    hold on; htheta=plot(zeros(1,n),'ko','markersize',6,'linewidth',2);
    set(gca,'fontsize',18);ylim([0 1]); drawnow
    xlabel('Actions (pitch, etc)'); ylabel('P(LO)');
    tith=title('0'); set(gca,'box','off');
    legend({'model','estimate'});
    
    figure(2); clf; hT=plot(zeros(1,n),'o'); xlabel('states'); ylabel('# visits');
    %r=-.004; % reward per LO event
    nR=15;
    %fign=fign+1;
    rs=-logspace(log10(0.01),log10(100),nR);
    %rs=-logspace(log10(0.0001),log10(0.1),nR);
    LOC=zeros(1,nR); LOCend=LOC;
    for j=1:nR
        T=zeros(1,n); LO=zeros(1,n);
        Tend=T; LOend=LO; % last 1000 trials
        r=rs(j);
        for i=1:N
            if 0% true
                % RH
                theta=(1+LO)./(1+T); thetan=(1+T-LO)./(1+T);
                theta0=(1+LO)./(2+T); theta0n=(2+T-LO)./(2+T);
                theta1=(2+LO)./(2+T); theta1n=(1+T-LO)./(2+T);
                DKL=[theta.*log(theta./theta0)+thetan.*log(thetan./theta0n);...
                    theta.*log(theta./theta1)+thetan.*log(thetan./(eps+theta1n))];
                pig=sum([thetan;theta].*DKL,1);
                %              pig=sum([theta0;theta1].*DKL,1);
            else
                % AZ
                theta1=(1+LO)./(1+T); theta0=(1+T-LO)./(1+T);
                theta01=(1+LO)./(2+T); theta00=(2+T-LO)./(2+T);
                theta11=(2+LO)./(2+T); theta10=(1+T-LO)./(2+T);
                DKL=[theta1.*log(theta1./theta11)+theta0.*log(theta0./(eps+theta10));...
                    theta1.*log(theta1./theta01)+theta0.*log(theta0./theta00)];
                pig=sum([theta1;theta0].*DKL,1);
            end
            
            h=LO; h(ncut+1:end)=0; % states ncut+1 to 40 are unrelated to LO
            [dummy,w]=max(log(pig)+r*(h./(eps+T))); % the r term is the expected reward
            %       [dummy,w]=max((pig)+r*(LO./(eps+T))); % the r term is the expected reward
            lo=rand(1)<sig(w);
            T(w)=T(w)+1; LO(w)=LO(w)+lo;
            if i>9*N/10
                Tend(w)=Tend(w)+1; LOend(w)=LOend(w)+lo;
            end
        end
        LOC(j)=sum(LO(1:ncut))/sum(T); % only states 1 to ncut trigger LO
        LOCend(j)=sum(LOend(1:ncut))/sum(Tend);
        fprintf('reward: %.2g, Light off contingency: %.2g\n',r,LOC(j));
        if j==5%ceil(nR/2)
            set(htheta,'ydata',LO./(eps+T));
            set(hT,'ydata',T);
            drawnow; %pause
            set(tith,'string',['r=' num2str(r)]) ;
            pause
        end
    end
    
    figure(3); semilogx(rs,LOC,colstr{k}); hold on;   xlabel('reward/LO'); ylabel('Contingency');%title(['n=' num2str(n)]);
    figure(4); semilogx(rs,LOCend,colstr{k}); hold on; xlabel('reward/LO'); ylabel('Contingency');%title(['n=' num2str(n)]);
    %    leg_str{k}=['n=' num2str(n)];
end
figure(3);legend(leg_str,'Location','Best');title([tit_str ': all trials']);
figure(4);legend(leg_str,'Location','Best'); title([tit_str ': last ' num2str(N/10) ' trials']);
legend