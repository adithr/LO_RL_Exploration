% model: predicted information gain (PIG)
% Friedrich Sommer Model
% author: RH

% the mode option specifies our hypothesis about the difference in hearing
% birds, what seems to work is option 3, hearing birds care about more
% non-LO states

% the total reward is the sum intrinsic (information) and extrinsic
% rewards. We assume logarithmic discounding of the PIG

mode=3; % 1=higher density, 2= more fully on+off states, 3=more on states

clear leg_str;
figure(3);clf; colstr={'k','b','g','m','c'}; %
figure(4);clf;
N=40000; % number of motor trials

for k=1:4
    n=k*16; % number of motor states
   % n=2*k+16; % number of motor states
    switch mode
        case 1
             sig=1./(1+exp(1/k*((1:n)-n/2)));
             tit_str='Low state density during deafness (black=deaf)';
       case 2
            sig=1./(1+exp((1:n)-n/2));
              tit_str='Fewer fully on and off states deafness (black=deaf)';
        case 3
            %sig=1./(1+exp((1:n)-8));
            sig=1./(1+exp((1:n)-n+8));
              tit_str='fewer light on states during deafness (black=deaf)';
    end
    figure(1);clf; plot(sig);
    hold on; htheta=plot(zeros(1,n),'ro'); ylim([0 1]); drawnow
    xlabel('Sensory states (Pitch, etc)'); ylabel('LO');
    
    figure(2); clf; hT=plot(zeros(1,n),'o'); xlabel('states'); ylabel('# visits');
    %r=-.004; % reward per LO event
    nR=20;
    %fign=fign+1;
    rs=-logspace(log10(0.1),log10(100),nR);
   %rs=-logspace(log10(0.0001),log10(0.1),nR);
    LOC=zeros(1,nR); LOCend=LOC;
    for j=1:nR
        T=zeros(1,n); LO=zeros(1,n);
        Tend=T; LOend=LO; % last 1000 trials
        r=rs(j);
        for i=1:N
            if 0% true
                % RH
                %                 theta=(1+LO)./(1+T); thetan=(1+T-LO)./(1+T);
                %                 theta0=(1+LO)./(2+T); theta0n=(1+T-LO)./(2+T);
                %                 theta1=(2+LO)./(2+T); theta1n=(T-LO)./(2+T);
                theta=(1+LO)./(1+T); thetan=(1+T-LO)./(1+T);
                theta0=(1+LO)./(2+T); theta0n=(2+T-LO)./(2+T);
                theta1=(2+LO)./(2+T); theta1n=(1+T-LO)./(2+T);               
                DKL=[theta.*log(theta./theta0)+thetan.*log(thetan./theta0n);...
                    theta.*log(theta./theta1)+thetan.*log(thetan./(eps+theta1n))];
         %       pig=sum([theta0;theta1].*DKL,1);
              pig=sum([thetan;theta].*DKL,1);
            else
            % AZ
                theta1=(1+LO)./(1+T); theta0=(1+T-LO)./(1+T);
                theta01=(1+LO)./(2+T); theta00=(2+T-LO)./(2+T);
                theta11=(2+LO)./(2+T); theta10=(1+T-LO)./(2+T);
                DKL=[theta1.*log(theta1./theta11)+theta0.*log(theta0./(eps+theta10));...
                    theta1.*log(theta1./theta01)+theta0.*log(theta0./theta00)];
                %DKL=-[theta11.*log(theta11./theta1)+theta10.*log(theta10./(eps+theta0));...
                %    theta01.*log(theta01./theta1)+theta00.*log(theta00./theta0)];
                pig=sum([theta1;theta0].*DKL,1);
            end
            
            %w=ceil(n*rand(1));
            [dummy,w]=max(log(pig)+r*(LO./(eps+T))); % the r term is the expected reward
     %       [dummy,w]=max((pig)+r*(LO./(eps+T))); % the r term is the expected reward
            lo=rand(1)<sig(w);
            T(w)=T(w)+1; LO(w)=LO(w)+lo;
            if i>9*N/10
                Tend(w)=Tend(w)+1; LOend(w)=LOend(w)+lo;
            end
        end
        LOC(j)=sum(LO)/sum(T);
        LOCend(j)=sum(LOend)/sum(Tend);
        fprintf('reward: %.2g, Light off contingency: %.2g\n',r,LOC(j));
        set(htheta,'ydata',LO./(eps+T));
        set(hT,'ydata',T);
        drawnow; %pause
    end
    
    figure(3); semilogx(rs,LOC,colstr{k}); hold on;   xlabel('reward/LO'); ylabel('Contingency');%title(['n=' num2str(n)]);
    figure(4); semilogx(rs,LOCend,colstr{k}); hold on; xlabel('reward/LO'); ylabel('Contingency');%title(['n=' num2str(n)]);
    leg_str{k}=['n=' num2str(n)];
end
figure(3);legend(leg_str,'Location','Best');title([tit_str ': all trials']);
figure(4);legend(leg_str,'Location','Best'); title([tit_str ': last ' num2str(N/10) ' trials']);
legend