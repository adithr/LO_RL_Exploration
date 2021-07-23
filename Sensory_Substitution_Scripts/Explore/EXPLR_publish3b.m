%%% EXPLR_publish %%%
% Model of Exploration in Hearing and Deaf Light-Off (LO) Birds
% Implementation of the model from the supplementary material of the
% manuscript:
% 
% Empowerment maximization

nn=3; % Number of Notes the bird is singing
kLO=2; % index of note that can trigger LO
na=6; % Number of Actions per note (should be even)
alpha=0.001; % learning rate TD learning
alpha0=alpha;
tau=1;  % forgetting time constant
verbose=0;
gamma=0; % Gamma parameter in SARSA learning
sc_fac=1e4;
sc_fac2=50;
offset=-0.0001;
ent_sc=0.01;

epsi=1e-40;

% Simulation parameters:
N=3000; % # trials (per note)
do_control=0; % add control condition without LO (0 = only LO, 1 = add control, 2 = remove entropy gain)
nbirds=2; % Number of Birds to simulate and average the result over

% Reward for LO
nR=12; % Number of different Reward values tested (reward value fixed for one  simulation)
rs=-logspace(log10(0.0001),log10(1000),nR); % reward per LO event
rs=rs(end:-1:1);

jplot=15; % plot VTA firing for reward # jplot
igC=0*1e-5;%1e-40;%1e-14; % for constant information gain, set to nonzero for getting rid of exploration bonus

% Number of States
nsH=na+2; % Number of States for Hearing bird (per note) corresponding to actual pitch sung
nsD=2; % Number of States for Deaf bird (per note), the second state is the silent state
nsA=nsH*nn; % total Number of States (for All notes), the last stte is the silent state

% Sensorimotor model: motor action -> sensory states
h1=zeros(nsH,1); h1(1:3)=[1/4 1/2 1/4]; h2=zeros(na,1); h2(1)=1/4;
wh=toeplitz(h1,h2); w=zeros(nsA,nn*na);
for i=1:nn
    w(1+(i-1)*nsH:i*nsH,1+(i-1)*na:i*na)=wh;
end
transitionAS=cumsum(w,1); % defines Transitions from Action to State (Fig. 4B)
figure(1); clf; imagesc(w);  gg=gray; colormap(gg(end:-1:1,:)); colorbar;
set(gca,'box','off'); xlabel('Actions'); ylabel('States');
title('Markov transition matrix: motor actions to sensory states');

% initialize variable quantifying model simulation
contingenciesH=zeros(nbirds,nR); % contingencies for Hearing birds
contingenciesD=contingenciesH; % same for Deaf birds
VsH=zeros(nR,na*nn); % Value function of States (mean Q values) for Hearing birds
VsD=VsH; % same for Deaf birds
VsallH=zeros(nbirds,nR,na*nn); % mean Value over all actions for Hearing birds
VsallD=VsallH; % same for Deaf
VTAsallplusH=zeros(nn,nbirds); VTAsallminusH=VTAsallplusH;
VTAsallplusD=VTAsallplusH; VTAsallminusD=VTAsallplusH;

aHold=ones(1,3); aDold=aHold; VTAHs=aHold; VTADs=aHold;
for c=1:do_control+1
    if c==1
        Non=1; % start LO on the first trial (contingent on sensory state)
    elseif c==2
        Non=N;  % never deliver LO
        fprintf('Control (no LO)\n')
    else
        Non=1;
        fprintf('Control (LO but no entropy gain)\n')
    end
    
    for b=1:nbirds % loop through Birds, 0=init bird
        VTAjH=zeros(nn,N); % VTA responses for different notes for specific reward Jplot
        VTAjD=VTAjH; % same for Hearing birds
        LOsH=zeros(1,N); LOsD=LOsH; % keeps track of LO (=1) or no LO (=0)

        fprintf('Bird %d\n',b)
        for r=1:nR % loop through rewards per LO
            R=rs(r); % reward per LO
            if verbose
                fprintf('j=%d/%d\n',r,nR);
            end
            
            % Initialize:
            % Hearing
            %V.PH=poli';%
            V.PH=1/na*ones(1,na*nn);   V.PHopt=V.PH;
            V.VH=zeros(1,na*nn); % action-Value function (Sarsa learning)
            V.ASCH=ones(nsA,na*nn); % Action-State Counter hearing (number of times action taken and state hit)
            V.ASCHall=V.ASCH;
            V.nLOHall=zeros(nsA,na*nn); % number of LO events
            
            % Deaf
            V.PD=1/na*ones(1,na*nn); V.PDopt=V.PD;
            V.VD=zeros(1,na*nn);
            V.ASCD=ones(nsD,na*nn); V.ASCDall=V.ASCD;
            V.nLODall=zeros(nsD,na*nn);
            
            % Hearing
            s_allH=zeros(1,N*nn); % state history
            a_allH=s_allH; % action history
            % Deaf
            s_allD=s_allH; a_allD=s_allH;
            
            poliH=1/nn*na*ones(1,nn*na);
            ci=0;
            a=zeros(1,N); bb=a; cc=a; dd=a;
            for i=1:N % loop over songs
                if i<10
                    alpha=0;
                else
                    alpha=alpha0;
                end
                for k=1:nn % loop over notes
                    ci=ci+1;
                    
                    % states and actions corresponding to current note:
                    ks=(1+(k-1)*nsH):k*nsH; ka=(1+(k-1)*na):k*na;
                    
                    
                    %% take action based on current action-value function

                    xH=V.PH(ka); %x=x-min(x); x=x/(eps+sum(x));
                    xH=max(0,xH); xH=xH/(eps+sum(xH));
                    aH=find(rand(1)<cumsum(xH),1,'first');
                   
                    aH=aH+(k-1)*na; a_allH(ci)=aH;
                    sH=find(rand(1)<transitionAS(:,aH),1,'first');   s_allH(ci)=sH; % markov transition
                    is_loH=(i>Non & k==kLO & sH>(kLO-1)*nsH+nsH/2); % simple, deterministic model
                    if b>0 && k==kLO && r==jplot
                        LOsH(i)=is_loH;
                    end
                    
                    % Deaf                  
                    xD=V.PD(ka);
                    xD=max(0,xD); xD=xD/(eps+sum(xD));                    
                    aD=find(rand(1)<cumsum(xD),1,'first');

                    aD=aD+(k-1)*na; a_allD(ci)=aD;
                    sD=find(rand(1)<transitionAS(:,aD),1,'first'); % state measured by experimenter
                    is_loD=(i>Non & k==kLO & sD>(kLO-1)*nsH+nsH/2); % simple, deterministic model
                    sD=2-is_loD;  s_allD(ci)=sD; % sensory state for deaf bird is binary
                    if b>0 && k==kLO  && r==jplot
                        LOsD(i)=is_loD;
                    end
                    
                    %% update action-state counter and estimated transition probability:
                    
                    V.ASCH_old=V.ASCH;
                    V.ASCD_old=V.ASCD;
                    
                    % update
                    %V.ASCH(sH,aH)=max(1,tau*V.ASCH(sH,aH)+1);
                    V.ASCH(sH,aH)=V.ASCH(sH,aH)+1;
                    V.ASCH=tau*V.ASCH;
                    V.ASCHall(sH,aH)=V.ASCHall(sH,aH)+1;
                    V.nLOHall(sH,aH)=V.nLOHall(sH,aH)+is_loH;
                    %V.ASCD(sD,aD)=max(1,tau*V.ASCD(sD,aD)+1);
                    V.ASCD(sD,aD)=V.ASCD(sD,aD)+1;
                    V.ASCD=tau*V.ASCD;
                    V.ASCDall(sD,aD)=V.ASCDall(sD,aD)+1;
                    V.nLODall(sD,aD)=V.nLODall(sD,aD)+is_loD;
                    
                    if (c==1 && i==Non) || (c==2 && i==1)
                        V.ASCHonall=V.ASCHall;  V.ASCDonall=V.ASCDall; V.nLODonall=V.nLODall; V.nLOHonall=V.nLOHall;
                    end
                    
                    %% Q learning: update value function
                    
                    % Hearing
                    if igC>0
                        igH=igC;
                    else
                        igH=calc_ig(V.ASCH,V.ASCH_old,aH);
                    end
                    if c<3
                        [xH,xHnew]=calc_pdf(V.PH,ka,aH,alpha);
                        [egH,DegH]=calc_emp2Diff(V.ASCH,xH/nn,xHnew/nn);
                        
                        [xHopt,xHnewopt]=calc_pdf(V.PHopt,ka,aH,alpha);
                        [egHopt,DegHopt]=calc_emp2Diff(V.ASCH,xHopt/nn,xHnewopt/nn);
                        V.PHopt(aH)=min(1,max(eps,V.PHopt(aH)+alpha*sign(DegHopt)));%*(sign(VTAHs(k))+gamma*V.VH(MaH)-0*V.VH(aHold(k)));
                        V.PHopt(ka)=V.PHopt(ka)/sum(V.PHopt(ka));
                        EntH=-sum(xHnew/2.*log(eps+xHnew/2));%+sum(xH.*log(eps+xH));
                    else
                        egH=0;
                    end
                    VTAHold=VTAHs(k);
                    V.VH(aH)=(1-alpha)*V.VH(aH)+alpha*is_loH*R;
                    %VTAH=0*log(igH+epsi)+DegH+is_loH*R-V.VH(aH); a(i)=egH;
                    VTAH=ent_sc*(EntH-log(na*nn))+DegH+sc_fac2*sum((xHnew-xH).*V.VH); a(i)=egH;
%                  VTAH=ent_sc*(EntH-log(na*nn))+DegH*rand(1)+sc_fac2*sum((xHnew-xH).*V.VH); a(i)=egH;
                 %   VTAH=DegH*(egH-egHopt>sc_fac*sum(xHnew.*V.VH))+sc_fac2*sum((xHnew-xH).*V.VH); a(i)=egH;
              %     VTAH=sign(DegH)*DegH^2*rand(1)+sc_fac2*sum((xHnew-xH).*V.VH)*rand(1); a(i)=egH;
               % VTAH=offset+DegH+sc_fac2*(R*is_loH-V.VH(aH)); a(i)=egH;
                    %  V.VH(aHold(k))=V.VH(aHold(k))+alpha*(xH(aHold(k))*VTAHs(k)+gamma*V.VH(MaH)-V.VH(aHold(k)));
                         V.PH(aH)=min(1,max(eps,V.PH(aH)+alpha*(0*15*(1/na-V.PH(aH))+sign(VTAH))));%*(sign(VTAHs(k))+gamma*V.VH(MaH)-0*V.VH(aHold(k)));
 %                 V.PH(aHold(k))=max(0,V.PH(aHold(k))+alpha*sign(VTAH));%*(sign(VTAHs(k))+gamma*V.VH(MaH)-0*V.VH(aHold(k)));
                     V.PH(ka)=V.PH(ka)/sum(V.PH(ka));
                     
                     VTAHs(k)=VTAH;
                    aHold(k)=aH;
                    %V.VH=poli';                   
                    
                    % Deaf
                    if igC>0
                        igD=igC;
                    else
                        igD=calc_ig(V.ASCD,V.ASCD_old,k);
                    end
                    if c<3
                        [xD,xDnew]=calc_pdf(V.PD,ka,aD,alpha);
                        [egD,DegD]=calc_emp2Diff(V.ASCD,xD/nn,xDnew/nn);
                        
                        [xDopt,xDnewopt]=calc_pdf(V.PDopt,ka,aD,alpha);
                        [egDopt,DegDopt]=calc_emp2Diff(V.ASCD,xDopt/nn,xDnewopt/nn);
                        V.PDopt(aD)=min(1,max(eps,V.PDopt(aD)+alpha*sign(DegDopt)));%*(sign(VTAHs(k))+gamma*V.VH(MaH)-0*V.VH(aHold(k)));
                        V.PDopt(ka)=V.PDopt(ka)/sum(V.PDopt(ka));
                        EntD=-sum(xDnew/2.*log(eps+xDnew/2));%+sum(xH.*log(eps+xH));
                    else
                        egD=0;
                    end
                    % if egD>0
                    %                   V.VD(aDold(k))=V.VD(aDold(k))+alpha*(VTADs(k)+gamma*V.VD(MaD)-V.VD(aDold(k)));
                    % V.PD(aDold(k))=V.PD(aDold(k))+alpha;
                    V.VD(aD)=(1-alpha)*V.VD(aD)+alpha*is_loD*R;
                    %  VTAD=0*log(igD+epsi)+DegD+is_loD*R-V.VD(aD);
                    %VTAD=DegD*(egD-egDopt>sc_fac*mean(V.VD))+200*sum((xDnew-xD).*V.VD); 
                    
                 %  VTAD=DegD*(egD-egDopt>sc_fac*sum(xDnew.*V.VD))+sc_fac2*sum((xDnew-xD).*V.VD);
%                     VTAD=ent_sc*(EntD-log(na*nn))+DegD*rand(1)+sc_fac2*sum((xDnew-xD).*V.VD); 
                     VTAD=ent_sc*(EntD-log(na*nn))+DegD+sc_fac2*sum((xDnew-xD).*V.VD); 
                 V.PD(aD)=min(1,max(0,V.PD(aD)+alpha*(0*15*(1/na-V.PD(aD))+sign(VTAD))));
                   
                   %  VTAD=offset+DegD+sc_fac2*(R*is_loD-V.VD(aD));
                   %             VTAD=sign(DegD)*DegD^2*rand(1)+sc_fac2*sum((xDnew-xD).*V.VD)*rand(1);
                   
                   bb(i)=egD-egDopt; cc(i)=mean(V.VD); dd(i)=VTAD;
                   
                   %VTAD=DegD+200*(xDnew(aD)-xD(aD)).*is_loD*R;
                   %    V.PD(aD)=max(0,V.PD(aD)+alpha*((randn(1)*alpha<(abs(VTAD))))*sign(VTAD));%*(sign(VTAHs(k))+gamma*V.VH(MaH)-0*V.VH(aHold(k)));
                   
%                    if DegD*sum((xDnew-xD).*V.VD)>0
%                        V.PD(aD)=min(1,max(0,V.PD(aD)+alpha*sign(DegD)));%*(sign(VTAHs(k))+gamma*V.VH(MaH)-0*V.VH(aHold(k)));
%                    elseif (DegD>0 && egD-egDopt<sc_fac2*sum(xDnew.*V.VD)) || ( DegD<0 && egD-egDopt>sc_fac2*sum(xDnew.*V.VD)) % far or close
%                        V.PD(aD)=min(1,max(0,V.PD(aD)+alpha));
%                     else
%                         V.PD(aD)=min(1,max(0,V.PD(aD)-alpha));
%                     end
                    
                    %end
                    %V.PD(setdiff(ka,aD))=min(1,max(0,V.PD(setdiff(ka,aD))-alpha*sign(VTAD)/(length(ka)-1)));
                      V.PD(ka)=V.PD(ka)/sum(V.PD(ka));
                     VTADs(k)=VTAD;
                    aDold(k)=aD;
                    
                    %                     if i>.8*N
                    %                         keyboard
                    %                     end
                    if r==jplot
                        VTAjH(k,i)=VTAH-V.PH(aH);
                        VTAjD(k,i)=VTAD-V.PD(aD);
                    end
                    
                end
            end
            VsH(r,:)=V.PH;
            VsD(r,:)=V.PD;
            
            contingenciesH(b,r)=sum(sum(V.nLOHall-V.nLOHonall))/(N-1);
            contingenciesD(b,r)=sum(sum(V.nLODall-V.nLODonall))/(N-1);
            
            if verbose
                fprintf('contingency: H=%.2g  D=%.2g\n', contingenciesH(b,r),contingenciesD(b,r));
                
                % plot action-state counts
                figure(11);clf; imagesc(V.ASCHall-V.ASCHonall); colormap(hot); colorbar;
                xlabel('actions'); ylabel('states'); title('Hearing action-state count');
                figure(12);clf; imagesc(V.ASCDall-V.ASCDonall); colormap(hot); colorbar;
                xlabel('actions'); ylabel('states'); title('Deaf action-state count');
                
                % plot distribution of states and actions visited XXX why Non=N for control birds?
                figure(15); clf; subplot(211); hist(s_allH(Non+1:end),1:nsA);   title('States hearing');
                subplot(212); hist(s_allD(Non+1:end),1:nsD,'r'); title('States deaf');
                figure(16); clf; subplot(211); hist(a_allH(Non+1:end),1:na*nn); title('Actions hearing');
                subplot(212); hist(a_allD(Non+1:end),1:na*nn,'r');  title('Actions deaf');
                
                % plot value function
                figure(21);clf; plot(V.PH,'k'); hold on;plot(V.PD,'r'); title('Q function');
                xlabel('action'); legend({'Hearing','Deaf'},'Location','best');
            end
        end
        
        VsallH(b,:,:)=VsH; VsallD(b,:,:)=VsD;
        if c==1
            VTAjH=VTAjH(:,N/2:N); LOsH=LOsH(N/2:N);
            VTAsallplusH(:,b)=mean(VTAjH(:,find(LOsH)),2);
            VTAsallminusH(:,b)=mean(VTAjH(:,find(~LOsH)),2);
            VTAsallplusD(:,b)=mean(VTAjD(:,find(LOsD)),2);
            VTAsallminusD(:,b)=mean(VTAjD(:,find(~LOsD)),2);
        end
    end
    
    if c==1
        figure(32);
        clf; semilogx(rs,mean(mean(VsallH,3),1),'k','linewidth',2); hold on;
        semilogx(rs,mean(mean(VsallD,3),1),'r','linewidth',2);
        set(gca,'box','off');axis tight
        legend({'Hearing','Deaf'},'Location','best');
        xlabel('r'); ylabel('mean Q'); title('Action averaged Q');
        
        figure(31); clf; semilogx(rs,mean(contingenciesH,1),'k','linewidth',2); hold on;
        semilogx(rs,mean(contingenciesD,1),'r','linewidth',2);
        semilogx(rs,.5*ones(1,nR),'k--','linewidth',2);
        set(gca,'box','off');
        legend({'Hearing','Deaf'},'Location','best');
        xlabel('Reward'); ylabel('LO contingency');
        axis tight
        
        if nn>3
        figure(33); clf;
        fplus=mean(VTAsallplusH,2); fminus=mean(VTAsallminusH,2);
        splus=std(VTAsallplusH')'; sminus=std(VTAsallminusH')';
        errorbar(1:nn,fplus,splus,'k','linewidth',2); hold on;
        errorbar(1:nn,fminus,sminus,'r--','linewidth',2);
        title(['R=' sprintf('%.2g',rs(jplot))]);
        legend({'LO','no LO'},'Location','Best');
        set(gca,'box','off')
        end
        
    elseif c==2
        figure(32);
        semilogx(rs,mean(mean(VsallH,3),1),'k--','linewidth',2);
        hold on; semilogx(rs,mean(mean(VsallD,3),1),'r--','linewidth',2);
        axis tight
        
    else
        figure(32);
        semilogx(rs,mean(mean(VsallH,3),1),'k:','linewidth',2);
        hold on; semilogx(rs,mean(mean(VsallD,3),1),'r:','linewidth',2);
        axis tight
        
        figure(31); semilogx(rs,mean(contingenciesH,1),'k:','linewidth',2); hold on;
        semilogx(rs,mean(contingenciesD,1),'r:','linewidth',2);
        axis tight
    end
    
end

% figure(31); print cont -dsvg; figure(32); print Q -dsvg; figure(33); print vta -dsvg;
figure(11); clf; plot(V.PH); title('Hearing');
figure(12);clf;plot(V.PD); title('Deaf');
 figure(13);plot(a); title('Empowerment');% ylim([.9 1.1])
%% DEFINE FUNCTIONS

function ig=calc_ig(ASC_new,ASC_old,a0)
% CALC_IG calculates information gain given for action a0

%   'ASC_new'   -   new action-state counter after taking action a0
%   'ASC_old'   -   old action-state counter
%   'a0'        -   current new action

% estimate of transition probability for the current note
T_old=ASC_old(:,a0)/(eps+sum(ASC_old(:,a0))); % old estimate
T_new=ASC_new(:,a0)/(eps+sum(ASC_new(:,a0))); % new estimate
dkl=sum(T_old.*log(T_old./T_new));
ig=dkl*1000; % for entropy
%ig=dkl;  % for impact
end

function emdist=calc_emd(ASC_new,ASC_old,a0)
% CALC_emd calculates earth mover's distance gain given new action
calc_emp2Diff
%   'ASC_new'   -   new action-state counter after taking action
%   'ASC_old'   -   old action-state counter
T_old=ASC_old(:,a0)/(eps+sum(ASC_old(:,a0))); % old estimate
T_new=ASC_new(:,a0)/(eps+sum(ASC_new(:,a0))); % new estimate
T0=zeros(size(T_old)); T0(end)=1;

% L1 dist
emdist=sum(abs(T_new-T0))-0*sum(abs(T_old-T0));
emdist=2*(emdist-2);

% true EMD
% enew=cumsum(T_new-T0);
% eold=cumsum(T_old-T0);
% emdist=sum(abs(enew))-sum(abs(eold));
% emdist=2000*emdist/length(T_new)^2;
end

function eg=calc_eg(ASC_new,ASC_old,a0)
% CALC_EG calculates entropy gain given new action

%   'ASC_new'   -   new action-state counter after taking action
%   'ASC_old'   -   old action-state counter

C=sum(ASC_old,2); % state count
P_old=C/sum(C); % state probability

%ASC(s0,a0)=tau*ASC(s0,a0)+1; % updated action-state counter
C=sum(ASC_new,2);
P_new=C/sum(C);

eg=-sum(P_new.*log(P_new))+sum(P_old.*log(P_old));

if 1
    C=sum(ASC_new,2);
    D=sum(ASC_new,1); % action count
    Ps_new=C/(eps+sum(C)); % state probability
    Pa_new=D/(eps+sum(C));
    
    C=sum(ASC_old,2);
    D=sum(ASC_old,1); % action count
    Ps_old=C/(eps+sum(C)); % state probability
    Pa_old=D/(eps+sum(C));
    
    empg=0;
    
    for a0=1:length(D)
        Tnew=ASC_new(:,a0)/(eps+sum(ASC_new(:,a0))); % new estimate
        Told=ASC_old(:,a0)/(eps+sum(ASC_old(:,a0))); % new estimate
        
        %empg=sum(Tnew*Pa_new.*log(Tnew./Ps_new))-sum(Told*Pa_old.*log(Told./Ps_old));
        %empg=empg+Pa_new(a0)*sum(Tnew*Pa_new(a0).*log(Tnew./Ps_new))-Pa_old(a0)*sum(Told*Pa_old(a0).*log(Told./Ps_old));
        empg=empg+sum(Tnew*Pa_new(a0).*log(Tnew./Ps_new));%-sum(Told*Pa_old(a0).*log(Told./Ps_old));
    end
    eg=empg;
end
% Tnew=ASC_new(:,a0)/(eps+sum(ASC_new(:,a0))); % new estimate
% Told=ASC_old(:,a0)/(eps+sum(ASC_old(:,a0))); % old estimate
%
% eg=eg+sum(Tnew.*log(Tnew))-sum(Told.*log(Told));


eg=3000*eg;
%eg=-10*exp(-eg);
end




