%%% EXPLR_publish %%%
% Model of Exploration in Hearing and Deaf Light-Off (LO) Birds
% Implementation of the model from the supplementary material of the
% manuscript:
% 
% Empowerment maximization
% note emp_opt yields the optimal policy (poli)

nn=3; % Number of Notes the bird is singing
kLO=2; % index of note that can trigger LO
na=6; % Number of Actions per note (should be even)
alpha=0.001/nn; % learning rate TD learning

tau=1;  % forgetting time constant, is tau=1 required for convergence to optimal policy?
verbose=0;

sc_fac2=50; % tradeoff between reward r and empowerment gain

ent_sc=0.01; % scaling of entropy term (making action choices uniform)


% Simulation parameters:
N=2500; % # trials (per note)
do_control=0; % add control condition without LO (0 = only LO, 1 = add control, 2 = remove entropy gain)
nbirds=2; % Number of Birds to simulate and average the result over

% Reward for LO
nR=12; % Number of different Reward values tested (reward value fixed for one  simulation)
rs=-logspace(log10(0.0001),log10(1000),nR); % reward per LO event
rs=rs(end:-1:1);

jplot=15; % plot VTA firing for reward # jplot

% Number of States
nsH=na+2; % Number of States for Hearing bird (per note) corresponding to actual pitch sung
nsD=2; % Number of States for Deaf bird (per note), the second state is the silent state
nsA=nsH*nn+1; % total Number of States (for All notes), the last stte is the silent state

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


VTAHs=ones(1,3); VTADs=VTAHs;
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
            V.PH=1/na*ones(1,na*nn); % Policy
%            V.PHopt=V.PH; % optimal policy (only empowerment, no punishment)
            V.VH=zeros(1,na*nn); % average punishment (per action)
            V.ASCH=ones(nsA,na*nn); % Action-State Counter hearing (number of times action taken and state hit)
            V.ASCHall=V.ASCH;
            V.nLOHall=zeros(nsA,na*nn); % number of LO events
            EntHold=log(na*nn);
            
            % Deaf
            V.PD=1/na*ones(1,na*nn); %V.PDopt=V.PD;
            V.VD=zeros(1,na*nn);
            V.ASCD=ones(nsD,na*nn); V.ASCDall=V.ASCD;
            V.nLODall=zeros(nsD,na*nn);
            EntDold=log(na*nn);
            
            % Hearing
            s_allH=zeros(1,N*nn); % state history
            a_allH=s_allH; % action history
            % Deaf
            s_allD=s_allH; a_allD=s_allH;
            
            poliH=1/nn*na*ones(1,nn*na);
            ci=0;
            a=zeros(1,N); bb=a; cc=a; dd=a;
            for i=1:N % loop over songs
                  for k=1:nn % loop over notes
                    ci=ci+1;
                    
                    % states and actions corresponding to current note:
                    ks=(1+(k-1)*nsH):k*nsH; ka=(1+(k-1)*na):k*na;
                    
                    
                    %% take action based on current policy

                    xH=V.PH(ka);% current policy for note k, sum(xH)=1
                    aH=find(rand(1)<cumsum(xH),1,'first'); % sample next action from policy
                   
                    aH=aH+(k-1)*na; a_allH(ci)=aH; % find correponding state and LO value
                    sH=find(rand(1)<transitionAS(:,aH),1,'first');   s_allH(ci)=sH; % markov transition
                    is_loH=(i>Non & k==kLO & sH>(kLO-1)*nsH+nsH/2); % simple, deterministic model
                    if b>0 && k==kLO && r==jplot
                        LOsH(i)=is_loH;
                    end
                    
                    % Deaf                  
                    xD=V.PD(ka);           
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
                    V.ASCH(sH,aH)=max(1,tau*V.ASCH(sH,aH)+1);
                    %V.ASCH(sH,aH)=V.ASCH(sH,aH)+1;
                    %V.ASCH=tau*V.ASCH;
                    V.ASCHall(sH,aH)=V.ASCHall(sH,aH)+1;
                    V.nLOHall(sH,aH)=V.nLOHall(sH,aH)+is_loH;
                    
                    V.ASCD(sD,aD)=max(1,tau*V.ASCD(sD,aD)+1);
                   % V.ASCD(sD,aD)=V.ASCD(sD,aD)+1;
                    %V.ASCD=tau*V.ASCD;
                    V.ASCDall(sD,aD)=V.ASCDall(sD,aD)+1;
                    V.nLODall(sD,aD)=V.nLODall(sD,aD)+is_loD;
                    
                    if (c==1 && i==Non) || (c==2 && i==1)
                        V.ASCHonall=V.ASCHall;  V.ASCDonall=V.ASCDall; V.nLODonall=V.nLODall; V.nLOHonall=V.nLOHall;
                    end
                    
                    %% update policy based on empowerment gain
                    
                    if i==500
                       disp('') 
                    end
                    % Hearing
                    if c<3
                        xHnn=V.PH; % this is the current policy, but normalized to sum(xHnn)=nn
                        xHnnew=calc_pdf(V.PH,ka,aH,alpha); % compute new policy to test
                        [egH,DegH]=calc_emp2Diff(V.ASCH,xHnn/nn,xHnnew/nn); % DegH = empowerment change 
                        EntHold=-sum(xHnn/nn.*log(eps+xHnn/nn)); % entropy of old policy
                        EntH=-sum(xHnnew/nn.*log(eps+xHnnew/nn)); % entropy of new policy
                    else
                        egH=0;
                    end
                    V.VH(aH)=(1-alpha)*V.VH(aH)+alpha*is_loH*R; % running average punishment  
                    %VTAH1=ent_sc*(EntH-log(na*nn)); % deviation from maximally informative policy
                    VTAH1=ent_sc*(EntH-EntHold); % deviation from maximally informative policy
                    VTAH2=sc_fac2*sum((xHnnew-xHnn).*V.VH); % change in expected punishment
 
                    VTAH=DegH+VTAH1+VTAH2;              % total valence change including empowerment change
                    a(i)=egH;
    
                    V.PH(aH)=min(1,max(eps,V.PH(aH)+alpha*sign(VTAH))); % adopt policy change if valenence increase, otherwise take opposite policy change
                    V.PH(ka)=V.PH(ka)/sum(V.PH(ka)); % renormalize policy
                     
                     VTAHs(k)=VTAH;
                     
                    % Deaf
                    if c<3
                        xDnn=V.PD;
                        xDnnew=calc_pdf(V.PD,ka,aD,alpha);
                        [egD,DegD]=calc_emp2Diff(V.ASCD,xDnn/nn,xDnnew/nn);
                        EntDold=-sum(xDnn/nn.*log(eps+xDnn/nn));
                        EntD=-sum(xDnnew/nn.*log(eps+xDnnew/nn));%+sum(xH.*log(eps+xH));
                    else
                        egD=0;
                    end
                    V.VD(aD)=(1-alpha)*V.VD(aD)+alpha*is_loD*R;
                    %VTAD=ent_sc*(EntD-log(na*nn))+DegD+sc_fac2*sum((xDnnew-xDnn).*V.VD);
                    VTAD=ent_sc*(EntD-EntDold)+DegD+sc_fac2*sum((xDnnew-xDnn).*V.VD);
                    V.PD(aD)=min(1,max(0,V.PD(aD)+alpha*sign(VTAD)));
                    
                    
                     bb(i)=egD;
                  %  cc(i)=mean(V.VD); dd(i)=VTAD;
                    
                    V.PD(ka)=V.PD(ka)/sum(V.PD(ka));
                     VTADs(k)=VTAD;
                      
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
figure(11); clf; plot(V.PH); title('Policy Hearing'); 
figure(12);clf;plot(V.PD); title('Policy Deaf');
 figure(13); clf; plot(a,'k'); hold on; plot(bb,'r'); legend({'Hearing','Deaf'}); title('Empowerment');% ylim([.9 1.1])
 
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




