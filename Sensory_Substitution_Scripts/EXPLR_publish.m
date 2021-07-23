%%% EXPLR_publish %%%
% Model of Exploration in Hearing and Deaf Light-Off (LO) Birds
% Implementation of the model from the supplementary material of the
% manuscript:
% "Deafening causes valence reversal of visual reinforcers of birdsong"
% AT Zai, S Cave-Lopez, M Rolland, N Giret, RHR Hahnloser

% Author: Richard Hahnloser and Anja Zai 2019
% Work address: Institute of Neuroinformatics, Winterthurerstrasse 8057,
% ETH Zurich and University of Zurich
% email: rich@ini.ethz.ch

% Model parameters:
% (capitalized letters in the variable description are used to indicate
% nameing of variables: 'N'umber of 'N'otes = nn)
nn=3; % Number of Notes the bird is singing
kLO=2; % index of note that can trigger LO
na=6; % Number of Actions per note (should be even)
alpha=0.05; % learning rate TD learning
tau=0.99;  % forgetting time constant
verbose=0;

% Simulation parameters:
N=600; % # trials (per note)
do_control=2; % add control condition without LO (0 = only LO, 1 = add control, 2 = remove entropy gain)
nbirds=10; % Number of Birds to simulate and average the result over

% Reward for LO
nR=15; % Number of different Reward values tested (reward value fixed for one  simulation)
rs=-logspace(log10(0.1),log10(100),nR); % reward per LO event
jplot=7; % plot VTA firing for reward # jplot
igC=1e-6; % for constant information gain (obsolete)

% Number of States
nsH=na+2; % Number of States for Hearing bird (per note) corresponding to actual pitch sung
nsD=2; % Number of States for Deaf bird (per note)
nsA=nsH*nn; % total Number of States (for All notes)

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
VTAjH=zeros(nn,N); % VTA responses for different notes for specific reward Jplot
VTAjD=VTAjH; % same for Hearing birds
VTAsallplusH=zeros(nn,nbirds); VTAsallminusH=VTAsallplusH;
VTAsallplusD=VTAsallplusH; VTAsallminusD=VTAsallplusH;
LOsH=zeros(1,N); LOsD=LOsH; % keeps track of LO (=1) or no LO (=0)

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
    
    for b=1:nbirds % loop through Birds
        fprintf('Bird %d\n',b)
        for r=1:nR % loop through rewards per LO
            R=rs(r); % reward per LO
            fprintf('j=%d/%d\n',r,nR);
            
            % Initialize:
            % Hearing
            VH=zeros(1,na*nn); % action-Value function (Sarsa learning)
            ASCH=ones(nsA,na*nn); % Action-State Counter hearing (number of times action taken and state hit)
            ASCHall=ASCH;
            nLOHall=zeros(nsA,na*nn); % number of LO events
            s_allH=zeros(1,N*nn); % state history
            a_allH=s_allH; % action history
            % Deaf
            VD=zeros(1,na*nn);
            ASCD=ones(nsD,na*nn); ASCDall=ASCD;
            nLODall=zeros(nsD,na*nn);
            s_allD=s_allH; a_allD=s_allH;
            
            ci=0;
            for i=1:N % loop over songs
                for k=1:nn % loop over notes
                    ci=ci+1;
                    
                    % states and actions corresponding to current note:
                    ks=(1+(k-1)*nsH):k*nsH; ka=(1+(k-1)*na):k*na; 
                    
                    
                    %% take action based on current action-value function
                    
                    % Hearing
                    [~,aH]=max(VH(ka)-0*rand(1,na)); % choose action
                    aH=aH+(k-1)*na;           a_allH(ci)=aH;
                    sH=find(rand(1)<transitionAS(:,aH),1,'first');   s_allH(ci)=sH; % markov transition
                    is_loH=(i>Non & k==kLO & sH>(kLO-1)*nsH+nsH/2); % simple, deterministic model
                    if k==2 && r==jplot
                        LOsH(i)=is_loH;
                    end
                    
                    % Deaf
                    [~,aD]=max(VD(ka)-0*rand(1,na));
                    aD=aD+(k-1)*na; a_allD(ci)=aD;
                    sD=find(rand(1)<transitionAS(:,aD),1,'first'); % state measured by experimenter
                    is_loD=(i>Non & k==kLO & sD>(kLO-1)*nsH+nsH/2); % simple, deterministic model
                    sD=is_loD+1;  s_allD(ci)=sD; % sensory state for deaf bird is binary
                    if k==2  && r==jplot
                        LOsD(i)=is_loD;
                    end
                    
                    %% update action-state counter and estimated transition probability:
                     
                    ASCH_old=ASCH;
                    ASCD_old=ASCD;
                    
                    % update
                    ASCH(sH,aH)=max(1,tau*ASCH(sH,aH)+1);
                    ASCHall(sH,aH)=ASCHall(sH,aH)+1;
                    nLOHall(sH,aH)=nLOHall(sH,aH)+is_loH;
                    ASCD(sD,aD)=max(1,tau*ASCD(sD,aD)+1);
                    ASCDall(sD,aD)=ASCDall(sD,aD)+1;
                    nLODall(sD,aD)=nLODall(sD,aD)+is_loD;
                    
                    if (c==1 && i==Non) || (c==2 && i==1)
                        ASCHonall=ASCHall;  ASCDonall=ASCDall; nLODonall=nLODall; nLOHonall=nLOHall;
                    end
                    
                    %% Q learning: update value function
                    
                    % Hearing
                    %ig=igC;
                    igH=calc_ig(ASCH,ASCH_old,aH);
                    if c<3
                        egH=calc_eg(ASCH,ASCH_old);
                    else
                        egH=0;
                    end
                    VTAH=log(igH+eps)+egH+is_loH*R-VH(aH);
                    VH(aH)=VH(aH)+alpha*(VTAH);
                    
                    % Deaf
                    %igD=igC;
                    igD=calc_ig(ASCD,ASCD_old,aD);
                    if c<3
                        egD=calc_eg(ASCD,ASCD_old); 
                    else
                        egD=0;
                    end
                    VTAD=log(igD+eps)+egD+is_loD*R-VD(aD);
                    VD(aD)=VD(aD)+alpha*VTAD;
                    
                    if r==jplot
                        VTAjH(k,i)=VTAH;
                        VTAjD(k,i)=VTAD;
                    end

                end
            end
            VsH(r,:)=VH;
            VsD(r,:)=VD;
            
            contingenciesH(b,r)=sum(sum(nLOHall-nLOHonall))/(N-1);
            contingenciesD(b,r)=sum(sum(nLODall-nLODonall))/(N-1);
            fprintf('contingency: H=%.2g  D=%.2g\n', contingenciesH(b,r),contingenciesD(b,r));
            
            if verbose
                % plot action-state counts
            figure(11);clf; imagesc(ASCHall-ASCHonall); colormap(hot); colorbar;
            xlabel('actions'); ylabel('states'); title('Hearing action-state count');
            figure(12);clf; imagesc(ASCDall-ASCDonall); colormap(hot); colorbar;
            xlabel('actions'); ylabel('states'); title('Deaf action-state count');
            
            % plot distribution of states and actions visited XXX why Non=N for control birds?
            figure(15); clf; subplot(211); hist(s_allH(Non+1:end),1:nsA);   title('States hearing');
            subplot(212); hist(s_allD(Non+1:end),1:nsD,'r'); title('States deaf');
            figure(16); clf; subplot(211); hist(a_allH(Non+1:end),1:na*nn); title('Actions hearing');
            subplot(212); hist(a_allD(Non+1:end),1:na*nn,'r');  title('Actions deaf');
            
            % plot value function
            figure(21);clf; plot(VH,'k'); hold on;plot(VD,'r'); title('Q function');
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
        
        figure(33); clf;
        fplus=mean(VTAsallplusH,2); fminus=mean(VTAsallminusH,2);
        splus=std(VTAsallplusH')'; sminus=std(VTAsallminusH')';
        errorbar(1:nn,fplus,splus,'k','linewidth',2); hold on;
        errorbar(1:nn,fminus,sminus,'r','linewidth',2);
        title(['R=' sprintf('%.2g',rs(jplot))]);
        legend({'LO','no LO'},'Location','Best');
        set(gca,'box','off')
        
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

%% DEFINE FUNCTIONS

function ig=calc_ig(ASC_new,ASC_old,a0)
% CALC_IG calculates information gain given for action a0

%   'ASC_new'   -   new action-state counter after taking action a0
%   'ASC_old'   -   old action-state counter
%   'a0'        -   current new action

% estimate of transition probability for the current note
T_old=ASC_old./(eps+ones(size(ASC_old,1),1)*sum(ASC_old,1)); % old estimate
T_new=ASC_new./(eps+ones(size(ASC_old,1),1)*sum(ASC_new,1)); % new estimate

% KL-divergence
%ig=0;
dkl=0;
for k2=1:size(T_old,1) % state loop for dkl
    dkl=dkl+T_old(k2,a0).*log(T_old(k2,a0)/T_new(k2,a0));
end
%ig=ig+T(k0,l)*dkl;
ig=dkl;
ig=ig*1000;
end


function eg=calc_eg(ASC_new,ASC_old)
% CALC_EG calculates entropy gain given new action

%   'ASC_new'   -   new action-state counter after taking action
%   'ASC_old'   -   old action-state counter

C=sum(ASC_old,2); % state count
P_old=C/sum(C); % state probability

%ASC(s0,a0)=tau*ASC(s0,a0)+1; % updated action-state counter
C=sum(ASC_new,2);
P_new=C/sum(C);

eg=-sum(P_new.*log(P_new))+sum(P_old.*log(P_old));
eg=3000*eg;
end


