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

% include_positive_rewards = fale, only use negative rewards
include_positive_rewards = false;

% Model parameters:
% (capitalized letters in the variable description are used to indicate
% nameing of variables: 'N'umber of 'N'otes = nn)
nn=3; % Number of Notes the bird is singing
kLO=2; % index of note that can trigger LO
na=6; % Number of Actions per note (should be even)
alpha=0.05; % learning rate TD learning
tau=0.99;  % forgetting time constant
verbose=0;
gamma=1; % Gamma parameter in SARSA learning

epsi=1e-40;

% Simulation parameters:
N=1000; % # trials (per note)
do_control=2; % add control condition without LO (0 = only LO, 1 = add control, 2 = remove entropy gain)
nbirds=3; % Number of Birds to simulate and average the result over

% Reward for LO
nR=15; % Number of different Reward values tested (reward value fixed for one  simulation)
if include_positive_rewards
    rs=[-logspace(log10(0.1),log10(50),nR) logspace(log10(0.1),log10(5),nR)]; % reward per LO event (or light on)
    nR=length(rs);
    rs=sort(rs);
    plot_f = @plot;
else
    rs=[-logspace(log10(0.1),log10(50),nR)]; % reward per LO event
    plot_f = @semilogx;
end

jplot=7; % plot VTA firing for reward # jplot
igC=0*1e-6;%1e-40;%1e-14; % for constant information gain, set to nonzero for getting rid of exploration bonus

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
aHold=zeros(3,1); aDold=aHold;

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
            V.VH=zeros(1,na*nn); % action-Value function (Sarsa learning)
            V.ASCH=ones(nsA,na*nn); % Action-State Counter hearing (number of times action taken and state hit)
            V.ASCHall=V.ASCH;
            V.nLOHall=zeros(nsA,na*nn); % number of LO events
            
            % Deaf
            V.VD=zeros(1,na*nn);
            V.ASCD=ones(nsD,na*nn); V.ASCDall=V.ASCD;
            V.nLODall=zeros(nsD,na*nn);
            
            % Hearing
            s_allH=zeros(1,N*nn); % state history
            a_allH=s_allH; % action history
            % Deaf
            s_allD=s_allH; a_allD=s_allH;
            
            ci=0;
            for i=1:N % loop over songs
                
                for k=1:nn % loop over notes
                    ci=ci+1;
                    
                    % states and actions corresponding to current note:
                    ks=(1+(k-1)*nsH):k*nsH; ka=(1+(k-1)*na):k*na;
                    
                    
                    %% take action based on current action-value function
                    
                    % Hearing
                    [temp,aH]=max(V.VH(ka)-0*rand(1,na)); % choose action | ~ -> ignores the first output
                    fprintf("VH, temp, aH %d %d %d\n",V.VH(ka),temp,aH)% Adith 21 July
                    disp(
                    aH=aH+(k-1)*na;           
                    a_allH(ci)=aH;
                    sH=find(rand(1)<transitionAS(:,aH),1,'first');   s_allH(ci)=sH; % markov transition
                    is_loH=(i>Non & k==kLO & sH>(kLO-1)*nsH+nsH/2); % simple, deterministic model
                    if b>0 && k==kLO && r==jplot
                        LOsH(i)=is_loH;
                    end
                    
                    % Deaf
                    
                    aDold(k)=aD; % Adith 21 July: aD is not declared before
                    [~,aD]=max(V.VD(ka)-0*rand(1,na));
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
                    V.ASCHall(sH,aH)=V.ASCHall(sH,aH)+1;
                    V.nLOHall(sH,aH)=V.nLOHall(sH,aH)+is_loH;
                    V.ASCD(sD,aD)=max(1,tau*V.ASCD(sD,aD)+1);
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
                        %         egH=calc_eg(V.ASCH,V.ASCH_old);
                       % egH=calc_impact(V.ASCH,V.ASCH_old,aH,a_allH(1:ci));
                         egH=calc_impact(V.ASCH,V.ASCH_old,aH);
                    else
                        egH=0;
                    end
                        VTAH=log(igH+epsi)+egH+is_loH*R-V.VH(aH);
                    %h=igH+egH+is_loH*R;
                    %VTAH=-exp(-h)-V.VH(aH);
                    V.VH(aH)=V.VH(aH)+alpha*(VTAH);
                    
                    % Deaf
                    if igC>0
                        igD=igC;
                    else
                        igD=calc_ig(V.ASCD,V.ASCD_old,aD);
                    end
                    if c<3
                        %    egD=calc_eg(V.ASCD,V.ASCD_old);
                        % egD=calc_impact(V.ASCD,V.ASCD_old,aD,a_allD(1:ci));
                        egD=calc_impact(V.ASCD,V.ASCD_old,aD);
                    else
                        egD=0;
                    end
                          VTAD=log(igD+epsi)+egD+is_loD*R-V.VD(aD);
                    %h=igD+egD+is_loD*R;
                    %VTAD=-exp(-h)-V.VD(aD);
                    V.VD(aD)=V.VD(aD)+alpha*VTAD;
                    %                     if i>.8*N
                    %                         keyboard
                    %                     end
                    if r==jplot
                        VTAjH(k,i)=VTAH;
                        VTAjD(k,i)=VTAD;
                    end
                    
                end
            end
            VsH(r,:)=V.VH;
            VsD(r,:)=V.VD;
            
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
                figure(21);clf; plot(V.VH,'k'); hold on;plot(V.VD,'r'); title('Q function');
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
        clf; 
        plot_f(rs,mean(mean(VsallH,3),1),'k','linewidth',2); hold on;
        plot_f(rs,mean(mean(VsallD,3),1),'r','linewidth',2);
        set(gca,'box','off');axis tight
        legend({'Hearing','Deaf'},'Location','best');
        xlabel('r'); ylabel('mean Q'); title('Action averaged Q');
        
        figure(31); clf; 
        plot_f(rs,mean(contingenciesH,1),'k','linewidth',2); hold on;
        plot_f(rs,mean(contingenciesD,1),'r','linewidth',2);
        plot_f(rs,.5*ones(1,nR),'k--','linewidth',2);
        set(gca,'box','off');
        legend({'Hearing','Deaf'},'Location','best');
        xlabel('Reward'); ylabel('LO contingency');
        axis tight
        
        figure(33); clf;
        fplus=mean(VTAsallplusH,2); fminus=mean(VTAsallminusH,2);
        splus=std(VTAsallplusH')'; sminus=std(VTAsallminusH')';
        errorbar(1:nn,fplus,splus,'k','linewidth',2); hold on;
        errorbar(1:nn,fminus,sminus,'r--','linewidth',2);
        title(['R=' sprintf('%.2g',rs(jplot))]);
        legend({'LO','no LO'},'Location','Best');
        set(gca,'box','off')
        
    elseif c==2
        figure(32);
        plot_f(rs,mean(mean(VsallH,3),1),'k--','linewidth',2);
        hold on; 
        plot_f(rs,mean(mean(VsallD,3),1),'r--','linewidth',2);
        axis tight
        
    else
        figure(32);
        plot_f(rs,mean(mean(VsallH,3),1),'k:','linewidth',2);hold on; 
        plot_f(rs,mean(mean(VsallD,3),1),'r:','linewidth',2);
        axis tight
        
        figure(31); 
        plot_f(rs,mean(contingenciesH,1),'k:','linewidth',2); hold on;
        plot_f(rs,mean(contingenciesD,1),'r:','linewidth',2);
        axis tight
    end
    
end

% figure(31); print cont -dsvg; figure(32); print Q -dsvg; figure(33); print vta -dsvg;

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
%ig=dkl*1000; % for entropy
ig=dkl;  % for impact
end

function emdist=calc_impact(ASC_new,ASC_old,a0)
% CALC_impact calculates the impact gain

%   'ASC_new'   -   new action-state counter after taking action
%   'ASC_old'   -   old action-state counter

%T_old=ASC_old(:,a0)/(eps+sum(ASC_old(:,a0))); % old estimate
T_new=ASC_new(:,a0)/(eps+sum(ASC_new(:,a0))); % new estimate
T0=zeros(size(T_new)); T0(end)=1;

if 1
    % DKL
    emdist=sum(T0.*log(eps+T0./(eps+T_new)));
    
else
    % L1 dist
    emdist=sum(abs(T_new-T0));%-0*sum(abs(T_old-T0));
    emdist=2*(emdist-2);
end

% true EMD earth mover's distance
% enew=cumsum(T_new-T0);
% eold=cumsum(T_old-T0);
% emdist=sum(abs(enew))-sum(abs(eold));
% emdist=2000*emdist/length(T_new)^2;
end

function empg=calc_empg(ASC_new,ASC_old,a0,a_all)
% CALC_EMPG calculates empowerment gain given new action
%   'ASC_new'   -   new action-state counter after taking action
%   'ASC_old'   -   old action-state counter
% if length(a_all)==1
%     a_all=[1 a_all];
% end
Told=ASC_old(:,a0)/(eps+sum(ASC_old(:,a0))); % old estimate
C=sum(ASC_old,2); % state count
D=sum(ASC_old,1); % action count
Ps_old=C/sum(C); % state probability
%Pa_old=sum(find(a_all(1:end-1)==a0))/(length(a_all)-1+eps);
Pa_old=D/sum(C);

Tnew=ASC_new(:,a0)/(eps+sum(ASC_new(:,a0))); % new estimate
C=sum(ASC_new,2);
D=sum(ASC_new,1); % action count
Ps_new=C/sum(C); % state probability
%Pa_new=sum(find(a_all==a0))/(length(a_all)+eps);
Pa_new=D/sum(C);

%empg=sum(Tnew*Pa_new.*log(Tnew./Ps_new))-sum(Told*Pa_old.*log(Told./Ps_old));
empg=sum(Tnew*Pa_new(a0).*log(Tnew./Ps_new))-sum(Told*Pa_old(a0).*log(Told./Ps_old));
empg=30000*empg;
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


