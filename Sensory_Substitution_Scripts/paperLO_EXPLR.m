%%% EXPLR_publish %%%
% Model of Exploration in Hearing and Deaf Light-Off (LO) Birds
% Implementation of the model from the supplementary material of the
% manuscript (please cite):
% Sensory substitution reveals a manipulation bias
% Anja T. Zai, Sophie Cave-Lopez, Manon Rolland, Nicolas Giret,
% Richard H.R. Hahnloser
% DOI: https://doi.org/10.1038/s41467-020-19686-w
%
% Copyright: (c) 2020 ETH Zurich, Anja T. Zai, Richard H.R. Hahnloser
%

% include_positive_rewards = fale, only use negative rewards
include_positive_rewards = false;

% Model parameters:
% (capitalized letters in the variable description are used to indicate
% nameing of variables: 'N'umber of 'N'otes = nn)
no_of_notes=3; % Number of Notes the bird is singing
lo_note=2; % index of note that can trigger LO
no_of_actions=6; % Number of Actions per note (should be even)
alpha=0.05; % learning rate TD learning
tau=0.99;  % forgetting time constant
verbose=0;
gamma=1; % Gamma parameter in SARSA learning

epsi=1e-40;

% Simulation parameters:
N=1000; % # trials (per note)
do_control=2; % add control condition without LO (0 = only LO, 1 = add control, 2 = remove entropy gain)
no_of_birds=3; % Number of Birds to simulate and average the result over

% Reward for LO
no_of_rewards=15; % Number of different Reward values tested (reward value fixed for one  simulation)
if include_positive_rewards
    reward_space=[-logspace(log10(0.1),log10(50),no_of_rewards) logspace(log10(0.1),log10(5),no_of_rewards)]; % reward per LO event (or light on)
    no_of_rewards=length(reward_space);
    reward_space=sort(reward_space);
    plot_f = @plot;
else
    reward_space=[-logspace(log10(0.1),log10(50),no_of_rewards)]; % reward per LO event
    plot_f = @semilogx;
end

jplot=7; % plot VTA firing for reward # jplot
igC=0*1e-6;%1e-40;%1e-14; % for constant information gain, set to nonzero for getting rid of exploration bonus

% Number of States
no_of_states_h=no_of_actions+2; % Number of States for Hearing bird (per note) corresponding to actual pitch sung
no_of_states_d=2; % Number of States for Deaf bird (per note), the second state is the silent state
total_no_of_states=no_of_states_h*no_of_notes+1; % total Number of States (for All notes), the last stte is the silent state

% Sensorimotor model: motor action -> sensory states
h1=zeros(no_of_states_h,1); h1(1:3)=[1/4 1/2 1/4]; h2=zeros(no_of_actions,1); h2(1)=1/4;
wh=toeplitz(h1,h2); w=zeros(total_no_of_states,no_of_notes*no_of_actions);
for i=1:no_of_notes
    w(1+(i-1)*no_of_states_h:i*no_of_states_h,1+(i-1)*no_of_actions:i*no_of_actions)=wh;
end
transitionAS=cumsum(w,1); % defines Transitions from Action to State (Fig. 4B)
figure(1); clf; imagesc(w);  gg=gray; colormap(gg(end:-1:1,:)); colorbar;
set(gca,'box','off'); xlabel('Actions'); ylabel('States');
title('Markov transition matrix: motor actions to sensory states');

% initialize variable quantifying model simulation
contingenciesH=zeros(no_of_birds,no_of_rewards); % contingencies for Hearing birds
contingenciesD=contingenciesH; % same for Deaf birds
value_states_h=zeros(no_of_rewards,no_of_actions*no_of_notes); % Value function of States (mean Q values) for Hearing birds
value_states_d=value_states_h; % same for Deaf birds
mean_value_states_h=zeros(no_of_birds,no_of_rewards,no_of_actions*no_of_notes); % mean Value over all actions for Hearing birds
mean_value_states_d=mean_value_states_h; % same for Deaf
vta_plus_h=zeros(no_of_notes,no_of_birds); vta_minus_h=vta_plus_h;
vta_plus_d=vta_plus_h; vta_minus_d = vta_plus_h;

for c=1:do_control+1
    if c==1
        Non=1; % start LO on the first trial (contingent on sensory state)
    elseif c==2
        Non=N;  % never deliver LO
        fprintf('Control (no LO)\n')
    else
        Non=1;
        fprintf('Control (LO but no impact)\n')
    end
    
    for b=1:no_of_birds % loop through Birds, 0=init bird
        vta_j_h=zeros(no_of_notes,N); % VTA responses for different notes for specific reward Jplot
        vta_j_d=vta_j_h; % same for Hearing birds
        lo_h=zeros(1,N); lo_d=lo_h; % keeps track of LO (=1) or no LO (=0)
        
        
        fprintf('Bird %d\n',b)
        for r=1:no_of_rewards % loop through rewards per LO
            R=reward_space(r); % reward per LO
            if verbose
                fprintf('j=%d/%d\n',r,no_of_rewards);
            end
            
            % Initialize:
            % Hearing
            V.v_sarsa_h=zeros(1,no_of_actions*no_of_notes); % action-Value function (Sarsa learning)
            V.a_s_counter_h=ones(total_no_of_states,no_of_actions*no_of_notes); % Action-State Counter hearing (number of times action taken and state hit)
            V.a_s_counter_h_all=V.a_s_counter_h;
            V.no_of_lo_events_h=zeros(total_no_of_states,no_of_actions*no_of_notes); % number of LO events
            
            % Deaf
            V.v_sarsa_d=zeros(1,no_of_actions*no_of_notes);
            V.a_s_counter_d=ones(no_of_states_d,no_of_actions*no_of_notes); V.a_s_counter_d_all=V.a_s_counter_d;
            V.no_of_lo_events_d=zeros(no_of_states_d,no_of_actions*no_of_notes);
            
            % Hearing
            state_history_h=zeros(1,N*no_of_notes); % state history
            action_history_h=state_history_h; % action history
            % Deaf
            state_history_d=state_history_h;
            action_history_d=state_history_h;


            ci=0;
            for i=1:N % loop over songs
                
                for k=1:no_of_notes % loop over notes
                    ci=ci+1;
                    
                    % states and actions corresponding to current note:
                    ks=(1+(k-1)*no_of_states_h):k*no_of_states_h; ka=(1+(k-1)*no_of_actions):k*no_of_actions;
                    
                    
                    %% take action based on current action-value function
                    
                    % Hearing
                    [~,action_current_h]=max(V.v_sarsa_h(ka)-0*rand(1,no_of_actions)); % choose action
                    action_current_h=action_current_h+(k-1)*no_of_actions;           
                    action_history_h(ci)=action_current_h;
                    state_next_h=find(rand(1)<transitionAS(:,action_current_h),1,'first');   
                    state_history_h(ci)=state_next_h; % markov transition
                    
                    is_loH=(i>Non & k==lo_note & state_next_h>(lo_note-1)*no_of_states_h+no_of_states_h/2); % simple, deterministic model
                    if b>0 && k==lo_note && r==jplot
                        lo_h(i)=is_loH;
                    end
                    
                    % Deaf
                    
                    [~,aD]=max(V.v_sarsa_d(ka)-0*rand(1,no_of_actions));
                    aD=aD+(k-1)*no_of_actions; action_history_d(ci)=aD;
                    sD=find(rand(1)<transitionAS(:,aD),1,'first'); % state measured by experimenter
                    is_loD=(i>Non & k==lo_note & sD>(lo_note-1)*no_of_states_h+no_of_states_h/2); % simple, deterministic model
                    fprintf("bird,%d is_lOD %d \n",b,is_loD);
                    sD=2-is_loD;  state_history_d(ci)=sD; % sensory state for deaf bird is binary
                    if b>0 && k==lo_note  && r==jplot
                        lo_d(i)=is_loD;
                    end
                    
                    %% update action-state counter and estimated transition probability:
                    
                    V.a_s_counter_h_old=V.a_s_counter_h;
                    V.a_s_counter_d_old=V.a_s_counter_d;
                    
                    % update
                    V.a_s_counter_h(state_next_h,action_current_h)=max(1,tau*V.a_s_counter_h(state_next_h,action_current_h)+1);
                    V.a_s_counter_h_all(state_next_h,action_current_h)=V.a_s_counter_h_all(state_next_h,action_current_h)+1;
                    V.no_of_lo_events_h(state_next_h,action_current_h)=V.no_of_lo_events_h(state_next_h,action_current_h)+is_loH;
                    V.a_s_counter_d(sD,aD)=max(1,tau*V.a_s_counter_d(sD,aD)+1);
                    V.a_s_counter_d_all(sD,aD)=V.a_s_counter_d_all(sD,aD)+1;
                    V.no_of_lo_events_d(sD,aD)=V.no_of_lo_events_d(sD,aD)+is_loD;
                    
                    if (c==1 && i==Non) || (c==2 && i==1)
                        V.a_s_counter_honall=V.a_s_counter_h_all;  V.a_s_counter_donall=V.a_s_counter_d_all; V.nLODonall=V.no_of_lo_events_d; V.nLOHonall=V.no_of_lo_events_h;
                    end
                    
                    %% Q learning: update value function
                    
                    % Hearing
                    if igC>0
                        igH=igC;
                    else
                        igH=calc_ig(V.a_s_counter_h,V.a_s_counter_h_old,action_current_h);
                    end
                    if c<3
                        egH=calc_impact(V.a_s_counter_h,V.a_s_counter_h_old,action_current_h);
                    else
                        egH=0;
                    end
                    VTAH=log(igH+epsi)+egH+is_loH*R-V.v_sarsa_h(aH);
                    V.v_sarsa_h(aH)=V.v_sarsa_h(aH)+alpha*(VTAH);
                    
                    % Deaf
                    if igC>0
                        igD=igC;
                    else
                        igD=calc_ig(V.a_s_counter_d,V.a_s_counter_d_old,aD);
                    end
                    if c<3
                        egD=calc_impact(V.a_s_counter_d,V.a_s_counter_d_old,aD);
                    else
                        egD=0;
                    end
                    VTAD=log(igD+epsi)+egD+is_loD*R-V.v_sarsa_d(aD);
                    V.v_sarsa_d(aD)=V.v_sarsa_d(aD)+alpha*VTAD;
                    
                    if r==jplot
                        vta_j_h(k,i)=VTAH;
                        vta_j_d(k,i)=VTAD;
                    end
                    
                end
            end
            value_states_h(r,:)=V.v_sarsa_h;
            value_states_d(r,:)=V.v_sarsa_d;
            
            contingenciesH(b,r)=sum(sum(V.no_of_lo_events_h-V.nLOHonall))/(N-1);
            contingenciesD(b,r)=sum(sum(V.no_of_lo_events_d-V.nLODonall))/(N-1);
            
            if verbose
                fprintf('contingency: H=%.2g  D=%.2g\n', contingenciesH(b,r),contingenciesD(b,r));
                
                % plot action-state counts
                figure(11);clf; imagesc(V.a_s_counter_h_all-V.a_s_counter_honall); colormap(hot); colorbar;
                xlabel('actions'); ylabel('states'); title('Hearing action-state count');
                figure(12);clf; imagesc(V.a_s_counter_d_all-V.a_s_counter_donall); colormap(hot); colorbar;
                xlabel('actions'); ylabel('states'); title('Deaf action-state count');
                
                % plot distribution of states and actions visited XXX why Non=N for control birds?
                figure(15); clf; subplot(211); hist(state_history_h(Non+1:end),1:total_no_of_states);   title('States hearing');
                subplot(212); hist(state_history_d(Non+1:end),1:no_of_states_d,'r'); title('States deaf');
                figure(16); clf; subplot(211); hist(action_history_h(Non+1:end),1:no_of_actions*no_of_notes); title('Actions hearing');
                subplot(212); hist(action_history_d(Non+1:end),1:no_of_actions*no_of_notes,'r');  title('Actions deaf');
                
                % plot value function
                figure(21);clf; plot(V.v_sarsa_h,'k'); hold on;plot(V.v_sarsa_d,'r'); title('Q function');
                xlabel('action'); legend({'Hearing','Deaf'},'Location','best');
            end
        end
        
        mean_value_states_h(b,:,:)=value_states_h; mean_value_states_d(b,:,:)=value_states_d;
        if c==1
            vta_j_h=vta_j_h(:,N/2:N); lo_h=lo_h(N/2:N);
            vta_plus_h(:,b)=mean(vta_j_h(:,find(lo_h)),2);
            vta_minus_h(:,b)=mean(vta_j_h(:,find(~lo_h)),2);
            vta_plus_d(:,b)=mean(vta_j_d(:,find(lo_d)),2);
            vta_minus_d(:,b)=mean(vta_j_d(:,find(~lo_d)),2);
        end
    end
    
    if c==1
        figure(32);
        clf;
        plot_f(reward_space,mean(mean(mean_value_states_h,3),1),'k','linewidth',2); hold on;
        plot_f(reward_space,mean(mean(mean_value_states_d,3),1),'r','linewidth',2);
        set(gca,'box','off');axis tight
        legend({'Hearing','Deaf'},'Location','best');
        xlabel('r'); ylabel('mean Q'); title('Action averaged Q');
        
        figure(31); clf;
        plot_f(reward_space,mean(contingenciesH,1),'k','linewidth',2); hold on;
        plot_f(reward_space,mean(contingenciesD,1),'r','linewidth',2);
        plot_f(reward_space,.5*ones(1,no_of_rewards),'k--','linewidth',2);
        set(gca,'box','off');
        legend({'Hearing','Deaf'},'Location','best');
        xlabel('Reward'); ylabel('LO contingency');
        axis tight
        
        figure(33); clf;
        fplus=mean(vta_plus_h,2); fminus=mean(vta_minus_h,2);
        splus=std(vta_plus_h')'; sminus=std(vta_minus_h')';
        errorbar(1:no_of_notes,fplus,splus,'k','linewidth',2); hold on;
        errorbar(1:no_of_notes,fminus,sminus,'r--','linewidth',2);
        title(['R=' sprintf('%.2g',reward_space(jplot))]);
        legend({'LO','no LO'},'Location','Best');
        set(gca,'box','off')
        
    elseif c==2
        figure(32);
        plot_f(reward_space,mean(mean(mean_value_states_h,3),1),'k--','linewidth',2);
        hold on;
        plot_f(reward_space,mean(mean(mean_value_states_d,3),1),'r--','linewidth',2);
        axis tight
        
    else
        figure(32);
        plot_f(reward_space,mean(mean(mean_value_states_h,3),1),'k:','linewidth',2);hold on;
        plot_f(reward_space,mean(mean(mean_value_states_d,3),1),'r:','linewidth',2);
        axis tight
        
        figure(31);
        plot_f(reward_space,mean(contingenciesH,1),'k:','linewidth',2); hold on;
        plot_f(reward_space,mean(contingenciesD,1),'r:','linewidth',2);
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
T_old=ASC_old(:,a0)/(eps+sum(ASC_old(:,a0))); % old estimate
T_new=ASC_new(:,a0)/(eps+sum(ASC_new(:,a0))); % new estimate
dkl=sum(T_old.*log(T_old./T_new));
ig=dkl;  % for impact
end

function emdist=calc_impact(ASC_new,ASC_old,a0)
% CALC_impact calculates the impact gain

%   'ASC_new'   -   new action-state counter after taking action
%   'ASC_old'   -   old action-state counter

%T_old=ASC_old(:,a0)/(eps+sum(ASC_old(:,a0))); % old estimate
T_new=ASC_new(:,a0)/(eps+sum(ASC_new(:,a0))); % new estimate
T0=zeros(size(T_new)); T0(end)=1;


% DKL
emdist=sum(T0.*log(eps+T0./(eps+T_new)));

end


