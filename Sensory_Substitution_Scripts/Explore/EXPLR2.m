% Model of Exploration in Deaf Light-Off Birds
% First run with do_control=0; then with do_control=1;
% logarithmic model
%
nsS=3; % number of states the bird is singing in (=1: simple model)
% =2: the bird sings first note, then second note
na=6; % # actions, should be even
N=600; % # trials
do_control=0;

nbirds=1; % number of birds to simulate

igC=-2;
if do_control % no LO
    Non=N;%N/4000; % first trial with LO
else
    Non=1;
end
nR=15; rs=-logspace(log10(0.0001),log10(1),nR); % LO rewards
jplot=7; % plot VTA firing for reward # jplot

% Hearing
ns=na+2; % # states
nsD=2; % # states

nsA=ns*nsS; % total number of (all) states
alpha=0.05; % learning rate TD learning

tau=0.99;  % forgetting time constant

% Sensorimotor model
h1=zeros(ns,1); h1(1:3)=[1/4 1/2 1/4]; h2=zeros(na,1); h2(1)=1/4;
wh=toeplitz(h1,h2);
w=zeros(nsA,nsS*na);
for i=1:nsS
    w(1+(i-1)*ns:i*ns,1+(i-1)*na:i*na)=wh;
end
W=cumsum(w,1);
figure(1); clf; imagesc(w);  gg=gray; colormap(gg(end:-1:1,:)); colorbar;
set(gca,'box','off');
xlabel('actions'); ylabel('states'); title('CMC');
%print CMC -dsvg

conts=zeros(nbirds,nR); % contingencies
contsD=conts;
Vs=zeros(nR,na*nsS); % mean Q values
VsD=Vs;
Vsall=zeros(nbirds,nR);
VsallD=Vsall;

VTAs=zeros(nsS,N);
VTAsD=VTAs;
VTAsallplus=zeros(nsS,nbirds); VTAsallminus=VTAsallplus;
VTAsallplusD=VTAsallplus; VTAsallminusD=VTAsallplus; 
LOs=zeros(1,N); LOsD=LOs;

method='PIG';
%method='CBE';

for nb=1:nbirds
    for j=1:nR
        r=rs(j);
        fprintf('j=%d/%d\n',j,nR);
        
        %   V=zeros(nsA,na*nsS); % State-Value function (Sarsa learning)
        V=zeros(1,na*nsS); % State-Value function (Sarsa learning)
        NAS=ones(nsA,na*nsS); % number of times action taken and state hit
        NASall=NAS;
        Tpls=zeros(nsA,nsA,na*nsS);
        nLO=zeros(nsA,na*nsS); % number of LO events
        nLOall=nLO;
        ss=zeros(1,N*nsS); % state history
        acts=ss; % action history
        igs=ss; egs=ss; % information and entropy history
        Rs=ss; % reinforcement history
        
        
        % Deaf
        %   VD=zeros(nsD,na*nsS);
        VD=zeros(1,na*nsS);
        NASD=ones(nsD,na*nsS); NASDall=NASD;
        TplsD=zeros(nsD,nsD,na*nsS); nLOD=zeros(nsD,na*nsS); nLODall=nLOD;% Deaf
        ssD=ss; actsD=ss; igsD=ss; egsD=ss; RsD=Rs;
        ci=0;
        for i=1:N % loop over songs
            for k=1:nsS % loop over notes
                ci=ci+1;
                % Hearing
                js=(1+(k-1)*ns):k*ns; ja=(1+(k-1)*na):k*na;
                T=NAS(js,ja)./(eps+ones(ns,1)*sum(NAS(js,ja),1)); % current estimate
                TD=NASD(:,ja)./(eps+ones(nsD,1)*sum(NASD(:,ja),1)); % current estimate
                switch method
                    case 'PIG'
                        % hearing
                        Tpls=xplr_next(NAS(js,ja),ns,na,tau);
                        %   pig=xplr_pig(T,Tpls,ns,na);
                        %  entr=xplr_peg(T,NAS(js,ja),ns,na);
                        
                        %%                      C=sum(NAS(js,ja),2);
                        %%                     eC=C'*T; entr=1./eC;
                        
                        % I=entr+log(pig+eps);
                        % R=I+r*sum(nLO(js,ja),1)./sum(NAS(js,ja),1);
                        
                        %Deaf
                        TplsD=xplr_next(NASD(:,ja),nsD,na,tau);
                        % pigD=xplr_pig(TD,TplsD,nsD,na);
                        % entrD=xplr_peg(TD,NASD,nsD,na);
                        
                        %%                     CD=sum(NASD,2);
                        %%                     eCD=CD'*TD; entrD=1./eCD;
                        
                        % ID=entrD+log(pigD+eps);
                        % RD=ID+r*sum(nLOD(:,ja),1)./sum(NASD(:,ja),1);
                        
                    case 'CBE' % Counter-based exploration, S Thrun 1992
                        C=sum(NAS(js,ja),2);
                        eC=C'*T; R=mean(eC)./eC+r*sum(nLO(js,ja),1)./sum(NAS(js,ja),1);
                        
                        CD=sum(NASD,2);
                        eCD=CD'*TD; RD=mean(eCD)./eCD+r*sum(nLOD(:,ja),1)./sum(NASD(:,ja),1);
                        % RD=RD+1e-10*rand(1,na);
                end
                
                % Hearing
                %[dummy,act]=max(R);
                [dummy,act]=max(V(ja)-0*rand(1,na));
                Rs(ci)=dummy;
                act=act+(k-1)*na;           acts(ci)=act;
                s=find(rand(1)<W(:,act),1,'first');   ss(ci)=s;
                is_lo=(i>Non & k==2 & s>3*ns/2); % simple, deterministic model
                if k==2 & j==jplot
                    LOs(i)=is_lo;
                end
                [dummy,actD]=max(VD(ja)-0*rand(1,na));
                RsD(ci)=dummy;
                actD=actD+(k-1)*na; actsD(ci)=actD;
                sD=find(rand(1)<W(:,actD),1,'first');
                is_loD=(i>Non & k==2 & sD>3*ns/2); % simple, deterministic model
                if k==2  & j==jplot
                    LOsD(i)=is_loD;
                end
                sD=is_loD+1;  ssD(ci)=sD; % deaf!
                
                % Q learning
                switch method
                    case 'PIG'
                        ig=igC;%
                        %ig=xplr_ig(T,Tpls,ns,s-(k-1)*ns,act-(k-1)*na); igs(ci)=ig;
                        eg=xplr_e(NAS(js,ja),s-(k-1)*ns,act-(k-1)*na,tau);   egs(ci)=eg;
                        %   V(s,act)=V(s,act)+alpha*(ig+eg+is_lo*r-V(s,act));
                        % VTA=log(ig+eps)+is_lo*r-V(act);
                          VTA=igC+eg+is_lo*r-V(act);
                       %VTA=ig+eg+is_lo*r-V(act);
                        V(act)=V(act)+alpha*(VTA);
                        
                        igD=igC;
                        %igD=xplr_ig(TD,TplsD,nsD,sD,actD-(k-1)*na); igsD(ci)=igD;
                        egD=xplr_e(NASD,sD,actD-(k-1)*na,tau); egsD(ci)=egD;
                        %  VD(sD,actD)=VD(sD,actD)+alpha*(igD+egD+is_loD*r-VD(sD,actD));
                         VTAD=igC+egD+is_loD*r-VD(actD);
                        %VTAD=igD+egD+is_loD*r-VD(actD);
                        VD(actD)=VD(actD)+alpha*VTAD;
                        if j==jplot
                            VTAs(k,i)=VTA;
                            VTAsD(k,i)=VTAD;
                        end
                        
                end
                
                NAS(s,act)=max(1,tau*NAS(s,act)+1);
                nLO(s,act)=max(0,tau*nLO(s,act)+is_lo);
                NASall(s,act)=NASall(s,act)+1;
                nLOall(s,act)=nLOall(s,act)+is_lo;
                
                %             NASD(sD,actD)=NASD(sD,actD)+1;
                %             nLOD(sD,actD)=nLOD(sD,actD)+is_loD;
                NASD(sD,actD)=max(1,tau*NASD(sD,actD)+1);
                nLOD(sD,actD)=max(0,tau*nLOD(sD,actD)+is_loD);
                NASDall(sD,actD)=NASDall(sD,actD)+1;
                nLODall(sD,actD)=nLODall(sD,actD)+is_loD;
                
                if i==Non
                    NASon=NAS; NASDon=NASD; nLODon=nLOD; nLOon=nLO;
                    NASonall=NASall;  NASDonall=NASDall; nLODonall=nLODall; nLOonall=nLOall;
                end
            end
        end
        Vs(j,:)=V;
        VsD(j,:)=VD;
        
        conts(nb,j)=sum(sum(nLOall-nLOonall))/(sum(sum(NASall(1:ns,1:na)-NASonall(1:ns,1:na))));
        contsD(nb,j)=sum(sum(nLODall-nLODonall))/(sum(sum(NASDall(1:nsD,1:na)-NASDonall(1:nsD,1:na))));
        fprintf('contingency: H=%.2g  D=%.2g\n', conts(nb,j),contsD(nb,j));
        
           
%         figure(12);clf; imagesc(NASall-NASonall); colormap(hot); colorbar;
%         xlabel('actions'); ylabel('states'); title('Hearing #(states,actions)-NAS');
%         
%         figure(13);clf; imagesc(NASDall-NASDonall); colormap(hot); colorbar;
%         xlabel('actions'); ylabel('states'); title('Deaf #(states,actions)-NAS');
%                
%         figure(15); clf; subplot(211); hist(ss(Non+1:end),1:nsA);   title('States hearing');
%         subplot(212); hist(ssD(Non+1:end),1:nsD,'r');
%         title('States deaf');
%         figure(16); clf; subplot(211); hist(acts(Non+1:end),1:na*nsS);
%         subplot(212); hist(actsD(Non+1:end),1:na*nsS,'r');  title('actions');
        
        
        figure(21);clf; plot(V,'k'); hold on;plot(VD,'r'); title('Q function');
        xlabel('action');
        legend({'Hearing','Deaf'},'Location','best');
        
        %imagesc(V); colormap(hot); colorbar; title(' Q function hearing');
        %figure(22);clf; imagesc(VD); colormap(hot); colorbar; title('Q function deaf');
        figure(22); clf; subplot(211); hist(Rs); xlabel('Reinforcement Hearing');
        legend(['Hearing: ' num2str(mean(Rs))],'Location','best');
        
        subplot(212); hist(RsD); xlabel('Reinforcement Deaf');
        legend(['Deaf: ' num2str(mean(RsD))],'Location','best');
        
        
        
           pause
    end
    
    if nbirds==1
        if do_control
            figure(32);
            semilogx(rs,mean(Vs,2),'k--','linewidth',2);
            hold on; semilogx(rs,mean(VsD,2),'r--','linewidth',2);
            axis tight
            %print LOcont -dsvg
        else
            figure(31); clf; semilogx(rs,conts(nb,:),'k','linewidth',2); hold on;
            semilogx(rs,contsD(nb,:),'r','linewidth',2);
            semilogx(rs,.5*ones(1,nR),'k--','linewidth',2);
            set(gca,'box','off');
            legend({'Hearing','Deaf'},'Location','best');
            xlabel('Reward'); ylabel('LO contingency');
            axis tight
            %print MeanQ -dsvg
            
            figure(32);
            clf; semilogx(rs,mean(Vs,2),'k','linewidth',2); hold on;
            semilogx(rs,mean(VsD,2),'r','linewidth',2);
            set(gca,'box','off');
            legend({'Hearing','Deaf'},'Location','best');
            xlabel('r'); ylabel('mean Q'); title('Action averaged Q');
        end
    end
    
    Vsall(nb,:)=mean(Vs,2)'; VsallD(nb,:)=mean(VsD,2)';
   % mF=mean(VTAs,2);
    VTAs=VTAs(:,N/2:N); LOs=LOs(N/2:N);
    VTAsallplus(:,nb)=mean(VTAs(:,find(LOs)),2);   
    VTAsallminus(:,nb)=mean(VTAs(:,find(~LOs)),2);   
    VTAsallplusD(:,nb)=mean(VTAsD(:,find(LOsD)),2);
    VTAsallminusD(:,nb)=mean(VTAsD(:,find(~LOsD)),2);
end


if do_control==0
    figure(31); clf; semilogx(rs,mean(conts,1),'k','linewidth',2); hold on;
    semilogx(rs,mean(contsD,1),'r','linewidth',2);
    semilogx(rs,.5*ones(1,nR),'k--','linewidth',2);
    set(gca,'box','off');
    legend({'Hearing','Deaf'},'Location','best');
    xlabel('Reward'); ylabel('LO contingency');
    axis tight
    %print LOcont -dsvg
    
    
    figure(32);
    clf; semilogx(rs,mean(Vsall,1),'k','linewidth',2); hold on;
    semilogx(rs,mean(VsallD,1),'r','linewidth',2);
    set(gca,'box','off');
    legend({'Hearing','Deaf'},'Location','best');
    xlabel('r'); ylabel('mean Q'); title('Action averaged Q');
    
    figure(33); clf;
    if nbirds==1
        mF=mean(VTAs,2);
        f1=mean(VTAs(:,find(LOs)),2)-mF; f2= mean(VTAs(:,find(~LOs)),2)-mF;
        plot(f1,'k'); hold on; plot(f2,'r');
        title(['R=' sprintf('%.2g',rs(jplot))]);
        % print VTA -dsvg
    else
        fplus=mean(VTAsallplus,2); fminus=mean(VTAsallminus,2);
        splus=std(VTAsallplus')'; sminus=std(VTAsallminus')';
        errorbar(1:nsS,fplus,splus,'k','linewidth',2); hold on; 
        errorbar(1:nsS,fminus,sminus,'r','linewidth',2);
        title(['R=' sprintf('%.2g',rs(jplot))]);
        legend({'LO','no LO'},'Location','Best');
        set(gca,'box','off')
    end
else
    figure(32);
    semilogx(rs,mean(Vsall,1),'k--','linewidth',2);
    hold on; semilogx(rs,mean(VsallD,1),'r--','linewidth',2);
    axis tight
    %print MeanQ -dsvg
    
end

% Tplus=zeros(ns,na,ns,na); % first state is s*, hypothetical outcome
%         for k=1:ns
%             for l=1:na
%                 % dkl(k,l)=T*log(
%                 h=NAS; h(k,l)=h(k,l)+1;
%                 h2=h./(eps+ones(ns,1)*sum(h,1));
%                 Tplus(:,:,k,l)=h2;
%             end
%         end

% PIG
%         pig=zeros(ns,na);
%         for k=1:ns % loop for pig
%             for l=1:na % loop for pig
%
%                 for k0=1:ns % loop for s*
%                     dkl=zeros(1,na);
%                     for k2=1:ns % state loop for dkl
%                         for l2=1:na
%                             h=zeros(1,na);
%                             h(:)=Tplus(k2,:,k0,l2);
%                             h2=T(k2,:).*log(T(k2,:)./(h));
%                             dkl(l2)=dkl(l2)+h2(l2);
%                         end
%                     end
%                     pig(k,l)=pig(k,l)+T(k0,l)*dkl(l);
%                 end
%             end
%         end

% in case nsS>1
%  pig=ones(ns,1)*pig1;
% pig=pig+(1e-8)*rand(ns,na); % add some random noise
%  R=log(eps+pig)+r*(nLO./NAS);
%  R=R-1e8*(w==0);
% [dummy,act]=max(R(:));
%  act=ceil(act/ns);