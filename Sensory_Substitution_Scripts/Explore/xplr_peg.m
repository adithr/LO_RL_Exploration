function entr_diff=xplr_peg(T,NAS,ns,na)
% predicted entropy gain
C=sum(NAS,2); % state count
P=C/sum(C); % state probability
entrC=-sum(P.*log(P));

entr_diff=zeros(1,na);
for i=1:na % next action chosen
    states=find(NAS(:,i)-1)';
    if isempty(states)
        states=1:ns;
    end
    for j=1:ns%states % possible states
        Cplus=C; Cplus(j)=Cplus(j)+1;
        Pplus=Cplus/sum(Cplus);
        entrPlus=-sum(Pplus.*log(Pplus));
        entr_diff(i)=entr_diff(i)+T(j,i)*(entrPlus-entrC);
    end
end

entr_diff=3000*entr_diff;%/length(P);

end


% entrPlus=zeros(1,na);
% for i=1:na % next action chosen
%     Cplus=C+T(:,i);
%     Pplus=Cplus/sum(Cplus);
%     entrPlus(i)=-sum(Pplus.*log(Pplus));
% end
% entr_diff=entrPlus+sum(P.*log(P));
% entr_diff=100*entr_diff;
% end

% % include next state
% Cplus=C*ones(1,na)+T;
% Pplus=Cplus./(ones(ns,1)*sum(Cplus,1));

%entr=-sum(Pplus.*log(Pplus),1) + sum(P.*log(P));
