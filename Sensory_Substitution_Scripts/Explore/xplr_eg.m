function entr_diff=xplr_eg(NAS,k0,l,tau)
% entropy gain
% k0 = current new state

C=sum(NAS,2); % state count
P=C/sum(C); % state probability

NAS(k0,l)=tau*NAS(k0,l)+1;
Cplus=sum(NAS,2);%+T(:,l);
%Cplus(k0)=Cplus(k0)+1;
Pplus=Cplus/sum(Cplus);

entrPlus=-sum(Pplus.*log(Pplus));

entr_diff=entrPlus+sum(P.*log(P));
entr_diff=3000*entr_diff;%/length(P);
end
