function entrPlus=xplr_e(NAS,k0,l,tau)
%  entropy
% k0 = current new state

%C=sum(NAS,2); % state count
%P=C/sum(C); % state probability

NAS(k0,l)=tau*NAS(k0,l)+1;

Cplus=sum(NAS,2);
Pplus=Cplus/sum(Cplus);

entrPlus=-sum(Pplus.*log(Pplus))/log(size(NAS,1));

end
