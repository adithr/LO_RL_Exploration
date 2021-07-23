function  [emp,Demp]=calc_emp2Diff(ASC,poli,poli_new)
% CALC_EMPG calculates empowerment and empowerment difference between two
% policies poli and poli_new
%
%  'ASC':  action-state counter, is used to estimate p(k) given policy P(j)
%   poli new is new policy

% emp is empowerment of poli_new
% Demp is empowerment difference E(poli_new)-E(poli)
emp=[];
Demp=[];

if abs(sum(poli)-1)>100*eps || abs(sum(poli_new)-1)>100*eps
    disp('policy not normalized');
    keyboard
end
ASC0=ASC;
ASC=ASC0.*(ones(size(ASC0,1),1)*(poli./(eps+sum(ASC0,1))));
ASC_new=ASC0.*(ones(size(ASC0,1),1)*(poli_new./(eps+sum(ASC0,1))));

C=sum(ASC,2);
Ps=C/(eps+sum(C)); % state probability

C=sum(ASC_new,2);
Ps_new=C/(eps+sum(C)); % state probability
emp=0; Demp=0;
for a=1:size(ASC_new,2)
    tnew=ASC_new(:,a)/(eps+sum(ASC_new(:,a)));
   t=ASC(:,a)/(eps+sum(ASC(:,a)));
   eh=sum(poli_new(a)*tnew.*log(eps+tnew./(eps+Ps_new)));
    emp=emp+eh;
    Demp=Demp+eh-sum(poli(a)*t.*log(eps+t./(eps+Ps)));
end
end