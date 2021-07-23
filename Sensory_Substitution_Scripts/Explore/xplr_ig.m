function ig=xplr_ig(T,Tpls,ns,k0,l)
% l = chosen action
% k0 = reached state (s*)
ig=0;
dkl=0;
for k2=1:ns % state loop for dkl
    dkl=dkl+T(k2,l).*log(T(k2,l)/Tpls(k2,k0,l));
end
%ig=ig+T(k0,l)*dkl;
ig=dkl;
ig=ig*1000;
end