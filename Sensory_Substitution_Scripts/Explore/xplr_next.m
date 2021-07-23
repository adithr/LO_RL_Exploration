function Tpls=xplr_next(NAS,ns,na,tau)

for k=1:ns
    for l=1:na
        h=NAS; h(k,l)=tau*h(k,l)+1;
        h2=h./(eps+ones(ns,1)*sum(h,1));
        Tpls(:,k,l)=h2(:,l);
    end
end
end