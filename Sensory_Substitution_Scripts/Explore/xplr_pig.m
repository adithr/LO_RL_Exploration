function pig=xplr_pig(T,Tpls,ns,na)
nsS=1;
% nsS=number of states the bird is singing in (=1)

pig=zeros(nsS,na);
for k=1:nsS
    for l=1:na % loop for pig
        for k0=1:ns % loop for s*
            
            dkl=zeros(1,na);
            for k2=1:ns % state loop for dkl
                h=zeros(1,na);
                h(:)=Tpls(k2,k0,:); %Tplus(k2,:,k0,l2);
                h2=T(k2,:).*log(T(k2,:)./h);
                dkl=dkl+h2;
            end
            pig(k,l)=pig(k,l)+T(k0,l)*dkl(l);
        end
    end
end
pig=pig*1000;
end
