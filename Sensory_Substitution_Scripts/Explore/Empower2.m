%% empowerment, two notes

n=1e5;
poptE=1/4*ones(2,4);%[1 0 0 0];
pLO_x=[1e-6 1e-6 1e-6 1e-6; 1e-6 1/4 3/4 1-1e-6];
pL_x=1-pLO_x;%[1 3/4 1/4 0.000001];
emp_opt=zeros(1,2); 
p=poptE;
for i=1:n
   % pLO=zeros(1,2); pL=pLO;
   pnew=p+randn(2,4)/10; pnew=max(0,min(1,pnew)); pnew=pnew./sum(pnew,2);
    for j=1:2
        pp=pnew(j,:);
        pLO=mean(sum(pLO_x.*pnew,2));
        pL=mean(sum(pL_x.*pnew,2));
        emp(j)=sum(pp.*pLO_x(j,:).*log(pLO_x(j,:)/pLO)+pp.*pL_x(j,:).*log(pL_x(j,:)./pL));
    end
    empA=sum(emp);
    if empA>emp_opt
        emp_opt=empA; poptE=p;
        p=pnew;
    end

end
fprintf('1-5=L, 8=LO\n');
p
%fprintf('empowerment: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n',poptE);
%poptE

