%% empowerment, only one note

n=1e6;
poptE=1/4*ones(1,4);%[1 0 0 0];
poptD=poptE;
pLO_x=[1e-6 1/4 3/4 1-1e-6];
pL_x=1-pLO_x;%[1 3/4 1/4 0.000001];
emp_opt=0; dkl_opt=0;
for i=1:n
    p=rand(1,4); p=p/sum(p);
    pLO=sum(pLO_x.*p);
    pL=sum(pL_x.*p);
    emp=sum(p.*pLO_x.*log(pLO_x/pLO)+p.*pL_x.*log(pL_x./pL));
    if emp>emp_opt
        emp_opt=emp; poptE=p;
    end
    
    dkl=-log(pL);
    if dkl>dkl_opt
        dkl_opt=dkl;
        poptD=p;
    end
    
end
fprintf('1=L, 4=LO\n');
fprintf('empowerment: %.2f %.2f %.2f %.2f\n',poptE);
%poptE

fprintf('DKL impact: %.2f %.2f %.2f %.2f\n',poptD);

%% DKL impact
% for i=1:n
%       p=rand(1,4); p=p/sum(p);
%   
% end

