w2=[1 1 .75 .25 0 0;0 0 .25 .75 1 1];

poli=1/6*ones(6,1);
e=-1000; best_poli=zeros(6,1); best_poli(1)=1;
for i=1:100000
    poli=poli+.01*randn(6,1); poli=max(eps,poli); poli=poli/sum(poli);
    [eprobe,~]=calc_emp2Diff(w(1:8,1:6).*(ones(8,1)*poli'),poli',poli');
  
    %eprobe=calc_emp2(w(1:8,1:6).*(ones(8,1)*poli'),poli'); % for hearing
 %   eprobe=calc_emp(w2.*(ones(2,1)*poli'),poli); % for deaf 
    if eprobe>e
        e=eprobe;
        best_poli=poli;
    else
        poli=best_poli;
    end
end
figure(1);clf;plot(best_poli);


