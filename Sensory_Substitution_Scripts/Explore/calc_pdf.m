function xHnew=calc_pdf(xH,ka,aH,alpha)
% compute next policy xHnew by increasing the probability of action aH by
% alpha

xHnew=xH; xHnew(aH)=min(1,xHnew(aH)+alpha);
xHnew(ka)=xHnew(ka)/(eps+sum(xHnew(ka)));
end