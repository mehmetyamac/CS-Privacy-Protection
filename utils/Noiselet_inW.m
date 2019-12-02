function y = Noiselet_inW(phi,inWav,t)

% wc=reshape(t,S1,S2); %estimated wavelet coefficients(sparse)
% %h=MakeONFilter('Coiflet',2);
% %im_hat=IWT2_PO(wc,3,h);
% im_hat=inWav(wc);    %estimated image
% S_hat=im_hat(:);     %estimated image into one vector
s_hat=inWav(t);
y=phi(s_hat(:));            %taking measurement from this vector.

end

