function x = Adj_Noiselet_inW(phi_T,Wav,t,N)

s_hat1=phi_T(t);
x = Wav(s_hat1(1:N));
% s_hat=s_hat1(1:S1*S2);
% img_hat=reshape(s_hat,S1,S2);
%  co=Wav(img_hat);
% % x=co(:);
% %h=MakeONFilter('Coiflet',2);
% %co=FWT2_PO(img_hat,3,h); %wavelet coeefficients of image
% x=co(:);
end

