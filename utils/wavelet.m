function x = wavelet(s, Wav, S1, S2)
    new_S1=ceil(log2(S1));

    new_S2=ceil(log2(S2));
    
    z_padded=zeros(2^new_S1,2^new_S2);

    z_padded(1:S1,1:S2)=reshape(s,S1,S2);

    x=Wav(z_padded);

end