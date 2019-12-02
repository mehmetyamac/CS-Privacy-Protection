function s = inwavelet(x, inWav, S1, S2)
    new_S1=ceil(log2(S1));
    new_S2=ceil(log2(S2));
    
    x_padded=reshape(x,2^new_S1,2^new_S2);
    
    s_padded=inWav(x_padded);
    
    S=s_padded(1:S1,1:S2);
    
    s=S(:);
end