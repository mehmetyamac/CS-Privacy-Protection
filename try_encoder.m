



n=15;
k=3;

%choose a method for bit correction
encoding_type='cyclic/binary';

[parity_matrix,gpol] = encoder_gen(k,n);
data = randi([0 1],k,1);

encData = encode(data,n,k,encoding_type,gpol);
tmp=randperm(n);


for i=1:5
    err_in=tmp(1:i);
    corruptedData=encData;
    corruptedData(err_in) = ~encData(err_in);
    decData = decode(corruptedData,n,k,'cyclic/binary',gpol,parity_matrix);
    numerr = biterr(data,decData);
    if numerr~=0
        disp(i)
    end        
end

