function [parity_matrix,gpol] = encoder_gen(k,n)
gpol = cyclpoly(n,k);
parmat = cyclgen(n,gpol);
parity_matrix = syndtable(parmat);
end

