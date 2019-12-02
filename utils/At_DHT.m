function v = At_DHT(y, OMEGA, n)
%

vn = zeros(n,1);
vn(OMEGA) = y;
%v =(1/sqrt(n))* my_DHT(vn);
v =idct(vn);
end

