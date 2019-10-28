function  s  = Adj_Noiselet( y,N,omega )

x=zeros(N,1);
x(omega)=y;

s=realnoiselet(x/sqrt(N));

end

