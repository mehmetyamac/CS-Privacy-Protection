function  y  = Noiselet(s,omega)

N=length(s);

y1=realnoiselet(s)/sqrt(N);
y=y1(omega);


end

