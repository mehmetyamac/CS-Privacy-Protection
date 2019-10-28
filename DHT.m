function  y= DHT(x,omega2  )

p=length(x);

dht_x=dct(x);
%dht_x=my_DHT(x/sqrt(p));
y= dht_x(omega2);

end

