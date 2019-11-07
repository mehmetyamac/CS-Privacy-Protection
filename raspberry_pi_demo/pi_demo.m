clear, clc
%read csv files
rng(1)

param.N=2^16;

param.em_power = 0.15;
param.degradation='binary';
%param.degradation='gauss';%'blurring'; %'gauss';
param.m1 = 7; %if blurring is the option than this is filter size
param.m2 = 7;
%param.matrix=0.9+0.05*randn(2^9,2^9);
samples = rand(1, 2^18);
% transform from uniform to standard normal distribution using inverse cdf
samples = sqrt(2) * erfinv(2 * samples - 1);
param.matrix=0.9+0.05*reshape(samples, 2^9, 2^9);

param.measurement_rate = 6;
param.mratio = param.measurement_rate./10;
param.y1 = [];
param.y2 = [];
param.x1 = [];
param.x2 = [];

param.S1 = 186;
param.S2 = 352;
param.S1 = 222;
param.S2 = 295;
param.M = 15003; %maximum leghts of the bits we can embed
load omega.mat
load omega2.mat
param.omega = omega;
param.omega2 = omega2;
param.type="rgb";
rng(1)


processFrame(param);


function processFrame(param)
    
    %load y_w.mat
    %load smean.mat
    %load v.mat
    %load coor.mat
    msg_size = 640 * 480 * 8;
    msg_size = (117966 + 3 + 3 + 4) * 8;
    TCPServer = tcpip('192.168.43.161', 5432, 'NetworkRole', 'server')
    TCPServer.InputBufferSize = 512;
    fopen(TCPServer);

    ee = 5;
    start = cputime;
    counter = 0;

    while(1)
        [y_w, smean, v, coor] = receive(TCPServer, msg_size);
        solA = userA_new(y_w, smean, param);
        if v(1) ~= 0
            solB = userB_new(y_w, smean, param, v, coor);
        end
        counter = counter + 1;
        if (cputime - start) > ee
            fprintf(strcat('FPS: ', num2str((cputime - start) / counter / (cputime - start) ), '\n'));
            counter = 0;
            start = cputime;
        end
    end
end



function [solImage] = userA_new(y_w, smean, param)
m = round(param.N*param.mratio);
M = param.M;
seed1 = 1;
seed2 = 2;
omega = param.omega + 1;
S1 = param.S1;
S2 = param.S2;
N = S1 * S2;

p1=m-M./3;
       
rng(seed2)

%%transforms
h=MakeONFilter('Coiflet',2);
Wav=@(t) FWT2_POE(t,3,h); %wavelet coeefficients of image
inWav= @(t) IWT2_POE(t,3,h);
Wav1=@(t) new_wavelet(t,Wav,S1,S2);
inWav1=@(t) new_inwavelet(t,inWav,S1,S2);

param.redundant=param.N-N;
    
   
%measurements
phi = @(t) Noiselet([t;zeros(param.redundant,1)],omega);
A   = @(t) Noiselet_inW(phi,inWav1,t);
%F   = @(t) DHT(t,omega2);
%FA  = @(t) DHT(A(t),omega2);


%Adjoints
phiT= @(t)  Adj_Noiselet(t,param.N,omega);
AT  = @(t)  Adj_Noiselet_inW(phiT,Wav1,t,N);
%FT  = @(t)  At_DHT( t,omega2,m );  % B
%FAT = @(t)  Adj_Noiselet_inW(phiT,Wav1,FT(t));

% regularization parameter
tau = 4;

% set tolA
tolA = 1.e-5;

if param.type=='rgb'
    loop =3;
else
    loop =1;
    
end

sol=zeros(S1,S2,3);
parfor i=1:loop
    

%%%% decoding part  %%%%%%%%
y_tild=y_w((i-1)*m+1:i*m);
% regularization parameter
tau = 4;

% set tolA
tolA = 1.e-7;
            
[~,x_tild1,~,~,~,~]= ...
                GPSR_BB(y_tild,A,tau,...
                'Debias',1,...
                'AT',AT,... 
                'Initialization',0,...
                'StopCriterion',1,...
                'ToleranceA',tolA,'ToleranceD',0.00001);
            
            %x_h=reshape(x_tild1,S1,S2);
            s_hat_h=inWav1(x_tild1);
            s_hat=reshape(s_hat_h,S1,S2);
                 s_hat=s_hat + smean(i);
                 sol(:,:,i) = s_hat;
            
end
solImage = uint8(cat(3, sol(:,:,1), sol(:,:,2), sol(:,:,3)));



end


function [solImage] = userB_new(y_w, smean,param, v, coor)
    omega = param.omega + 1;
    omega2 = param.omega2 + 1;
    S1 = param.S1;
    S2 = param.S2;
    N = S1*S2;
    m = round(param.N*param.mratio);
    M = param.M;
    seed2 = 2;

    p1=m-M./3;

    rng(seed2)

    %%transforms
    h=MakeONFilter('Coiflet',2);
    Wav=@(t) FWT2_POE(t,3,h); %wavelet coeefficients of image
    inWav= @(t) IWT2_POE(t,3,h);
    Wav1=@(t) new_wavelet(t,Wav,S1,S2);
    inWav1=@(t) new_inwavelet(t,inWav,S1,S2);

    param.redundant=param.N-N;


    %measurements
    phi = @(t) Noiselet([t;zeros(param.redundant,1)],omega);
    A   = @(t) Noiselet_inW(phi,inWav1,t);
    F   = @(t) DHT(t,omega2);
    FA  = @(t) DHT(A(t),omega2);


    %Adjoints
    phiT= @(t)  Adj_Noiselet(t,param.N,omega);
    AT  = @(t)  Adj_Noiselet_inW(phiT,Wav1,t,N);
    FT  = @(t)  At_DHT( t,omega2,m );  % B
    FAT = @(t)  Adj_Noiselet_inW(phiT,Wav1,FT(t),N);


    % regularization parameter
    tau = 4;

    % set tolA
    tolA = 1.e-5;

    if param.type=='rgb'
        loop =3;
    else
        loop =1;

    end


    temp2=ones(m,1);
    temp2(omega2)=0;
    in=find(temp2==1);


    parfor i=1:loop


        %%%% decoding part  %%%%%%%%
        y_tild=F(y_w((i-1)*m+1:i*m));
        % regularization parameter
        tau = 4;

        % set tolA
        tolA = 1.e-7;

        [~,x_tild1,objective,times,debias_start,mses]= ...
                        GPSR_BB(y_tild,FA,tau,...
                        'Debias',1,...
                        'AT',FAT,... 
                        'Initialization',0,...
                        'StopCriterion',1,...
                        'ToleranceA',tolA,'ToleranceD',0.00001);


        %%%%%%%%%%%%%%%%% Reconstruct watermark messege %%%%%
        new_y = y_w((i-1)*m+1:i*m) - A(x_tild1);
        w_t = DHT(new_y,in);
        w_h = zeros(size(w_t));
        w_h(find(w_t>=0)) = v(i)*1;
        w_h(find(w_t<0)) = v(i)*-1;

        www_hat(:,i) = w_h;
    end


    %convert the estimated information into bits
    w_hat=zeros(size(www_hat));        
    w_hat(www_hat>0)=1;
    w_hat(www_hat<0)=0;

    k=1:3:M;
    l=2:3:M;
    d=3:3:M;
    www_h(k)=w_hat(:,1);
    www_h(l)=w_hat(:,2);
    www_h(d)=w_hat(:,3);

    param.x1=bin2dec(num2str(www_h(1:8)));
    param.x2=bin2dec(num2str(www_h(9:16)));
    param.y1=bin2dec(num2str(www_h(17:24)));
    param.y2=bin2dec(num2str(www_h(25:32)));
    
    param.x1 = coor(1) + 1;
    param.y1 = coor(2) + 1;
    param.x2 = coor(3) - coor(1);
    param.y2 = coor(4) - coor(2);

    s_hat1=inWav1(x_tild1(:));
    s_hat1=reshape(s_hat1,S1,S2);
    s_hat=s_hat1 + smean(3);

    mask = zeros(size(s_hat));
    mask(param.y1:(param.y1 + param.y2), param.x1:(param.x1 + param.x2)) = 1;

    area_mask=sum(sum(mask));

    www_h(www_h==0)=-1;
    tmp=www_h(33:33+area_mask-1);

    inside=zeros(S1,S2);


    inside(mask == 1) = tmp;

    %D=watermark_inf.D;
    %infor.total_error=sum(sum(D~=inside));
    D=inside;
    D2=D.*param.matrix(S1,S2);
    %outside = (inside-1).*(-1);
    %measurements
    outside = (mask-1).*(-1);


    aa=(33+area_mask-1)./3;

    www_hat((aa+1):end,:)=0;
     m1=param.m1;%=3;
     m2=param.m2;%3;
     P=1/(m1*m2)*ones(m1,m2); % uniform 3 ? 3 blur

        switch param.degradation
            case 'blurring'
              phi_D = @(t) phi(outside(:).*t+mask(:).*blur(t,P,S1,S2));
              phiT_Dt = @(t) outside(:).*new_phi_T(t,phiT,N) +D(:).*new_phi_T(t,phiT,N);   
            case 'binary'
                phi_D = @(t) phi(outside(:).*t+D(:).*t);
                phiT_Dt = @(t) outside(:).*new_phi_T(t,phiT,N) +D(:).*new_phi_T(t,phiT,N);
            case 'gauss'
                phi_D = @(t) phi(outside(:).*t+D2(:).*t);
                phiT_Dt = @(t) outside(:).*new_phi_T(t,phiT,N) +D2(:).*new_phi_T(t,phiT,N);
        end


    %phi_D = @(t) phi(outside(:).*t+inside(:).*t); %sadece burasi degisecek
    A_D   = @(t) Noiselet_inW(phi_D,inWav1,t);

    %phiT_Dt = @(t) outside(:).*phiT(t) +inside(:).*phiT(t); %sadece burasi degisecek
    AT_Dt = @(t)  Adj_Noiselet_inW(phiT_Dt,Wav1,t,N);

    sol=zeros(S1,S2,3);

    for i =1:loop
        newy2=y_w((i-1)*m+1:i*m) -At_DHT(www_hat(:,i),in,m);
        [~,x_debias3,~,~,~,~]= ...
                    GPSR_BB(newy2,A_D,tau,...
                    'Debias',1,...
                    'AT',AT_Dt,... 
                    'Initialization',2,...
                    'StopCriterion',1,...
                    'ToleranceA',tolA,'ToleranceD',0.0001);

                     s_hat_h=inWav1(x_debias3);
                     s_hat=reshape(s_hat_h,S1,S2);


                %x_h=reshape(x_debias3,S1,S2);
                 %    s_hat=inWav(x_h);
                     s_hat=s_hat + smean(i);
                     sol(:,:,i) = s_hat;
                     %psnr3=measerr(s,s_hat)


    end

    solImage = uint8(cat(3, sol(:,:,1), sol(:,:,2), sol(:,:,3)));
end

function out=new_phi_T(t,phi_T,N)
    s_hat1=phi_T(t);
    out = s_hat1(1:N);
end

function x=new_wavelet(s,Wav,S1,S2)
    new_S1=ceil(log2(S1));

    new_S2=ceil(log2(S2));
    
    z_padded=zeros(2^new_S1,2^new_S2);

    z_padded(1:S1,1:S2)=reshape(s,S1,S2);

    x=Wav(z_padded);

end

function [y_w, smean, v, coor] = receive(TCPServer, msg_size)
frame = [];
 while (1)
     while(1)       % Waits for incoming CSI data
         nBytes = get(TCPServer,'BytesAvailable');
         if nBytes > 0
             break;
         end
     end

     data = uint8(fread(TCPServer, TCPServer.BytesAvailable, 'uint8'));
     frame = [frame; data];
     %flushinput(TCPServer);
     
     if length(frame) >= msg_size
         image = typecast(frame(1:msg_size), 'double');
         %imshow(reshape(uint8(image), 640, 480))
         frame(1:msg_size) = [];
         y_w = image(1:(msg_size/8) - 10);
         smean = image((msg_size/8) - 9:(msg_size/8) - 7);
         v = image((msg_size/8) - 6:(msg_size/8) - 4);
         coor = image((msg_size/8) - 3:(msg_size/8));
         break
     end 
 end
end

function s=new_inwavelet(x,inWav,S1,S2)
    new_S1=ceil(log2(S1));
    new_S2=ceil(log2(S2));
    
    x_padded=reshape(x,2^new_S1,2^new_S2);
    
    s_padded=inWav(x_padded);
    
    S=s_padded(1:S1,1:S2);
    
    s=S(:);
end

function y=blur(t,P,S1,S2)
    X =reshape(t,S1,S2);
    Y=imfilter(X,P,'symmetric');
    y=Y(:);

end