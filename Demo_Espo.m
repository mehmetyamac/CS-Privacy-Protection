%Demo Espo
clear all;close all;
cam = webcam('Integrated Camera')
cam.Resolution = '640x480';
img = imrotate(snapshot(cam), 180);
%%sparse image
%img=(double((imread('./youtube/Abdullah_1.jpg'))));
param.type='rgb'
[S1_init,S2_init,S3_init]=size(img)
param.N=2^18;
img2=imresize(img,sqrt((param.N)/(S1_init*S2_init))); %burda bir yanlislik var
img2 = img2(1:end-1,1:end-1, :);
param.em_power=0.12; % 0.085

param.seed1=10;
param.seed2=20;


[S1, S2,S3] =size(img2);
N=S1*S2;

if param.N <N
   S1= S1_init;
   S2= S2_init;
   param.redundant=param.N-S1_init*S2_init;
   N=S1*S2;
   
else
   param.redundant = param.N-N;
   img=img2;
end





imshow(uint8(img),[],'InitialMagnification','fit');                                % Show original image.
axis image


hold on

x = [];                                      % Initialize Point.x
y = [];                                      % Initialize Point.y
n = 0;

%creating mesage box for users
G=cell(2,1);
G{1}=sprintf('%s%',('Click Left mouse button to pick points of curve: '));
G{2}=sprintf('%s%',('Click Right mouse button to pick last point of curve: '));
msgbox(G,'Create an initial curve','error');


delta=0.01;
buton = 1;                                  %continue drawing curve until user push right button
while buton == 1                        
      [a, b, buton] = ginput(1);
      n = n + 1;
      x(n,1) = a;
      y(n,1) = b;
      plot(x, y, 'g-');
end   

plot([x;x(1,1)],[y;y(1,1)],'r-');
hold off
drawnow;
% sampling
x = [x;x(1,1)]; 
y = [y;y(1,1)]; 
t = 1:n+1; 
ts = [1:delta:n+1]'; 
xi = interp1(t,x,ts); 
yi = interp1(t,y,ts); 
n = length(xi); 
x = floor(xi(1:n-1)); 
y = floor(yi(1:n-1)); 
n =length(x);





%figure();
%imshow(img,[]);
mask=zeros(S1,S2)
for i=1:length(xi)
    mask(floor(yi(i)),floor(xi(i)))=1;
end

inside=bwfill(boundarymask(mask), 'holes');  %fill the inside of mask with all 1s

figure,imshow(inside,'InitialMagnification','fit')
outside = (inside-1).*(-1);

%Mask
pr=0.5;
MM=(rand(S1,S2)<pr);
MM=MM*2 - 1;
D=inside.*MM;

%D(D==1)=2;
d= D(:);


inx=find(inside==1);

%finding boundaries if you want to transmit just boundaries
[b] = bwboundaries(inside);
mask2=false([S1, S2]);
for i = 1:length(b)
    for j = 1:length(b{i})
        ind = b{i}(j,:);
        mask2(ind(1),ind(2))=1;
    end
end

inside2=imfill(mask2,26, 'holes');

if ~isequal(inside,inside2)
    disp('something wrong in mask operator')
end






%create watermark for cordinates.
temp=double(dec2bin(b{1}(:,1),9))-48;
temp2=double(dec2bin(b{1}(:,2),9))-48;
temp3=double(dec2bin(size(temp,1),9))-48;
w_c=[temp3';temp(:);temp2(:)];
%create watermark for mask. 
w_m = d(inx);
w_c=w_c';
w_c(w_c==0)=-1;
ww = [w_c(:);w_m];

%Arranging the watermark length and measurement ratio

if length(ww)<15000 %maximum leghts of the bits we can embed
    param.mratio=0.65;
    param.M = 15003;
elseif length(ww)>15000 && length(ww)<21000
    param.M = 21003;
    param.mratio=0.71;
else 
    param.M = 24003;
    param.mratio=0.75;
end
%concatanate all watermark
www=zeros(param.M,1);
www(1:length(ww)) =ww;



if strcmp(param.type,'rgb')
    %split the watermark data for each channel
    k=1:3:param.M;
    l=2:3:param.M;
    d=3:3:param.M;

    www1=www(k);
    www2=www(l);
    www3=www(d);
end

    
m=round(param.mratio*param.N); %number of measurements for CS (each channel)


if strcmp(param.type,'rgb')
    s1= double(img(:,:,1));
    s2= double(img(:,:,2));
    s3 =double(img(:,:,3));
    S =[s1(:) s2(:) s3(:)];
else
    S=double(img(:));
end

 smean=mean(S);
 S=S-smean;
 
 
   
 
 rng(param.seed1)
 temp1=randperm(N);
 omega=temp1(1:m);  % Pick up m measurements randomnly
 
 %measurements
  phi = @(t) Noiselet([t;zeros(param.redundant,1)],omega);
  phi_D = @(t) phi(outside(:).*t+D(:).*t);
  tic 
  if strcmp(param.type,'rgb')
       y1= phi_D(S(:,1));
       y2= phi_D(S(:,2));
       y3= phi_D(S(:,3));
       y=[y1;y2;y3];
  else
      y= phi_D(S);
  end
  toc
  
  %watermark embedding
 
  
  
   p1=m-(param.M)./3;
   rng(param.seed2)
   temp2=randperm(m);
   omega2=temp2(1:p1);
  
   temp2=ones(m,1);
   temp2(omega2)=0;
   in=find(temp2==1);
   
   
   
   if strcmp(param.type,'rgb')
        bw1=At_DHT(www1,in,m);
        bw2=At_DHT(www2,in,m);
        bw3=At_DHT(www3,in,m);
        alpha1=norm(y1).*(param.em_power);
        bw1=(bw1./norm(bw1)).*alpha1;
        alpha2=norm(y2).*(param.em_power);
        bw2=(bw2./norm(bw2)).*alpha2;
        alpha3=norm(y3).*(param.em_power);
        bw3=(bw3./norm(bw3)).*alpha3;
        
        watermark_inf.v1(1)=check_watermark(bw1,in,www1);
        watermark_inf.v1(2)=check_watermark(bw2,in,www2);
        watermark_inf.v1(3)=check_watermark(bw3,in,www3);
        
        y_w = [y1+bw1;y2+bw2;y3+bw3];
        
        
    else
        bw=At_DHT(www,in,param.m1);
        alpha=norm(y)*param.em_power;
        bw=(bw./norm(bw)).*alpha;
         %check watermark
      
        watermark_inf.v1=check_watermark(bw,in,www);
        y_w = y + bw;
   end
   
   param.S1=S1;
   param.S2=S2;
   watermark_inf.w=www;
   watermark_inf.b=b;
%    %%transforms
% h=MakeONFilter('Coiflet',2);
% Wav=@(t) FWT2_POE(t,3,h); %wavelet coeefficients of image
% inWav= @(t) IWT2_POE(t,3,h);
% Wav1=@(t) new_wavelet(t,Wav,S1,S2);
% inWav1=@(t) new_inwavelet(t,inWav,S1,S2);
% 
% param.redundant=param.N-N;
%     
%    
% %measurements
% phi = @(t) Noiselet([t;zeros(param.redundant,1)],omega);
% A   = @(t) Noiselet_inW(phi,inWav1,t);
% %F   = @(t) DHT(t,omega2);
% %FA  = @(t) DHT(A(t),omega2);
% 
% 
% %Adjoints
% phiT= @(t)  Adj_Noiselet(t,param.N,omega);
% AT  = @(t)  Adj_Noiselet_inW(phiT,Wav1,t,N);
% %FT  = @(t)  At_DHT( t,omega2,m );  % B
% %FAT = @(t)  Adj_Noiselet_inW(phiT,Wav1,FT(t));
% 
% % regularization parameter
% tau = 4;
% 
% % set tolA
%    % regularization parameter
% tau = 4;
% 
% % set tolA
% tolA = 1.e-7;
%             
% [~,x_tild1,~,~,~,~]= ...
%                 GPSR_BB(y1,A,tau,...
%                 'Debias',1,...
%                 'AT',AT,... 
%                 'Initialization',0,...
%                 'StopCriterion',1,...
%                 'ToleranceA',tolA,'ToleranceD',0.00001);
%             
%             %x_h=reshape(x_tild1,S1,S2);
%             s_hat_h=inWav1(x_tild1);
%             s_hat=reshape(s_hat_h,S1,S2);
%                  s_hat=s_hat + smean(i);
%                  sol(:,:,i) = s_hat;
mm = m + 30 - mod(m, 30);
tt=factor(mm);
ss2=tt(1)*tt(end);
ss1=prod(tt(2:end-1));

Encoded_Image=uint8(cat(3, reshape([y1; zeros(30-mod(m, 30), 1)],ss1,ss2),...,
    reshape([y2; zeros(30-mod(m, 30), 1)],ss1,ss2), reshape([y3; zeros(30-mod(m, 30), 1)],ss1,ss2)));
figure,subplot(1,3,1)
imshow(Encoded_Image,[]), ylabel('Encrypted and Compressed Signal');
drawnow; 
solImageA = userA(y_w, smean,param);
subplot(1,3,2)
imshow(solImageA,[]), title("User-A, Semi - Authorization");
drawnow;   
solImageB= userB(y_w,watermark_inf, smean,param);
subplot(1,3,3)
imshow(solImageB,[]), title("User-B, Full - Authorization");
drawnow;
sgtitle("Reversible Privacy Preservation using Multi-level Encryption and Compressive Sensing")
 function v =check_watermark(bw,in,www)
 %check watermark
        bw_in=DHT(bw,in);
        v = abs(bw_in(1));

        w= v*www;

        w_h=zeros(size(bw_in));
        w_h(find(bw_in>=0.1))=v*1;
        w_h(find(bw_in<-0.1))=v*-1;

        error_in= (sum((w_h-w)>10^-1));

        if error_in ~=0
            error('something wrong in watermark')
        end
        
end
  
function [solImage] = userA(y_w, smean,param)
S1=param.S1;
S2=param.S2;
N=S1*S2;
m=round(param.N*param.mratio);
M=param.M;
seed1=param.seed1;
seed2=param.seed2;

rng(seed1)
temp1=randperm(N);
omega=temp1(1:m);  % Pick up m measurements randomnly


p1=m-M./3;
    
    
rng(seed2)
temp2=randperm(m);
omega2=temp2(1:p1);

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


function [solImage] = userB(y_w,watermark_inf, smean,param)
S1=param.S1;
S2=param.S2;
N=S1*S2;
m=round(param.N*param.mratio);
M=param.M;
seed1=param.seed1;
seed2=param.seed2;

rng(seed1)
temp1=randperm(N);
omega=temp1(1:m);  % Pick up m measurements randomnly


p1=m-M./3;
    
    
rng(seed2)
temp2=randperm(m);
omega2=temp2(1:p1);

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
    
    
    % regularization parameters for GPSR solver
    tau = 4;
    % set tolA
    tolA = 1.e-7;

        parfor i=1:loop


        %%%% Coarse Estimation of sparse signal %%%%%%%%
        y_tild=F(y_w((i-1)*m+1:i*m));
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
        w_h(find(w_t>=0)) = watermark_inf.v1(i)*1;
        w_h(find(w_t<0)) = watermark_inf.v1(i)*-1;
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
    
    
    %%% Decode the mask region %%%
    
    masklength=bin2dec(num2str(www_h(1:9)));
    
    b{1} = zeros(masklength,2);
    
    temp=www_h(10:(masklength*9+9));
    temp=reshape(temp,masklength,9);
    
    b{1}(:,1)=bin2dec(num2str(temp))
    
    temp2=www_h((masklength*9+10):masklength*9*2+9);
    temp2=reshape(temp2,masklength,9);
    
    b{1}(:,2)=bin2dec(num2str(temp2));
    
    
    
    % find the boundaries and use them 
    mask2=false([S1, S2]);
    for i = 1:length(b)
        for j = 1:length(b{i})
            ind = b{i}(j,:);
            mask2(ind(1),ind(2))=1;
        end
    end
    
    %figure,imshow(mask2)
    
    inside2=bwfill(mask2,'holes');
    
    %figure,imshow(inside2)
    
    idx=find(inside2==1);
    
    www_h(www_h==0)=-1;
    
    D=zeros(S1,S2); D(idx)=www_h(masklength*9*2+10:masklength*9*2+9+length(idx));
    
   
    
    %%% Zero-out the unused bits
    
    www_h(masklength*9*2+10+length(idx):end)=0; 
    
    %%% Define the measurement system and its adjoint
    outside = (inside2-1).*(-1);
    phi_D = @(t) phi(outside(:).*t+D(:).*t);
    phiT_Dt = @(t) outside(:).*new_phi_T(t,phiT,N) +D(:).*new_phi_T(t,phiT,N);
    A_D   = @(t) Noiselet_inW(phi_D,inWav1,t);
    AT_Dt = @(t)  Adj_Noiselet_inW(phiT_Dt,Wav1,t,N);

    
    %%% Reconstruct the original image
    
    
    sol=zeros(S1,S2,3);

parfor i =1:loop
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


function x=new_wavelet(s,Wav,S1,S2)
    new_S1=ceil(log2(S1));

    new_S2=ceil(log2(S2));
    
    z_padded=zeros(2^new_S1,2^new_S2);

    z_padded(1:S1,1:S2)=reshape(s,S1,S2);

    x=Wav(z_padded);

end


function s=new_inwavelet(x,inWav,S1,S2)
    new_S1=ceil(log2(S1));
    new_S2=ceil(log2(S2));
    
    x_padded=reshape(x,2^new_S1,2^new_S2);
    
    s_padded=inWav(x_padded);
    
    S=s_padded(1:S1,1:S2);
    
    s=S(:);
end

function out=new_phi_T(t,phi_T,N)
    s_hat1=phi_T(t);
    out = s_hat1(1:N);
end






