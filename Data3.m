clear, clc
%read csv files

path1 = 'C:\Users\ahishalm\Local\pri_pre\new-data\youtube\YouTubeFaces\frame_images_DB\';
path2 = 'C:\Users\ahishalm\Local\pri_pre\mehmet_face_Tuesday\Results\measurement_';

txt_files = dir(fullfile(path1,'*txt'));
%png_files = dir(fullfile(path1,'*jpg'));
cordinates = importdata('C:\Users\ahishalm\Local\pri_pre\new-data\youtube\YouTubeFaces\frame_images_DB\Abdullah_Gul.labeled_faces.txt');
%S1=256;
%S2=256;
i=1;
param.N=2^16;

NoPersons=1595;
param.em_power=0.15; % 0.085
param.degradation='binary';
%param.degradation='gauss';%'blurring'; %'gauss';
param.m1=7; %if blurring is the option than this is filter size
param.m2=7;
param.matrix=0.9+0.05*randn(2^9,2^9);
param.dataset='TUT';
param.dataset='youtube';
measurement_rate = 6;
count=1;
param.y1 = [];
param.y2 = [];

param.x1 = [];
param.x2 = [];
rng(1)
for k=1:length(txt_files)
    cordinates = importdata(strcat(path1, txt_files(k).name));
    if isempty(cordinates) || length(cordinates.data) < 30
        continue
    end
    
    ind = randperm(length(cordinates.data), 30);
    NofFrames = length(ind);
    png_files = cordinates.textdata(ind);
    
    for i = 1:NofFrames
        performanceA(i) = struct('PSNR', zeros(1), 'PSNR_insidemask', zeros(1), 'PSNR_outsidemask', zeros(1));
        performanceB(i) = struct('PSNR', zeros(1), 'err', zeros(1), 'PSNR_insidemask', zeros(1), 'PSNR_outsidemask', zeros(1), 'fail', zeros(1), 'big', zeros(1));
    end

    switch param.dataset
        case 'TUT'
            y1 = cordinates(:,2);
            y2 = cordinates(:,4) - y1;
            x1 = cordinates(:,5);
            x2 = cordinates(:,3) - x1;
        case 'youtube'
            x1 = cordinates.data(:,2)+round(cordinates.data(:,4)./2);
            y1 = cordinates.data(:,3)+round(cordinates.data(:,5)./2);
            y2 = cordinates.data(:,3)-round(cordinates.data(:,5)./2);
            x2 = cordinates.data(:,2)-round(cordinates.data(:,4)./2);
    end
            
    x1 = x1(ind);
    x2 = x2(ind);
    y1 = y1(ind);
    y2 = y2(ind);
    x1(x1 < 1) = 1;
    x2(x2 < 1) = 1;
    y1(y1 < 1) = 1;
    y2(y2 < 1) = 1;
    parfor i = 1:NofFrames
        z = i;
        [performanceA(i), performanceB(i), x1(i), x2(i), y1(i), y2(i)] = processFrame(measurement_rate, x1(i), x2(i), y1(i), y2(i), z, ...
                            png_files{i}, performanceA(i), performanceB(i), path1, path2, param);
    end
  %k
  %save images and results
  splits = strsplit(png_files{1},'\');
  identity = splits{1};
  outDir = strcat(path2,int2str(measurement_rate), '\', identity);

  ptt = strcat(outDir,'_results.mat');
  save(ptt,'performanceA','performanceB','measurement_rate');
  ptt2 = strcat(outDir,'_labels.mat');
  save(ptt2,'png_files','x1','x2', 'y1', 'y2');
  drawnow;
%   free = java.io.File('C:\').getFreeSpace()*1e-9; % Bazen hata veriyor
%   if free < 25.0
%       break
%   end
end


function [performanceA, performanceB, x1, x2, y1, y2] = processFrame(measurement_rate, x1, x2, y1, y2, i, png_files, performanceA, performanceB, path1, path2,param)
    
    %param.em_power=0.085;  %0.065; %increase this if the results are bad
        
    param.mratio =measurement_rate./10;
    param.y1 = y1;
    param.y2 = y2;

    param.x1 = x1;
    param.x2 = x2;

   % param.N=256*256;

    param.type="rgb";
    
    %read images

    filename=png_files;
    imagename=strcat(path1,filename);
    I0 = imread(imagename);
    [S1_init,S2_init,~]=size(I0);
    I1= imresize(I0,sqrt((param.N)/(S1_init*S2_init)));
    param.S1=size(I1,1);
    param.S2=size(I1,2);
    %re-arange the coardinates of the faces
    param.x1 = round(x1.*((param.S1./S1_init)));
    param.x2 = round(x2.*((param.S1./S1_init)));
    
    param.y1 = round(y1.*((param.S2./S2_init)));
    param.y2 = round(y2.*((param.S2./S2_init)));
    %if (param.S2*param.S1)>param.N
        ll=0;
        while (param.S2*param.S1)>param.N
            if mod(ll,2)
                param.S1=param.S1-1;
                param.x1 = param.x1-1;
            elseif mod(ll,2)
                param.S2=param.S2-1;
                param.y1 = param.y1-1;
            end
                ll=ll+1;
        end
        
        if mod(param.S1,2)
            param.S1=param.S1-1;
            param.x1 = param.x1-1;
        end
        
        if mod(param.S2,2)
            param.S2=param.S2-1;
            param.x2 = param.x2-1;
        end
        
    %elseif (param.S2*param.S1)<param.N
        
        
    %end
        
    I =I1(1:param.S1,1:param.S2,:);   
    
    param.S1=size(I,1);
    param.S2=size(I,2);
    clear I0, clear I1;
    img1=(double(rgb2gray(I)));
    
    param.seed1=i+7;
    param.seed2=i+8;
    param.M=15003; %maximum leghts of the bits we can embed
    
    %re-arange the coardinates of the faces
    param.x1 = round(x1.*((param.S1./S1_init)));
    param.x2 = round(x2.*((param.S1./S1_init)));
    param.y1 = round(y1.*((param.S2./S2_init)));
    param.y2 = round(y2.*((param.S2./S2_init)));
    param.x1(param.x1 < 1) = 1;
    param.x2(param.x2 < 1) = 1;
    param.y1(param.y1 < 1) = 1;
    param.y2(param.y2 < 1) = 1;
    y1 = param.y1;
    y2 = param.y2;
    x1 = param.x1;
    x2 = param.x2;
    
    [mask,h]= create_mask(img1,param);
    if sum(sum(mask)) > 15000 || size(I, 1) ~= param.S1 || size(mask, 1) ~= param.S1 || size(I, 2) ~= param.S2 || size(mask, 2) ~= param.S2
        performanceB.big = 1;
        return
    end
    
%%transforms
h1=MakeONFilter('Coiflet',2);
Wav=@(t) FWT2_POE(t,3,h1); %wavelet coeefficients of image
inWav= @(t) IWT2_POE(t,3,h1);

[S1, S2] =size(img1);
%tic
x=new_wavelet(img1(:),Wav,S1,S2);
s1=new_inwavelet(x,inWav,S1,S2);
%toc
s_hat=reshape(s1,S1,S2);

SNR =measerr(img1,s_hat);

    if SNR<100
        disp("something wrong with wavelet")
    elseif SNR <50
        error(("choose wavelet wisely"))
    end
    
    
    [y_w,watermark_inf, smean]=transmitter(double(I), mask,param);

    outside = (mask-1).*(-1);
        
    
    % Write original frame
    splits = strsplit(filename,'\');
    pp1 = strcat(path2, int2str(measurement_rate), '\original\', splits{1}, '\', splits{2});
    pp2 = strcat(path2, int2str(measurement_rate), '\userA\', splits{1}, '\', splits{2});
    pp3 = strcat(path2, int2str(measurement_rate), '\userB\', splits{1}, '\', splits{2});
    if ~exist(pp1, 'dir')
       mkdir(pp1)
    end
   
    if ~exist(pp2, 'dir')
       mkdir(pp2)
    end
    
    if ~exist(pp3, 'dir')
       mkdir(pp3)
    end
    
    imwrite(I, strcat(path2,int2str(measurement_rate), '\original\', filename));
    
    %User A reconstruction
    [sol] = userA(y_w, smean,param);
    %figure,imshow((solImage),[])
    pth_personA= strcat(path2,int2str(measurement_rate),'/userA/',int2str(i),'.png');
    %imwrite(sol,pth_personA)
    imwrite(sol, strcat(path2,int2str(measurement_rate), '\userA\', filename));

    performanceA.PSNR=measerr((I),(sol));
    I1 = imcrop(I,h.Position);
    I2 = imcrop(sol,h.Position);
    performanceA.PSNR_insidemask=measerr(I1,I2);
    performanceA.PSNR_outsidemask=measerr((I.*uint8(outside)),(sol.*uint8(outside)));
    
    param.dataset='youtube';
    [sol, infor, fail] = userB(y_w,watermark_inf, smean, param);
    %figure,imshow((sol),[])
    pth_personB= strcat(path2,int2str(measurement_rate),'/userB/',int2str(i),'.png');
    %imwrite(sol,pth_personB)
    imwrite(sol, strcat(path2,int2str(measurement_rate), '\userB\', filename));

    [~,h]= create_mask(img1, param);
    I2 = imcrop(sol,h.Position);
    performanceB.fail = fail;
    performanceB.PSNR=measerr((I),(sol));
    performanceB.err = infor.total_error; 
    performanceB.PSNR_insidemask=measerr(I1,I2);
    performanceB.PSNR_outsidemask=measerr((I.*uint8(outside)),(sol.*uint8(outside)));

    %measurement_rate=10*param.mratio;
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
F   = @(t) DHT(t,omega2);
FA  = @(t) DHT(A(t),omega2);


%Adjoints
phiT= @(t)  Adj_Noiselet(t,param.N,omega);
AT  = @(t)  Adj_Noiselet_inW(phiT,Wav1,t,N);
FT  = @(t)  At_DHT( t,omega2,m );  % B
FAT = @(t)  Adj_Noiselet_inW(phiT,Wav1,FT(t));

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
for i=1:loop
    

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


function [solImage, infor, fail] = userB(y_w,watermark_inf, smean,param)
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


for i=1:loop
    

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
v = watermark_inf.v1(i);
new_y = y_w((i-1)*m+1:i*m) - A(x_tild1);
w_t=DHT(new_y,in);
w_h=zeros(size(w_t));
w_h(find(w_t>=0))=v*1;
w_h(find(w_t<0))=v*-1;

www_hat(:,i)=w_h;
end


%convert information into bits
w_hat=zeros(size(www_hat));        
w_hat(www_hat>0)=1;
w_hat(www_hat<0)=0;

k=1:3:M;
l=2:3:M;
d=3:3:M;
www_h(k)=w_hat(:,1);
www_h(l)=w_hat(:,2);
www_h(d)=w_hat(:,3);

pp.x1=bin2dec(num2str(www_h(1:8)));
pp.x2=bin2dec(num2str(www_h(9:16)));
pp.y1=bin2dec(num2str(www_h(17:24)));
pp.y2=bin2dec(num2str(www_h(25:32)));

fail = 0;
if (param.x1 ~= round(pp.x1)) || (param.x2 ~= round(pp.x2)) || (param.y1 ~= round(pp.y1)) || (param.y2 ~= round(pp.y2))
    fail = 1;
    pp.x1 = param.x1;
    pp.x2 = param.x2;
    pp.y1 = param.y1;
    pp.y2 = param.y2;
end
s_hat1=inWav1(x_tild1(:));
s_hat1=reshape(s_hat1,S1,S2);
s_hat=s_hat1 + smean(3);

mask =create_mask(s_hat,param);

area_mask=sum(sum(mask));

www_h(www_h==0)=-1;
tmp=www_h(33:33+area_mask-1);

inside=zeros(S1,S2);


inside(mask == 1) = tmp;

D=watermark_inf.D;
infor.total_error=sum(sum(D~=inside));
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

function out =new_A(A,t,N,m )
out(1:m) = A(t(1:N));
out(m+1:2*m)=A(t(N+1:2*N));
out(2*m+1:3*m)=A(t(2*N+1:3*N));
end

function out = new_AT(AT,t,N,m)
out(1:N)=AT(t(1:m));
out(N+1:2*N)=AT(t(m+1:2*m));
out(2*N+1:3*N)=AT(t(2*m+1:3*m));
end

function [y_w,watermark_inf, smean] =transmitter(img, mask,param)
    
    M=param.M;
    size3=size(img,3);
    
    %%degration%%

    %Mask
    pr=0.5;
    MM=(rand(param.S1,param.S2)<pr);
    MM=MM*2 - 1;

    D=mask.*MM;   % D(j,k)= {-1,1}, bunu gauss yapalım ya da deblurring
    D2=(mask.*MM).*param.matrix(1:param.S1,1:param.S2);
    d= D(:);
    %message 1
    inx=find(mask==1);
    watermark_inf.D=D;

    %create watermark for cordinates.
    w_c =double(dec2bin([param.x1; param.x2; param.y1; param.y2;255]))-48;
    w_c(5,:) = [];
    w_c(w_c==0)=-1;
    %create watermark for mask. 
    w_m = d(inx);
    w_c=w_c';
    ww = [w_c(:);w_m];
    
    
    %M=size(ww,1);
    %create watermark for seeds. 
    seed1=param.seed1;
    seed2=param.seed2;
    
    w_si =double(dec2bin([seed1; seed2; M]))-48;
    
    w_s=[w_si(1,:)' ; w_si(2,:)'];
    w_s(w_s==0)=-1;
    M2 = length(w_s);
    
    %concatanate all watermark
    www=zeros(M,1);
    www(1:length(ww)) =ww;
    %www(end-length(w_s)+1:end)=w_s;
    
    %split the watermark data for each channel
    k=1:3:M;
    l=2:3:M;
    d=3:3:M;
    
    www1=www(k);
    www2=www(l);
    www3=www(d);
    
    N=param.S1*param.S2;
    m=round(param.mratio*param.N); %number of measurements for CS (each channel)
     
    
    if size3==3
       s1= double(img(:,:,1));
       s2= double(img(:,:,2));
       s3 =double(img(:,:,3));
       S =[s1(:) s2(:) s3(:)];
    elseif size3==1
        S=double(img(:));
    else
        disp("image should be rgb or grayscale")
    end
    %S=double(img(:));
    smean=mean(S);
    S=S-smean;
    
    [S1,S2]=size(s1);
    %seed=10;
    rng(seed1)
    temp1=randperm(N);
    omega=temp1(1:m);  % Pick up m measurements randomnly
    
    
    p1=m-M./3;
    
    
    rng(seed2)
    temp2=randperm(m);
    omega2=temp2(1:p1);
    
   

    m1=param.m1;%=3;
    m2=param.m2;%3;
    P=1/(m1*m2)*ones(m1,m2); % uniform 3 ? 3 blur
    
    outside = (mask-1).*(-1);
    
    param.redundant=param.N-N;
    %measurements
    phi = @(t) Noiselet([t;zeros(param.redundant,1)],omega);
    %A   = @(t) Noiselet_inW(phi,inWav,t,S1,S2);
    %F   = @(t) DHT(t,omega2);
    %FA  = @(t) DHT(A(t),omega2);
       %Adjoints
    phiT= @(t)  Adj_Noiselet(t,N,omega);
    %AT  = @(t)  Adj_Noiselet_inW(phiT,Wav,t,S1,S2);
    %FT  = @(t)  At_DHT( t,omega2,m );
    %FAT = @(t)  Adj_Noiselet_inW(phiT,Wav,FT(t),S1,S2);
    %phi_D = @(t) (phi(t)+phi(d.*t));
    switch param.degradation
        case 'blurring'
          phi_D = @(t) phi(outside(:).*t+mask(:).*blur(t,P,S1,S2));
          %phiT_Dt = @(t) outside(:).*phiT(t) +D(:).*phiT(t);   
        case 'binary'
            phi_D = @(t) phi(outside(:).*t+D(:).*t);
            %phiT_Dt = @(t) outside(:).*phiT(t) +D(:).*phiT(t);
        case 'gauss'
            phi_D = @(t) phi(outside(:).*t+D2(:).*t);
            %phiT_Dt = @(t) outside(:).*phiT(t) +D2(:).*phiT(t);
    end
    %phi_D = @(t) phi(outside(:).*t+D(:).*t);
    %A_D   = @(t) Noiselet_inW(phi_D,inWav,t,S1,S2);
 

    %phiT_Dt = @(t) (outside(:).*phiT(t)+d.*phiT(t));
    %phiT_Dt = @(t) outside(:).*phiT(t) +D(:).*phiT(t);  
    %AT_Dt = @(t)  Adj_Noiselet_inW(phiT_Dt,Wav,t,S1,S2);

    
    %take measurements;
    if size3==3
        y1= phi_D(S(:,1));
        y2= phi_D(S(:,2));
        y3= phi_D(S(:,3));
        y=[y1;y2;y3];
    else
        y= phi_D(S);
    end

    %Construct Encoding Matrix
    
%     if size3==3
%         temp2=ones(param.m1,1);
%         temp2(omega2)=0;
%         in=find(temp2==1);
%         bw=At_DHT(www,in,3*m);
%         
%     else
%         temp2=ones(3*m,1);
%         temp2(omega2)=0;
%         in=find(temp2==1);
%         bw=At_DHT(www,in,m);
%     end
    temp2=ones(m,1);
    temp2(omega2)=0;
    in=find(temp2==1);
    
    if size3==3
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
    
    
    
    
    
    
    %Construct Second 
    %F2=randn(m,length(w_s));
    %F2=orth(F2);
   
    
%     bw2=B2*w_s;
%     
%     alpha2=norm(y)*.065;
%     bw2=(bw2./norm(bw2)).*alpha;
%     
%     %check watermark
%     bw_in=B2'*bw2;
%     v = abs(bw_in(1));
% 
%     w= v*w_s;
% 
%     w_h=zeros(size(bw_in));
%     w_h(find(bw_in>=0.1))=v*1;
%     w_h(find(bw_in<-0.1))=v*-1;
% 
%     error_in= (sum((w_h-w)>10^-1))
%     if error_in ~=0
%         error('something wrong in watermark')
%     end
%     watermark_inf.v2=v;
%     
%     %watermark the measurements
%     y_w=zeros(size(y));
%     y_w(1:param.m1) = y(1:param.m1) + bw;
%     y_w(param.m1+1:end)=y(param.m1+1:end) + bw2;
    


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


function y=blur(t,P,S1,S2)
    X =reshape(t,S1,S2);
    Y=imfilter(X,P,'symmetric');
    y=Y(:);

end

function  [mask, h]=create_mask(I,param)

%figure(88), imshow(I,[])
%h = drawrectangle('Position',[param.x1,param.y1,param.x2,param.y2],'StripeColor','r');
%mask = createMask(h);
mask = zeros(size(I));
switch param.dataset
    case 'TUT'
        mask(param.y1 + 1:param.y2 + param.y1 + 1, param.x1 + 1:param.x1 + param.x2 + 1) = 1;
        h.Position = [param.x1,param.y1,param.x2,param.y2];
    case 'youtube'
        mask(param.y2:param.y1, param.x2:param.x1) = 1;
        h.Position =[param.x2 param.y2 param.x1-param.x2 param.y1-param.y2];
end

%imwrite(uint8(I), 'mask1.png')
end