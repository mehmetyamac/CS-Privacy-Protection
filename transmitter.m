function [y_w,watermark_inf, smean] = transmitter(img, mask, param, i)

%%%%%%%%%%%%%%%
% This function takes the input image, mask for the privacy sensitive parts
% and the required parameters for the algorith, and it produces encripted
% and compressed signal as the output: y_w = y_d + Bw = (A + M)s + Bw.
%%%%%%%%%%%%%%%

    M = param.M;
    
    %%Degration%%

    % Mask of the privacy sensitive part.
    pr = 0.5; % Privacy preservation costant, the probability of being corrupted pixel in the privacy sensitive part.
    rng(i); % Since  a different  randomized  corruption  matrix  is  employed  for  eachframe.
    MM = (rand(param.S1,param.S2) < pr);
    MM = MM*2 - 1;

    D = mask.*MM; % Binary corruption of the mask.
    D2 = (mask.*MM).*param.matrix(1:param.S1,1:param.S2); % Binary + Gaussian corruption of the mask.
    % Message 1
    watermark_inf.D = D;

    % Create watermark for cordinates.
    w_c = double(dec2bin([param.x1; param.x2; param.y1; param.y2;255])) - 48;
    w_c(5,:) = [];
    w_c(w_c == 0) = -1;
    % Create watermark for mask.
    d = D(:);
    inx = find(mask == 1);
    w_m = d(inx);
    w_c = w_c';
    ww = [w_c(:); w_m];
    
    % Concatanate coor and mask watermarks.
    www=zeros(M,1);
    www(1:length(ww)) =ww;
    
    % Split the watermark data for each channel.
    k = 1:3:M;
    l = 2:3:M;
    d = 3:3:M;
    www1 = www(k);
    www2 = www(l);
    www3 = www(d);
    
    N = param.S1 * param.S2;
    m = round(param.mratio * param.N); % Number of measurements for CS (each channel)
    
    % Seperete the image into channels.
    s1 = double(img(:,:,1));
    s2 = double(img(:,:,2));
    s3 = double(img(:,:,3));
    S = [s1(:) s2(:) s3(:)];
    
    smean = mean(S); % Mean normalization.
    S = S - smean;
    
    % Encoding matrix for the signal s
    param.redundant = param.N - N; % How much we have left for the max. signal length per channel.
    rng(1)
    temp1= randperm(N);
    omega = temp1(1:m);  % Pick up m measurements randomly.
    phi = @(t) Noiselet([t;zeros(param.redundant,1)],omega);
    % Adjoints
    phiT= @(t)  Adj_Noiselet(t,N,omega);
    % Corrupted encoding matrix for the signal s: (A + M)
    outside = (mask-1).*(-1); % Outside region.
    switch param.degradation
        case 'binary'
            phi_D = @(t) phi(outside(:).*t + D(:).*t);
        case 'gauss'
            phi_D = @(t) phi(outside(:).*t + D2(:).*t);
    end
    
    % Construct Encoding for the watermark.
    rng(2)
    temp2 = randperm(m);
    p1 = round(m-M./3);
    omega2 = temp2(1:p1);  % Pick up m measurements randomly.
    temp2 = ones(m, 1);
    temp2(omega2) = 0;
    in = find(temp2 == 1);
    
    % Take measurements from the signal for each color channel;
    y1 = phi_D(S(:,1));
    y2 = phi_D(S(:,2));
    y3 = phi_D(S(:,3));
    y = [y1; y2; y3];
    
    bw1 = At_DHT(www1,in,m); % Encoding the watermark, B*w
    bw2 = At_DHT(www2,in,m); % Encoding the watermark, B*w
    bw3 = At_DHT(www3,in,m); % Encoding the watermark, B*w
    % Normalization
    bw1 = bw1./norm(bw1);
    bw2 = bw2./norm(bw2);
    bw3 = bw3./norm(bw3);
        
    % Embedding power = Bw/y_d for ensuring good reconstruction quality for the User-A.
    alpha1 = norm(y1).*(param.em_power);
    bw1 = bw1.*alpha1;
    alpha2 = norm(y2).*(param.em_power);
    bw2 = bw2.*alpha2;
    alpha3 = norm(y3).*(param.em_power);
    bw3 = bw3.*alpha3;
        
    watermark_inf.v1(1) = check_watermark(bw1,in,www1);
    watermark_inf.v1(2) = check_watermark(bw2,in,www2);
    watermark_inf.v1(3) = check_watermark(bw3,in,www3);
        
    y_w = [y1+bw1; y2+bw2; y3+bw3]; % Obtaining final copressed and encripted signal, y_w = y_d + Bw
        
end

function v =check_watermark(bw,in,www)
    % Check watermark
    bw_in = DHT(bw,in);
    v = abs(bw_in(1));

    w = v*www;

    w_h=zeros(size(bw_in));
    w_h(bw_in >= 0.1) = v*1;
    w_h(bw_in <- 0.1) = v*-1;

    error_in= (sum((w_h-w)>10^-1));

    if error_in ~=0
        error('something wrong in watermark')
    end
        
end