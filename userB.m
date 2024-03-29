function [solImage, infor, fail] = userB(y_w,watermark_inf, smean,param)

%%%%%%%%%%%%%%%
% This function takes the compressed + encripted signal, and the parameters
% of the framework as the input, and it reconstrcts the signal s for User-B
%%%%%%%%%%%%%%%

    S1 = param.S1; % Image dimensions.
    S2 = param.S2;
    N = S1*S2; % Total signal size per channel.
    m = round(param.N*param.mratio); % Compressed signal size per channel.
    M = param.M; % Max. length of the bits for embedding the watermark.
    
    % Transforms
    h=MakeONFilter('Coiflet',2);
    Wav=@(t) FWT2_POE(t,3,h); % Wavelet coeefficients of image.
    inWav= @(t) IWT2_POE(t,3,h);
    Wav1=@(t) wavelet(t,Wav,S1,S2);
    inWav1=@(t) inwavelet(t,inWav,S1,S2);

    % Measurements
    % Encoding matrix (Measurement matrix) for the signal s.
    rng(1)
    temp1 = randperm(N);
    omega = temp1(1:m);  % Pick up m measurements randomnly
    param.redundant=param.N-N; % How much we have left for the max. signal length per channel.
    phi = @(t) Noiselet([t;zeros(param.redundant,1)],omega);
    A   = @(t) Noiselet_inW(phi,inWav1,t);
    
    % Encoding for the watermark.
    rng(2)
    temp2 = randperm(m);
    p1 = m-M./3;
    omega2 = temp2(1:p1);
    F   = @(t) DHT(t,omega2);
    FA  = @(t) DHT(A(t),omega2);


    % Adjoints of the above matrices.
    phiT= @(t)  Adj_Noiselet(t,param.N,omega);
    AT  = @(t)  Adj_Noiselet_inW(phiT,Wav1,t,N);
    FT  = @(t)  At_DHT( t,omega2,m );  % B
    FAT = @(t)  Adj_Noiselet_inW(phiT,Wav1,FT(t),N);

    temp2=ones(m,1);
    temp2(omega2)=0;
    in=find(temp2==1);

    %%%%  Decoding Part %%%%%%%%
    % Regularization parameter
    tau = 4;
    % Set tolA
    tolA = 1.e-7;
    
    for i=1:3 % Reconstruction of the watermark.

        y_tild = F(y_w((i-1)*m+1:i*m));
        % First estimation of x:
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
        w_t = DHT(new_y,in);
        w_h = zeros(size(w_t));
        w_h(w_t >= 0) = v*1;
        w_h(w_t < 0) = v*-1;

        www_hat(:,i) = w_h;
    end

    % Convert the recovered watermark information into bits
    w_hat = zeros(size(www_hat));        
    w_hat(www_hat>0) = 1;
    w_hat(www_hat<0) = 0;

    % Collect the watermark information from each channel.
    k=1:3:M;
    l=2:3:M;
    d=3:3:M;
    www_h(k)=w_hat(:,1);
    www_h(l)=w_hat(:,2);
    www_h(d)=w_hat(:,3);

    % Recover the mask region, this is for the demo script.
    if isempty(param.x1)
        masklength=bin2dec(num2str(www_h(1:9)));
        b{1} = zeros(masklength,2);
    
        temp = www_h(10:(masklength*9+9));
        temp = reshape(temp,masklength,9);

        b{1}(:,1) = bin2dec(num2str(temp));

        temp2 = www_h((masklength*9+10):masklength*9*2+9);
        temp2 = reshape(temp2,masklength,9);

        b{1}(:,2) = bin2dec(num2str(temp2));

        fail = 0;
        if ~isequal(b, param.b)
            fail = 1; % The recovery of the privacy sensitive locations is failed.
            if param.emLOC == false % Check if the flag is set to use original ground-truth instead.
                b = param.b;
            end
        end
        
        % Find the boundaries and use them .
        mask2 = false([param.S1, param.S2]);
        for i = 1:length(b)
            for j = 1:length(b{i})
                ind = b{i}(j,:);
                mask2(ind(1),ind(2))=1;
            end
        end
        
        % Obtain M and calculate the error.
        inside = bwfill(mask2,'holes');
        outside = (inside-1).*(-1);
        idx = find(inside == 1);
        www_h(www_h == 0)=-1;

        D=zeros(S1,S2); % Binary mask.
        D(idx)=www_h(masklength*9*2+10:masklength*9*2+9+length(idx));
        D2 = D.*param.matrix(1:param.S1,1:param.S2); % Binary masked Gaussian degradation.
        infor.total_error = sum(sum(D ~= watermark_inf.D));
        
        %%% Zero-out the unused bits
        aa = round((masklength*9*2+10+length(idx)-1)/3);
        www_hat((aa+1):end, :) = 0; % Zero-out the unused bits.

    else
        % Recover face locations. this is for benchmarking over YouTube dataset.
        pp.x1=bin2dec(num2str(www_h(1:8)));
        pp.x2=bin2dec(num2str(www_h(9:16)));
        pp.y1=bin2dec(num2str(www_h(17:24)));
        pp.y2=bin2dec(num2str(www_h(25:32)));

        fail = 0;
        if (param.x1 ~= round(pp.x1)) || (param.x2 ~= round(pp.x2)) || (param.y1 ~= round(pp.y1)) || (param.y2 ~= round(pp.y2))
            fail = 1; % The recovery of the face locations is failed.
            if param.emLOC == false % Check if the flag is set to use original ground-truth instead.
                pp.x1 = param.x1;
                pp.x2 = param.x2;
                pp.y1 = param.y1;
                pp.y2 = param.y2;
            end
        end

        s_hat1 = inWav1(x_tild1(:)); % x -> s
        s_hat1 = reshape(s_hat1,S1,S2);
        s_hat = s_hat1 + smean(3); % Add substracted mean.

        % Create the mask for the privacy sensitive pixels.
        mask = zeros(size(s_hat));
        mask(pp.y2:pp.y1, pp.x2:pp.x1) = 1;

        % Obtain M and calculate the error.
        area_mask = sum(sum(mask));
        www_h(www_h == 0) = -1;
        tmp = www_h(33:33+area_mask-1);
        inside = zeros(S1,S2);
        inside(mask == 1) = tmp;
        D = watermark_inf.D;
        infor.total_error = sum(sum(D~=inside));
        D = inside; % Binary mask.
        D2 = D.*param.matrix(1:param.S1,1:param.S2); % Binary masked Gaussian degradation
        outside = (mask-1).*(-1);


        aa = (33+area_mask-1)./3;

        www_hat((aa+1):end,:) = 0; % Zero-out the unused bits.
    end
    
    switch param.degradation
        case 'binary'
            phi_D = @(t) phi(outside(:).*t+D(:).*t);
            phiT_Dt = @(t) outside(:).*new_phi_T(t,phiT,N) +D(:).*new_phi_T(t,phiT,N);
        case 'gauss'
            phi_D = @(t) phi(outside(:).*t+D2(:).*t);
            phiT_Dt = @(t) outside(:).*new_phi_T(t,phiT,N) +D2(:).*new_phi_T(t,phiT,N);
    end


    A_D   = @(t) Noiselet_inW(phi_D,inWav1,t);
    AT_Dt = @(t)  Adj_Noiselet_inW(phiT_Dt,Wav1,t,N);

    sol=zeros(S1,S2,3);
    % Regularization parameter
    tau = 4;
    % Set tolA
    tolA = 1.e-5;
    % Final estimation of x to form s,
    for i =1:3 % Estimate s for each channel.
        newy2 = y_w((i-1)*m+1:i*m) - At_DHT(www_hat(:,i),in,m); 
        [~,x_debias3,~,~,~,~]= ...
                    GPSR_BB(newy2,A_D,tau,...
                    'Debias',1,...
                    'AT',AT_Dt,... 
                    'Initialization',2,...
                    'StopCriterion',1,...
                    'ToleranceA',tolA,'ToleranceD',0.0001);

         s_hat_h=inWav1(x_debias3); % Inverse wavelet to compute s from x.
         s_hat=reshape(s_hat_h,S1,S2);

         s_hat=s_hat + smean(i); % Add substracted mean from the transmitter.
         sol(:,:,i) = s_hat;

    end

    solImage = uint8(cat(3, sol(:,:,1), sol(:,:,2), sol(:,:,3))); % Collect reconstructed signal for each channel.
end

function out = new_phi_T(t,phi_T,N)
    s_hat1=phi_T(t);
    out = s_hat1(1:N);
end