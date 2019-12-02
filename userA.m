function [solImage] = userA(y_w, smean, param)

%%%%%%%%%%%%%%%
% This function takes the compressed + encripted signal, and the parameters
% of the framework as the input, and it reconstrcts the signal s for User-A
%%%%%%%%%%%%%%%

    S1 = param.S1; % Image dimensions
    S2 = param.S2;
    N = S1*S2; % Total signal size per channel.
    m = round(param.N * param.mratio); % Compressed signal size per channel.
    M = param.M; % Max. length of the bits for embedding the watermark.

    % Transforms
    h = MakeONFilter('Coiflet',2);
    Wav = @(t) FWT2_POE(t,3,h); % Wavelet coeefficients of image.
    inWav = @(t) IWT2_POE(t,3,h);
    Wav1=@(t) wavelet(t,Wav,S1,S2);
    inWav1=@(t) inwavelet(t,inWav,S1,S2);

    
    % Measurements
    % Encoding matrix (Measurement matrix) for the signal s.
    rng(1)
    temp1 = randperm(N);
    omega = temp1(1:m);  % Pick up m measurements randomnly
    param.redundant = param.N-N; % How much we have left for the max. signal length per channel.
    phi = @(t) Noiselet([t;zeros(param.redundant,1)],omega);
    A   = @(t) Noiselet_inW(phi,inWav1,t);
    
    % Encoding for the watermark.
    rng(2)
    temp2 = randperm(m);
    p1 = m - M./3;
    omega2 = temp2(1:p1);
    F   = @(t) DHT(t,omega2);
    FA  = @(t) DHT(A(t),omega2);


    % Adjoints of the above matrices
    phiT= @(t)  Adj_Noiselet(t,param.N,omega);
    AT  = @(t)  Adj_Noiselet_inW(phiT,Wav1,t,N);
    FT  = @(t)  At_DHT( t,omega2,m );  % B
    FAT = @(t)  Adj_Noiselet_inW(phiT,Wav1,FT(t));
    
    %%%%   Decoding Part  %%%%%%%%
    % Regularization parameter
    tau = 4;
    % Set tolA
    tolA = 1.e-7;
    
    sol=zeros(S1,S2,3);
    for i = 1:3 % Estimate s for each channel.
        y_tild = y_w((i-1)*m+1:i*m);
       

        [~,x_tild1,~,~,~,~]= ...
                        GPSR_BB(y_tild,A,tau,...
                        'Debias',1,...
                        'AT',AT,... 
                        'Initialization',0,...
                        'StopCriterion',1,...
                        'ToleranceA',tolA,'ToleranceD',0.00001);

        s_hat_h = inWav1(x_tild1); % Inverse wavelet to compute s from x.
        s_hat = reshape(s_hat_h, S1, S2);
        s_hat = s_hat + smean(i); % Add substracted mean from the transmitter.
        sol(:,:,i) = s_hat;

    end
    solImage = uint8(cat(3, sol(:,:,1), sol(:,:,2), sol(:,:,3))); % Collect reconstructed signal for each channel.

end

