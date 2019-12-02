%%% Include Dependicies %%%
addpath(genpath('Measurements')) % Noiselet measurement mex file.
addpath(genpath('Wavelab850')) % https://statweb.stanford.edu/~wavelab/
%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('utils'))

numWorkers = 8; % Desired # of parallel workers.
%parpool(numWorkers);
rng(1);

path1 = 'C:\Users\ahishalm\Desktop\projects\pri_pre\YouTubeFaces\frame_images_DB'; % Dataset folder.
path2 = 'C:\Users\ahishalm\Desktop\projects\pri_pre\Results\measurement_'; % Results folder

txt_files = dir(fullfile(path1,'*txt'));

param.N = 2^16; % Max. length of the signal.

param.em_power = 0.15; % 0.085 % Embedding power, see the defination in the paper.
param.degradation = 'binary'; % 'binary' or 'gauss'
param.matrix = 0.9 + 0.1*randn(2^9, 2^9); % Gaussian matrix for binary masked Gaussian degradation option.
measurement_rate = 6; % MR = measurement_rate/10;
param.mratio =measurement_rate./10;
param.M = 30003; % Maximum leghts of the bits for embeding.

% Embed Face flag; True: use the reconstructed face locations, or
% False: use the ground-truth face locations in Type - B decoder.
param.emLOC = false;

% If the face region is larger compared to the image dimensions, the
% recovery of Type-II decoder decreases significantly. Hence, an option is
% provided to exlude frames with large faces from the performance evaluations.
param.bigInclude = 1; % Choose 1 to include or 0 to exclude.

IDcount = 0;
NoPersons = 100; % # of identities for the evaluation.
NofFrames = 30; % # of frames per identity for the evaluation.

param.y1 = [];
param.y2 = [];

param.x1 = [];
param.x2 = [];

for k=1:length(txt_files)
    cordinates = importdata(fullfile(path1, txt_files(k).name));
    % Search for identities that have at least NofFrames + 20 samples (20 samples for the face recognition trainset).
    if isempty(cordinates) || length(cordinates.data) < NofFrames + 20
        continue
    end
    IDcount = IDcount + 1;
    ind = randperm(length(cordinates.data), NofFrames + 20); % Randomly select samples for this identity.
    png_files = cordinates.textdata(ind); % Sample image names.
    
    % Arrange face coordinates.
    x1 = cordinates.data(:,2)+round(cordinates.data(:,4)./2);
    y1 = cordinates.data(:,3)+round(cordinates.data(:,5)./2);
    y2 = cordinates.data(:,3)-round(cordinates.data(:,5)./2);
    x2 = cordinates.data(:,2)-round(cordinates.data(:,4)./2);        
    x1 = x1(ind); % Coors. for randomly selected samples.
    x2 = x2(ind);
    y1 = y1(ind);
    y2 = y2(ind);
    x1(x1 < 1) = 1;
    x2(x2 < 1) = 1;
    y1(y1 < 1) = 1;
    y2(y2 < 1) = 1;
    
    for i = 1:NofFrames % Allocate structs for the performance evaluations for User-A and User-B.
        performanceA(i) = struct('PSNR', zeros(1), 'PSNR_insidemask', zeros(1), 'ssimVal', zeros(1), 'PSNR_outsidemask', zeros(1));
        performanceB(i) = struct('PSNR', zeros(1), 'err', zeros(1), 'PSNR_insidemask', zeros(1), 'ssimVal', zeros(1), 'PSNR_outsidemask', zeros(1), 'fail', zeros(1), 'big', zeros(1));
    end
    
    parfor i = 1:NofFrames % Process frames
        z = i;
        [performanceA(i), performanceB(i)] = processFrame(measurement_rate, x1(i), x2(i), y1(i), y2(i), z, ...
                            png_files{i}, performanceA(i), performanceB(i), path1, path2, param);
    end
    
    %%% Save results. %%%
    splits = strsplit(png_files{1},'\');
    identity = splits{1};
    outDir = fullfile(strcat(path2, int2str(measurement_rate)), identity);

    ptt = strcat(outDir,'_results.mat');
    save(ptt,'performanceA','performanceB','measurement_rate');
    drawnow;
   
    %%% Attack Type - I: Known Plain-text (original faces), known labels.%%%
    %%% Save 20 non-overlapping training samples for the face recognition. %%%
    trDir = fullfile(strcat(path2, int2str(measurement_rate)), 'train_facesAttackI', identity); % Directory for saving the train face images.
    if ~exist(trDir, 'dir'); mkdir(trDir); end
    
    for ii = NofFrames + 1:NofFrames + 20
        % Read the image and crop the face part.
        if ispc % If Windows OS.
            splits = strsplit(png_files{ii}, '\');
        else % Other OS.
            png_files{ii} = strrep(png_files{ii}, '\', '/');
            splits = strsplit(png_files{ii}, '/');
        end
        imagename = fullfile(path1, png_files{ii});
        I0 = imread(imagename);
        face = I0(y2(ii):y1(ii), x2(ii):x1(ii), :);
        
        imwrite(face, fullfile(trDir, splits{3})) % Write the train sample.
    end
   
    %%% Attack Type - II: Known Plain-text (original faces), known anonymized and their labels.%%%
    %%% Save 20 non-overlapping original faces + 10 Anonymized faces for the training of the face recognition. %%%
    trDir2 = fullfile(strcat(path2, int2str(measurement_rate)), 'train_facesAttackII', identity); % Directory for saving the train face images.
    UserADir = fullfile(strcat(path2, int2str(measurement_rate)), 'userA_faces', identity); % User-A directory (anonymized faces).
    newUserADir = fullfile(strcat(path2, int2str(measurement_rate)), 'userA_faces_AttackII', identity); % New User-A directory.
    if ~exist(trDir2, 'dir'); mkdir(trDir2); end
    if ~exist(newUserADir, 'dir'); mkdir(newUserADir); end
    
    copyfile(trDir, trDir2); % Copy the original faces.
    copyfile(UserADir, newUserADir); % Copy the original faces.
    
    for ii = 1:10 % Include 10 anonymized face samples to the query of face recognition.
        if ispc % If Windows OS.
            splits = strsplit(png_files{ii}, '\');
        else % Other OS.
            png_files{ii} = strrep(png_files{ii}, '\', '/');
            splits = strsplit(png_files{ii}, '/');
        end
        imagename = fullfile(newUserADir, splits{3});
        movefile(imagename, fullfile(trDir2, splits{3}));
    end
   
   if IDcount >= NoPersons % Check how many identities are processed.
       fprintf("Run over %d identities is completed, results are saved to %s \n\n", IDcount, strcat(path2, int2str(measurement_rate)))
       analyse(strcat(path2, int2str(measurement_rate)), param.bigInclude); % Calculate the average PSNRs, SSIMs over identities
       break
   end
end


function [performanceA, performanceB] = processFrame(measurement_rate, x1, x2, y1, y2, i, filename, performanceA, performanceB, path1, path2,param)
    
    if ~ispc % If not Windows OS.
        filename = strrep(filename, '\', '/');
    end

    param.y1 = y1;
    param.y2 = y2;
    param.x1 = x1;
    param.x2 = x2;

    % Read the image and crop or add some pixels to make sure it has width
    % and height with power of 2 for wavelet transform.
    imagename = fullfile(path1, filename);
    I0 = imread(imagename);
    [S1_init, S2_init, ~]=size(I0);
    I1 = imresize(I0,sqrt((param.N)/(S1_init*S2_init)));
    param.S1 = size(I1,1);
    param.S2 = size(I1,2);
    % Re-arange the coardinates of the faces
    param.x1 = round(x1.*((param.S1./S1_init)));
    param.x2 = round(x2.*((param.S1./S1_init)));
    param.y1 = round(y1.*((param.S2./S2_init)));
    param.y2 = round(y2.*((param.S2./S2_init)));
    ll=0;
    while (param.S2*param.S1)>param.N
        if mod(ll,2)
            param.S1 = param.S1-1;
            param.x1 = param.x1-1;
        elseif mod(ll,2)
            param.S2 = param.S2-1;
            param.y1 = param.y1-1;
        end
        ll = ll+1;
    end
    
    if mod(param.S1,2)
        param.S1=param.S1-1;
        param.x1 = param.x1-1;
    end
        
    if mod(param.S2,2)
        param.S2=param.S2-1;
        param.x2 = param.x2-1;
    end
        
        
    I = I1(1:param.S1, 1:param.S2, :); % New cropped image.
 
    clear I0, clear I1;
    img1 = double(rgb2gray(I));
    
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
    
    % Create mask for the face.
    mask = zeros(size(img1));
    mask(param.y2:param.y1, param.x2:param.x1) = 1;
    h.Position = [param.x2 param.y2 param.x1-param.x2 param.y1-param.y2];
    
    if sum(mask(:)) > (param.M - 3)/2 || size(I, 1) ~= param.S1 || size(mask, 1) ~= param.S1 || size(I, 2) ~= param.S2 || size(mask, 2) ~= param.S2
        performanceB.big = 1; % Too big face part.
        return
    end


    [y_w, watermark_inf, smean]=transmitter(double(I), mask, param, i); % Transmitter implementation for compression + encription.

    % User-A reconstruction
    [solA] = userA(y_w, smean,param);
        
    % User-B reconstruction
    [solB, infor, fail] = userB(y_w, watermark_inf, smean, param);
    
    %%% Performance Evaluations %%%:
    % User-A
    performanceA.PSNR = measerr((I),(solA)); % PSNR of the whole frame.
    I1 = imcrop(I, h.Position);
    I2 = imcrop(solA, h.Position);
    outside = (mask-1).*(-1);
    performanceA.PSNR_insidemask = measerr(I1,I2); % PSNR of the privacy sensitive part.
    performanceA.ssimVal = ssim(I1, I2); % SSIM of the privacy sensitive part.
    performanceA.PSNR_outsidemask = measerr((I.*uint8(outside)), (solA.*uint8(outside))); % PSNR of the outside of the privacy sensitive part.
    
    % User-B
    performanceB.fail = fail; % Check if decoding of the face coordinates is failed.
    performanceB.PSNR = measerr((I),(solB)); % PSNR of the whole frame.
    performanceB.err = infor.total_error;  % Total error.
    I2 = imcrop(solB,h.Position); % Crop the face.
    performanceB.PSNR_insidemask = measerr(I1,I2); % PSNR of the privacy sensite part.
    performanceB.ssimVal = ssim(I1, I2); % SSIM of the privacy sensitive part.
    performanceB.PSNR_outsidemask = measerr((I.*uint8(outside)), (solB.*uint8(outside))); % PSNR of the outside of the privacy sensitive part.
    
    %%% Directories for saving original, User-A and User-B frames%%%
    if ispc % If Windows OS.
        splits = strsplit(filename, '\');
    else
        splits = strsplit(filename, '/');
    end
    pp1 = fullfile(strcat(path2, int2str(measurement_rate)), 'original', splits{1});
    pp2 = fullfile(strcat(path2, int2str(measurement_rate)), 'userA', splits{1});
    pp3 = fullfile(strcat(path2, int2str(measurement_rate)), 'userB', splits{1});
    if ~exist(pp1, 'dir')
       mkdir(pp1)
    end
   
    if ~exist(pp2, 'dir')
       mkdir(pp2)
    end
    
    if ~exist(pp3, 'dir')
       mkdir(pp3)
    end
    imwrite(I, fullfile(strcat(path2, int2str(measurement_rate)), 'original', splits{1}, splits{3})); % Write original frame.
    imwrite(solA, fullfile(strcat(path2, int2str(measurement_rate)), 'userA', splits{1}, splits{3})); % Write User-A frame.
    imwrite(solB, fullfile(strcat(path2,int2str(measurement_rate)), 'userB', splits{1}, splits{3})); % Write User-B frame.
    
    %%% Write the faces for the face recognition evaluations.%%%
    pp1 = fullfile(strcat(path2, int2str(measurement_rate)), 'original_faces', splits{1});
    pp2 = fullfile(strcat(path2, int2str(measurement_rate)), 'userA_faces', splits{1});
    pp3 = fullfile(strcat(path2, int2str(measurement_rate)), 'userB_faces', splits{1});
    if ~exist(pp1, 'dir')
       mkdir(pp1)
    end
   
    if ~exist(pp2, 'dir')
       mkdir(pp2)
    end
    
    if ~exist(pp3, 'dir')
       mkdir(pp3)
    end
    faceI = I(y2:y1, x2:x1, :); % Face from the original frame
    faceA = solA(y2:y1, x2:x1, :); % Face from the User-A
    faceB = solB(y2:y1, x2:x1, :); % Fce from the User-B
    imwrite(faceI, fullfile(strcat(path2, int2str(measurement_rate)), 'original_faces', splits{1}, splits{3})); % Write original frame.
    imwrite(faceA, fullfile(strcat(path2, int2str(measurement_rate)), 'userA_faces', splits{1}, splits{3})); % Write User-A frame.
    imwrite(faceB, fullfile(strcat(path2, int2str(measurement_rate)), 'userB_faces', splits{1}, splits{3})); % Write User-B frame.

end