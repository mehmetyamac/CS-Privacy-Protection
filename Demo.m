%%% Web-Cam Demo %%%
clear; close all;
%%% Include Dependicies %%%
addpath(genpath('Measurements')) % Noiselet measurement mex file.
addpath(genpath('Wavelab850')) % https://statweb.stanford.edu/~wavelab/
%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('utils')) % Functions...

rng(1);

cam = webcam('Logitech HD Webcam C270'); % Change accordingly for your Web-Cam.
cam.Resolution = '640x480';
I0 = imrotate(snapshot(cam), 180);

param.N = 2^18; % Max. length of the signal.

param.em_power = 0.15; % 0.085 % Embedding power, see the defination in the paper.
param.degradation = 'binary'; % 'binary' or 'gauss'
param.matrix = 0.9 + 0.1*randn(2^10, 2^10); % Gaussian matrix for binary masked Gaussian degradation option.
measurement_rate = 4; % MR = measurement_rate/10;
param.mratio = measurement_rate./10;
param.M = 21003; % Maximum leghts of the bits for embeding.

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


% Crop or add some pixels to the frame for making sure that it has width
% and height dimensions divisible by 2 for wavelet transform.
[S1_init, S2_init, ~]=size(I0);
I1 = imresize(I0,sqrt((param.N)/(S1_init*S2_init)));
param.S1 = size(I1,1);
param.S2 = size(I1,2);

%img2 = img2(1:end-1,1:end-1, :);
ll=0;
while (param.S2*param.S1)>param.N
    if mod(ll,2)
        param.S1 = param.S1-1;
    elseif mod(ll,2)
        param.S2 = param.S2-1;
    end
        ll = ll+1;
end
    
if mod(param.S1,2)
    param.S1=param.S1-1;
end
        
if mod(param.S2,2)
    param.S2=param.S2-1;
end

I = I1(1:param.S1, 1:param.S2, :); % New cropped image.

imshow(uint8(I),[],'InitialMagnification','fit');                                % Show original image.
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


mask = zeros(param.S1,param.S2);
for i=1:length(xi)
    mask(floor(yi(i)),floor(xi(i))) = 1;
end

% Find region boundaries of segmentation...
mask = bwfill(boundarymask(mask), 'holes'); % Fill the inside of mask with all 1s.

figure,imshow(mask,'InitialMagnification','fit')

if sum(mask(:)) > (param.M - 3)
    error("Too big region is selected")
end
% Finding boundaries if you want to transmit just boundaries
[b] = bwboundaries(mask);

param.b = b;
[y_w,watermark_inf, smean] = transmitter(I, mask, param, i);

solImageA = userA(y_w, smean,param);
[solImageB, infor, fail] = userB(y_w, watermark_inf, smean,param);

%%% Performance Evaluations %%%:
% User-A
PSNRA = measerr(I, solImageA); % PSNR of the whole frame.
PSNRA_insidemask = measerr(I(repelem(mask, 1, 1, 3) == 1), solImageA(repelem(mask, 1, 1, 3) == 1)); % PSNR of the privacy sensitive part.
A_ssimVal = ssim(I(repelem(mask, 1, 1, 3) == 1), solImageA(repelem(mask, 1, 1, 3) == 1)); % SSIM of the privacy sensitive part.
A_PSNR_outsidemask = measerr(I(repelem(mask, 1, 1, 3) == 0), solImageA(repelem(mask, 1, 1, 3) == 0)); % PSNR of the outside of the privacy sensitive part.
    
% User-B
PSNRB = measerr(I, solImageB); % PSNR of the whole frame.
PSNRB_insidemask = measerr(I(repelem(mask, 1, 1, 3) == 1), solImageB(repelem(mask, 1, 1, 3) == 1)); % PSNR of the privacy sensite part.
B_ssimVal = ssim(I(repelem(mask, 1, 1, 3) == 1), solImageB(repelem(mask, 1, 1, 3) == 1)); % SSIM of the privacy sensitive part.
B_PSNR_outsidemask = measerr(I(repelem(mask, 1, 1, 3) == 0), solImageB(repelem(mask, 1, 1, 3) == 0)); % PSNR of the outside of the privacy sensitive part.


% For imshow purposes, add some zero if necessary to the encoded signal.
chSize = length(y_w)/3; % Length of one channel.
newchSize = chSize + 30 - mod(chSize, 30);
tt = factor(newchSize);
ss2 = tt(1)*tt(end); % New ss1, ss2 dimensions for imshow.
ss1 = prod(tt(2:end-1));

rr = reshape([y_w(1:chSize); zeros(30 - mod(chSize, 30), 1)], ss1, ss2);
gg = reshape([y_w(chSize + 1 : 2 * chSize); zeros(30 - mod(chSize, 30), 1)], ss1, ss2);
bb = reshape([y_w(2 * chSize + 1 : end); zeros(30 - mod(chSize, 30), 1)], ss1, ss2);

Encoded_Image = cat(3, rr, gg, bb); % Encripted and compressed...

% Showing and printing the results...

disp(strcat("SSIM of privacy sensitive part for User-A: ", num2str(A_ssimVal)));
disp(strcat("SSIM of privacy sensitive part for User-B: ", num2str(B_ssimVal)));

disp(strcat("PSNR A:             ", num2str(PSNRA)));
disp(strcat("PSNR B:             ", num2str(PSNRB)));

disp(strcat("PSNR inside box A:  ", num2str(PSNRA_insidemask)));
disp(strcat("PSNR inside box B:  ", num2str(PSNRB_insidemask)));

disp(strcat("PSNR outside box A: ", num2str(A_PSNR_outsidemask)));
disp(strcat("PSNR outside box B: ", num2str(B_PSNR_outsidemask)));

disp(strcat("Recovered the location of privacy sensitive part: ", num2str(fail)));
disp(strcat("Total Errors:                              ", num2str(infor.total_error)));

figure,subplot(1,3,1)
imshow(Encoded_Image,[]), ylabel('Encrypted and Compressed Signal');
subplot(1,3,2)
imshow(solImageA,[]), title("User-A, Semi - Authorization");
subplot(1,3,3)
imshow(solImageB,[]), title("User-B, Full - Authorization");
sgtitle("Reversible Privacy Preservation using Multi-level Encryption and Compressive Sensing")