
close all
clear all
%clc

% %%%Toy input signal
% n = 512;
% t = 0:0.001:(n*0.001)-0.001;
% yn = square(2*pi*10*t)+0.02*randn(size(t));
% plot(yn)
% grid on
% title('Noisy Signal')
% %%%%%%%%%transforms%%%%%%%%%%
% Level =4;
% waveletName = 'db6';
% Wav = @(t) wavedec(t,Level,waveletName);
% inWav = @(t,l) waverec(t,l,waveletName);
% [a,b]=Wav(yn);
% b
% figure,plot(a)
% 
% s_hat=inWav(a,b);
% figure,plot(s_hat)
% 
% norm(s_hat-yn)



path1 = '/Volumes/VERBATIM/Abdullah_Gul/1/';
path2 = '/Users/mehmetyamac/Desktop/mehmet_face/Results/measurement_';

png_files = dir(fullfile(path1,'*jpg'));
filename=png_files(1).name;
imagename=strcat(path1,filename);
f=imread(imagename);

s=rgb2gray(f);
%%%%%%%%%transforms%%%%%%%%%%
Level =5;
waveletName = 'coif2';
Wav = @(t) wavedec2(t,Level,waveletName);
inWav = @(t,l) waverec2(t,l,waveletName);
tic
[a,b]=Wav(s);




s_hat=inWav(a,b);
toc
%figure,imshow(reshape(a,360,480));
map = colormap; rv = length(map);
mode='square';
plotwavelet2(a,b,Level,waveletName,rv,mode)

figure,imshow(s_hat,[])

norm(s_hat-double(s),'fro')
SNR =measerr(s,s_hat)

figure,plot(sort(abs(a(1:10000)),'descend'))


%Second option

%%transforms
h=MakeONFilter('Coiflet',2);
Wav=@(t) FWT2_POE(t,3,h); %wavelet coeefficients of image
inWav= @(t) IWT2_POE(t,3,h);

[S1, S2] =size(s);
tic
x=new_wavelet(s,Wav,S1,S2);
s1=new_inwavelet(x,inWav,S1,S2);
toc
s_hat=reshape(s1,S1,S2);

% new_S1=ceil(log2(S1));
% 
% new_S2=ceil(log2(S2));
% 
% %zero padding
% 
% z_padded=zeros(2^new_S1,2^new_S2);
% 
% z_padded(1:S1,1:S2)=s;
% 
% z_padden_coef=Wav(z_padded);
% 
% z_padden_s_hat=inWav(z_padden_coef);
% 
% figure,imshow(z_padden_s_hat,[])
% 
% s_hat=z_padden_s_hat(1:S1,1:S2);

SNR =measerr(s,s_hat)

figure,imshow(s_hat,[])

function x=new_wavelet(s,Wav,S1,S2)
    new_S1=ceil(log2(S1));

    new_S2=ceil(log2(S2));
    
    z_padded=zeros(2^new_S1,2^new_S2);

    z_padded(1:S1,1:S2)=s;

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



function plotwavelet2(C,S,level,wavelet,rv,mode)

%   Plot wavelet image (2D) decomposition.
%   A short and simple function for displaying wavelet image decomposition
%   coefficients in 'tree' or 'square' mode
%
%   Required : MATLAB, Image Processing Toolbox, Wavelet Toolbox
%
%   plotwavelet2(C,S,level,wavelet,rv,mode)
%
%   Input:  C : wavelet coefficients (see wavedec2)
%           S : corresponding bookkeeping matrix (see wavedec2)
%           level : level decomposition 
%           wavelet : name of the wavelet
%           rv : rescale value, typically the length of the colormap
%                (see "Wavelets: Working with Images" documentation)
%           mode : 'tree' or 'square'
%
%   Output:  none
%
%   Example:
%
%     % Load image
%     load wbarb;
%     % Define wavelet of your choice
%     wavelet = 'haar';
%     % Define wavelet decomposition level
%     level = 2;
%     % Compute multilevel 2D wavelet decomposition
%     [C S] = wavedec2(X,level,wavelet);
%     % Define colormap and set rescale value
%     colormap(map); rv = length(map);
%     % Plot wavelet decomposition using square mode
%     plotwavelet2(C,S,level,wavelet,rv,'square');
%     title(['Decomposition at level ',num2str(level)]);
%
%
%   Benjamin Tremoulheac, benjamin.tremoulheac@univ-tlse3.fr, Apr 2010

A = cell(1,level); H = A; V = A; D = A;

for k = 1:level
    A{k} = appcoef2(C,S,wavelet,k); % approx
    [H{k} V{k} D{k}] = detcoef2('a',C,S,k); % details  
    
    A{k} = wcodemat(A{k},rv);
    H{k} = wcodemat(H{k},rv);
    V{k} = wcodemat(V{k},rv);
    D{k} = wcodemat(D{k},rv);
end

if strcmp(mode,'tree')
    
    aff = 0;
    
    for k = 1:level
        subplot(level,4,aff+1); image(A{k});
        title(['Approximation A',num2str(k)]);
        subplot(level,4,aff+2); image(H{k});
        title(['Horizontal Detail ',num2str(k)]);
        subplot(level,4,aff+3); image(V{k});
        title(['Vertical Detail ',num2str(k)]);
        subplot(level,4,aff+4); image(D{k});
        title(['Diagonal Detail ',num2str(k)]);
        aff = aff + 4;
    end
    
elseif strcmp(mode,'square')
    
    dec = cell(1,level);
    dec{level} = [A{level} H{level} ; V{level} D{level}];
    
    for k = level-1:-1:1
        dec{k} = [imresize(dec{k+1},size(H{k})) H{k} ; V{k} D{k}];
    end
    
    image(dec{1});
    
end

end













