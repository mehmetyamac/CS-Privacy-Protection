clear, clc, close all
outDir = 'C:\Users\ahishalm\Local\face_mehmet\mehmet_face_Tuesday\Results\labels';
txt_files = dir(fullfile(outDir,'*.mat'));

outDir2 = 'C:\Users\ahishalm\Local\face_mehmet\mehmet_face_Tuesday\Results\measurement_6_gauss';
txt_files2 = dir(fullfile(outDir2,'*_results.mat'));

gtd_path = 'C:\Users\ahishalm\Local\face_mehmet\new-data\youtube\YouTubeFaces\txt\';
txt_files_gtd = dir(fullfile(gtd_path,'*txt'));

user = 'userB\';

frameSize = 2908;
identity = 500;

count = 0;
rng(10500);
for i = 1:identity
    cordinates = importdata(strcat(gtd_path, txt_files_gtd(i).name));
    if isempty(cordinates) || length(cordinates.data) < 30
        continue
    end
    s1 = split(txt_files_gtd(i).name, '.')
    if strcmp(s1{1}, 'Abdullah')
        continue
    end
    while 1
        count = count + 1;
        if count > 14
            load(fullfile(txt_files2(count+1).folder, txt_files2(count+1).name)); % Load Results
        else
            load(fullfile(txt_files2(count).folder, txt_files2(count).name)); % Load Results
        end
        load(fullfile(txt_files(count).folder, txt_files(count).name)); % Load Labels
        s2 = split(txt_files(count).name, '.');
        s22 = split(txt_files2(count).name, '.');
       
        if strcmp(s1{1}, s2{1})
            break
        end
    end
    
    p = 0;
    for c = 1:30
        p = p + performanceB(c).big;
    end
    if p == 30
        continue
    end
    
    x1 = cordinates.data(:,2)+round(cordinates.data(:,4)./2);
    y1 = cordinates.data(:,3)+round(cordinates.data(:,5)./2);
    y2 = cordinates.data(:,3)-round(cordinates.data(:,5)./2);
    x2 = cordinates.data(:,2)-round(cordinates.data(:,4)./2);
    x1(x1 < 1) = 1;
    x2(x2 < 1) = 1;
    y1(y1 < 1) = 1;
    y2(y2 < 1) = 1;
        
    ind = randperm(length(cordinates.data), 49);
    png_files_gtd = cordinates.textdata(ind);
    x1 = x1(ind);
    x2 = x2(ind);
    y1 = y1(ind);
    y2 = y2(ind);
    samples = 0;
    for j = 1:length(png_files_gtd)
        if any(strcmp(png_files,png_files_gtd{j}))
            continue
        end
        samples = samples + 1;
        I = imread(strcat('C:\Users\ahishalm\Local\face_mehmet\new-data\youtube\YouTubeFaces\frame_images_DB\', png_files_gtd{j}));
        face = I(y2(j):y1(j), x2(j):x1(j), :);
        splits = split(png_files_gtd{j}, '\');
        faceFile = strcat('C:\Users\ahishalm\Local\face_mehmet\new-data\youtube\YouTubeFaces\faces\', splits{1});
        if ~exist(faceFile, 'dir')
           mkdir(faceFile)
        end
        imwrite(face, strcat(faceFile,'\', splits{3}));
        if samples > 19
            break
        end
    end
    if samples < 20
        continue
    end
end