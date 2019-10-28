clear, clc, close all
outDir = 'C:\Users\ahishalm\Local\face_mehmet\mehmet_face_Tuesday\Results\measurement_6_gauss';
%outDir = 'C:\Users\ahishalm\Local\face_mehmet\mehmet_face_Tuesday\Results\new_results\measurement_8_0.15_gauss';
txt_files = dir(fullfile(outDir,'*labels.mat'));

outDir2 = 'C:\Users\ahishalm\Local\face_mehmet\mehmet_face_Tuesday\Results\measurement_6_gauss';
%outDir2 = 'C:\Users\ahishalm\Local\face_mehmet\mehmet_face_Tuesday\Results\new_results\measurement_8_0.15_gauss';
txt_files2 = dir(fullfile(outDir,'*_results.mat'));

user = 'original\';

frameSize = 2908;
identity = 104;

count = 0;
for i = 1:identity
    %txt_files(i).name
    load(fullfile(txt_files2(i).folder, txt_files2(i).name)); % Load Results
    load(fullfile(txt_files(i).folder, txt_files(i).name)); % Load Labels
    for j = 1:length(performanceA)
        if performanceB(j).big == 1
            continue
        end
        count = count + 1;
        I = imread(strcat(outDir, '\', user, png_files{j}));
        %I = insertShape(I, 'Rectangle', [x2(j), y2(j), x1(j) - x2(j), y1(j) - y2(j)]);
        %imshow(I)
        face = I(y2(j):y1(j), x2(j):x1(j), :);
        splits = split(png_files{j}, '\');
        title = strcat(outDir, '\faces_', user, splits{1});
        if ~exist(title, 'dir')
           mkdir(title)
        end
        imwrite(face, fullfile(title, splits{3}))
        %figure, imshow(face)
    end
end