clear, clc, close all
faces_20 = 'C:\Users\ahishalm\Local\face_mehmet\new-data\youtube\YouTubeFaces\faces_20';
%faces_20_tr2 = 'C:\Users\ahishalm\Local\face_mehmet\new-data\youtube\YouTubeFaces\faces_20_tr2_mr8_gauss';
faces_20_tr2 = 'C:\Users\ahishalm\Local\face_mehmet\new-data\youtube\YouTubeFaces\new_results_0.15\faces_20_tr2_mr7_bin';
%corrupted0 = 'C:\Users\ahishalm\Local\face_mehmet\mehmet_face_Tuesday\Results\measurement_8_gauss\faces_userA';
corrupted0 = 'C:\Users\ahishalm\Local\face_mehmet\mehmet_face_Tuesday\Results\new_results\measurement_7_0.15_bin\faces_userA';
%corrupted = 'C:\Users\ahishalm\Local\face_mehmet\mehmet_face_Tuesday\Results\measurement_8_gauss\faces_userA_tr2_temp';
corrupted = 'C:\Users\ahishalm\Local\face_mehmet\mehmet_face_Tuesday\Results\new_results\measurement_7_0.15_bin\faces_userA_tr2_temp';

copyfile (corrupted0, corrupted);
copyfile (faces_20, faces_20_tr2);

files = dir(faces_20_tr2);

rng(10500)
for i = 3:length(files)
    cor = dir(fullfile(corrupted, files(i).name, '*.jpg'));
    ind = randperm(length(cor), 10);
    for j = 1:length(ind)
        movefile(fullfile(cor(ind(j)).folder, cor(ind(j)).name),  fullfile(files(i).folder, files(i).name, cor(ind(j)).name))
    end
    
end