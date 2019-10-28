clc, clear, close all
outDir = 'C:\Users\ahishalm\Local\face_mehmet\mehmet_face_Tuesday\Results\measurement_4_gauss\';
path = strcat(outDir, 'faces_userA\');
path2 = strcat(outDir, 'faces_userB\');
ref_path = strcat(outDir, 'faces_original');
folders = dir(path);
folders = folders(3:end);
ssimValA = 0;
ssimValB = 0;
classNo = 100;
count = 0;
for i = 1:classNo
   imageNameA = dir(fullfile(path, folders(i).name));
   imageNameB = dir(fullfile(path2, folders(i).name));
   ref_imageName = dir(fullfile(ref_path, folders(i).name));
   imageNameA = imageNameA(3:end);
   imageNameB = imageNameB(3:end);
   ref_imageName = ref_imageName(3:end);
   for j = 1:length(imageNameA)
       count = count + 1;
       faceA = imread(fullfile(imageNameA(j).folder, imageNameA(j).name));
       faceB = imread(fullfile(imageNameB(j).folder, imageNameB(j).name));
       ref = imread(fullfile(ref_imageName(j).folder, ref_imageName(j).name));
       ssimValA = ssim(faceA, ref) + ssimValA;
       ssimValB = ssim(faceB, ref) + ssimValB;
   end
end
disp(ssimValA/count)
disp(ssimValB/count)