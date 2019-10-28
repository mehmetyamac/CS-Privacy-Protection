clear, clc
outDir = 'C:\Users\ahishalm\Local\face_mehmet\mehmet_face_Tuesday\Results\measurement_6_gauss';
outDir = 'C:\Users\ahishalm\Local\face_mehmet\mehmet_face_Tuesday\Results\new_results\measurement_7_0.15_bin';
txt_files = dir(fullfile(outDir,'*_results.mat'));

frameSize = 2908;
%frameSize = 120;
identity = 104;
%identity = 4;
fail = zeros(frameSize, 1);
PSNRA = zeros(frameSize, 1);
PSNRA_insidemask = zeros(frameSize, 1);
PSNRA_outsidemask = zeros(frameSize, 1);

PSNRB = zeros(frameSize, 1);
PSNRB_insidemask = zeros(frameSize, 1);
PSNRB_outsidemask = zeros(frameSize, 1);
totalerror = zeros(frameSize, 1);
count = 0;
for i = 1:identity
    %txt_files(i).name
    load(fullfile(txt_files(i).folder, txt_files(i).name));

    for j = 1:length(performanceA)
        if performanceB(j).big == 1
            continue
        end
        count = count + 1;
        PSNRA(count, 1) = performanceA(j).PSNR;
        PSNRA_insidemask(count, 1) = performanceA(j).PSNR_insidemask;
        PSNRA_outsidemask(count, 1) = performanceA(j).PSNR_outsidemask;
        
        PSNRB(count, 1) = performanceB(j).PSNR;
        PSNRB_insidemask(count, 1) = performanceB(j).PSNR_insidemask;
        PSNRB_outsidemask(count, 1) = performanceB(j).PSNR_outsidemask;
        totalerror(count, 1) = performanceB(j).err;
        fail(count, 1) = performanceB(j).fail;
    end
    
end
disp(strcat("PSNR A:             ", num2str(mean(PSNRA))));
disp(strcat("PSNR B:             ", num2str(mean(PSNRB))));

disp(strcat("PSNR inside box A:  ", num2str(mean(PSNRA_insidemask))));
disp(strcat("PSNR inside box B:  ", num2str(mean(PSNRB_insidemask))));

disp(strcat("PSNR outside box A: ", num2str(mean(PSNRA_outsidemask))));
disp(strcat("PSNR outside box B: ", num2str(mean(PSNRB_outsidemask))));

disp(strcat("Fails:              ", num2str(sum(fail(:)))));
disp(strcat("Total Errors:              ", num2str(sum(totalerror(:)))));
        
