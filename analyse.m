function analyse(outDir, bigInclude)
    mat_files = dir(fullfile(outDir,'*_results.mat')); % Saved results files.
    frameNu = 3000; % # of frames.
    
    %%%%%% Allocate performance metrics for User-A and User-B
    
    % User-A
    ssimValA = zeros(frameNu, 1); % SSIM of the privacy sensitive part.
    PSNRA = zeros(frameNu, 1); % PSNR of the whole frame.
    PSNRA_insidemask = zeros(frameNu, 1); % PSNR of the privacy sensitive part.
    PSNRA_outsidemask = zeros(frameNu, 1); % PSNR of the outside of the privacy sensitive part.

    % User-B
    ssimValB = zeros(frameNu, 1); % SSIM of the privacy sensitive part.
    fail = zeros(frameNu, 1); % Check if decoding of the face coordinates is failed.
    PSNRB = zeros(frameNu, 1);  % PSNR of the whole frame.
    PSNRB_insidemask = zeros(frameNu, 1); % PSNR of the privacy sensitive part.
    PSNRB_outsidemask = zeros(frameNu, 1); % PSNR of the outside of the privacy sensitive part.
    totalerror = zeros(frameNu, 1); % Total number of errors in recovered mask of concealed region.
    count = 0;
    bigg = 0;
    %%%%%%
    
    for i = 1:length(mat_files) % Loop over identites.
        load(fullfile(mat_files(i).folder, mat_files(i).name)); % Load the saved PSNRs for this identity.

        for j = 1:length(performanceA) % Loop over the samples of the identity.
            if performanceB(j).big == 1 % Check if the concealed region (face part) is larger than 15000.
                bigg = bigg + 1;
                % If chosen, do not include larger faces for performance evaluation.
                if bigInclude == 0; continue; end
            end
            count = count + 1; % Counter for how many frames included for performance evaluations.
            
            %%% Collecting measured performance metrics:
            ssimValA(count, 1) = performanceA(j).ssimVal;
            PSNRA(count, 1) = performanceA(j).PSNR;
            PSNRA_insidemask(count, 1) = performanceA(j).PSNR_insidemask;
            PSNRA_outsidemask(count, 1) = performanceA(j).PSNR_outsidemask;

            ssimValB(count, 1) = performanceB(j).ssimVal;
            PSNRB(count, 1) = performanceB(j).PSNR;
            PSNRB_insidemask(count, 1) = performanceB(j).PSNR_insidemask;
            PSNRB_outsidemask(count, 1) = performanceB(j).PSNR_outsidemask;
            totalerror(count, 1) = performanceB(j).err;
            fail(count, 1) = performanceB(j).fail;
        end

    end
    
    % Averaged PSNRs, total number of failure during recovery of face coordinate, and total number of errors in recovered mask of concealed region:
    disp(strcat("Total number of frames included to performance evaluation: ", num2str(count)))
    disp(strcat("Averaged SSIM of privacy sensitive part for User-A: ", num2str(sum(ssimValA(:))/count)));
    disp(strcat("Averaged SSIM of privacy sensitive part for User-B: ", num2str(sum(ssimValB(:))/count)));

    disp(strcat("PSNR A:             ", num2str(mean(PSNRA(1:count)))));
    disp(strcat("PSNR B:             ", num2str(mean(PSNRB(1:count)))));

    disp(strcat("PSNR inside box A:  ", num2str(mean(PSNRA_insidemask(1:count)))));
    disp(strcat("PSNR inside box B:  ", num2str(mean(PSNRB_insidemask(1:count)))));

    disp(strcat("PSNR outside box A: ", num2str(mean(PSNRA_outsidemask(1:count)))));
    disp(strcat("PSNR outside box B: ", num2str(mean(PSNRB_outsidemask(1:count)))));

    disp(strcat("Total fails for recovering face locations: ", num2str(sum(fail(:)))));
    disp(strcat("Total Errors:                              ", num2str(sum(totalerror(:)))));
end