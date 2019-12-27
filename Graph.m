clc, clear, close all
emb = 10.^[-2, -1.8, -1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4, -0.2];

paths = dir('C:\Users\ahishalm\Desktop\projects\pri_pre\Results_graph');
paths = paths(3:end);

for i = 1:10
    outDir = fullfile(paths(i).folder, paths(i).name, 'measurement_6');
    analyse(outDir, 1);
end

% Not include big parts...
outUserA = [38.4995, 38.2194, 37.5582, 36.2129, 34.0698, 31.3637, 28.3782, 25.2285, 21.9523, 18.6041];
inUserB =  [13.8627, 15.8578, 19.1146, 23.3614, 28.014, 32.5889, 36.1149, 37.7184, 38.1695, 38.0637];

% Include big parts...
outUserA = [37.927, 37.6605, 37.0317, 35.7502, 33.6997, 31.0902, 28.1828, 25.0878, 21.8468, 18.5199];
inUserB =  [13.6537, 15.5222, 18.5721, 22.5541, 26.9293, 31.2589, 34.6604, 36.3705, 37.184, 37.5106];

figure, plot(log10(emb), outUserA, 'LineWidth',3), grid, hold on
plot(log10(emb), inUserB, 'LineWidth',3)
ylabel('Peak signal-to-noise ratios (PSNRs, dB)', 'FontSize', 15)
xlabel('log(||Bw||/||y_d||)', 'FontSize', 15)
legend('Concealed Region - User B', 'Outside Conc. Region - User A', 'FontSize', 12)
