install;
clear all;

load('data/balance.mat');
d = size(xTr,1);

params.par.knn = 3;
params.par.max_iters = 20000;
%params.par.k2 = 10; % for approximating the miss neighbors
params.par.approx = 1; % approx=0 for standard SGD

rng(123456);
% learn a Mahalanobis distance metric
M = LMDMLA(xTr, yTr, params);

% classification accuracy
pred1 = knnClassifier(xTr, yTr, 3, xTe, eye(d));
pred2 = knnClassifier(xTr, yTr, 3, xTe, M);

fprintf('\n----------------------------------------------\n');
fprintf('the 3-NN accuracy:\n');
fprintf('Euclidean = %.2f\n', mean(pred1 == yTe)*100);
fprintf('LMDML     = %.2f\n', mean(pred2 == yTe)*100);
fprintf('----------------------------------------------\n');