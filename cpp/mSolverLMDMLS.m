function [M, cost] = mSolverLMDMLS(X, Y, Targets, nClass, B, c, aprox, max_iters, Impostors)
%% solving DML by SGD in C code
% X         : input examples  d x n
% Y         : class labels    n x 1
% Targets   : Target examples k x n
% nClass    : number of classes
% B         : trace bound
% c         : initial learning rate
% approx    : (0) is SGD : (1) approximate SGD
% max_iters : maximum number of iteration

    if issparse(X) 
        X = full(X);
    end
    if nargin == 8,
        [M, cost] = mexSolverLMDMLS(X, Y, Targets, nClass, B, c, aprox, max_iters);
    else
        [M, cost] = mexSolverLMDMLS(X, Y, Targets, nClass, B, c, aprox, max_iters, Impostors);
    end
end