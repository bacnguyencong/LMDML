function [M, ctime] = LMDMLA(xTr, yTr, params)
%%=========================================================================
% Learn a distance metric learning by maximizing margin
% INPUT:
%       xTr     : training examples by columns   
%       yTr     : training class label by colummn vector
%       params  : parameters for method
% OUPUT:
%       M       : the Mahalanobis matrix
%==========================================================================
% author: Bac Nguyen Cong 
% email : bac.nguyencong@ugent.be
%%=========================================================================

    %parameters for initial configurations
    k = params.par.knn;
    start = tic;
    % parameters for targets and triplets constraints    
    Targets = int32(getTargets(xTr, yTr, k));
    
    if isfield(params.par, 'k2'),
        Impostors = int32(getImpostors(xTr, yTr, params.par.k2));
    end
    ctime = toc(start);
    % make class 0 - (nClass - 1)
    [dummy, nClass] = filterClass(yTr); dummy = int32(dummy);
    
    best_acc= -Inf;
    lr      = 1;    
    
    if size(xTr,2) > 5000,
        CVO = cvpartition(yTr,'HoldOut', 5000/size(xTr,2));
        xte = xTr(:, CVO.test); yte = yTr(CVO.test);
    else
        xte = xTr; yte = yTr;
    end
    
    for c =  10.^(-1:2),       
        if isfield(params.par, 'k2'),
            [tM, cost]= mSolverLMDMLS(xTr,dummy,...
                       Targets,nClass,c,lr,params.par.approx,params.par.max_iters, Impostors);
        else
            [tM, cost]= mSolverLMDMLS(xTr,dummy,...
                       Targets,nClass,c,lr,params.par.approx,params.par.max_iters);
        end
        preds = kNearestNeighborsM(tM, xTr, xte, k+1);
        preds = yTr(preds(2:k+1,:));
        if k > 1, preds = mode(preds, 1); end    
        preds = preds(:);
        accTemp = 100 * mean(preds == yte);
        fprintf('#c=%.3f, cost=%.5f, acc=%.2f\n', c, cost, accTemp);
        if (accTemp > best_acc)
            M = tM;
            best_acc = accTemp;
        end
    end
    fprintf('Acc=%.2f\n', best_acc);   
end