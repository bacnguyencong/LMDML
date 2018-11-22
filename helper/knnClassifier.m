function preds = knnClassifier(xTr, labels, k, xTe, M)
%%=========================================================================
% Do k nearest neighbors classifier
% INPUT:
%       xTr     : training examples by columns   
%       labels  : class label of input examples by column vector
%       k       : number of nearest neighbors
%       xTe     : testing examples by columns
% OUPUT:
%       preds   :  predicted label for each instance in xTe by colums
%==========================================================================
% author: Bac Nguyen Cong 
% email : nguyencongbacbk@gmail.com
%%=========================================================================
    if exist('M', 'var')
        ind = kNearestNeighborsM(M, xTr, xTe, k);
    else
        ind = kNearestNeighbors(xTr, xTe, k);
    end
    preds = labels(ind);
    if size(ind, 1) > 1  %check if is 1 nearest neighbor classifier
        preds = mode(preds, 1);
    end    
    preds = preds(:);
end