function [iknn, dist] = kNearestNeighborsM(M, xTr, xTe, k)
%%=========================================================================
% Find k nearest neighbors for each testing instance in the training set
% INPUT:
%       M   : Mahalanobis matrix
%       xTr : training examples by columns   
%       xTe : testing examples by columns
%       k   : number of nearest neighbors
% OUPUT:
%       iknn: index of k nearest neighbors by columns for each instance
%       dist: squared Euclidean distance of k nearest neighbors by columns
%==========================================================================
% author: Bac Nguyen Cong 
% email : nguyencongbacbk@gmail.com
%%=========================================================================
    nTrains   = size(xTr,2);
    nTests    = size(xTe,2);
    maxBlocks = 700;
    
    % check number of nearest neighbors
    if k > nTrains
        k = nTrains;
        warning('k is bigger than training examples'); %#ok<WNTAG>
    end
    
    iknn = zeros(k, nTests);
    dist = zeros(k, nTests);
    X    = sum(xTr.*(M*xTr), 1)';
    
    % compute by each block
    for i=1:maxBlocks:nTests
        addBlocks = min(maxBlocks,nTests-i+1)-1;        
        x         = xTe(:,i:i+addBlocks);
		Dist      = bsxfun(@plus,X, bsxfun(@plus, sum(x.*(M*x)), - 2*xTr'*(M*x)));
        [v, ind]  = sort(Dist,1); clear('Dist');
        % save result
        iknn(:,i:i+addBlocks) = ind(1:k,:); clear('ind');
        dist(:,i:i+addBlocks) = v(1:k,:);   clear('v');
    end
    
end
%%=======================================================================