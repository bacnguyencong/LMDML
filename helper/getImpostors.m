function [Imp] = getImpostors(xTr, yTr, k)
%%=========================================================================
% Find impostor neighbour for each instance in the training set
% INPUT:
%       xTr : training examples by columns   
%       yTr : labels for each example by column
% OUPUT:
%       Imp : index of k impostor neighbors by columns for each instance
%==========================================================================
% author: Bac Nguyen Cong 
% email : nguyencongbacbk@gmail.com
%%=========================================================================

   nInst = size(xTr, 2);
   labels= unique(yTr);
    
   index = 1:nInst;
   
   Imp   = zeros(k, nInst);
    
    for i=1:length(labels)       
        fprintf('Finding impostor for class %d ... \n', labels(i));       
        indj     = index(:, yTr == labels(i)); 
        indk     = index(:, yTr ~= labels(i));      
        x        = xTr(:, yTr == labels(i)); 
        X        = xTr(:, yTr ~= labels(i));
        if nInst <= 20000,
            kknn = kNearestNeighbors(X, x, k);    
        else
            kknn = fastKNN(X, x, k);
        end
        
        Imp(:,indj)= indk(kknn);
        clear('indj', 'indk', 'x', 'X', 'kknn');
    end
end

function iknn = fastKNN(xTr, xTe, k)
    fprintf('Using fast knn...\n');
    tree = buildmtreemex(xTr, 10);
    iknn = usemtreemex(xTe, xTr, tree, k);
end