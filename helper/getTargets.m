function T = getTargets(xTr, yTr, k)
%%=========================================================================
% Find target neighbours for each instance in the training set and
% Universum samples
% INPUT:
%       xTr : training examples by columns   
%       yTr : labels for each example by column
%       k   : number of nearest neighbors
% OUPUT:
%       T   : index of k target neighbors by columns for each instance
%==========================================================================
% author: Bac Nguyen Cong 
% email : nguyencongbacbk@gmail.com
%%=========================================================================

    nInst = size(xTr, 2);
    labels= unique(yTr);
    
    index = 1:nInst;
    
    T     = zeros(k, nInst); % targets for training    
    sort(labels);
    
    for i=1:length(labels)       
        fprintf('Find Targets of class (%d)', i);
        x              = xTr(:, yTr == labels(i)); 
        indi           = index(:, yTr == labels(i));
        if nInst <= 20000,
            iknn = kNearestNeighbors(x, x, k+1);
        else
            iknn = fastKNN(x, x, k+1);
        end            
        T(:,indi)      = indi(iknn(2:k+1,:));        
        clear('x', 'indi', 'iknn');
        fprintf('.\n');
    end   
end

function iknn = fastKNN(xTr, xTe, k)
    fprintf('Using fast knn...\n');
    tree = buildmtreemex(xTr, 10);
    iknn = usemtreemex(xTe, xTr, tree, k);
end

