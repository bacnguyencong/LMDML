function [dummy, nClass] = filterClass( yTr )
%% Filter classes to 0 - (nClass-1) for all examples
%%
    Y      = unique(yTr);
    nClass = length(Y);
    dummy  = zeros(length(yTr),1);    
    for i=1:nClass,
        dummy(yTr == Y(i)) = i - 1;
    end
end

