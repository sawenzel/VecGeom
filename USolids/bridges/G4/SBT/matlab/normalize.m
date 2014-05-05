function res = normalize(a)
    normA = max(a) - min(a);               % this is a vector
    normA = repmat(normA, [length(a) 1]);  % this makes it a matrix
                                           % of the same size as A
    res = a./normA;  % your normalized matrix
end

