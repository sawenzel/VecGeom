function dif=subarray(dif, first, count)
    if (~isempty(dif))
        if (nargin < 2)
            first = 1;
        end
        if (nargin < 3)
            count = -1;
        end
        if (count < 0)
            count = length(dif) - (first - 1);
        end
        last = count + (first - 1);
        dif = dif(first:last,:);
    end
end
