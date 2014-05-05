function dif = dif(filename1, filename2)
    if nargin < 2
        filename2 = '';
    end
    if ~isempty(filename1)
        dif = load (filename1);
        if ~isempty(filename2)
            var2 = load (filename2);
            dif = dif - var2;
        end
    else
        dif = [];
    end
    width = size(dif,1);
    height = size(dif,2);
    for i=1:width
       for j=1:height
           val = dif(i,j);
           if (val > 1e99) 
               dif(i,j) = 1e21;
           end
           if (val < -1e99) 
               dif(i,j) = -1e21;
           end    
       end
    end
end
