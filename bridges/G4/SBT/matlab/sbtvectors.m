
function res = sbtvectors(method, name, nameValues1, nameValues2, first, count, color)
    filenamePoints = [method 'Points.dat'];
    filenameVectors = [method name '.dat'];
    filenameValues1 = '';
    filenameValues2 = '';

%     if nameValues are specified, concerete values are mapped to the
%     vector
    if (nargin >= 3 && ~isempty(nameValues1))
        filenameValues1 = [method nameValues1 '.dat'];
    end
    if (nargin >= 4 && ~isempty(nameValues2))
        filenameValues2 = [method nameValues2 '.dat'];
    end
    if (nargin < 7)
        color = 'k';
    end    
    if (nargin < 5)
        first = 1;
    end
    if (nargin < 6)
        count = -1;
    end
    
    realcount = count;
    if count == 1
        realcount = 2;
    end
    values = subarray (dif(filenameValues1, filenameValues2), first, realcount);
    points = subarray (load (filenamePoints), first, realcount);
    vectors = subarray (load (filenameVectors), first, realcount);
    x = points(:,1);
    y = points(:,2);
    z = points(:,3);

%      vectors = vectors * size;

    n = size(points, 1);
    mulFactor = linspace (1000*n, 1, n) / n + 1;
    mulFactor = 10 * mulFactor';
    
    u = vectors(:,1);
    v = vectors(:,2);
    w = vectors(:,3);
    
    if (count == 1)
        u(2) = 0;
        v(2) = 0;
        w(2) = 0;
    end
    
    if (~isempty(values))
        values = 10 + 200*normalize(values);    
        values = values ./ max(values);
        values = values .* 3000;
        u = u .* values;
        v = v .* values;
        w = w .* values;
    end
    hold on;
    sizeLines = 2; % value 0 means no automatic scaling
    if (isempty(values))
        quiver3(x,y,z,u,v,w,sizeLines,color);
    else
        quiver3(x,y,z,u,v,w,color);
    end
    grid on;
    hold off;
end
