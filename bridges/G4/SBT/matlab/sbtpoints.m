function sbtpoints (method, name1, name2, first, count)    
    maxPoints = 100000;
    numargin = nargin;
    filename2 = '';    
    if (nargin < 3)
        name2 = '';
    else
        if ~isempty(name2)
            filename2 = [method name2 '.dat'];        
        end
    end
    if (nargin < 4)
        first = 1;
    end
    if (nargin < 5)
        count = -1;
    end
    realcount = count;
    if count==1
        realcount = 2;
    end
    filenamePoints = [method 'Points.dat'];
    filename1 = [method name1 '.dat'];
    figure;
    hold on;
    
    points = subarray (dif (filenamePoints, ''), first, realcount);
    len = size(points, 1);    
    tit = ['Folder: ' dirname() '; Method: ' method '; Software: '];
    values = subarray (dif(filename1, filename2), first, realcount);    
    if (~isempty (name2))
       indexes = sbtdifferences(values, first, count);
       tit = [tit name1 '-' name2 ';' 'Total number of different points: ' int2str(length(indexes))];
       bp = arrayfilter(points,indexes);
       bpvalues = arrayfilter(values,indexes);
        if count==1
            bp(2,:) = bp(1,:);
            bpvalues(2) = bpvalues(1);
            bpvalues(1) = 0;
        end       
       if len <= maxPoints
%            scatter3 (bp(:,1), bp(:,2), bp(:,3), 200, 'r');
%           scatter3 (bp(:,1), bp(:,2), bp(:,3), 100, bpvalues, 'filled');
       end
    else
        tit = [tit name1];
    end
    if len <= maxPoints
        values = magnitude(values);
        pointsize = 25;
        pointsizes = linspace (10, 100, length(values));
        pointsizes = pointsizes';
        colormap('default');
    %     values = linspace (0, 1, length(values));
        if count==1
            points(2,:) = points(1,:);
            val = values(1);
            values(1) = val;
            values(2) = val;
            caxis([0.99*val,1.01*val]);
        end
        if count == 1 && (strcmp(method, SafetyFromInside) || strcmp(method,SafetyFromOutside))
            diameter = values(2);
            center = points(2,:);
            plotSphere(diameter, center);
        end
         scatter3 (points(:,1), points(:,2), points(:,3), pointsize, values, 'filled'); % sizes        
         colorbar;
         title(tit);
        hold off;
    else
        disp(['Too many points ' num2str(len) ' , set max parameter to visualize maximally ' num2str(maxPoints)]);
    end
    axis square;            %# Make the scaling on the x, y, and z axes equal    
end

function mag = magnitude (values)
    len = length(values);
    mag = zeros(len, 1);
    for i=1:len
       mag(i,:) = norm(values(i,:));
    end
end

function plotSphere(r,center)
    [x,y,z] = sphere(100);      %# Makes a 21-by-21 point unit sphere 
    obj = surface(r*x+center(1),r*y+center(2),r*z+center(3),'Facecolor','b','EdgeColor','m');  %# Plot the surface, multiplying unit coordinates with radii 
    alpha(obj, 0.1);
end

function res = arrayfilter(array, indexes)
    len = length(indexes);
    res = array(1:len,:);
    for i=1:len
        index = indexes(i);
        res(i,:) = array(index,:);
    end
end
