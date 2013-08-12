function res = sbtplotpart(method, software1, software2, first, count, color, submethod)
    realcount = count;
    if count == 1
        count = 2;
    end
    if nargin < 7
      submethod = '';
    end
    filename1 = [method submethod software1 '.dat'];
    if isempty(software2)
        software = software1;
        filename2 = '';
    else
        filename2 = [method submethod software2 '.dat'];
        software = [software1 ' - ' software2];
    end
    if (strcmp (software2, 'Directions'))
       values = sbtdot(filename1, filename2);
    else
       values = dif(filename1 , filename2);
    end
    valuesall = values;
    values = subarray (values, first, count);
    if (~isempty(software2))
        points = load([method 'Points.dat']);
        directions = load([method 'Directions.dat']);
        methodSoftware = [ method submethod ' (' software1 ' - ' software2 ')'];
        disp(['Evaluating differences for ' methodSoftware]);
        res = sbtdifferences(values, first, realcount, points, directions);
        differencesCount = length(res);
        if (differencesCount)
            difstr = ['Total number of different points for ' methodSoftware ': ' int2str(differencesCount)];
            disp(difstr);
        end
    end
    if nargin < 6
      color = 'b';
    end
    filenameLegend = [method 'Legend.dat'];
    counts = load (filenameLegend);
    maxPointsInside = counts(1);
    maxPointsSurface = counts(2);
    maxPointsOutside = counts(3);
    offsetInside = 0;
    offsetSurface = maxPointsInside;
    offsetOutside = offsetSurface + maxPointsSurface;    
    grid on;
    valid = strcmp( submethod, 'Valid');
    if ((strcmp( method, 'Normal') || strcmp( submethod, 'Normal') || strcmp( submethod, 'SurfaceNormal')) && valid == 0)
        if iscell(color)
            h = plot (values(1));
            set(h,'Color', rgb(color{1}));
            h = plot (values(2));
            set(h,'Color', rgb(color{2}));
            h = plot (values(3));    
            set(h,'Color', rgb(color{3}));
        else
            plot(values);
        end;
        legend([method submethod ' X'], [method submethod ' Y'], [method submethod ' Z']);
    else
        len = length(values);
        range = first:1:first+len-1;        
        if (realcount == 1)
%             range(2) = range(1);
            values(2) = values(1);
        end
        
%         h = plot (range, values, color); original was like this
%        set(h,'Color', color);

%         if (count < 0)
%           count = length(valuesall) - (first - 1);
%         end
        name = {};  
        
         hold on;
         if maxPointsInside
           h = subplot(valuesall, first, count, 1 + offsetInside, maxPointsInside);
           if h
            set(h,'Color', 'r');
            name = [name; {[method submethod ' (Inside)']}];         
           end
         end
         hold on;
         if maxPointsSurface
           h = subplot(valuesall, first, count, 1 + offsetSurface, maxPointsSurface);
           if h
             set(h,'Color', 'm');
             name = [name; {[method submethod ' (Surface)']}];         
           end
         end
         hold on;
          if maxPointsOutside
            h = subplot(valuesall, first, count, 1 + offsetOutside, maxPointsOutside);
            if h          
              set(h,'Color', 'b');
              name = [name; {[method submethod ' (Outside)']}];         
            end
          end
        legend(name);
    end
    label = '';
    if maxPointsInside
        label = [label 'Inside=[' int2str(offsetInside+1) ' .. ' int2str(offsetInside + maxPointsInside) '], '];
    end    
    if maxPointsSurface
        label = [label 'Surface=[' int2str(offsetSurface+1) ' .. ' int2str(offsetSurface + maxPointsSurface) '], '];
    end
    if maxPointsOutside
        label = [label 'Outside=[' int2str(offsetOutside+1) ' .. ' int2str(offsetOutside + maxPointsOutside) ']']; 
    end    
    tit = ['Folder: ' dirname() ' ; Software: ' software];
    title(tit);
    xlabel(label);
end

function h = subplot(values, first, count, offset, max)
           labels = 1:length(values);
           labels = labels';
           values = subarray(values, offset, max);
           labels = subarray(labels, offset, max);
           offset = first - offset + 2;
           values = subarray(values, offset, count);
           labels = subarray(labels, offset, count);
           if (~isempty(values))
                h = plot(labels, values);
           else
               h = 0;
           end
           
end
