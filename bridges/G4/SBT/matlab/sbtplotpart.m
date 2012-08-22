function res = sbtplotpart(method, software1, software2, first, count, color)
    realcount = count;
    if count == 1
        count = 2;
    end
    filename1 = [method software1 '.dat'];
    if isempty(software2)
        software = software1;
        filename2 = '';
    else
        filename2 = [method software2 '.dat'];
        software = [software1 ' - ' software2];
    end
    if (strcmp (software2, 'Directions'))
       values = sbtdot(filename1, filename2);
    else
       values = dif(filename1, filename2);
    end    
    values = subarray (values, first, count);
    if (~isempty(software2))
        methodSoftware = [ method ' (' software1 '-' software2 ')'];
        disp(['Evaluating differences for ' methodSoftware]);
        res = sbtdifferences(values, first, realcount);
        differencesCount = length(res);
        if (differencesCount)
            difstr = ['Total number of different points for ' methodSoftware ': ' int2str(differencesCount)];
            disp(difstr);
        end
    end
    if nargin < 6
      color = 'b';
    end
    grid on;
    if (strcmp( method, 'Normal'))
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
        legend([method ' X'], [method ' Y'], [method ' Z']);    
    else
        len = length(values);
        range = first:1:first+len-1;        
        if (realcount == 1)
%             range(2) = range(1);
            values(2) = values(1);
        end
        h = plot (range, values, color);
        set(h,'Color', color);
        legend(method);
    end
    filenameLegend = [method 'Legend.dat'];
    counts = load (filenameLegend);
    maxPointsInside = counts(1);
    maxPointsSurface = counts(2);
    maxPointsOutside = counts(3);
    offsetInside = 0;
    offsetSurface = maxPointsInside;
    offsetOutside = offsetSurface + maxPointsSurface;
    label = '';
    if maxPointsInside
        label = [label 'Inside=[' int2str(offsetInside) ' .. ' int2str(offsetInside + maxPointsInside) '], '];
    end    
    if maxPointsSurface > 0
        label = [label 'Surface=[' int2str(offsetSurface) ' .. ' int2str(offsetSurface + maxPointsSurface) '], '];
    end
    if maxPointsOutside
        label = [label 'Outside=[' int2str(offsetOutside) ' .. ' int2str(offsetOutside + maxPointsOutside) ']']; 
    end    
    tit = ['Folder: ' dirname() ' ; Software: ' software];
    title(tit);
    xlabel(label);
end
