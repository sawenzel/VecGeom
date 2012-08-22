function res = sbtplot(method, software, color, var)
if nargin < 3
  color = 'b';
end
if nargin < 4
    filename = [method software '.dat'];
    var = load (filename);
end
plot (var, color);
grid on % Turn on grid lines for this plot
if (strcmp( method, 'Normal'))
    legend([method ' X'], [method ' Y'], [method ' Z - ']);
else
    legend([method]);    
end
filenameLegend = [method 'Legend.txt'];
lines = textread(filenameLegend,'%s','delimiter','\n','whitespace','');
line = lines{1};
dirname = cd();
paths = strsplit(dirname, '\');
folder = paths{length(paths)};
tit = ['Folder: ' folder ' ; Software: ' software];
title(tit);
xlabel(line);
end
