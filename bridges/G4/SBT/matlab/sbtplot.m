function var = sbtplot(method, software1, software2, first, count, color)
if nargin < 2
    disp('Syntax is sbtplot(method, software1, [software2] [first] [count] [color])');
    disp('Example: sbtplot(Inside, Geant4, Root, 1);');
    return;
end
if nargin < 3
    software2 = '';
end
if nargin < 4
    first = 1;
end
if nargin < 5
    count = -1;
end
if nargin < 6
    color = '';
end
cell = {'', 'Normal', 'Convex', 'SurfaceNormal', 'Valid'};
len = length(cell);
for i=1:len
  submethod = cell{i};
  filename = [method submethod software1 '.dat'];
  exists = exist(filename, 'file');
  if exists
    figure;
    sbtplotpart(method, software1, software2, first, count, color, submethod);
  end
end
