function var = sbtplot(method, software1, software2, first, count, color)
if nargin < 2
    disp('Syntax is sbtplot(method,software1, [software2] [color])');
    disp('Example: sbtplot(''Inside'', ''Geant4'', ''Root'');');
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
    color = 'b';
end
figure;
sbtplotpart(method, software1, software2, first, count, color);
