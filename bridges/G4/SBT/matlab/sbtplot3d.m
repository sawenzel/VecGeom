function res = sbtplot3d(method, software1, software2, first, count)
    if (nargin < 4)
        first = 1;
    end
    if (nargin < 5)
        count = -1;
    end
if nargin < 2
    disp('Syntax is sbtplot3d(method,software1, [software2])');
    disp('Example: sbtplot3d(''Inside'', ''Geant4'', ''Root'');');
    return;
end
if nargin < 3
    sbtpoints(method, software1, '', first, count);
else
    sbtpoints(method, software1, software2, first, count);
end
sbtpolyhedra(method);
xlabel('X axis');
ylabel('Y axis');
zlabel('Z axis');
