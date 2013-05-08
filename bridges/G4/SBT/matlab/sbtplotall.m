function sbtplotall(software1, software2, first, count)
    if nargin < 1
        disp('Syntax is sbtplotall(software1, [software2] [first] [count])');
        disp('Example: sbtplotall(''Geant4'', ''Root'');');
        return;
    end
    if nargin < 3
      first = 1;
    end
    if nargin < 4
      count = -1;
    end    
    if nargin < 2
      software2 = '';
    end    
    sbtplot('Inside', software1, software2, first, count);
    sbtplot('Normal', software1, software2, first, count);
    sbtplot('SafetyFromInside', software1, software2, first, count);
    sbtplot('SafetyFromOutside', software1, software2, first, count);
    sbtplot('DistanceToIn', software1, software2, first, count);
    sbtplot('DistanceToOut', software1, software2, first, count);
end
