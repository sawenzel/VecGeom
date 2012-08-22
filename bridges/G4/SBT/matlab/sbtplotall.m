function sbtplotall(software1, software2)
    if nargin < 1
        disp('Syntax is sbtplotall(software1, [software2])');
        disp('Example: sbtplotall(''Geant4'', ''Root'');');
        return;
    end
    if nargin < 2
      software2 = '';
    end
    sbtplot('Inside', software1, software2);
    sbtplot('Normal', software1, software2);
    sbtplot('SafetyFromInside', software1, software2);
    sbtplot('SafetyFromOutside', software1, software2);
    sbtplot('DistanceToIn', software1, software2);    
    sbtplot('DistanceToOut', software1, software2);
end
