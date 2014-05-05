function sbtplotallone(software1, software2, first, count)
    if nargin < 1
        disp('Syntax is sbtplotallone (software1, [software2])');
        disp('Example: sbtplotallone(''Geant4'', ''Root'');');
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
    figure;
    hold on;
    sbtplotpart('Inside', software1, software2, first, count, 'k');
    sbtplotpart('SafetyFromInside', software1, software2,  first, count, 'r');
    sbtplotpart('SafetyFromOutside', software1, software2,  first, count, 'g');
    sbtplotpart('DistanceToOut', software1, software2, first, count, 'b');
    sbtplotpart('DistanceToIn', software1, software2, first, count, 'm');
    sbtplotpart('Normal', software1, software2,  first, count, {'c', 'y', 'orange'});
    legend('Inside', 'SafetyFromInside', 'SafetyFromOutside', 'DistanceToOut', 'DistanceToIn', 'Normal-X', 'Normal-Y', 'Normal-Z');
    xlabel('All points');
    hold off;
end
