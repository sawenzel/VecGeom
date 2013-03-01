function res = sbtperf(scale)
     % disp('Syntax is sbtperf [log]');
  labels = textread('PerformanceLabels.txt','%s');
  perftab = load ('Performance.dat');
  scrsz = get(0,'ScreenSize');
  figure('Position',[100 100 scrsz(3)/3 scrsz(4)/3])
  s = sum(perftab);  
  names = {};
  perftabreduced = perftab;
  perftabreduced( :, ~any(perftabreduced,1) ) = [];  %columns
  if s(1,1) > 0
      names{end+1} = 'Geant4';
%       names{end} = 'G4TessellatedSolid';
  end
  if s(1,2) > 0
      names{end+1} = 'ROOT';
  end
  if s(1,3) > 0
      names{end+1} = 'USolid';
%       names{end} = 'New G4TessellatedSolid';
  end
  bar(perftabreduced);
  set(gca,'XTickLabel',labels);
  if nargin > 0
    set(gca,'YScale', scale);
  end
  legend(names);
  xlabel ('Method');
  ylabel ('Time per one method call [nanoseconds]');
  tit = ['Performance of methods at folder ' dirname()];
  title (tit);
  [m,n] = size(perftab);
  for i=1:m
      g4 = perftab(i,1);      
      us = perftab(i,3);
      rt = perftab(i,2);
      if rt > 0
        ratio = rt / g4;
        lab = labels{i};
        bonus = '';
        if ratio > 0
            bonus = [' (' num2str(1 / ratio) ')'];
        disp(['Ratio for Geant4 vs. Root for ', lab, ' is '  num2str(ratio) bonus]);
        end
      end      
      if us > 0
        ratio = us / g4;
        lab = labels{i};
        bonus = '';
        if ratio > 0
            bonus = [' (' num2str(1 / ratio) ')'];
        disp(['Ratio for Geant4 vs. USolids for ', lab, ' is '  num2str(ratio) bonus]);
        end
      end
  end
end
