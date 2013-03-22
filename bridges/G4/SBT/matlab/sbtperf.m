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
  perftab1st3 = perftab(1:3, :);
  s2 = sum(perftab1st3);
  if s(1,1) > 0
      names{end+1} = 'Geant4';
%       names{end} = 'G4TessellatedSolid';
      disp(['Total for Geant4: ' num2str(s(1,1)) ' | 1st 3 methods: ' num2str(s2(1,1))]);
  end
  if s(1,2) > 0
      names{end+1} = 'ROOT';
      disp(['Total for ROOT: ' num2str(s(1,2)) ' | 1st 3 methods: ' num2str(s2(1,2))]);
  end
  if s(1,3) > 0
      names{end+1} = 'USolids';
      disp(['Total for USolids: ' num2str(s(1,3)) ' | 1st 3 methods: ' num2str(s2(1,3))]);
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
