function res = perf()
  labels = textread('PerformanceLabels.txt','%s');
  perftab = load ('Performance.dat');
  scrsz = get(0,'ScreenSize');
  figure('Position',[100 100 scrsz(3)/3 scrsz(4)/3])
  bar(perftab);
  set(gca,'XTickLabel',labels);
  legend('Geant4','ROOT','USolid');
%   legend('A','B','C');
  xlabel ('Method');
  ylabel ('Time per one method call [nanoseconds]');
  tit = ['Performance of methods at folder ' dirname()];
  title (tit);
end
