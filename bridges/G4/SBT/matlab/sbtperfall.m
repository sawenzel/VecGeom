function res = sbtperfall(method, folders)

folder = folders{1};
filenameLabels = [folder '/PerformanceLabels.txt'];
disp(['Method ' method ' ...']);
labels = textread(filenameLabels,'%s');
% folders = cell(nargin-1);
foldersCount = length(folders);
perf = zeros (foldersCount,3);
len = length(labels);
index = 0;
for i = 1:len
    if (strcmp (labels{i}, method))
        index = i;
    end
end
if (index == 0)
   disp(['Unknown method ' method]); 
   return
end
ticks = folders;
for i = 1:foldersCount
    folder = folders{i};
    disp(['Reading folder ' folder ' ...']);
    filename = [folder '/' 'Performance.dat'];
    perftab = load (filename);
    perf(i,:) = perftab(index,:);
    [~,tick] = strtok(folder, '-');
    a = strsplit(folder, '-');
    count = a{3};
%     tick = strrep(folder, '-', ' ');
    ticks{i} = tick(2:length(tick));
    ticks{i} = count;
end
figure;
bar(perf);
set(gca,'XTickLabel',ticks);
legend('Geant4','ROOT','USolid');
% legend('A','B','C');
xlabel ('Count of solid parts');
ylabel ('Time [nanoseconds per operation]');
dirname = cd();
paths = strsplit(dirname, '\');
folder = paths{length(paths)};
tit = ['Performance of method ' method ' at folder ' folder];
title (tit);
end
