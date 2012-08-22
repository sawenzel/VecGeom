
% datasets = {'polyhedra-4-p10k', 'polyhedra-9-p10k', 'polyhedra-12-p10k', 'polyhedra-22-p10k', 'polyhedra-32-p10k', 'polyhedra-42-p10k', 'polyhedra-52-p10k', 'polyhedra-62-p10k', 'polyhedra-72-p10k', 'polyhedra-82-p10k', 'polyhedra-92-p10k'};
% datasets = {'polyhedra-5-4-p10k', 'polyhedra-5-9-p10k', 'polyhedra-5-12-p10k', 'polyhedra-5-22-p10k', 'polyhedra-5-32-p10k', 'polyhedra-5-42-p10k', 'polyhedra-5-52-p10k', 'polyhedra-5-62-p10k', 'polyhedra-5-72-p10k', 'polyhedra-5-82-p10k', 'polyhedra-5-92-p10k'};
% datasets = {'polyhedra-8-4-p10k', 'polyhedra-8-9-p10k', 'polyhedra-8-12-p10k', 'polyhedra-8-22-p10k', 'polyhedra-8-32-p10k', 'polyhedra-8-42-p10k', 'polyhedra-8-52-p10k', 'polyhedra-8-62-p10k', 'polyhedra-8-72-p10k', 'polyhedra-8-82-p10k', 'polyhedra-8-92-p10k'};

datasets = {'polycone-3-p10k','polycone-5-p10k','polycone-7-p10k','polycone-10-p10k'};
datasets = {'polycone-3-p10k','polycone-5-p10k','polycone-7-p10k','polycone-10-p10k', 'polycone-20-p10k', 'polycone-30-p10k', 'polycone-40-p10k',  'polycone-50-p10k',  'polycone-60-p10k',  'polycone-70-p10k',  'polycone-80-p10k', 'polycone-90-p10k'};
datasets = {'multiunion-1-p10k','multiunion-2-p10k','multiunion-3-p10k','multiunion-4-p10k', 'multiunion-5-p10k','multiunion-6-p10k','multiunion-7-p10k','multiunion-8-p10k', 'multiunion-9-p10k', 'multiunion-10-p10k', 'multiunion-15-p10k', 'multiunion-25-p10k', 'multiunion-50-p10k' , 'multiunion-100-p10k'};
n = length(datasets);

sbtperfall('Inside', datasets(:,1:n-1));
sbtperfall('Normal', datasets(:,1:n-3));

sbtperfall('SafetyFromInside', datasets(:,1:n-2));
sbtperfall('SafetyFromOutside', datasets(:,1:n-2));

sbtperfall('DistanceToIn', datasets(:,1:n));
sbtperfall('DistanceToOut', datasets(:,1:n-4));

% cd polyconep10k-1; sbtperf; cd ..
% cd polyconep10k-2; sbtperf; cd ..
% cd polyconep10k-3; sbtperf; cd ..
% cd polyconep10k-4; sbtperf; cd ..
% cd polyconep10k-5; sbtperf; cd ..
% cd polyconep10k-6; sbtperf; cd ..
% cd polyconep10k-7; sbtperf; cd ..
% cd polyconep10k-8; sbtperf; cd ..
% cd polyconep10k-9; sbtperf; cd ..
%    
