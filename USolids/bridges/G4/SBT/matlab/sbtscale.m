function res=sbtscale()
    figure;
    hold on;
%    plotone('Voxel', 'm');
%     plotone('VoxelOriginal', 'g');
     plotone('Voxel2', 'r');
    plotone('Base', 'b');
    legend('New Voxels Implementation', 'New Voxels Implementation Test', 'Old Voxels implementation', 'Base implementation');
%     legend('Multi-Union 1st version', 'Multi-Union 2nd version', 'Boolean solid');
    legend('Multi-Union', 'Boolean solid');
    xlabel('Number of nodes');
    ylabel('Time of execution [s]');
    title('Scaling of Multi-Union inside method with boxes');
end

function plotone(base,color)
    times = load(['times' base '.dat']);
    nodes = load(['nodes' base '.dat']);
    total = sum(times);
    disp(['Total time of ' base ' ' num2str(total)]);
    plot(nodes,times,color);
end
