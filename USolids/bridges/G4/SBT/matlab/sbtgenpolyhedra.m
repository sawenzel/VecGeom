
rBeg = [0];
zBeg = [1];
rMidOrig = [2, 5,  5, 15,  8, 8.1, 8.1,  8, 10, 10];
zMidOrig = [2, 7, 12, 20, 25,  26,  27, 30, 32, 35];
rEnd = [0];
zEnd = [40];

for i = 1 : 9
    len = length(rMidOrig);
    rMid = zeros(1, len * i); 
    zMid = zeros(1, len * i);
    for j = 1 : i
        for k = 1 : len
            index = (j - 1) * len + k;
            rMid(index) = rMidOrig(k);
            zMid(index) = 40 * (j - 1) + zMidOrig(k);
        end
    end
    zEnd = 40 * i;    
    r = [rBeg, rMid, rEnd];
    z = [zBeg, zMid, zEnd];
    len = length(r);
    outr = sprintf('%g,', r);
    outz = sprintf('%g,', z);
    outr = outr(1:end-1);
    outz = outz(1:end-1);
    disp(['# ## Polyhedra ' num2str(len) ' #######################################################################']);
    disp('#');
    disp(['/solid/G4Polyhedra 0 360 8 ' num2str(len) ' (' outr ') (' outz ')']);
    disp(['/performance/errorFileName log/polyhedra-' num2str(len) '-p10k/polyhedrap.a1.log']);
%     disp('/performance/repeat 100');
    disp('/control/execute geant4/performance.geant4');
    disp('#');     
    figure; plot(z,r);
end

% C=[A;B] ( [A,B] will put them vertically )