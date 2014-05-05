
rBeg = [0, 1, 3, 3, 1,  4,  1, 0];
zBeg = [1, 3, 3, 4, 10, 15, 15, 18];
rMidOrig = [2,  2 ,  1,  1,  2,  2,  1,  1,  2,  2,  1,  1,  2,  2,  1,  1,  2,  2,  1,  1,  2,  2,  1,  1,  2,  2,  1,  1,  2,  2,  1,  1,  2,  2,  1,  1,  2,  2,  1,  1,  2,  2,  1,  1,  2,  2,  1,  1,  2,  2,  1,  1,  2,  2,  1,  1,  2,  2];
zMidOrig = [18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47];
rEnd = [1, 0];
zEnd = [184, 185];

for j = 1 : 9
    e = 5*j;
    len = 2*e;
    rMid = zeros(1, len); 
    zMid = zeros(1, len);
    for i = 1 : e
        val = 1 + mod(i, 2);
        rMid(2*i) =  val;
        rMid(2*i - 1) = val;
        z = 18 + i;
        zMid(2*i) = z;
        zMid(2*i - 1) = z-1;
        zEnd = [2*z-1, 2*z];
    end
    r = [rBeg, rMid, rEnd];
    z = [zBeg, zMid, zEnd];
    len = length(r);
    outr = sprintf('%g,', r);
    outz = sprintf('%g,', z);
    outr = outr(1:end-1);
    outz = outz(1:end-1);
    disp(['# ## Polycone ' num2str(len) ' #######################################################################']);
    disp('#');
    disp(['/solid/G4Polycone 0 360 ' num2str(len) ' (' outr ') (' outz ')']);
    disp(['/performance/errorFileName log/polycone-' num2str(len) '-p10k/polyconep.a1.log']);
    disp('/performance/repeat 100');
    disp('/control/execute geant4/performance.geant4');
    disp('#');     
    figure; plot(z,r);
end

% C=[A;B] ( [A,B] will put them vertically )