r = [1 0 0];       %# start
w = [.9 .9 .9];    %# middle
b = [0 0 1];       %# end

len = 128

%# colormap of size 64-by-3, ranging from red -> white -> blue
c1 = zeros(len,3); c2 = zeros(len,3);
for i=1:3
    c1(:,i) = linspace(r(i), w(i), len);
    c2(:,i) = linspace(w(i), b(i), len);
end
c = [c1(1:end-1,:);c2];

surf(peaks), shading interp
caxis([-8 8]), colormap(c), colorbar
