function res = sbtdot(filename1, filename2)
    var1 = load (filename1);
    var2 = load (filename2);
    len = length(var2);
    res = zeros(len, 1);
    for i=1:len
        d1 = var1(i,:);
        d2 = var2(i,:);
        d = dot (d1, d2);
        res(i) = d;         
    end
end
