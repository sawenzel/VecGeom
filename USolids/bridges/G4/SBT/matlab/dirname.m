function folder = dirname(dir)
    if nargin < 1
        dir = cd();
    end
    paths = strsplit(dir, '\');
    folder = paths{length(paths)};
end
