function res = sbtpolyhedra(method)
    filenameVertices = [method 'Vertices.dat'];
    filenameTriangles = [method 'Triangles.dat'];
    filenameQuads = [method 'Quads.dat'];
    vertices = load(filenameVertices);
    quads = load(filenameQuads);
    triangles = load(filenameTriangles);
    hold on;
    h = patch('vertices',vertices,'faces',quads,'facecolor','c','edgecolor','b'); % draw faces in green
    alpha(h,.1);
    h = patch('vertices',vertices,'faces',triangles,'facecolor','c','edgecolor','b'); % draw faces in green
    alpha(h,.1);
    view(3), grid on;% default view with grid
    axis equal;
end
