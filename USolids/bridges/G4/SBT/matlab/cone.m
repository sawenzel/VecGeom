
% Load the data. The winds data set contains six 3-D arrays: u, v, and w specify
% the vector components at each of the coordinates specified in x, y, and z. The 
% coordinates define a lattice grid structure where the data is sampled within the 
% volume.
load wind

% Now establish the range of the data to place the slice planes and to specify 
% where you want the cone plots (min, max):
xmin = min(x(:));
xmax = max(x(:));
ymin = min(y(:));
ymax = max(y(:));
zmin = min(z(:));

% Use daspect to set the data aspect ratio of the axes before calling coneplot.
daspect([2,2,1])

% Decide where in data space you want to plot cones. This example selects the 
% full range of x and y in eight steps and the range 3 to 15 in four steps in z 
% using linspace and meshgrid.
xrange = linspace(xmin,xmax,8);
yrange = linspace(ymin,ymax,8);
zrange = 3:4:15;
[cx cy cz] = meshgrid(xrange,yrange,zrange);

% Draw the cones, setting the scale factor to 5 to make the cones larger than 
% the default size:
hcones = coneplot(x,y,z,u,v,w,cx,cy,cz,5);
% 'quiver'
% Set the coloring of each cone using FaceColor and EdgeColor:
set(hcones,'FaceColor','red','EdgeColor','none')

return;

% Calculate the magnitude of the vector field (which represents wind speed) to 
% generate scalar data for the slice command:
hold on
wind_speed = sqrt(u.^2 + v.^2 + w.^2);
% Create slice planes along the x-axis at xmin and xmax, along the y-axis at 
% ymax, and along the z-axis at zmin:
hsurfaces = slice(x,y,z,wind_speed,[xmin,xmax],ymax,zmin);
% Specify interpolated face color so the slice coloring indicates wind speed, 
% and do not draw edges (hold, slice, FaceColor, EdgeColor):
set(hsurfaces,'FaceColor','interp','EdgeColor','none')
hold off

% Use the axis command to set the axis limits equal to the range of the data.
axis tight; 
% Orient the view to azimuth = 30 and elevation = 40. (rotate3d is a useful 
% command for selecting the best view.)
view(30,40);axis off
% Select perspective projection to provide a more realistic looking volume 
% using camproj:
camproj perspective; 
% Zoom in on the scene a little to make the plot as large as possible using camzoom:
camzoom(1.5)

% The light source affects both the slice planes (surfaces) and the cone plots 
% (patches). However, you can set the lighting characteristics of each independently:

% Add a light source to the right of the camera and use Phong lighting to give the 
% cones and slice planes a smooth, three-dimensional appearance using camlight and lighting:
camlight right; lighting phong

% Increase the value of the AmbientStrength property for each slice plane to improve 
% the visibility of the dark blue colors: 
set(hsurfaces,'AmbientStrength',.6)
% Increase the value of the DiffuseStrength property of the cones to brighten particularly
% those cones not showing specular reflections:
set(hcones,'DiffuseStrength',.8)
