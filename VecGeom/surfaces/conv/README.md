# Key Information about Surface Convertors

## Useful insights for all shapes

- Quadrilateral and Triangle Masks require a certain order of the points given as parameters. The points should be entered counter-clockwise in the (xOy) plane. The normal vector of the surface will point according to the right-hand rule.

## Box
A simple box.

### Variables:
- *dx*  - half length of the *X* edge
- *dy*  - half length of the *Y* edge
- *dz*  - half length of the *Z* edge

## Parallelepiped
Inclined prism having as top and bottom faces identical parallelograms parallel with the (xOy) plane, placed at +/-dz. Each parallelogram has two edges parallel with Ox and the other two inclined with the angle alpha with respect to Oy. The vector connecting the center of the bottom and top parallelograms crosses the origin of the reference frame and has theta/phi inclination with respect to the z axis in spherical coordinates.

### Variables:
- *dx*    - half length of the top/bottom parallelogram edges parallel with *X*
- *dy*    - half length of the projection of the top/bottom parallelogram edges over *Y*
- *dz*    - half distance along *Oz* between the top and bottom planes
- *theta* - polar angle of the parallelepiped axis in radians
- *phi*   - azimuthal angle of the parallelepiped axis in radians
- *alpha* - angle between *Oy* axis and the *Y* edge in radians.

## Tube
A z-aligned tube segment having inner and outer radii, having a cut in a range of the azimuthal angle.

### Variables:
- *z*    - half length along the z axis
- *rmin* - inner radius
- *rmax* - outer radius
- *sphi* - starting phi angle of the segment in radians
- *dphi* - delta angle of the segment in radians, the interval which contains the shape.

## Cone
A z-aligned cone segment having inner and outer conical surfaces, having a cut in a range of the azimuthal angle.

### Variables:
- *dz*    - half length along the z axis
- *rmin1* - inner radius at *-dz*
- *rmax1* - outer radius at *-dz*
- *rmin2* - inner radius at *+dz*
- *rmax2* - outer radius at *+dz*
- *sphi*  - starting angle of the segment in radians;
- *dphi*  - delta angle of the segment in radians, the interval which contains the shape.

## Trapezoid
A trapezoidal inclined prism having as top and bottom faces similar trapezes parallel with the (xOy) plane, placed at +/-dz. Each trapeze has two edges parallel with Ox and the segment connecting their centers is inclined with the angle alpha with respect to Oy. The vector connecting the center of the bottom and top faces crosses the origin of the reference frame and has theta/phi inclination with respect to the z axis in spherical coordinates.

### Variables:
- *dz*     - half length along the z axis
- *theta*  - polar angle of the parallelepiped axis in radians
- *phi*    - azimuthal angle of the parallelepiped axis in radians
- *dy1*    - half length of the projection of the bottom trapeze on the y axis
- *dx1*    - half length of the bottom trapeze x-parallel lower edge
- *dx2*    - half length of the bottom trapeze x-parallel upper edge
- *alpha1* - angle between the segment connecting the centers of the bottom trapeze parallel x edges with the y axis
- *dy2*    - half length of the projection of the top trapeze on the y axis
- *dx3*    - half length of the top trapeze x-parallel lower edge
- *dx4*    - half length of the top trapeze x-parallel upper edge
- *alpha2* - angle between the segment connecting the centers of the top trapeze parallel x edges with the y axis

### Important note
The condition of validity for the trapezoids is that the bottom and top trapezes are similar (scaled). So not all combinations of paramemters are valid, the following identities must hold:
- dy1/dy2 = dx1/dx3 = dx2/dx4
- alpha1 = alpha2

## Extruded
Extrusion solid created by translating and scaling a blueprint arbitrary polygon parallel with (xOy) to an arbitrary number of z positions. The blueprint polygon vertices are defined counter-clockwise in the xOy plane (as per PlanarPolygon::GetOrientation). The corresponding polygon vertices between any two consecutive z sections are connected with lines.

### Variables:
- *nvertices* - number of vertices of the blueprint polygon
- *vertices*  - (x,y) coordinates of each vertex of the blueprint polygon;
- *nsections* - number of z sections;
- *sections*  - origin vector and scaling factor for each section;

## Trd
A simple right trapezoid prism having top and bottom rectangular faces parallel with (xOy).

### Variables:
- *x1* - half length of the *X* edge at -dz
- *x2* - half length of the *X* edge at +dz
- *y1* - half length of the *Y* edge at -dz
- *y2* - half length of the *Y* edge at +dz
- *z*  - half length along *Z*

## Polyhedron
A z-connected series of right prisms. The (xOy)-parallel faces are segments in a defined phi range of regular polygons having N edges. For each z section of cut of the shape, the polygon extends between an inner and an outer radii, i.e. the outer polygon has an inner hole. The radii of the outer and inned *inscribed* circles are given as parameters per section. The corresponding inner and outer vertices at each section are connected with straight lines.

### Variables:
- *phiStart*    - starting angle in radians
- *phiDelta*    - delta angle in radians, the interval which contains the shape
- *sideCount*   - number of sides of the polygon
- *zPlaneCount* - number of different z sections
- *zPlanes*     - z positions of the sections
- *rMin*        - radius of the inscribed circel of the inner polygons at each section
- *rMax*        - radius of the inscribed circle of the outer polygons at each section
