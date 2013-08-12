//
// ********************************************************************
// * License and Disclaimer																					 *
// *																																	*
// * The	Geant4 software	is	copyright of the Copyright Holders	of *
// * the Geant4 Collaboration.	It is provided	under	the terms	and *
// * conditions of the Geant4 Software License,	included in the file *
// * LICENSE and available at	http://cern.ch/geant4/license .	These *
// * include a list of copyright holders.														 *
// *																																	*
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work	make	any representation or	warranty, express or implied, *
// * regarding	this	software system or assume any liability for its *
// * use.	Please see the license in the file	LICENSE	and URL above *
// * for the full disclaimer and the limitation of liability.				 *
// *																																	*
// * This	code	implementation is the result of	the	scientific and *
// * technical work of the GEANT4 collaboration.											*
// * By using,	copying,	modifying or	distributing the software (or *
// * any work based	on the software)	you	agree	to acknowledge its *
// * use	in	resulting	scientific	publications,	and indicate your *
// * acceptance of all terms of the Geant4 Software license.					*
// ********************************************************************
//
//
// $Id: UPolyhedron.h,v 1.25 2009-10-28 13:38:54 allison Exp $
// GEANT4 tag $Name: not supported by cvs2svn $
//
//
// Class Description:
// UPolyhedron is an intermediate class between description of a shape
// and visualization systems. It is intended to provide some service like:
//	 - polygonization of shapes with triangulization (quadrilaterization)
//		 of complex polygons;
//	 - calculation of normals for faces and vertices;
//	 - finding result of boolean operation on polyhedra;
//
// Public constructors:
//
//	 UPolyhedronBox (dx,dy,dz)
//																				- create polyhedron for Box;
//	 UPolyhedronTrd1 (dx1,dx2,dy,dz)
//																				- create polyhedron for Trd1;
//	 UPolyhedronTrd2 (dx1,dx2,dy1,dy2,dz)
//																				- create polyhedron for Trd2;
//	 UPolyhedronTrap (dz,theta,phi, h1,bl1,tl1,alp1, h2,bl2,tl2,alp2)
//																				- create polyhedron for Trap;
//	 UPolyhedronPara (dx,dy,dz,alpha,theta,phi)
//																				- create polyhedron for Para;
//	 UPolyhedronTube (rmin,rmax,dz)
//																				- create polyhedron for Tube;
//	 UPolyhedronTubs (rmin,rmax,dz,phi1,dphi)
//																				- create polyhedron for Tubs;
//	 UPolyhedronCone (rmin1,rmax1,rmin2,rmax2,dz)
//																				- create polyhedron for Cone;
//	 UPolyhedronCons (rmin1,rmax1,rmin2,rmax2,dz,phi1,dphi)
//																				- create polyhedron for Cons;
//	 UPolyhedronPgon (phi,dphi,npdv,nz, z(*),rmin(*),rmax(*))
//																				- create polyhedron for Pgon;
//	 UPolyhedronPcon (phi,dphi,nz, z(*),rmin(*),rmax(*))
//																				- create polyhedron for Pcon;
//	 UPolyhedronSphere (rmin,rmax,phi,dphi,the,dthe)
//																				- create polyhedron for Sphere;
//	 UPolyhedronTorus (rmin,rmax,rtor,phi,dphi)
//																				- create polyhedron for Torus;
//	 UPolyhedronEllipsoid (dx,dy,dz,zcut1,zcut2)
//																				- create polyhedron for Ellipsoid;
// Public functions:
//
//	 GetNoVertices ()			 - returns number of vertices;
//	 GetNoFacets ()				 - returns number of faces;
//	 GetNextVertexIndex (index,edgeFlag) - get vertex indeces of the
//														quadrilaterals in order;
//														returns false when finished each face;
//	 GetVertex (index)			- returns vertex by index;
//	 GetNextVertex (vertex,edgeFlag) - get vertices with edge visibility
//														of the quadrilaterals in order;
//														returns false when finished each face;
//	 GetNextVertex (vertex,edgeFlag,normal) - get vertices with edge
//														visibility and normal of the quadrilaterals
//														in order; returns false when finished each face;
//	 GetNextEdgeIndeces (i1,i2,edgeFlag) - get indeces of the next edge;
//														returns false for the last edge;
//	 GetNextEdgeIndeces (i1,i2,edgeFlag,iface1,iface2) - get indeces of
//														the next edge with indeces of the faces
//														to which the edge belongs;
//														returns false for the last edge;
//	 GetNextEdge (p1,p2,edgeFlag) - get next edge;
//														returns false for the last edge;
//	 GetNextEdge (p1,p2,edgeFlag,iface1,iface2) - get next edge with indeces
//														of the faces to which the edge belongs;
//														returns false for the last edge;
//	 GetFacet (index,n,nodes,edgeFlags=0,normals=0) - get face by index;
//	 GetNextFacet (n,nodes,edgeFlags=0,normals=0) - get next face with normals
//														at the nodes; returns false for the last face;
//	 GetNormal (index)			- get normal of face given by index;
//	 GetUnitNormal (index)	- get Unit normal of face given by index;
//	 GetNextNormal (normal) - get normals of each face in order;
//														returns false when finished all faces;
//	 GetNextUnitNormal (normal) - get normals of Unit length of each face
//														in order; returns false when finished all faces;
//	 GetSurfaceArea()			 - get surface area of the polyhedron;
//	 GetVolume()						- get volume of the polyhedron;
//	 GetNumberOfRotationSteps()	 - get number of steps for whole circle;
//	 SetNumberOfRotationSteps (n) - Set number of steps for whole circle;
//	 ResetNumberOfRotationSteps() - reset number of steps for whole circle
//														to default value;
// History:
//
// 20.06.96 Evgeni Chernyaev <Evgueni.Tcherniaev@cern.ch> - initial version
//
// 23.07.96 John Allison
// - added GetNoVertices, GetNoFacets, GetNextVertex, GetNextNormal
//
// 30.09.96 E.Chernyaev
// - added GetNextVertexIndex, GetVertex by Yasuhide Sawada
// - added GetNextUnitNormal, GetNextEdgeIndeces, GetNextEdge
// - improvements: angles now expected in radians
//								 int -> int, double -> double	
// - UVector3 replaced by either UPoint3D or UNormal3D
//
// 15.12.96 E.Chernyaev
// - private functions UPolyhedronAlloc, UPolyhedronPrism renamed
//	 to AllocateMemory and CreatePrism
// - added private functions GetNumberOfRotationSteps, RotateEdge,
//	 RotateAroundZ, SetReferences
// - rewritten UPolyhedronCons;
// - added UPolyhedronPara, ...Trap, ...Pgon, ...Pcon, ...Sphere, ...Torus,
//	 so full List of implemented shapes now looks like:
//	 BOX, TRD1, TRD2, TRAP, TUBE, TUBS, CONE, CONS, PARA, PGON, PCON,
//	 SPHERE, TORUS
//
// 01.06.97 E.Chernyaev
// - RotateAroundZ modified and SetSideFacets added to allow Rmin=Rmax
//	 in bodies of revolution
//
// 24.06.97 J.Allison
// - added static private member fNumberOfRotationSteps and static public
//	 functions void SetNumberOfRotationSteps (int n) and
//	 void ResetNumberOfRotationSteps ().	Modified
//	 GetNumberOfRotationSteps() appropriately.	Made all three functions
//	 inline (at end of this .hh file).
//	 Usage:
//		UPolyhedron::SetNumberOfRotationSteps
//		 (ffVerticesiew -> GetViewParameters ().GetNoOfSides ());
//		pPolyhedron = solid.CreatePolyhedron ();
//		UPolyhedron::ResetNumberOfRotationSteps ();
//
// 19.03.00 E.Chernyaev
// - added boolean operations (add, subtract, intersect) on polyhedra;
//
// 25.05.01 E.Chernyaev
// - added GetSurfaceArea() and GetVolume();
//
// 05.11.02 E.Chernyaev
// - added createTwistedTrap() and createPolyhedron();
//
// 06.03.05 J.Allison
// - added IsErrorBooleanProcess
//
// 20.06.05 G.Cosmo
// - added UPolyhedronEllipsoid
//
// 21.10.09 J.Allison
// - removed IsErrorBooleanProcess (now error is returned through argument)
//

#ifndef U_POLYHEDRON_HH
#define U_POLYHEDRON_HH

#include "UTypes.hh"
#include "UTransform3D.hh"

#ifndef DEFAULT_NUMBER_OF_STEPS
#define DEFAULT_NUMBER_OF_STEPS 24
#endif


typedef UVector3 UPoint3D;
typedef UVector3 UNormal3D;
typedef UVector3 UVector3D;

struct UEdge
{
    int v; // list of vertices
	int f; // neighbouring facet of each edge in given facet
};

class UFacet {
    friend class UPolyhedron;
    friend std::ostream& operator<<(std::ostream&, const UFacet &facet);

 private:
     UEdge edge[4];
 public:
    UFacet(int v1=0, int f1=0, int v2=0, int f2=0, 
           int v3=0, int f3=0, int v4=0, int f4=0)
    { 
        edge[0].v=v1; edge[0].f=f1; edge[1].v=v2; edge[1].f=f2;
        edge[2].v=v3; edge[2].f=f3; edge[3].v=v4; edge[3].f=f4;
    }
};

class UPolyhedron {
    friend std::ostream& operator<<(std::ostream&, const UPolyhedron &ph);

 protected:
    static int fNumberOfRotationSteps;

    std::vector<UPoint3D> fVertices;
    std::vector<UFacet>	fFacets;

	// Re-allocate memory for UPolyhedron
	void AllocateMemory(int nVert, int nFace);

	// Find neighbouring facet
	int FindNeighbour(int iFace, int iNode, int iOrder) const;

	// Find normal at node
	UNormal3D FindNodeNormal(int iFace, int iNode) const;

	// Create UPolyhedron for prism with quadrilateral base
	void CreatePrism();

	// Generate facets by revolving an edge around Z-axis
	void RotateEdge(int k1, int k2, double r1, double r2,
						int v1, int v2, int vEdge,
						bool ifWholeCircle, int ns, int &kface);

	// Set side facets for the case of incomplete rotation
	void SetSideFacets(int ii[4], int vv[4], std::vector<int> &kk, std::vector<double> &r,
						double dphi, int ns, int &kface);

	// Create UPolyhedron for body of revolution around Z-axis
	void RotateAroundZ(int nstep, double phi, double dphi,
						int np1, int np2,
						const std::vector<double> &z, std::vector<double> &r,
						int nodeVis, int edgeVis);

	// For each edge Set reference to neighbouring facet
	void SetReferences();

	// Invert the order on nodes in facets
	void InvertFacets();

 public:

   UPolyhedron() {}

   inline virtual ~UPolyhedron() {}

	// Copy constructor
	UPolyhedron(const UPolyhedron &from);

	// Assignment
	UPolyhedron &operator=(const UPolyhedron &from);

	// Get number of vertices
	int GetNoVertices() const { return fVertices.size() - 1; }

	// Get number of facets
	int GetNoFacets() const { return fFacets.size() - 1; }

	// Transform the polyhedron
//	UPolyhedron &Transform(const UTransform3D &t); // not yet implemented

	// Get next vertex index of the quadrilateral
	bool GetNextVertexIndex(int &index, int &edgeFlag) const;

	// Get vertex by index 
	UPoint3D GetVertex(int index) const;

	// Get next vertex + edge visibility of the quadrilateral
	bool GetNextVertex(UPoint3D &vertex, int &edgeFlag) const;

	// Get next vertex + edge visibility + normal of the quadrilateral
	bool GetNextVertex(UPoint3D &vertex, int &edgeFlag,
											 UNormal3D &normal) const;

	// Get indeces of the next edge with indeces of the faces
	bool GetNextEdgeIndeces(int &i1, int &i2, int &edgeFlag,
							int &iface1, int &iface2) const;

	// Get indeces of the next edge
	bool GetNextEdgeIndeces(int &i1, int &i2, int &edgeFlag) const;

	// Get next edge
	bool GetNextEdge(UPoint3D &p1, UPoint3D &p2, int &edgeFlag) const;

	// Get next edge
	bool GetNextEdge(UPoint3D &p1, UPoint3D &p2, int &edgeFlag,
										 int &iface1, int &iface2) const;

	void GetFacet(int iFace, std::vector<int> &facet);

	// Get face by index
	void GetFacet(int iFace, int &n, int *iNodes,
								int *edgeFlags = 0, int *iFaces = 0) const;

	// Get face by index
	void GetFacet(int iFace, int &n, UPoint3D *nodes,
								int *edgeFlags=0, UNormal3D *normals=0) const;

	// Get next face with normals at the nodes
	bool GetNextFacet(int &n, UPoint3D *nodes, int *edgeFlags=0,
						UNormal3D *normals=0) const;

	// Get normal of the face given by index
	UNormal3D GetNormal(int iFace) const;

	// Get Unit normal of the face given by index
	UNormal3D GetUnitNormal(int iFace) const;

	// Get normal of the next face
	bool GetNextNormal(UNormal3D &normal) const;

	// Get normal of Unit length of the next face 
	bool GetNextUnitNormal(UNormal3D &normal) const;

	/*
	// Boolean operations 
	UPolyhedron add(const UPolyhedron &p) const;
	UPolyhedron subtract(const UPolyhedron &p) const;
	UPolyhedron intersect(const UPolyhedron &p) const;
	*/

	// Get area of the surface of the polyhedron
	double GetSurfaceArea() const;

	// Get volume of the polyhedron
	double GetVolume() const;

	// Get number of steps for whole circle
	static int GetNumberOfRotationSteps();

	// Set number of steps for whole circle
	static void SetNumberOfRotationSteps(int n);

	// Reset number of steps for whole circle to default value
	static void ResetNumberOfRotationSteps();

	/**
	 * Creates polyhedron for twisted trapezoid.
	 * The trapezoid is given by two bases perpendicular to the z-axis.
	 * 
	 * @param	Dz	half length in z
	 * @param	xy1 1st base (at z = -Dz)
	 * @param	xy2 2nd base (at z = +Dz)
	 * @return status of the operation - is non-zero in case of problem
	 */
	int createTwistedTrap(double Dz,
                           const double xy1[][2], const double xy2[][2]);

	/**
	 * Creates user defined polyhedron.
	 * This function allows to the user to define arbitrary polyhedron.
	 * The faces of the polyhedron should be either triangles or planar
	 * quadrilateral. Nodes of a face are defined by indexes pointing to
	 * the elements in the xyz array. Numeration of the elements in the
	 * array starts from 1 (like in fortran). The indexes can be positive
	 * or negative. Negative sign means that the corresponding edge is
	 * invisible. The normal of the face should be directed to exterior
	 * of the polyhedron. 
	 * 
	 * @param	Nnodes number of nodes
	 * @param	Nfaces number of faces
	 * @param	xyz		nodes
	 * @param	faces	faces (quadrilaterals or triangles)
	 * @return status of the operation - is non-zero in case of problem
	 */
	int createPolyhedron(int Nnodes, int Nfaces,
				const double xyz[][3], const int faces[][4]);
};

class UPolyhedronTrd2 : public UPolyhedron
{
 public:
	UPolyhedronTrd2(double Dx1, double Dx2,
						double Dy1, double Dy2, double Dz);
	virtual ~UPolyhedronTrd2();
};

class UPolyhedronTrd1 : public UPolyhedronTrd2
{
 public:
	UPolyhedronTrd1(double Dx1, double Dx2,
						double Dy, double Dz);
	virtual ~UPolyhedronTrd1();
};

class UPolyhedronBox : public UPolyhedronTrd2
{
 public:
	UPolyhedronBox(double Dx, double Dy, double Dz);
	virtual ~UPolyhedronBox();
};

class UPolyhedronTrap : public UPolyhedron
{
 public:
	UPolyhedronTrap(double Dz, double Theta, double Phi,
						double Dy1,
						double Dx1, double Dx2, double Alp1,
						double Dy2,
						double Dx3, double Dx4, double Alp2);
	virtual ~UPolyhedronTrap();
};

class UPolyhedronPara : public UPolyhedronTrap
{
 public:
	UPolyhedronPara(double Dx, double Dy, double Dz,
						double Alpha, double Theta, double Phi);
	virtual ~UPolyhedronPara();
};

class UPolyhedronParaboloid : public UPolyhedron
{
 public:
	UPolyhedronParaboloid(double r1,
						double r2,
						double dz,
						double Phi1,
                        double Dphi);
	virtual ~UPolyhedronParaboloid();
};

class UPolyhedronHype : public UPolyhedron
{
 public:
	UPolyhedronHype(double r1,
					double r2,
					double tan1,
					double tan2,
					double halfZ);
	virtual ~UPolyhedronHype();
};

class UPolyhedronCons : public UPolyhedron
{
 public:
	UPolyhedronCons(double Rmn1, double Rmx1, 
										double Rmn2, double Rmx2, double Dz,
										double Phi1, double Dphi); 
	virtual ~UPolyhedronCons();
};

class UPolyhedronCone : public UPolyhedronCons
{
 public:
	UPolyhedronCone(double Rmn1, double Rmx1, 
                    double Rmn2, double Rmx2, double Dz);
	virtual ~UPolyhedronCone();
};

class UPolyhedronTubs : public UPolyhedronCons
{
 public:
	UPolyhedronTubs(double Rmin, double Rmax, double Dz, double Phi1, double Dphi);
	virtual ~UPolyhedronTubs();
};

class UPolyhedronTube : public UPolyhedronCons
{
 public:
	UPolyhedronTube (double Rmin, double Rmax, double Dz);
	virtual ~UPolyhedronTube();
};

class UPolyhedronPgon : public UPolyhedron
{
 public:
	UPolyhedronPgon(double phi, double dphi, int npdv, int nz,
					const std::vector<double> &z,
					const std::vector<double> &rmin,
					const std::vector<double> &rmax);
	virtual ~UPolyhedronPgon();
};

class UPolyhedronPcon : public UPolyhedronPgon
{
 public:
	UPolyhedronPcon(double phi, double dphi, int nz,
        const std::vector<double> &z,
        const std::vector<double> &rmin,
        const std::vector<double> &rmax);
	virtual ~UPolyhedronPcon();
};

class UPolyhedronSphere : public UPolyhedron
{
 public:
 	UPolyhedronSphere(double rmin, double rmax,
					double phi, double dphi,
					double the, double dthe);
	virtual ~UPolyhedronSphere();
};

class UPolyhedronTorus : public UPolyhedron
{
 public:
	UPolyhedronTorus(double rmin, double rmax, double rtor,
					double phi, double dphi);
	virtual ~UPolyhedronTorus();
};

class UPolyhedronEllipsoid : public UPolyhedron
{
 public:
	UPolyhedronEllipsoid(double dx, double dy, double dz, 
					double zcut1, double zcut2);
	virtual ~UPolyhedronEllipsoid();
};

class UPolyhedronEllipticalCone : public UPolyhedron
{
 public:
	UPolyhedronEllipticalCone(double dx, double dy, double z,
															double zcut1);
	virtual ~UPolyhedronEllipticalCone();
};

#endif /* U_POLYHEDRON_HH */
