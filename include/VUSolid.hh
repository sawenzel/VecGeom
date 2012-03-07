#ifndef USOLIDS_VUSolid
#define USOLIDS_VUSolid
////////////////////////////////////////////////////////////////////////////////
//  "Universal" Solid Interface
//  Authors: J. Apostolakis, G. Cosmo, A. Gheata, A. Munnich, T. Nikitina (CERN)
//
//  Created: 25 May 2011
//
////////////////////////////////////////////////////////////////////////////////

#ifndef USOLIDS_Utypes
#include "Utypes.hh"
#endif

#ifndef USOLIDS_UVector3
#include "UVector3.hh"
#endif

class VUSolid
{
public:

enum EnumInside { eInside=0, eSurface=1, eOutside=2 }; 
   // Use eInside < eSurface < eOutside: allows "max(,)" to combine Inside of surfaces
   // Potentially replace eSurface with eInSurface, eOutSurface 

enum EAxisType { eXaxis, eYaxis, eZaxis };

protected:
static double     fgTolerance;
static double     frTolerance;

public:
  VUSolid();
  VUSolid(const char *name);
  virtual ~VUSolid();
  
  // Navigation methods
  virtual EnumInside Inside (const UVector3 &aPoint) const = 0;
  //
  // Evaluate if point is inside, outside or on the surface within the tolerance
  virtual double  SafetyFromInside ( const UVector3 &aPoint, 
                                     bool aAccurate=false) const = 0;
  virtual double  SafetyFromOutside( const UVector3 &aPoint, 
                                     bool aAccurate=false) const = 0;
  // 
  // Estimates isotropic distance to the surface of the solid. This must
  // be either accurate or an underestimate. 
  //  Two modes: - default/fast mode, sacrificing accuracy for speed
  //             - "precise" mode,  requests accurate value if available.
  //  For both modes, if at a large distance from solid ( > ? ) 
  //    it is expected that a simplified calculation will be made if available.

  virtual double DistanceToIn(  const UVector3 &aPoint, 
				                    const UVector3 &aDirection,
                                double aPstep = UUtils::kInfinity) const = 0;
  virtual double DistanceToOut( const UVector3 &aPoint,
                                const UVector3  &aDirection,
                                UVector3       &aNormalVector,
                                bool           &aConvex,
                                double aPstep = UUtils::kInfinity) const = 0;
  // 
  // o return the exact distance (double) from a surface, given a direction
  // o compute the normal on the surface, returned as argument, calculated
  //      within the method to verify if it is close to the surface or not
  // o for DistanceToOut(), normal-vector and convexity flag could be optional (to decide).
  //    If normal cannot be computed (or shape is not convex), set 'convex' to 'false'.
  // o for DistanceToIn(), the normal-vector could be added as optional

  virtual bool Normal( const UVector3& aPoint, UVector3 &aNormal ) const = 0; 
  // Computes the normal on a surface and returns it as a unit vector
  //   In case a point is further than tolerance_normal from a surface, set validNormal=false
  //   Must return a valid vector. (even if the point is not on the surface.)
  //
  //   On an edge or corner, provide an average normal of all facets within tolerance

  // Decision: provide or not the Boolean 'validNormal' argument for returning validity 

  virtual void Extent(EAxisType aAxis, double &aMin, double &aMax) const = 0;
  // Returns the minimum and maximum extent along the specified Cartesian axis
  virtual void Extent( UVector3 &aMin, UVector3 &aMax ) const = 0;
  // Return the minimum and maximum extent along all Cartesian axes
  // For both the Extent methods
  // o Expect mostly to use a GetBBox()/CalculateBBox() method internally to compute the extent
  // o Decision: whether to store the computed BBox (containing or representing 6 double values), 
  //     and whether to compute it at construction time.
  // Methods are *not* const to allow caching of the Bounding Box
 virtual UGeometryType  GetEntityType() const =0;
       // Provide identification of the class of an object.
       // (required for persistency and STEP interface)
  
  const char     *GetName() const {return fName.c_str();}
  void            SetName(const char *aName) {fName = aName;} 

  // Auxiliary methods
  virtual double Capacity()    = 0 ;   //  like  CubicVolume() 
  virtual double SurfaceArea() = 0 ;
  //  Expect the solids to cache the values of Capacity and Surface Area 
  
  // Sampling
  virtual void    SamplePointsInside(int aNpoints, UVector3 *aArray) const {}
  virtual void    SamplePointsOnSurface(int aNpoints, UVector3 *aArray) const {}
  virtual void    SamplePointsOnEdge(int aNpoints, UVector3 *aArray) const {}
  // o generates points on the edges of a solid - primarily for testing purposes
  // o for solids composed only of curved surfaces(like full spheres or toruses) or 
  //        where an implementation is not available, it defaults to PointOnSurface.

  virtual VUSolid* Clone() const =0; 
  // o provide a new object which is a clone of the solid
  
  // Visualization
  //
  //From G4Visualisation
  virtual void GetParametersList(int aNumber,double *aArray) const =0;
  virtual UPolyhedron* GetPolyhedron() const =0;

  virtual void    SetMeshPoints(double *aArray) const {}
  // o used in TGeo for visualization and overlap checking. 
  //   Implementation existing for all solids.
  virtual void    FillMesh(UBuffer3D &aBuffer) const {}
  // o An internal buffer to be filled, converted to TBuffer3D in TGeo bridge
  // class and fed into GL for visualization. Algorithms existing.   
  virtual void    GetMeshNumbers( int &/*nvertices*/, int &/*nsegments*/, int &/*npolygons*/) const {}
  // o Fills the number of points, segments and polygons of the visualization
  // mesh of the solid
  static double   Tolerance() {return fgTolerance;}

protected:
  virtual void    ComputeBBox(UBBox *aBox, bool aStore = false) = 0;
  // o Compute the bounding box for the solid. Called automatically and stored ? 
  // o Can throw an exception if the solid is invalid
private:
  UString         fName;  // Name of the solid
  UBBox          *fBBox;   // Bounding box
};
#endif

