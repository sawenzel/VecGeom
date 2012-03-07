#ifndef USOLIDS_UBox
#define USOLIDS_UBox
////////////////////////////////////////////////////////////////////////////////
//
//  A simple box defined by half-lengths on the three axis. The center of the
//  box matches the origin of the local reference frame.  
//
////////////////////////////////////////////////////////////////////////////////

#ifndef USOLIDS_VUSolid
#include "VUSolid.hh"
#endif

#ifndef USOLIDS_UUtils
#include "UUtils.hh"
#endif

class UBox : public VUSolid
{

public:
   UBox() : VUSolid(), fDx(0), fDy(0), fDz(0) {}
   UBox(const char *name, double dx, double dy, double dz);
   virtual ~UBox() {}
   
   // Navigation methods
   EnumInside     Inside (const UVector3 &aPoint) const;   

   virtual double  SafetyFromInside ( const UVector3 &aPoint, 
                                      bool aAccurate=false) const;
   virtual double  SafetyFromOutside( const UVector3 &aPoint, 
                                      bool aAccurate=false) const;
   virtual double  DistanceToIn     ( const UVector3 &aPoint, 
                                      const UVector3 &aDirection,
                                      // UVector3       &aNormalVector,
                                
									  double aPstep = UUtils::kInfinity) const;                               

   virtual double DistanceToOut     ( const UVector3 &aPoint,
                                      const UVector3 &aDirection,
                                      UVector3       &aNormalVector, 
                                      bool           &aConvex,
                                      double aPstep = UUtils::kInfinity) const;
   virtual bool Normal ( const UVector3& aPoint, UVector3 &aNormal ) const; 
   virtual void Extent ( EAxisType aAxis, double &aMin, double &aMax ) const;
   virtual void Extent (UVector3 &aMin, UVector3 &aMax) const;
   virtual double Capacity() {return 8.*fDx*fDy*fDz;}
   virtual double SurfaceArea() {return 8.*(fDx*fDy+fDx*fDz+fDy*fDz);}
   virtual VUSolid* Clone() const {return 0;}
   virtual UGeometryType GetEntityType() const { return "Box";}
   virtual void    ComputeBBox(UBBox *aBox, bool aStore = false) {}

  //G4Visualisation
   virtual void GetParametersList(int aNumber,double *aArray) const{
    aNumber=3;aArray[0]=fDx;aArray[1]=fDy;aArray[2]=fDz;} 
   virtual UPolyhedron* GetPolyhedron() const{return 0;}
   
private:  
   double                fDx;   // Half-length on X
   double                fDy;   // Half-length on Y
   double                fDz;   // Half-length on Z
};
#endif
