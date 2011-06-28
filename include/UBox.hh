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

   virtual double  SafetyFromInside ( const UVector3 aPoint, 
                                      bool aAccurate=false) const;
   virtual double  SafetyFromOutside( const UVector3 aPoint, 
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
   virtual bool Normal ( const UVector3& aPoint, UVector3 &aNormal ); 
   virtual void Extent ( EAxisType aAxis, double &aMin, double &aMax );
   virtual void Extent ( double aMin[3], double aMax[3] ); 
   virtual double Capacity() {return 8.*fDx*fDy*fDz;}
   virtual double SurfaceArea() {return 8.*(fDx*fDy+fDx*fDz+fDy*fDz);}
   virtual VUSolid* Clone() const {return 0;}
   virtual UGeometryType GetEntityType() const { return "UBox";}
   virtual void    ComputeBBox(UBBox *aBox, bool aStore = false) {}
   
private:  
   double                fDx;   // Half-length on X
   double                fDy;   // Half-length on Y
   double                fDz;   // Half-length on Z
};
#endif
