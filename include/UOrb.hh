#ifndef USOLIDS_UOrb
#define USOLIDS_UOrb
////////////////////////////////////////////////////////////////////////////////
//
//  A simple orb defined by half-lengths on the three axis. The center of the
//  orb matches the origin of the local reference frame.  
//
////////////////////////////////////////////////////////////////////////////////

#ifndef USOLIDS_VUSolid
#include "VUSolid.hh"
#endif

#ifndef USOLIDS_UUtils
#include "UUtils.hh"
#endif

#define _USE_MATH_DEFINES
#include <math.h>


class UOrb : public VUSolid
{

public:
   UOrb() : VUSolid(), fR(0), fRTolerance(0) {}
   UOrb(const char *name, double pRmax);
   virtual ~UOrb() {}
   
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
   virtual double Capacity() {return (4*M_PI/3)*fR*fR*fR;}
   virtual double SurfaceArea() {return (4*M_PI)*fR*fR;}
   virtual VUSolid* Clone() const {return 0;}
   virtual UGeometryType GetEntityType() const { return "UOrb";}
   virtual void    ComputeBBox(UBBox *aBox, bool aStore = false) {}
   
private:  
    double fR;
    double fRTolerance;

};
#endif
