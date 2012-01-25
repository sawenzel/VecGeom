/**
* @file
* @author  John Doe <jdoe@example.com>
* @version 0.1
*
* @section LICENSE
*
* @section DESCRIPTION
*
* A simple Orb defined by half-lengths on the three axis. The center of the Orb matches the origin of the local reference frame.
*/

#ifndef USOLIDS_UOrb
#define USOLIDS_UOrb

#include "VUSolid.hh"
#include "UUtils.hh"

class UOrb : public VUSolid
{

public:
   UOrb() : VUSolid(), fR(0), fRTolerance(0) {}
   UOrb(const char *name, double pRmax);
   virtual ~UOrb() {}
   
   // Navigation methods
   EnumInside     Inside (const UVector3 &aPo6int) const;   

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

   virtual bool Normal ( const UVector3& aPoint, UVector3 &aNormal ); 
   virtual void Extent ( EAxisType aAxis, double &aMin, double &aMax );
   virtual void Extent ( double aMin[3], double aMax[3] ); 
   virtual double Capacity();
   virtual double SurfaceArea();
   virtual VUSolid* Clone() const {return 0;}
   virtual UGeometryType GetEntityType() const { return "UOrb";}
   virtual void ComputeBBox(UBBox *aBox, bool aStore = false) {}

  //G4Visualisation
   virtual void GetParametersList(int aNumber,double *aArray) const{} 
   virtual UPolyhedron* GetPolyhedron() const{return 0;}
   
private:  
    double fR;
    double fRTolerance;

	virtual double DistanceToOutForOutsidePoints(const UVector3 &p, const UVector3 &v, UVector3 &n) const;

};
#endif
