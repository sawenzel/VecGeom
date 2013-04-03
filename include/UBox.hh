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
	UBox(const std::string &name, double dx, double dy, double dz);
	virtual ~UBox();

	UBox(const UBox& rhs);
    UBox& operator=(const UBox& rhs); 
    // Copy constructor and assignment operator

	void Set(double dx, double dy, double dz);

	void Set(const UVector3 &vec);

	// Navigation methods
	EnumInside     Inside (const UVector3 &aPoint) const;   

	double  SafetyFromInside ( const UVector3 &aPoint, 
		bool aAccurate=false) const;
	double  SafetyFromOutside( const UVector3 &aPoint, 
		bool aAccurate=false) const;
	double  DistanceToIn     ( const UVector3 &aPoint, 
		const UVector3 &aDirection,
		// UVector3       &aNormalVector,

		double aPstep = UUtils::kInfinity) const;                               

	double DistanceToOut     ( const UVector3 &aPoint,
		const UVector3 &aDirection,
		UVector3       &aNormalVector, 
		bool           &aConvex,
		double aPstep = UUtils::kInfinity) const;
	bool Normal ( const UVector3& aPoint, UVector3 &aNormal ) const; 
//	void Extent ( EAxisType aAxis, double &aMin, double &aMax ) const;
	void Extent (UVector3 &aMin, UVector3 &aMax) const;
	double Capacity() {return 8.*fDx*fDy*fDz;}
	double SurfaceArea() {return 8.*(fDx*fDy+fDx*fDz+fDy*fDz);}
	VUSolid* Clone() const 
	{
		return new UBox(GetName(), fDx, fDy, fDz);
	}
	UGeometryType GetEntityType() const { return "Box";}
	void    ComputeBBox(UBBox * /*aBox*/, bool /*aStore = false*/) {}

	//G4Visualisation
	void GetParametersList(int aNumber,double *aArray) const{
		aNumber=3;aArray[0]=fDx;aArray[1]=fDy;aArray[2]=fDz;} 

	UVector3 GetPointOnSurface() const;

	std::ostream& StreamInfo(std::ostream& os) const;

	UPolyhedron* CreatePolyhedron () const;

	UPolyhedron* GetPolyhedron() const{return CreatePolyhedron();}

  inline double GetDz()
  {
    return fDz;
  }

  inline void SetDz(double v)
  {
    fDz = v;
  }

private:  
	double                fDx;   // Half-length on X
	double                fDy;   // Half-length on Y
	double                fDz;   // Half-length on Z
};
#endif
