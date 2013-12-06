/*
 * PhysicalTube.h
 *
 *  Created on: Nov 8, 2013
 *      Author: swenzel
 */

#ifndef PHYSICALTUBE_H_
#define PHYSICALTUBE_H_

#include "GlobalDefs.h"
#include "PhysicalBox.h"
#include "Vector3D.h"
#include "TransformationMatrix.h"
#include <type_traits>
#include "Utils.h"
#include "TubeTraits.h"
#include "Vc/vector.h"
#include "Vc/common/types.h"




template <typename T=double>
// T is a floating point type
class TubeParameters // : ShapeParameters
{
private:
	//**** members as Vc vectors ******//
	// Vc::Vector<T> vcRminSqr __attribute__((aligned(ALIGNMENT_BOUNDARY)));
	// Vc::Vector<T> vcRmaxSqr __attribute__((aligned(ALIGNMENT_BOUNDARY)));
	// Vc::Vector<T> vcZ __attribute__((aligned(ALIGNMENT_BOUNDARY)));

	T dRmin; // inner radius
	T dRmax; // outer radius
	T dZ; // halfLength in z direction
	T dSPhi; // starting angle in radians
	T dDPhi; // delta angle of segment in radians

	// for caching
	T cacheRminSqr; // rminsquared
	T cacheRmaxSqr; // rmaxsquared
	T cacheTolORminSqr; // tolerant outer radius of rmin
	T cacheTolORmaxSqr; // tolerant outer radius of rmax

	T cacheTolIRminSqr; // tolerant inner radius of rmin
	T cacheTolIRmaxSqr; // tolerant inner radius of rmax

	//**** we save normals to phi - planes *****//
	Vector3D normalPhi1; // ( probably need to worry about alignment )
	Vector3D normalPhi2; // ( probably need to worry about alignment
	//**** vectors along radial direction of phi-planes
	Vector3D alongPhi1;
	Vector3D alongPhi2;

public:
	TubeParameters(T pRmin, T pRmax, T pDZ, T pPhiMin, T pPhiMax) :
		dRmin(pRmin),
		dRmax(pRmax),
		dZ(pDZ),
		dSPhi(pPhiMin),
		dDPhi(pPhiMax)

		// vcZ( )
{
		//vcZ = Vc::One*dZ;
		//vcRminSqr = Vc::One*dRmin*dRmin;
		//vcRmaxSqr = Vc::One*dRmax*dRmax;

			// calculate caches
			cacheRminSqr=dRmin*dRmin;
			cacheRmaxSqr=dRmax*dRmax;

			if ( dRmin > UUtils::GetRadHalfTolerance() )
			{
				// CHECK IF THIS CORRECT ( this seems to be inversed with tolerance for ORmax
				cacheTolORminSqr = (dRmin - UUtils::GetRadHalfTolerance()) * (dRmin - UUtils::GetRadHalfTolerance());
				cacheTolIRminSqr = (dRmin + UUtils::GetRadHalfTolerance()) * (dRmin + UUtils::GetRadHalfTolerance());
			}
			else
			{
				cacheTolORminSqr = 0.0;
				cacheTolIRminSqr = 0.0;
			}

			cacheTolORmaxSqr = (dRmax + UUtils::GetRadHalfTolerance()) * (dRmax + UUtils::GetRadHalfTolerance());
			cacheTolIRmaxSqr = (dRmax - UUtils::GetRadHalfTolerance()) * (dRmax - UUtils::GetRadHalfTolerance());

			// calculate normals
			PhiUtils::GetNormalVectorToPhiPlane(dSPhi, normalPhi1, true);
			PhiUtils::GetNormalVectorToPhiPlane(dSPhi + dDPhi, normalPhi2, false);
			// get alongs
			PhiUtils::GetAlongVectorToPhiPlane(dSPhi, alongPhi1);
			PhiUtils::GetAlongVectorToPhiPlane(dSPhi + dDPhi, alongPhi2);

			normalPhi1.print();
			normalPhi2.print();
	};

//	virtual void inspect() const;

	inline T GetRmin() const {return dRmin;}
	inline T GetRmax() const {return dRmax;}
	inline T GetDZ() const {return dZ;}
	inline T GetSPhi() const {return dSPhi;}
	inline T GetDPhi() const {return dDPhi;}

	virtual ~TubeParameters(){};
	// The placed boxed can easily access the private members
	template<int,int,class,class> friend class PlacedUSolidsTube;

	// template<int,int,int> friend class PlacedRootTube;
} __attribute__((aligned(ALIGNMENT_BOUNDARY)));

template<TranslationIdType tid, RotationIdType rid, class TubeType=TubeTraits::HollowTubeWithPhi, class T=double>
class PlacedUSolidsTube : public PhysicalVolume
{
private:
	TubeParameters<T> const * tubeparams;

public:
	PlacedUSolidsTube( TubeParameters<T> const * _tb, TransformationMatrix const *m ) : PhysicalVolume(m), tubeparams(_tb) {
		this->bbox = new PlacedBox<1,0>( new BoxParameters(tubeparams->dRmax, tubeparams->dRmax, tubeparams->dZ), new TransformationMatrix(0,0,0,0,0,0) );
	};

	// ** functions to implement
	__attribute__((always_inline))
	virtual double DistanceToIn( Vector3D const &, Vector3D const &, double ) const;

	virtual double DistanceToOut( Vector3D const &, Vector3D const &, double ) const {return 0.;}
	virtual bool   Contains( Vector3D const & ) const;

	__attribute__((always_inline))
	inline
	virtual bool   UnplacedContains( Vector3D const & ) const;

	virtual double SafetyToIn( Vector3D const & ) const {return 0.;}
	virtual double SafetyToOut( Vector3D const & ) const {return 0.;}

	// for basket treatment (supposed to be dispatched to particle parallel case)
	virtual void DistanceToIn( Vectors3DSOA const &, Vectors3DSOA const &, double const * /*steps*/, double * /*result*/ ) const;

	// for basket treatment (supposed to be dispatched to loop case over (SIMD) optimized 1-particle function)
	virtual void DistanceToInIL( Vectors3DSOA const &, Vectors3DSOA const &, double const * /*steps*/, double * /*result*/ ) const;
	virtual void DistanceToInIL( Vector3D const *, Vector3D const *, double const * /*steps*/, double * /*result*/, int /*size*/ ) const;

	// a template version for T = Vc or T = Boost.SIMD or T= double
	template<typename VectorType = Vc::Vector<T> >
	inline
	__attribute__((always_inline))
	void DistanceToIn( VectorType const & /*x-vec*/, VectorType const & /*y-vec*/, VectorType const & /*z-vec*/,
					   VectorType const & /*dx-vec*/, VectorType const & /*dy-vec*/, VectorType const & /*dz-vec*/, VectorType const & /*step*/, VectorType & /*result*/ ) const;


// some helper functions ( mainly for debugging )
	Vector3D RHit( Vector3D const &x, Vector3D const &y, double & distance, double radius, bool wantMinusSolution ) const;
	Vector3D PhiHit ( Vector3D const &x, Vector3D const &y, double & distance, bool isPhi1Solution ) const;
	Vector3D ZHit ( Vector3D const &x, Vector3D const &y, double & distance) const;
	bool isInRRange( Vector3D const & x ) const;
	bool isInPhiRange( Vector3D const & x) const;
	bool isInZRange( Vector3D const & x ) const;
	virtual void DebugPointAndDirDistanceToIn( Vector3D const & x, Vector3D const & y) const;
	void printInfoHitPoint( char const * c, Vector3D const & vec, double distance ) const;
	//

	template<typename VectorType = Vc::Vector<T> >
	inline
	__attribute__((always_inline))
	typename VectorType::Mask determineRmaxHit( VectorType const & /*x-vec*/, VectorType const & /*y-vec*/, VectorType const & /*z-vec*/,
												VectorType const & /*dirx-vec*/, VectorType const & /*diry-vec*/, VectorType const & /*dirz-vec*/, VectorType const & /**/ ) const;

	template<typename VectorType = Vc::Vector<T> >
	inline
	__attribute__((always_inline))
	typename VectorType::Mask determineZHit( VectorType const & /*x-vec*/, VectorType const & /*y-vec*/, VectorType const & /*z-vec*/,
													VectorType const & /*dirx-vec*/, VectorType const & /*diry-vec*/, VectorType const & /*dirz-vec*/, VectorType const & /**/ ) const;

};

template<int tid, int rid, class TubeType, typename T>
Vector3D
PlacedUSolidsTube<tid,rid, TubeType, T>::ZHit( Vector3D const & pos, Vector3D const & dir, double & distance ) const
{
	distance = - (tubeparams->dZ - std::abs(pos.z))/std::abs(dir.z);
	Vector3D vec( pos.x + distance*dir.x, pos.y + distance*dir.y, pos.z + distance*dir.z );
	return vec;
}

template<int tid, int rid, class TubeType, typename T>
Vector3D
PlacedUSolidsTube<tid,rid, TubeType,T>::PhiHit ( Vector3D const & pos, Vector3D const & dir, double & distance, bool isPhi1Solution ) const
{
	Vector3D const & normal = ( isPhi1Solution ) ? tubeparams->normalPhi1 : tubeparams->normalPhi2;
	T scalarproduct1 = normal.x*pos.x + normal.y*pos.y;
	T N1dotDir = normal.x*dir.x + normal.y*dir.y;

	distance = -scalarproduct1/N1dotDir;
	Vector3D vec( pos.x + distance*dir.x, pos.y + distance*dir.y, pos.z + distance*dir.z );
	return vec;
}

template<int tid, int rid, class TubeType, typename T>
bool
PlacedUSolidsTube<tid,rid, TubeType,T>::isInRRange( Vector3D const & pos) const
{
	double d = Vector3D::scalarProductInXYPlane(pos, pos);
	return ( tubeparams->cacheRminSqr <= d && d <= tubeparams->cacheRmaxSqr );
}

template<int tid, int rid, class TubeType, typename T>
bool
PlacedUSolidsTube<tid,rid, TubeType,T>::isInZRange( Vector3D const & pos) const
{
	return ( std::abs(pos.z) <= tubeparams->dZ + UUtils::GetCarHalfTolerance() );
}

template<int tid, int rid, class TubeType, typename T>
bool
PlacedUSolidsTube<tid,rid, TubeType,T>::isInPhiRange( Vector3D const & pos) const
{
	return PhiUtils::PointIsInPhiSector<T>( tubeparams->normalPhi1, tubeparams->normalPhi2, pos );
}

template<int tid, int rid, class TubeType, typename T>
Vector3D
PlacedUSolidsTube<tid,rid,TubeType,T>::RHit( Vector3D const & pos, Vector3D const & dir, double & distance, double radius, bool wantMinusSolution ) const
{
	T rdotr = pos.x * pos.x + pos.y*pos.y;
	T dirdotdir = 1 - dir.z * dir.z;
	T rdotn = pos.x * dir.x + pos.y* dir.y;

	T c = rdotr - radius*radius;
	T a = dirdotdir;
	T b = 2 * rdotn;
	T discriminant = b*b - 4*a*c;

	if ( discriminant >= 0)
	{
		if(wantMinusSolution) distance = ( -b - sqrt(discriminant) ) / (2*a);
		if(!wantMinusSolution) distance = ( -b + sqrt(discriminant) ) / (2*a);
	}
	else
	{
		distance = UUtils::kInfinity;
	}
	Vector3D vec( pos.x + distance*dir.x, pos.y + distance*dir.y, pos.z + distance*dir.z );
	return vec;
}

template<int tid, int rid, class TubeType, typename T>
void
PlacedUSolidsTube<tid,rid,TubeType,T>::printInfoHitPoint( char const * c, Vector3D const & vec, double distance ) const
{
	std::cout << c << " " << vec << "\t\t at dist " << distance << " in z " << isInZRange(vec)
					<< " inPhi " << isInPhiRange( vec )
					<< " inR " << isInRRange( vec )
					<< std::endl;
}


template<int tid, int rid, class TubeType, typename T>
void
PlacedUSolidsTube<tid,rid,TubeType,T>::DebugPointAndDirDistanceToIn( Vector3D const & x, Vector3D const & y ) const
{
	std::cout << "INSPECTING POINT - TUBE INTERACTION ALONG FLIGHT-PATH " << std::endl;
	std::cout << "PARTICLE POSITION " << std::endl;
	double distance=0;
	printInfoHitPoint("ParticlePos", x, distance );
	std::cout << "PARTICLE DIRECTION " << std::endl;
	std::cout << y << std::endl;
	std::cout << "RMAX INTERACTION " << std::endl;
	Vector3D vmax = this->RHit( x, y, distance, tubeparams->dRmax, true);
	printInfoHitPoint("hitpointrmax", vmax, distance);

	if( TubeTraits::NeedsRminTreatment<TubeType>::value )
	{
		std::cout << "RMIN INTERACTION " << std::endl;
		Vector3D vmin = this->RHit( x, y, distance, tubeparams->dRmin, false);
		printInfoHitPoint("hitpointrmin", vmin, distance);
	}

	if( TubeTraits::NeedsPhiTreatment<TubeType>::value )
	{
		std::cout << "PHI INTERACTION " << std::endl;
		Vector3D vphi1 = this->PhiHit( x, y, distance, true);
		printInfoHitPoint("hitpointphi1", vphi1, distance);
		Vector3D vphi2 = this->PhiHit( x, y, distance, false);
		printInfoHitPoint("hitpointphi2", vphi2, distance);
	}

	std::cout << "Z INTERACTION " << std::endl;
	Vector3D vZ = this->ZHit( x, y, distance );
	printInfoHitPoint("hitpointZ", vZ, distance);
}


template<int tid, int rid, class TubeType, typename T>
inline
double
PlacedUSolidsTube<tid,rid,TubeType,T>::DistanceToIn( Vector3D const &xm, Vector3D const &vm, double cPstep ) const
{

  // Intersection point variables
  T sd, xi, yi, zi;

  // Intersection with Z surfaces
  T tolODz, tolIDz;
  tolIDz = tubeparams->dZ - UUtils::GetCarHalfTolerance();
  tolODz = tubeparams->dZ + UUtils::GetCarHalfTolerance();

  Vector3D x,v;
  // do coordinate transformation
  matrix->MasterToLocal<tid,rid>(xm,x);
  matrix->MasterToLocalVec<rid>(vm,v);

  // might do here a comparison of safety with pcstep

  T abspz = std::fabs(x.z);
  if ( abspz >= tolIDz )
  {
    if (x.z * v.z < 0)   // at +Z going in -Z or visa versa
    {
      sd = (abspz - tubeparams->dZ) / std::fabs(v.z); // Z intersect distance
      sd = (sd<0.)? 0. : sd;

      xi   = x.x + sd * v.x;              // Intersection coords
      yi   = x.y + sd * v.y;
      T rho2 = xi * xi + yi * yi;

      // Check validity of intersection
      if (( tubeparams->cacheTolIRminSqr <= rho2) && (rho2 <= tubeparams->cacheTolIRmaxSqr))
      {
    	  // check here phi segmented case
    	  //else
          return sd;
      }

      // if not hit we investigate further below
    }
    else // going away
    {
    	return UUtils::kInfinity;  // On/outside extent, and heading away
    }
  }

  // -> Can not intersect z surfaces
  // Intersection with rmax (possible return) and rmin (must also check phi)
  // Intersection point (xi,yi,zi) on line x=p.x+t*v.x etc.
  // Intersects with x^2+y^2=R^2
  // Hence (v.x^2+v.y^2)t^2+ 2t(p.x*v.x+p.y*v.y)+p.x^2+p.y^2-R^2=0
  // t1    t2                t3
  T t1, t2, t3, b, c, d;    // Quadratic solver variables
  t1 = 1.0 - v.z * v.z; // ( same as v.x*v.x + v.y*v.y )
  t2 = x.x * v.x + x.y * v.y;
  t3 = x.x * x.x + x.y * x.y;

  T snxt = UUtils::kInfinity;

  if ( t1 > 0 )          // Check not || to z axis
  {
    	b = t2 / t1;
    	c = t3 - tubeparams->cacheRmaxSqr;
    	if ( (t3 >= tubeparams->cacheTolORmaxSqr) && (t2 < 0) ) // This also handles the tangent case
    	{
    		// Try outer cylinder intersection
    		//          c=(t3-fRMax*fRMax)/t1;
    		c /= t1;
    		d = b * b - c;

    		if (d >= 0) // If real root
    		{
    			sd = c / (-b + std::sqrt(d));
    			if (sd >= 0)  // If 'forwards'
    			{
    				// Check z intersection
    				zi = x.z + sd * v.z;
    				if (std::fabs(zi) <= tolODz)
    				{
    					// Z ok. Check phi intersection if reqd
    					return sd;
    				} //  end if std::fabs(zi)
    			}   //  end if (sd>=0)
    		}     //  end if (d>=0)
    	}       //  end if (r>=fRMax)
    	else
    	{
    		// Inside outer radius :
    		// check not inside, and heading through tubs (-> 0 to in)
    		if ((t3 > tubeparams->cacheTolIRminSqr ) && (t2 < 0) && ( abspz <= tolIDz))
    		{
    			// Inside both radii, delta r -ve, inside z extent

    			// < check here if not full tube>

    			// In the old version, the small negative tangent for the point
    			// on surface was not taken in account, and returning 0.0 ...
    			// New version: check the tangent for the point on surface and
    			// if no intersection, return UUtils<>::kInfinity, if intersection instead
    			// return sd.
    			c = t3 - tubeparams->cacheRmaxSqr;
    			if (c <= 0.0)
    			{
    				return 0.0;
    			}
    				else
    				{
    					c = c/t1;
    					d = b*b-c;
    					if (d >= 0.0)
    					{
    						snxt = c / (-b + std::sqrt(d)); // using safe solution
    						// for quadratic equation
    						return ( snxt < UUtils::GetCarHalfTolerance() )? 0 : snxt;
    					}
    					else
    					{
    						return UUtils::kInfinity;
    					}
    				}
            }  // end if   (t3>tolIRMin2)
    	}    // end if   (Inside Outer Radius)

    	// if we have inner cylinder
    	if (tubeparams->dRmin)     // Try inner cylinder intersection
    	{
    		c = (t3 - tubeparams->cacheRminSqr) / t1;
    		d = b * b - c;
    		if (d >= 0.0)    // If real root
    		{
    			// Always want 2nd root - we are outside and know rmax Hit was bad
    			// - If on surface of rmin also need farthest root

    			sd = (b > 0.) ? c / (-b - std::sqrt(d)) : (-b + std::sqrt(d));
    			if (sd >= -UUtils::GetCarHalfTolerance())  // check forwards
    			{
    				// Check z intersection
    				//
    				if (sd < 0.0)
    				{
    					sd = 0.0;
    				}
    				zi = x.z + sd * v.z;
    				if (std::fabs(zi) <= tolODz)
    				{
    					// Z ok. Check phi
    					//
    					return sd;
    				}
    			}         //    end if (sd>=0)
    		}           //    end if (d>=0)
     	}             //    end if (fRMin)
  } // end check t1 != 0
  return ( snxt < UUtils::GetCarHalfTolerance()) ? 0 : snxt;
}

template<int tid, int rid, typename TubeType, typename ValueType>
template<typename VectorType>
inline
__attribute__((always_inline))
typename VectorType::Mask PlacedUSolidsTube<tid,rid,TubeType, ValueType>::determineRmaxHit( VectorType const & x, VectorType const & y, VectorType const & z,
											VectorType const & dirx, VectorType const & diry, VectorType const & dirz, VectorType const & distanceRmax ) const
{
	if( ! TubeTraits::NeedsRminTreatment<TubeType>::value &&  ! TubeTraits::NeedsPhiTreatment<TubeType>::value )
	{
		return distanceRmax > 0 && Vc::abs(z+distanceRmax*dirz) < tubeparams->dZ;
	}

	if( ! TubeTraits::NeedsRminTreatment<TubeType>::value &&  TubeTraits::NeedsPhiTreatment<TubeType>::value )
	{
		// need to have additional look if hitting point on zylinder is not in empty phi range
		VectorType xhit = x + distanceRmax*dirx;
		VectorType yhit = y + distanceRmax*diry;
		return distanceRmax > 0 && Vc::abs(z+distanceRmax*dirz) < tubeparams->dZ && ! PhiUtils::PointIsInPhiSector<ValueType>(
					tubeparams->normalPhi1.x, tubeparams->normalPhi1.y, tubeparams->normalPhi2.x, tubeparams->normalPhi2.y, xhit, yhit );
	}
	else
	{
		return typename VectorType::Mask(false);
	}
}


template<int tid, int rid, typename TubeType, typename ValueType>
template<typename VectorType>
inline
__attribute__((always_inline))
typename VectorType::Mask PlacedUSolidsTube<tid,rid,TubeType, ValueType>::determineZHit( VectorType const & x, VectorType const & y, VectorType const & z,
											VectorType const & dirx, VectorType const & diry, VectorType const & dirz, VectorType const & distancez ) const
{
	if( ! TubeTraits::NeedsRminTreatment<TubeType>::value &&  ! TubeTraits::NeedsPhiTreatment<TubeType>::value )
	{
		VectorType xhit = x + distancez*dirx;
		VectorType yhit = y + distancez*diry;
		return distancez > 0 && ((xhit*xhit + yhit*yhit) < tubeparams->cacheRmaxSqr);
	}

	if( ! TubeTraits::NeedsRminTreatment<TubeType>::value &&  TubeTraits::NeedsPhiTreatment<TubeType>::value )
	{
		// need to have additional look if hitting point on zylinder is not in empty phi range
		VectorType xhit = x + distancez*dirx;
		VectorType yhit = y + distancez*diry;
		return distancez > 0 && ((xhit*xhit + yhit*yhit) < tubeparams->cacheRmaxSqr) && ! PhiUtils::PointIsInPhiSector<ValueType>(
					tubeparams->normalPhi1.x, tubeparams->normalPhi1.y, tubeparams->normalPhi2.x, tubeparams->normalPhi2.y, xhit, yhit );
	}

	if( TubeTraits::NeedsRminTreatment<TubeType>::value &&  ! TubeTraits::NeedsPhiTreatment<TubeType>::value )
	{
		VectorType xhit = x + distancez*dirx;
		VectorType yhit = y + distancez*diry;
		VectorType hitradiussquared = xhit*xhit + yhit*yhit;
		return distancez > 0 && hitradiussquared <= tubeparams->cacheRmaxSqr && hitradiussquared >= tubeparams->cacheRminSqr;
	}
	else // this should be totally general case ( Rmin and Rmax and general phi )
	{
		VectorType xhit = x + distancez*dirx;
		VectorType yhit = y + distancez*diry;
		VectorType hitradiussquared = xhit*xhit + yhit*yhit;
		return distancez > 0 && hitradiussquared <= tubeparams->cacheRmaxSqr
							 && hitradiussquared >= tubeparams->cacheRminSqr
							 && ( ! PhiUtils::PointIsInPhiSector<ValueType>(
									tubeparams->normalPhi1.x, tubeparams->normalPhi1.y, tubeparams->normalPhi2.x, tubeparams->normalPhi2.y, xhit, yhit ) );
	}
}


// a template version targeted at T = Vc or T = Boost.SIMD or T= double
// this is the kernel operating on type T
template<int tid, int rid, typename TubeType, typename ValueType>
template<typename VectorType>
inline
void PlacedUSolidsTube<tid,rid,TubeType,ValueType>::DistanceToIn( VectorType const & xm, VectorType const & ym, VectorType const & zm,
					   VectorType const & dirxm, VectorType const & dirym, VectorType const & dirzm, VectorType const & stepmax, VectorType & distance ) const
{
	typedef typename VectorType::Mask MaskType;

	MaskType done_m(false); // which particles in the vector are ready to be returned == aka have been treated
	distance = UUtils::kInfinityVc; // initialize distance to infinity

	VectorType x,y,z;
	matrix->MasterToLocal<tid,rid,VectorType>(xm,ym,zm,x,y,z);
	VectorType dirx, diry, dirz;
	matrix->MasterToLocalVec<tid,rid,VectorType>(dirxm,dirym,dirzm,dirx,diry,dirz);

	// do some inside checks
	// if safez is > 0 it means that particle is within z range
	// if safez is < 0 it means that particle is outside z range

	VectorType safez = tubeparams->dZ - Vc::abs(z);
	MaskType inz_m = safez > UUtils::fgToleranceVc;
	done_m = !inz_m && ( z*dirz >= Vc::Zero ); // particle outside the z-range and moving away

	VectorType r2 = x*x + y*y;
	VectorType n2 = Vc::One-dirz*dirz; // dirx_v*dirx_v + diry_v*diry_v; ( dir is normalized !! )
	VectorType rdotn = x*dirx + y*diry;
//	MaskType inrmax_m = (r2 - tubeparams->cacheRmaxSqr ) <= UUtils::frToleranceVc;
//	MaskType inrmin_m = (tubeparams->cacheRminSqr - r2) <= UUtils::frToleranceVc;

	// QUICK CHECK IF OUTER RADIUS CAN BE HIT AT ALL
	// BELOW WE WILL SOLVE A QUADRATIC EQUATION OF THE TYPE
	// a * t^2 + b * t + c = 0
	// if this equation has a solution at all ( == hit )
	// the following condition needs to be satisfied
	// DISCRIMINANT = b^2 -  4 a*c > 0
	//
	// THIS CONDITION DOES NOT NEED ANY EXPENSIVE OPERATION !!
	//
	// then the solutions will be given by
	//
	// t = (-b +- SQRT(DISCRIMINANT)) / (2a)
	//
	// b = 2*(dirx*x + diry*y)  -- independent of shape
	// a = dirx*dirx + diry*diry -- independent of shape
	// c = x*x + y*y - R^2 = r2 - R^2 -- dependent on shape
	VectorType c = r2 - tubeparams->cacheRmaxSqr;
	VectorType a = n2;
	VectorType inverse2a = 1./(2*a);

	VectorType b = 2*rdotn;
	VectorType discriminant = b*b-4*a*c;
	MaskType   canhitrmax = ( discriminant >= Vc::Zero );

	done_m |= ! canhitrmax;

	// this might be optional
	if( done_m.isFull() )
	{
		// joint z-away or no chance to hit Rmax condition
#ifdef LOG_EARLYRETURNS
		std::cerr << " RETURN1 IN DISTANCETOIN " << std::endl;
#endif
		return;
	}

	// Check outer cylinder (only r>rmax has to be considered)
	// this IS ALWAYS the MINUS (-) solution
	VectorType distanceRmax( UUtils::kInfinityVc );
	distanceRmax( canhitrmax ) = (-b - Vc::sqrt( discriminant ))*inverse2a;

	// this determines which vectors are done here already
	MaskType rmaxdone = determineRmaxHit( x, y, z, dirx, diry, dirz, distanceRmax );
	distanceRmax( ! rmaxdone ) = UUtils::kInfinityVc;
	distance( ! done_m && rmaxdone ) = distanceRmax;
	done_m |= rmaxdone;

	// **** inner tube ***** only compiled in for tubes having inner hollow tube ******/
	if ( TubeTraits::NeedsRminTreatment<TubeType>::value )
	{
		// only parameter "a" changes
		// c = r2 - tubeparams->cacheRminSqr;
		discriminant =  discriminant - 4*a*( tubeparams->cacheRmaxSqr - tubeparams->cacheRminSqr);
		MaskType canhitrmin = ( discriminant >= Vc::Zero );
		VectorType distanceRmin ( UUtils::kInfinityVc );
		// this is always + solution
		distanceRmin ( canhitrmin ) = (-b + Vc::sqrt( discriminant ))*inverse2a;

		VectorType zRmin = z + distanceRmin*dirz;
		distanceRmin ( ( Vc::abs(zRmin) > tubeparams->dZ ) || distanceRmin < 0 ) = UUtils::kInfinityVc;

		// reduction of distances
		distanceRmax = Vc::min( distanceRmax, distanceRmin );
	}

	// now do Z-Face
	VectorType distancez = -safez/Vc::abs(dirz);
	MaskType zdone = determineZHit(x,y,z,dirx,diry,dirz,distancez);
	distance ( ! done_m && zdone ) = distancez;
	done_m |= zdone;

	// now PHI

	// **** PHI TREATMENT FOR CASE OF HAVING RMAX ONLY ***** only compiled in for tubes having phi sektion ***** //
	if ( TubeTraits::NeedsPhiTreatment<TubeType>::value && ! ( TubeTraits::NeedsRminTreatment<TubeType>::value ) )
	{
		// all particles not done until here have the potential to hit a phi surface
		// phi surfaces require divisions so it might be useful to check before continuing

		if( ! done_m.isFull() )
		{
			VectorType distphi;
			PhiUtils::DistanceToPhiPlanes<ValueType>(tubeparams->dZ, tubeparams->cacheRmaxSqr,
					tubeparams->normalPhi1.x, tubeparams->normalPhi1.y, tubeparams->normalPhi2.x, tubeparams->normalPhi2.y,
					tubeparams->alongPhi1, tubeparams->alongPhi2,
					x, y, z, dirx, diry, dirz, distphi);
			distance ( ! done_m ) = distphi;
		}
	}
}


template<int tid, int rid, typename TubeType, typename ValueType>
inline
void PlacedUSolidsTube<tid,rid,TubeType,ValueType>::DistanceToIn( Vectors3DSOA const & points_v, Vectors3DSOA const & dirs_v, double const * steps, double * distance ) const
{
	int i=0;
	typedef typename Vc::Vector<ValueType> VectorType;
	for( i=0; i < points_v.size; i += Vc::Vector<ValueType>::Size )
		{
			VectorType x( &points_v.x[i] );
			VectorType y( &points_v.y[i] );
			VectorType z( &points_v.z[i] );
			VectorType xd( &dirs_v.x[i] );
			VectorType yd( &dirs_v.y[i] );
			VectorType zd( &dirs_v.z[i] );
			VectorType step( &steps[i] );
			VectorType dist;
			DistanceToIn< VectorType >(x, y, z, xd, yd, zd, step, dist);

			// store back result
			dist.store( &distance[i] );
		}
}


template<int tid, int rid, typename TubeType, typename T>
void PlacedUSolidsTube<tid,rid,TubeType,T>::DistanceToInIL( Vectors3DSOA const & points, Vectors3DSOA const & dirs, const double * steps, double * distances ) const
{
	for(auto i=0;i<points.size;i++)
	{
		distances[i]=this->PlacedUSolidsTube<tid,rid,TubeType,T>::DistanceToIn(points.getAsVector(i),dirs.getAsVector(i),steps[i]);
	}
}


template<int tid, int rid, typename TubeType, typename T>
void PlacedUSolidsTube<tid,rid,TubeType,T>::DistanceToInIL( Vector3D const * points, Vector3D const * dirs, const double * steps, double * distances, int np ) const
{
	for(auto i=0;i<np;i++)
	{
		distances[i]=this->PlacedUSolidsTube<tid,rid,TubeType,T>::DistanceToIn(points[i],dirs[i],steps[i]);
	}
}

template<int tid, int rid, typename TubeType, typename T>
bool   PlacedUSolidsTube<tid,rid,TubeType,T>::Contains( Vector3D const & x ) const
{
	Vector3D xp;
	matrix->MasterToLocal<tid,rid>(x,xp);
	return this->PlacedUSolidsTube<tid,rid,TubeType,T>::UnplacedContains(xp);
}


template<int tid, int rid, typename TubeType, typename T>
inline
bool PlacedUSolidsTube<tid,rid,TubeType,T>::UnplacedContains( Vector3D const & x) const
{
	// checkContainedZ
	if( std::abs(x.z) > tubeparams->dZ ) return false;

	// checkContainmentR
	T r2 = x.x*x.x + x.y*x.y;
	if( r2 > tubeparams->cacheRmaxSqr ) return false;

	if ( TubeTraits::NeedsRminTreatment<TubeType>::value )
	{
		if( r2 < tubeparams->cacheRminSqr ) return false;
	}

	if ( TubeTraits::NeedsPhiTreatment<TubeType>::value )
	{
		if ( ( Vector3D::scalarProduct(x, tubeparams->normalPhi1 ) > 0 )
				&& Vector3D::scalarProductInXYPlane(x, tubeparams->normalPhi2 ) > 0 ) return false;
	}
	return true;
}


#endif /* PHYSICALTUBE_H_ */
