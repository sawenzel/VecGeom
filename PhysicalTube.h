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




template <typename T=double>
// T is a floating point type
class TubeParameters // : ShapeParameters
{
private:
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
	Vector3D normalPhi1;
	Vector3D normalPhi2;

public:
	TubeParameters(T pRmin, T pRmax, T pDZ, T pPhiMin, T pPhiMax) :
		dRmin(pRmin),
		dRmax(pRmax),
		dZ(pDZ),
		dSPhi(pPhiMin),
		dDPhi(pPhiMax) {
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
			PhiToPlane::GetNormalVectorToPhiPlane(dSPhi, normalPhi1, true);
			PhiToPlane::GetNormalVectorToPhiPlane(dSPhi + dDPhi, normalPhi2, false);

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
};

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
	virtual void DistanceToIn( Vectors3DSOA const &, Vectors3DSOA const &, double const * /*steps*/, double * /*result*/ ) const {};

	// for basket treatment (supposed to be dispatched to loop case over (SIMD) optimized 1-particle function)
	virtual void DistanceToInIL( Vectors3DSOA const &, Vectors3DSOA const &, double const * /*steps*/, double * /*result*/ ) const;
	virtual void DistanceToInIL( Vector3D const *, Vector3D const *, double const * /*steps*/, double * /*result*/, int /*size*/ ) const;

	// a template version for T = Vc or T = Boost.SIMD or T= double
	template<typename VectorType>
	inline
	__attribute__((always_inline))
	void DistanceToIn( VectorType const & /*x-vec*/, VectorType const & /*y-vec*/, VectorType const & /*z-vec*/,
					   VectorType const & /*dx-vec*/, VectorType const & /*dy-vec*/, VectorType const & /*dz-vec*/, VectorType const & /*step*/, VectorType & /*result*/ ) const;

};



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
    			// if no intersection, return UUtils::kInfinity, if intersection instead
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


// a template version targeted at T = Vc or T = Boost.SIMD or T= double
// this is the kernel operating on type T
template<int tid, int rid, typename TubeType, typename ValueType>
template<typename VectorType>
inline
void PlacedUSolidsTube<tid,rid,TubeType,ValueType>::DistanceToIn( VectorType const & x, VectorType const & y, VectorType const & z,
					   VectorType const & dirx, VectorType const & diry, VectorType const & dirz, VectorType const & stepmax, VectorType & distance ) const
{
	// Static method to compute distance from outside point to a set of tubes with given parameters
/*
	VectorTraits<T>::MaskType done_m(false); // which particles are ready to be returned
	T s_v(1.E30); // distance to each segment

	// check Z planes
	T zi_v = dz_v - Vc::abs(z_v);
	vdm inz_m = zi_v >= 0;

	done_m = !inz_m && (z_v*dir[2] >= 0); // particle outside the z-range and moving away
	if( done_m.isFull() ) return s_v;

	s_v(!done_m) = -zi_v/TMath::Abs(dir[2]);
	vd xi_v = x + s_v*dir[0];
	  vd yi_v = y + s_v*dir[1];
	  vd ri2_v = xi_v*xi_v + yi_v*yi_v;
	  vd rmin2_v = rmin_v*rmin_v;
	  vd rmax2_v = rmax_v*rmax_v;

	  done_m |= !inz_m && (rmin2_v <= ri2_v) && (ri2_v <= rmax2_v);
	  if( done_m.isFull() ) return s_v;

	  // check outer cyl. surface
	  Double_t r2  = x*x + y*y;
	  vd b_v(0.);
	  vd d_v(0.);
	  vdm inrmax_m = (r2 - rmax2_v) <= tol_v;
	  vdm inrmin_m = (rmin2_v - r2) <= tol_v;

	  // TO DO: CHECK IF IT'S INSIDE (BOUNDARY WITHIN MACHINE PRECISION, TOO SPECIFIC)

	  // check outer cylinder (consider only rmax)
	  Double_t n2 = dir[0]*dir[0] + dir[1]*dir[1];
	  Double_t rdotn = x*dir[0] + y*dir[1];

	  if (TMath::Abs(n2)<TGeoShape::Tolerance())
	    {
	      s_v(!done_m) = 1.E30;
	      return s_v;
	    }

	  TGeoTube_v::DistToTubes_v4(r2, n2, rdotn, rmax2_v, b_v, d_v);

	  vdm d_m = d_v > 0;
	  s_v(!done_m && !inrmax_m && d_m) = -b_v - d_v;
	  zi_v = z_v + s_v*dir[2];
	  done_m |= !inrmax_m && d_m && (s_v > 0) && (Vc::abs(zi_v) <= dz_v);

	  s_v(!done_m) = 1.E30;
	  done_m |= (rmin_v <=0);
	  if ( done_m.isFull() ) return s_v;

	  // Check inner cylinder (MAYBE SOME OPERATIONS CAN BE DONE ONCE AND REPEATED ONLY IF ONE OF THE PARTICULAR CASES HAPPENS)

	  TGeoTube_v::DistToTubes_v4(r2, n2, rdotn, rmin2_v, b_v, d_v);
	  d_m = d_v > 0;
	  s_v(!done_m && d_m) = -b_v + d_v;
	  zi_v = z_v + s_v*dir[2];
	  done_m |= d_m && (s_v > 0) && (Vc::abs(zi_v) <= dz_v);
	  if ( done_m.isFull() ) return s_v;

	  s_v(!done_m) = 1.E30;
	  return s_v;
	}
*/

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
				&& Vector3D::scalarProduct(x, tubeparams->normalPhi2 ) > 0 ) return false;
	}
	return true;
}


#endif /* PHYSICALTUBE_H_ */
