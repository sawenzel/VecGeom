/*
 * PhysicalBox.h
 *
 *  Created on: Nov 8, 2013
 *      Author: swenzel
 */

#ifndef PHYSICALBOX_H_
#define PHYSICALBOX_H_

#include <iostream>
#include "Vc/vector.h"

#include "PhysicalVolume.h"
#include "LogicalVolume.h"
#include "Vector3D.h"
#include "Vector3DFast.h"
#include "TransformationMatrix.h"

#include "UBox.hh"
#include "TGeoBBox.h"

class BoxParameters // : ShapeParameters
{
private:
	double dX; // half distances in x-y-z direction
	double dY;
	double dZ;

	// internal representation as a 3D vector
	Vector3DFast internalvector;

public:
	// for proper memory allocation on the heap
	static 
	void * operator new(std::size_t sz)
	{
	  std::cerr  << "overloaded new called for Boxparams" << std::endl; 
	  void *aligned_buffer=_mm_malloc( sizeof(BoxParameters), 32 );
	  return ::operator new(sz, aligned_buffer);
	}

	static
	  void operator delete(void * ptr)
	{
	  _mm_free(ptr);
	}



	BoxParameters(double x,double y, double z) : dX(x), dY(y), dZ(z), internalvector() {
		internalvector.SetX(dX);
		internalvector.SetY(dY);
		internalvector.SetZ(dZ);
	}

	virtual void inspect() const;

	double GetDX() const {return dX;}
	double GetDY() const {return dY;}
	double GetDZ() const {return dZ;}

	Vector3DFast const & GetAsVector3DFast() const { return internalvector; }

	double GetVolume() const {return 4*dX*dY*dZ;}

	virtual ~BoxParameters(){};
	// The placed boxed can easily access the private members
	template<int,int> friend class PlacedBox;
};


//template<TranslationIdType tid, RotationIdType rid>
template<TranslationIdType tid, RotationIdType rid>
class PlacedBox : public PhysicalVolume
{

private:
	BoxParameters const * boxparams;
	//friend class GeoManager;
	template<bool in>
	double Safety( Vector3D const &) const;
	PlacedBox<0,1296> * unplacedbox;

public:
  //void foo() const;
	double GetDX() const {return boxparams->GetDX();}
	double GetDY() const {return boxparams->GetDY();}
	double GetDZ() const {return boxparams->GetDZ();}

	//will provide a private constructor
	PlacedBox(BoxParameters const * bp, TransformationMatrix const *m) : PhysicalVolume(m), boxparams(bp)
	{
		// the bounding box of this volume is just the box itself
		// just forget about translation and rotation
		this->bbox = reinterpret_cast<PlacedBox<0, 1296>*>(this);
		analogoususolid = new UBox("internal_ubox", GetDX(), GetDY(), GetDZ());
		analogousrootsolid = new TGeoBBox("internal_tgeobbox", GetDX(), GetDY(), GetDZ());
		if(! ( tid==0 && rid==1296 ) )
		{
			unplacedbox = new PlacedBox<0,1296>( bp, m );
		}
	}

	__attribute__((always_inline))
	virtual inline double DistanceToIn( Vector3D const &, Vector3D const &, double cPstep ) const;

	__attribute__((always_inline))
	virtual double DistanceToOut( Vector3D const & point, Vector3D const & dir, double cPstep ) const {
	  bool convex;
	  UVector3 normal;
	  double s = analogoususolid->DistanceToOut(reinterpret_cast<UVector3 const &>(point), reinterpret_cast<UVector3 const &>(dir), normal, convex);
	    return s; 
	}

	virtual inline bool Contains( Vector3D const & ) const;
	virtual inline bool UnplacedContains( Vector3D const & ) const;
	virtual inline bool Contains( Vector3D const &, Vector3D & ) const;
	virtual inline bool Contains( Vector3D const &, Vector3D &, TransformationMatrix * ) const;
	virtual double SafetyToIn( Vector3D const &) const;
	virtual double SafetyToOut( Vector3D const &) const;

	// for fast vectors
	virtual double DistanceToIn( Vector3DFast const &, Vector3DFast const &, double /*step*/ ) const; // done
	// virtual double DistanceToOut( Vector3DFast const &, Vector3DFast const &, double /*step*/ ) const; // done
	virtual double DistanceToInAndSafety( Vector3DFast const &, Vector3DFast const &, double /*step*/, double & ) const {};
	virtual double DistanceToOutAndSafety( Vector3DFast const &, Vector3DFast const &, double /*step*/, double & ) const {};
	virtual bool   Contains( Vector3DFast const & ) const; // done
	virtual bool   UnplacedContains( Vector3DFast const & ) const; // done
	// same as Contains but returning the transformed point for further processing
	// this function is a "specific" version for locating points in a volume hierarchy
	// it also modifies the global matrix
	virtual bool   Contains( Vector3DFast const &, Vector3DFast & ) const; // done
	// this version modifies the global matrix additionally
	virtual bool   Contains( Vector3DFast const &, Vector3DFast &, FastTransformationMatrix * ) const; // done


	virtual void DistanceToInIL( Vectors3DSOA const &, Vectors3DSOA const &, double const * /*steps*/, double * /*result*/ ) const;
	//	virtual void DistanceToInIL( std::vector<Vector3D> const &, std::vector<Vector3D> const &, double const * /*steps*/, double * /*result*/ ) const;
	virtual void DistanceToInIL( Vector3D const *, Vector3D const *, double const * /*steps*/, double * /*result*/, int /**/ ) const;


	// for the basket treatment
	virtual void DistanceToIn( Vectors3DSOA const &, Vectors3DSOA const &, double const *, double * ) const;
	virtual void DistanceToOut( Vectors3DSOA const &, Vectors3DSOA const &, double const *, double * ) const;
	virtual void DistanceToOutCheck( Vectors3DSOA const &, Vectors3DSOA const &, double const *, double * ) const;

	template<typename T>
	inline
	void DistanceToInT(Vectors3DSOA const &, Vectors3DSOA const &, double const *, double * ) const;


        template<typename T>
	inline
	void DistanceToOutT(Vectors3DSOA const &, Vectors3DSOA const &, double const *, double * ) const;

// a template version for T = Vc or T = Boost.SIMD or T= double
	template<typename T>
	inline
	__attribute__((always_inline))
	void DistanceToIn( T const & /*x-vec*/, T const & /*y-vec*/, T const & /*z-vec*/,
			   T const & /*dx-vec*/, T const & /*dy-vec*/, T const & /*dz-vec*/, T const & /*step*/, T & /*result*/ ) const;

	template<typename T>
	inline
	__attribute__((always_inline))
	void DistanceToOut( T const & /*x-vec*/, T const & /*y-vec*/, T const & /*z-vec*/,
			    T const & /*dx-vec*/, T const & /*dy-vec*/, T const & /*dz-vec*/, T const & /*step*/, T & /*result*/ ) const;

	virtual ~PlacedBox(){};

	virtual PhysicalVolume const * GetAsUnplacedVolume() const
	{
		if( tid ==0 && rid == 1296 )
		{
			return this;
		}
		else
		{
			return unplacedbox;
		}
	}
	//a factory method that produces a specialized box based on params and transformations
	//static PhysicalBox* MakeBox( BoxParameters *param, TransformationMatrix *m );
};




template <TranslationIdType tid, RotationIdType rid>
inline
double
PlacedBox<tid,rid>::DistanceToIn(Vector3D const &x, Vector3D const &y, double cPstep ) const
{
	// Computes distance from a point presumably outside the solid to the solid
	// surface. Ignores first surface if the point is actually inside. Early return
	// infinity in case the safety to any surface is found greater than the proposed
	// step aPstep.
	// The normal vector to the crossed surface is filled only in case the box is
	// crossed, otherwise aNormal.IsNull() is true.

	// Compute safety to the closest surface on each axis.
	// Early exits if safety bigger than proposed step.
	const double delta = 1E-9;

	// here we do the point transformation
	Vector3D aPoint;
	matrix->MasterToLocal<tid,rid>(x, aPoint);

	//   aNormal.SetNull();
	double safx = std::abs(aPoint.x) - boxparams->dX;
	double safy = std::abs(aPoint.y) - boxparams->dY;
	double safz = std::abs(aPoint.z) - boxparams->dZ;

	if ((safx > cPstep) || (safy > cPstep) || (safz > cPstep))
		return 1E30;

	// only here we do the directional transformation
	Vector3D aDirection;
	matrix->MasterToLocalVec<rid>(y, aDirection);

	// Check if numerical inside
	bool outside = (safx > 0) || (safy > 0) || (safz > 0);
	if ( !outside ) {
		// If point close to this surface, check against the normal
		if ( safx > -delta ) {
			return ( aPoint.x * aDirection.x > 0 ) ? Utils::kInfinity : 0.0;
		}
		if ( safy > -delta ) {
			return ( aPoint.y * aDirection.y > 0 ) ? Utils::kInfinity : 0.0;
		}
		if ( safz > -delta ) {
			return ( aPoint.z * aDirection.z > 0 ) ? Utils::kInfinity : 0.0;
		}
		// Point actually "deep" inside, return zero distance, normal un-defined
		return 0.0;
	}

	// The point is really outside. Only axis with positive safety to be
	// considered. Early exit on each axis if point and direction components
	// have the same .sign.
	double dist = 0.0;
	double coordinate = 0.0;
	if ( safx > 0 ) {
		if ( aPoint.x * aDirection.x >= 0 ) return Utils::kInfinity;
		dist = safx/std::abs(aDirection.x);
		coordinate = aPoint.y + dist*aDirection.y;
		if ( std::abs(coordinate) < boxparams->dY ) {
			coordinate = aPoint.z + dist*aDirection.z;
			if ( std::abs(coordinate) < boxparams->dZ ) {
				return dist;
			}
		}
	}
	if ( safy > 0 ) {
		if ( aPoint.y * aDirection.y >= 0 ) return Utils::kInfinity;
		dist = safy/std::abs(aDirection.y);
		coordinate = aPoint.x + dist*aDirection.x;
		if ( std::abs(coordinate) < boxparams->dX ) {
			coordinate = aPoint.z + dist*aDirection.z;
			if ( std::abs(coordinate) < boxparams->dZ ) {
				return dist;
			}
		}
	}
	if ( safz > 0 ) {
		if ( aPoint.z * aDirection.z >= 0 ) return Utils::kInfinity;
		dist = safz/std::abs(aDirection.z);
		coordinate = aPoint.x + dist*aDirection.x;
		if ( std::abs(coordinate) < boxparams->dX ) {
			coordinate = aPoint.y + dist*aDirection.y;
			if ( std::abs(coordinate) < boxparams->dY ) {
				return dist;
			}
		}
	}
	return Utils::kInfinity;
}

// a template version for T = Vc or T = Boost.SIMD or T= double
// this is the kernel operating on type T
template<int tid, int rid>
template<typename VecType>
inline
void PlacedBox<tid,rid>::DistanceToIn( VecType const & x, VecType const & y, VecType const & z,
					   VecType const & dirx, VecType const & diry, VecType const & dirz, VecType const & stepmax, VecType & distance ) const
{
	typedef typename VecType::Mask MaskType;
	MaskType in;

	VecType saf[3];
	VecType newpt[3];
	VecType tiny(1e-20);
	VecType big(Utils::kInfinity);

	// should be done in the box
	VecType par[3]={ boxparams->dX, boxparams->dY, boxparams->dZ }; // very convenient

	// new thing: do coordinate transformation in place here
	matrix->MasterToLocal<tid,rid,VecType>(x,y,z,newpt[0],newpt[1],newpt[2]);
	//
	saf[0] = Vc::abs(newpt[0])-par[0];
	saf[1] = Vc::abs(newpt[1])-par[1];
	saf[2] = Vc::abs(newpt[2])-par[2];
	 /*
	   MaskType faraway; // initializing all components to zero
	   faraway = saf[0]>=stepmax || saf[1]>=stepmax || saf[2]>=stepmax;
	   */
	in = saf[0]<Vc::Zero && saf[1]<Vc::Zero && saf[2]<Vc::Zero;
	distance=big;

	   /*
	   if( faraway == Vc::One )
	       return; // return big
*/

	   // new thing:  do coordinate transformation for directions here
	   VecType localdirx, localdiry, localdirz;
	   matrix->MasterToLocalVec<tid,rid,VecType>(dirx, diry, dirz, localdirx, localdiry, localdirz);
	   //

	   // proceed to analysis of hits
	   // might still be optimized a bit
	   VecType snxt[3];
	   snxt[0] = saf[0]/(Vc::abs(localdirx)+tiny); // distance to y-z face
	   VecType coord1=newpt[1]+snxt[0]*localdiry; // calculate new y and z coordinate
	   VecType coord2=newpt[2]+snxt[0]*localdirz;

	   MaskType hit0 =  saf[0] > 0 && newpt[0]*localdirx < 0 && ( Vc::abs(coord1) <= par[1] && Vc::abs(coord2) <= par[2] ); // if out and right direction
	   distance(hit0) = snxt[0];
	   MaskType done=hit0;
	   //if( done.isFull() ) return;

	   snxt[1] = saf[1]/(Vc::abs(localdiry)+tiny); // distance to x-z face
	   coord1=newpt[0]+snxt[1]*localdirx; // calculate new x and z coordinate
	   coord2=newpt[2]+snxt[1]*localdirz;

	   MaskType hit1 = saf[1] > 0 && newpt[1]*localdiry < 0 && ( Vc::abs(coord1) <= par[0] && Vc::abs(coord2) <= par[2] ); // if out and right direction
	   distance(!done && hit1) = snxt[1];
	   done|=hit1;
	   //if( done.isFull() ) return;

	   snxt[2] = saf[2]/(Vc::abs(localdirz)+tiny); // distance to x-y face
	   coord1=newpt[0]+snxt[2]*localdirx; // calculate new x and y coordinate
	   coord2=newpt[1]+snxt[2]*localdiry;
	   MaskType miss2 = saf[2] <= 0 | newpt[2]*localdirz >= 0 | ( Vc::abs(coord1) > par[0] | Vc::abs(coord2) > par[1] ); // if out and right direction
	   distance(!done && !miss2 ) = snxt[2];

	   distance(in)=Vc::Zero;
	   return;
}


// a template version for T = Vc or T = Boost.SIMD or T= double
// this is the kernel operating on type T
template<int tid, int rid>
template<typename VecType>
inline
void PlacedBox<tid,rid>::DistanceToOut( VecType const & x, VecType const & y, VecType const & z,
					VecType const & dirx, VecType const & diry, VecType const & dirz, VecType const & stepmax, VecType & distance ) const
{
    typedef typename VecType::Mask MaskType;
    VecType tiny(1e-20);
    VecType big(Utils::kInfinity);

    // should be done in the box
    VecType par[3] = { boxparams->dX, boxparams->dY, boxparams->dZ }; // very convenient

    // new thing: do coordinate transformation in place here
    VecType newpt[3];
    matrix->MasterToLocal<tid,rid,VecType>(x,y,z,newpt[0],newpt[1],newpt[2]);

    /*
      MaskType faraway; // initializing all components to zero
      faraway = saf[0]>=stepmax || saf[1]>=stepmax || saf[2]>=stepmax;
    */
    VecType saf[3];
    saf[0] = Vc::abs(newpt[0])-par[0];
    saf[1] = Vc::abs(newpt[1])-par[1];
    saf[2] = Vc::abs(newpt[2])-par[2];

    MaskType inside = saf[0]<Vc::Zero && saf[1]<Vc::Zero && saf[2]<Vc::Zero;
    distance( !inside ) = big;

    //   If all particles are outside volume, can return immediately -- a good shortcut!???
    //if(!in) return;
    if(!inside) printf("Shortcut flagged!\n");

    // new thing:  do coordinate transformation for directions here
    VecType localdirx, localdiry, localdirz;
    matrix->MasterToLocalVec<tid,rid,VecType>(dirx, diry, dirz, localdirx, localdiry, localdirz);

    auto invdirx = 1.0/localdirx;
    auto invdiry = 1.0/localdiry;
    auto invdirz = 1.0/localdirz;

    MaskType mask;
    auto distx = (par[0]-newpt[0]) * invdirx;
    mask = dirx<0;
    distx( mask ) = (-par[0]-newpt[0]) * invdirx;

    auto disty = (par[1]-newpt[1]) * invdiry;
    mask = diry<0;
    disty( mask ) = (-par[1]-newpt[1]) * invdiry;

    auto distz = (par[2]-newpt[2]) * invdirz;
    mask = dirz<0;
    distz( mask ) = (-par[2]-newpt[2]) * invdirz;

    distance = distx;
    mask = distance>disty;
    distance(mask) = disty;
    mask = distance>distz;
    distance(mask) = distz;

    return;
}


// for the basket treatment
// this is actually a very general theme valid for any shape so it should become
// a template method !!
// like a for_each over chunks of vectors and some functor

template<int tid, int rid>
template<typename T>
inline
void PlacedBox<tid,rid>::DistanceToInT( Vectors3DSOA const & points_v, Vectors3DSOA const & dirs_v, double const * steps, double * distance ) const
{
	for( int i=0; i < points_v.size; i += T::Size )
		{
			T x( &points_v.x[i] );
			T y( &points_v.y[i] );
			T z( &points_v.z[i] );
			T xd( &dirs_v.x[i] );
			T yd( &dirs_v.y[i] );
			T zd( &dirs_v.z[i] );
			T step( &steps[i] );
			T dist;
			DistanceToIn<T>(x, y, z, xd, yd, zd, step, dist);

			// store back result
			dist.store( &distance[i] );
		}
}

template<int tid, int rid>
template<typename T>
inline
void PlacedBox<tid,rid>::DistanceToOutT( Vectors3DSOA const & points_v, Vectors3DSOA const & dirs_v, double const * steps, double * distance ) const
{
	for( int i=0; i < points_v.size; i += T::Size )
		{
			T x( &points_v.x[i] );
			T y( &points_v.y[i] );
			T z( &points_v.z[i] );
			T xd( &dirs_v.x[i] );
			T yd( &dirs_v.y[i] );
			T zd( &dirs_v.z[i] );
			T step( &steps[i] );
			T dist;
			DistanceToOut<T>(x, y, z, xd, yd, zd, step, dist);

			// store back result
			dist.store( &distance[i] );
		}
}

// dispatching the virtual method to some concrete method
template<int tid, int rid>
void PlacedBox<tid,rid>::DistanceToIn( Vectors3DSOA const & points_v, Vectors3DSOA const & dirs_v, double const * steps, double * distance ) const
{
	// PlacedBox<tid,rid>::DistanceToInT< Vc::double_v, Vc >(points_v, dirs_v, step, distance);
	this->template DistanceToInT<Vc::double_v>( points_v, dirs_v, steps, distance);
}


template<int tid, int rid>
void PlacedBox<tid,rid>::DistanceToOut( Vectors3DSOA const & points_v, Vectors3DSOA const & dirs_v, double const * steps, double * distance ) const {
    // PlacedBox<tid,rid>::DistanceToOutT< Vc::double_v, Vc >(points_v, dirs_v, step, distance);
    this->template DistanceToOutT<Vc::double_v>( points_v, dirs_v, steps, distance);
}


template<int tid, int rid>
void PlacedBox<tid,rid>::DistanceToInIL( Vectors3DSOA const & points, Vectors3DSOA const & dirs, const double * steps, double * distances ) const
{
	for(auto i=0;i<points.size;i++)
	{
		distances[i]=this->PlacedBox<tid,rid>::DistanceToIn(points.getAsVector(i),dirs.getAsVector(i),steps[i]);
	}
}


template<int tid, int rid>
void PlacedBox<tid,rid>::DistanceToInIL( Vector3D const * points, Vector3D const * dirs, const double * steps, double * distances, int np ) const
{
	for(auto i=0;i<np;i++)
	{
		// this inlined successfully
		distances[i]=this->PlacedBox<tid,rid>::DistanceToIn(points[i],dirs[i],steps[i]);
	}
}


template<int tid, int rid>
inline
bool PlacedBox<tid,rid>::Contains( Vector3D const & point ) const
{
	// here we do the point transformation
	Vector3D localPoint;
	matrix->MasterToLocal<tid,rid>(point, localPoint);
	return this->PlacedBox<tid,rid>::UnplacedContains( localPoint );
}

template<int tid, int rid>
inline
bool PlacedBox<tid,rid>::Contains( Vector3D const & point, Vector3D & localPoint ) const
{
	// here we do the point transformation
	matrix->MasterToLocal<tid,rid>(point, localPoint);
	return this->PlacedBox<tid,rid>::UnplacedContains( localPoint );
}

template<int tid, int rid>
inline
bool PlacedBox<tid,rid>::Contains( Vector3D const & point, Vector3D & localPoint, TransformationMatrix * globalm ) const
{
	// here we do the point transformation
	matrix->MasterToLocal<tid,rid>(point, localPoint);
	bool in = this->PlacedBox<tid,rid>::UnplacedContains( localPoint );
	if( in )
	{
		globalm->MultiplyT<tid,rid>( matrix );
	}
	return in;
}


template<int tid, int rid>
inline
bool PlacedBox<tid,rid>::Contains( Vector3DFast const & point ) const
{
	// here we do the point transformation
	Vector3DFast localPoint;
	fastmatrix->MasterToLocal<tid,rid>(point, localPoint);
	return this->PlacedBox<tid,rid>::UnplacedContains( localPoint );
}

template<int tid, int rid>
inline
bool PlacedBox<tid,rid>::Contains( Vector3DFast const & point, Vector3DFast & localPoint ) const
{
	// here we do the point transformation
	fastmatrix->MasterToLocal<tid,rid>(point, localPoint);
	return this->PlacedBox<tid,rid>::UnplacedContains( localPoint );
}

template<int tid, int rid>
inline
bool PlacedBox<tid,rid>::Contains( Vector3DFast const & point, Vector3DFast & localPoint, FastTransformationMatrix * globalm ) const
{
	// here we do the point transformation
	fastmatrix->MasterToLocal<tid,rid>(point, localPoint);
	bool in = this->PlacedBox<tid,rid>::UnplacedContains( localPoint );
	if( in )
	{
		globalm->Multiply<tid,rid>( fastmatrix );
	}
	return in;
}

template<int tid, int rid>
inline
bool PlacedBox<tid,rid>::UnplacedContains( Vector3D const & point ) const
{
	// this could be vectorized also
	if ( std::abs(point.x) > boxparams->dX ) return false;
	if ( std::abs(point.y) > boxparams->dY ) return false;
	if ( std::abs(point.z) > boxparams->dZ ) return false;
	return true;
}


template<int tid, int rid>
inline
bool PlacedBox<tid,rid>::UnplacedContains( Vector3DFast const & point ) const
{
	 Vector3DFast tmp = point.Abs();
	 return ! tmp.IsAnyLargerThan( boxparams->GetAsVector3DFast() );
}


/*
template<int tid, int rid>
inline
double PlacedBox<tid,rid>::DistanceToOut( Vector3DFast const & point, Vector3DFast const & dir, double step ) const
{
	Vector3DFast safetyPlus = boxparams->GetAsVector3DFast() + point;
	Vector3DFast safetyMinus = boxparams->GetAsVector3DFast() - point;

	// gather right safeties
	Vector3DFast rightSafeties = Vector3DFast::ChooseComponentsBasedOnConditionFast( safetyPlus, safetyMinus, dir );

	Vector3DFast distances = rightSafeties / dir.Abs();
	return distances.MinButNotNegative();
}
*/


template<int tid, int rid>
inline
double PlacedBox<tid,rid>::DistanceToIn( Vector3DFast const & point, Vector3DFast const & dir, double cPstep ) const
{
	const double delta = 1E-9;
	// here we do the point transformation
	Vector3DFast localPoint;
	fastmatrix->MasterToLocal<tid,rid>(point, localPoint);
	//   aNormal.SetNull();
	Vector3DFast safety = localPoint.Abs() - boxparams->GetAsVector3DFast();
	
	// check this::
	if(safety.IsAnyLargerThan(cPstep))
		return Utils::kInfinity;

	// here we do the directional transformation
	Vector3DFast localDirection;
	fastmatrix->MasterToLocalVec<rid>(dir, localDirection);
	// Check if numerical inside

	bool outside = safety.IsAnyLargerThan( 0 );
	if ( !outside )
	{
		std::cerr << "particle actually outside; not implemented " << std::endl;
	}
	// check any early return stuff ( because going away ) here:
	Vector3DFast pointtimesdirection = localPoint*localDirection;
	if( Vector3DFast::ExistsIndexWhereBothComponentsPositive(safety, pointtimesdirection)) return Utils::kInfinity;

	// compute distance to surfaces
	Vector3DFast distv = safety/localDirection.Abs();

	// compute target points ( needs some reshuffling )
	// might be suboptimal for SSE or really scalar
	Vector3DFast hitxyplane = localPoint + distv.GetZ()*localDirection;
	// the following could be made faster ( maybe ) by calculating the abs on the whole vector
	if(    std::abs(hitxyplane.GetX()) < boxparams->GetDX()
		&& std::abs(hitxyplane.GetY()) < boxparams->GetDY())
		return distv.GetZ();
	Vector3DFast hitxzplane = localPoint + distv.GetY()*localDirection;
	if(    std::abs(hitxzplane.GetX()) < boxparams->GetDX()
		&& std::abs(hitxzplane.GetZ()) < boxparams->GetDZ())
		return distv.GetY();
	Vector3DFast hityzplane = localPoint + distv.GetX()*localDirection;
	if(	   std::abs(hityzplane.GetY()) < boxparams->GetDY()
		&& std::abs(hityzplane.GetZ()) < boxparams->GetDZ())
		return distv.GetX();
	return Utils::kInfinity;
}


template<int tid, int rid>
template<bool in>
double PlacedBox<tid, rid>::Safety( Vector3D const & point ) const
{

   double safe, safy, safz;
   if (in) {
	   Vector3D localPoint;
	   matrix->MasterToLocal<tid,rid>(point, localPoint);
	   safe = boxparams->dX - std::fabs(localPoint.x);
	   safy = boxparams->dY - std::fabs(localPoint.y);
	   safz = boxparams->dZ - std::fabs(localPoint.z);
	   if (safy < safe) safe = safy;
	   if (safz < safe) safe = safz;
	   }
   else {
	   Vector3D localPoint;
	   matrix->MasterToLocal<tid,rid>(point, localPoint);
	   safe = -boxparams->dX + std::fabs(localPoint.x);
	   safy = -boxparams->dY + std::fabs(localPoint.y);
	   safz = -boxparams->dZ + std::fabs(localPoint.z);
	   if (safy > safe) safe = safy;
	   if (safz > safe) safe = safz;
   }
   return safe;
}

template<int tid, int rid>
double PlacedBox<tid,rid>::SafetyToIn( Vector3D const & point ) const
{
	return Safety<false>( point );
}

template<int tid, int rid>
double PlacedBox<tid,rid>::SafetyToOut( Vector3D const & point ) const
{
	return Safety<true>( point );
}

/// a simple cross-check function: should use only doubles internally
template<int tid, int rid>
void PlacedBox<tid,rid>::DistanceToOutCheck( Vectors3DSOA const & points_v, Vectors3DSOA const & dirs_v, double const * steps, double * distance ) const {

  auto tiny = double{1.e-20};
  auto big = double{1.e+30};

   // should be done in the box
   double par[3] = { boxparams->dX, boxparams->dY, boxparams->dZ }; // very convenient

   double x,y,z,xd,yd,zd,step;
   double newpt[3],saf[3];
   for( auto i=0; i < points_v.size; ++i ) {
     x = points_v.x[i];
     y = points_v.y[i];
     z = points_v.z[i];
     xd = dirs_v.x[i];
     yd = dirs_v.y[i];
     zd = dirs_v.z[i];
     step = steps[i];

     // new thing: do coordinate transformation in place here
     matrix->MasterToLocal<tid,rid,double>(x,y,z,newpt[0],newpt[1],newpt[2]);

     saf[0] = fabs(newpt[0])-par[0];
     saf[1] = fabs(newpt[1])-par[1];
     saf[2] = fabs(newpt[2])-par[2];

     if( saf[0]>0 || saf[1]>0 || saf[2]>0 ) {
       // all particles are outside volume
       distance[i] = big;
     }
     else {

       // new thing:  do coordinate transformation for directions here
       double localdirx, localdiry, localdirz;
       matrix->MasterToLocalVec<tid,rid,double>(xd, yd, zd, localdirx, localdiry, localdirz);

       double distx,disty,distz;
       distx = (localdirx>0.) ? (par[0]-newpt[0]) / localdirx : (-par[0]-newpt[0]) / localdirx ;
       disty = (localdiry>0.) ? (par[1]-newpt[1]) / localdiry : (-par[1]-newpt[1]) / localdiry ;
       distz = (localdirz>0.) ? (par[2]-newpt[2]) / localdirz : (-par[2]-newpt[2]) / localdirz ;

       assert(distx>0);
       assert(disty>0);
       assert(distz>0);

       distance[i] = (distx<disty) ? distx : disty ;
       if(distz < distance[i]) distance[i] = distz;
     }
   }
}

#endif /* PHYSICALBOX_H_ */
