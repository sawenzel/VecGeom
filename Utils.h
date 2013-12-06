/*
 * Utils.h
 *
 *  Created on: Nov 15, 2013
 *      Author: swenzel
 */


#ifndef UTILS_H_
#define UTILS_H_


#include "tbb/tick_count.h" // timing from Intel TBB
#include "cmath"
#include "Vc/vector.h"

struct StopWatch
{
  tbb::tick_count t1;
  tbb::tick_count t2;
  void Start(){  t1=tbb::tick_count::now(); }
  void Stop(){  t2=tbb::tick_count::now(); }
  double getDeltaSecs() { return (t2-t1).seconds(); }
};



struct QuadraticSolver
{




};


// a kernel function solving a quadratic equation

struct DistanceFunctions
{
	// calculates/
	template <typename FloatType>
	static void DistToTube( FloatType const & rxy2, FloatType const & dirxy2,
							FloatType const & rdotdirxy, FloatType const & radius2, FloatType & b, FloatType & delta )
	{
		// r2 is the norm squared of position vector in x-y plane only
		// n2 is the norm squared of direction vector in x-y plane only
		// rdotn is scalar product between r and direction vector in x-y plane
		// radius2 is square of tube radius to be considered

		// return vale:
		// the distance will be b +- delta or infinity if delta==-1
		FloatType t1=FloatType(1)/dirxy2;
		FloatType t3=rxy2-radius2;
		b=t1*rdotdirxy;
		delta=b*b-t1*t3;
		if( delta>0 )
			delta=std::sqrt(delta);
		else
			delta=-1;
	}

	// special version for Vc
	static void DistToTubeVc( Vc::double_v const & rxy2, Vc::double_v const & dirxy2,
							Vc::double_v const & rdotdirxy, Vc::double_v const & radius2, Vc::double_v & b, Vc::double_v & delta )
	{
		// r2 is the norm squared of position vector in x-y plane only
		// n2 is the norm squared of direction vector in x-y plane only
		// rdotn is scalar product between r and direction vector in x-y plane
		// radius2 is square of tube radius to be considered

		// return vale:
		// the distance will be b +- delta or infinity if delta==-1
		Vc::double_v t1 = Vc::One/dirxy2;
		Vc::double_v t3 = rxy2-radius2;
		b = t1*rdotdirxy;
		delta = b*b-t1*t3;

		// maybe better to split this part off to allow for early return

		Vc::double_m deltamask = delta > 0;
		delta(deltamask)  = Vc::sqrt(delta); // is this executed even if no hit at all ??
		delta(!deltamask) = -1;
	}
};


struct PhiUtils
{
	template <typename T>
	static
	inline
	void GetNormalVectorToPhiPlane( T phi , Vector3D & v, bool inverse )
	{
		if( ! inverse )
		{
			v.x=-std::sin( phi );
			v.y=+std::cos( phi );
			v.z=T(0);
		}
		else
		{
			v.x=std::sin( phi );
			v.y=-std::cos( phi );
			v.z=T(0);
		}
	}

	template <typename T>
	static
	inline
	void GetAlongVectorToPhiPlane( T phi , Vector3D & v )
	{
		v.x=std::cos( phi );
		v.y=std::sin( phi );
		v.z=T(0);
	}


	template <typename T>
	static
	inline
	void PointIsInPhiSector( T const & phi1normalx, T const & phi1normaly,
							 T const & phi2normalx, T const & phi2normaly,
							 Vc::Vector<T> const & xcoord, Vc::Vector<T> const & ycoord,
							 typename Vc::Vector<T>::Mask & isinphi )
	{
		// method based on calculating the scalar product of position vectors with the normals of the (empty) phi sektor
		// avoids taking the atan2

		// this method could be template specialized in case DeltaPhi = 180^o
		Vc::Vector<T> scalarproduct1 = phi1normalx*xcoord + phi1normaly*ycoord;
		Vc::Vector<T> scalarproduct2 = phi2normalx*xcoord + phi2normaly*ycoord;
		isinphi = (scalarproduct1 > Vc::Zero && scalarproduct2 > Vc::Zero);
	}

	template <typename T>
	static
	inline
	typename Vc::Vector<T>::Mask PointIsInPhiSector( T const & phi1normalx, T const & phi1normaly,
							 T const & phi2normalx, T const & phi2normaly,
							 Vc::Vector<T> const & xcoord, Vc::Vector<T> const & ycoord )
	{
		// method based on calculating the scalar product of position vectors with the normals of the (empty) phi sektor
		// avoids taking the atan2

		// this method could be template specialized in case DeltaPhi = 180^o
		if( phi1normalx*phi2normalx + phi1normaly*phi2normaly >= 0)
		{
			Vc::Vector<T> scalarproduct1 = phi1normalx*xcoord + phi1normaly*ycoord;
			Vc::Vector<T> scalarproduct2 = phi2normalx*xcoord + phi2normaly*ycoord;
			return (scalarproduct1 > Vc::Zero && scalarproduct2 > Vc::Zero);
		}
		else // here sektor is larger than Pi
		{
			Vc::Vector<T> scalarproduct1 = phi1normalx*xcoord + phi1normaly*ycoord;
			Vc::Vector<T> scalarproduct2 = phi2normalx*xcoord + phi2normaly*ycoord;
			return (scalarproduct1 > Vc::Zero || scalarproduct2 > Vc::Zero);
		}
	}

	template <typename T>
	static
	inline
	bool PointIsInPhiSector( Vector3D const & phi1normal, Vector3D const & phi2normal,
			Vector3D const & pos )
		{
			// method based on calculating the scalar product of position vectors with the normals of the (empty) phi sektor
			// a	voids taking the atan2

			// this method could be template specialized in case DeltaPhi = 180^o
			T scalarproduct1 = Vector3D::scalarProductInXYPlane( phi1normal, pos);
			T scalarproduct2 = Vector3D::scalarProductInXYPlane( phi2normal, pos);
			return (scalarproduct1 > 0 && scalarproduct2 > 0);
		}

	template <typename T>
	static
	inline
	void DistanceToPhiPlanes( T const & dz, T const & radius2, T const & phi1normalx, T const & phi1normaly,
							  T const & phi2normalx, T const & phi2normaly,
							  Vector3D const & phiAlong1, Vector3D const & phiAlong2,
							  Vc::Vector<T> const & xcoord, Vc::Vector<T> const & ycoord, Vc::Vector<T> const & zcoord,
							  Vc::Vector<T> const & xdir, Vc::Vector<T> const & ydir, Vc::Vector<T> const & zdir,
							  Vc::Vector<T> & distToPlane )
	{
		// method based on calculating the scalar product of position vectors with the normals of the (empty) phi sektor
		// avoids taking the atan2

		// this method could be template specialized on case DeltaPhi = 180^o to avoid double division
		Vc::Vector<T> scalarproduct1 = phi1normalx*xcoord + phi1normaly*ycoord;
		Vc::Vector<T> scalarproduct2 = phi2normalx*xcoord + phi2normaly*ycoord;
		Vc::Vector<T> N1dotDir = phi1normalx*xdir + phi1normaly*ydir;
		Vc::Vector<T> N2dotDir = phi2normalx*xdir + phi2normaly*ydir;
		Vc::Vector<T> distToPlane1 = UUtils::kInfinityVc;
		distToPlane = UUtils::kInfinityVc;

		Vc::Vector<T> temp = -scalarproduct1/N1dotDir;
		Vc::Vector<T> zi = zcoord + temp*zdir;
		Vc::Vector<T> xi = xcoord + temp*xdir;
		Vc::Vector<T> yi = ycoord + temp*ydir;

		//

		// check the conditions -- tolerance is coming later
		distToPlane1 ( temp > 0 && Vc::abs(zi) < dz && xi*xi + yi*yi < radius2 && xi * phiAlong1.x + yi * phiAlong1.y >= 0 ) = temp;

		temp = -scalarproduct2/N2dotDir;
		zi = zcoord + temp*zdir;
		xi = xcoord + temp*xdir;
		yi = ycoord + temp*ydir;

		// check the conditions -- tolerance is coming later
		distToPlane ( temp > 0 && Vc::abs(zi) < dz && xi*xi + yi*yi < radius2 && xi * phiAlong2.x + yi * phiAlong2.y >= 0 ) = temp;

		distToPlane = Vc::min(distToPlane1, distToPlane);
	}
};


#endif /* UTILS_H_ */
