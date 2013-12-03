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


struct PhiToPlane
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
};


#endif /* UTILS_H_ */
