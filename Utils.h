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
/*
struct DistanceFunctions
{
	// calculates/
	template <typename T>
	static void DistToTube( T const & r2, T const & n2, T const & rdotn, T const & radius2, T & b, T & delta )
	{
		// r2 is the scalar product of position vector
		// n2 is the norm of norm vector
		T t1=T(1)/n2;
		T t3=r2-radius2;
		b=t1*rdotn;
		T cv=t1*t3;
		delta=b*b-c;
		delta=sqrt(delta);
	}


};
*/


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
