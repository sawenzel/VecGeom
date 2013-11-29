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
