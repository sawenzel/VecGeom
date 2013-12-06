/*
 * GlobalDefs.h
 *
 *  Created on: Nov 15, 2013
 *      Author: swenzel
 */

#ifndef GLOBALDEFS_H_
#define GLOBALDEFS_H_


#ifndef ALIGNMENT_BOUNDARY
#define	ALIGNMENT_BOUNDARY 32
#endif

// Other ideas: we should be able to configure the library for different purposes
// For example: the question whether we treat the surface or not should be configurable
// Could solve this via template type construction or via simple ifdefs

#ifndef GEOM_SURFACE_MODE
#define GEOM_SURFACE_MODE false
#endif

/*
struct SurfaceEnumType
{
	enum { Outside, OnSurface, Inside };
};

template <bool B>
struct SurfaceReturnType
{
	typedef bool type;
};
template <>
struct SurfaceReturnType<true>
{
	typedef typename SurfaceEnumType type;
};
*/

#include "Vc/vector.h"

//template<typename ValueType=double>
//struct UUtils
namespace UUtils
{
	static const double kInfinity=1E30;
	static const double kPi=3.14159265358979323846;
	static const double kTwoPi=2.0 * kPi;
	static const double kRadToDeg=180.0 / kPi;
	static const double kDegToRad=kPi / 180.0;
	static const double kSqrt2=1.4142135623730950488016887242097;

	static const double fgTolerance=1E-9;
	static const double frTolerance=1E-9;
	static const double faTolerance=1E-9;
	static const double fgHalfTolerance = fgTolerance*0.5;
	static const double frHalfTolerance = frTolerance*0.5;
	static const double faHalfTolerance = faTolerance*0.5;

	typedef Vc::double_v ValueType;

	static const ValueType kInfinityVc=1E30;
	static const ValueType kPiVc=3.14159265358979323846;
	static const ValueType kTwoPiVc=2.0 * kPi;
	static const ValueType kRadToDegVc=180.0 / kPi;
	static const ValueType kDegToRadVc=kPi / 180.0;
	static const ValueType kSqrt2Vc=1.4142135623730950488016887242097;

	static const ValueType fgToleranceVc=1E-9;
	static const ValueType frToleranceVc=1E-9;
	static const ValueType faToleranceVc=1E-9;
	static const ValueType fgHalfToleranceVc = fgTolerance*0.5;
	static const ValueType frHalfToleranceVc = frTolerance*0.5;
	static const ValueType faHalfToleranceVc = faTolerance*0.5;

	static
	bool IsSameWithinTolerance( double x, double y )
	{
		return std::abs(x-y) < fgHalfTolerance;
	}

	/*
	static void init()
	{
		kInfinity=1E30;
		kPi       = 3.14159265358979323846;
		kTwoPi = 2.0 * kPi;
		kRadToDeg = 180.0 / kPi;
		kDegToRad = kPi / 180.0;
		kSqrt2    = 1.4142135623730950488016887242097;

		fgTolerance = 1E-9;
		frTolerance = 1E-9;
		faTolerance = 1E-9;
		fgHalfTolerance = fgTolerance*0.5;
		frHalfTolerance = frTolerance*0.5;
		faHalfTolerance = faTolerance*0.5;
	}
*/

	static
	inline double GetCarTolerance()  {return fgTolerance;}
	static
	inline double GetRadTolerance()  {return frTolerance;}
	static
	inline double GetAngTolerance()  {return faTolerance;}

	static
	inline double GetCarHalfTolerance()  {return fgHalfTolerance;}
	static
	inline double GetRadHalfTolerance()  {return frHalfTolerance;}
	static
	inline double GetAngHalfTolerance()  {return faHalfTolerance;}

};


#endif /* GLOBALDEFS_H_ */
