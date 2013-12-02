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

struct UUtils
{
	static constexpr double kInfinity=1E30;
	static constexpr double kPi       = 3.14159265358979323846;
	static constexpr double kTwoPi    = 2.0 * kPi;
	static constexpr double kRadToDeg = 180.0 / kPi;
	static constexpr double kDegToRad = kPi / 180.0;
	static constexpr double kSqrt2    = 1.4142135623730950488016887242097;

	static constexpr double fgTolerance = 1E-9;
	static constexpr double frTolerance = 1E-9;
	static constexpr double faTolerance = 1E-9;
	static constexpr double fgHalfTolerance = fgTolerance*0.5;
	static constexpr double frHalfTolerance = frTolerance*0.5;
	static constexpr double faHalfTolerance = faTolerance*0.5;

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
