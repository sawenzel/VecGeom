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

template<typename ValueType=double>
struct UUtils
{
	static constexpr ValueType kInfinity=1E30;
	static constexpr ValueType kPi       = 3.14159265358979323846;
	static constexpr ValueType kTwoPi    = 2.0 * kPi;
	static constexpr ValueType kRadToDeg = 180.0 / kPi;
	static constexpr ValueType kDegToRad = kPi / 180.0;
	static constexpr ValueType kSqrt2    = 1.4142135623730950488016887242097;

	static constexpr ValueType fgTolerance = 1E-9;
	static constexpr ValueType frTolerance = 1E-9;
	static constexpr ValueType faTolerance = 1E-9;
	static constexpr ValueType fgHalfTolerance = fgTolerance*0.5;
	static constexpr ValueType frHalfTolerance = frTolerance*0.5;
	static constexpr ValueType faHalfTolerance = faTolerance*0.5;

	static
	inline ValueType GetCarTolerance()  {return fgTolerance;}
	static
	inline ValueType GetRadTolerance()  {return frTolerance;}
	static
	inline ValueType GetAngTolerance()  {return faTolerance;}

	static
	inline ValueType GetCarHalfTolerance()  {return fgHalfTolerance;}
	static
	inline ValueType GetRadHalfTolerance()  {return frHalfTolerance;}
	static
	inline ValueType GetAngHalfTolerance()  {return faHalfTolerance;}

	// static vd tol_v = 1.E-10;
};


#endif /* GLOBALDEFS_H_ */
