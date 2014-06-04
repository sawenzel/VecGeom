/// @file TubeImplementation.h
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_TUBEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_TUBEIMPLEMENTATION_H_


#include "base/global.h"
#include "base/transformation3d.h"
#include "volumes/kernel/BoxImplementation.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/UnplacedTrd.h"
#include "volumes/kernel/shapetypes/TrdTypes.h"

namespace VECGEOM_NAMESPACE {

namespace TrdUtilities {

	/*
	 * Checks whether a point (x, y) falls on the left or right half-plane
	 * of a line. The line is defined by a (vx, vy) vector, extended to infinity.
	 *
	 * Of course this can only be used for lines that pass through (0, 0), but
	 * you can supply transformed coordinates for the point to check for any line.
	 *
	 * This simply calculates the magnitude of the cross product of vectors (px, py) 
	 * and (vx, vy), which is defined as |x| * |v| * sin theta.
	 *
	 * If the cross product is positive, the point is clockwise of V, or the "right"
	 * half-plane. If it's negative, the point is CCW and on the "left" half-plane. 
	 */

	template<typename Backend>
	VECGEOM_INLINE
	VECGEOM_CUDA_HEADER_BOTH
	void PointLineOrientation(typename Backend::precision_v const &px, typename Backend::precision_v const &py,
                            Precision const &vx, Precision const &vy,
                            typename Backend::precision_v &crossProduct) {
		crossProduct = vx * py - vy * px;
	}


} // Trd utilities

template <TranslationCode transCodeT, RotationCode rotCodeT, typename trdTypeT>
struct TrdImplementation {

  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void UnplacedContains(
      UnplacedTrd const &trd,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside) {

    using namespace TrdUtilities;
    using namespace TrdTypes;  
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    Float_t pzPlusDz = point.z()+trd.dz();

    // inside Z?
    Bool_t inz = Abs(point.z()) <= trd.dz();

    // inside X?
    Float_t cross;
    PointLineOrientation<Backend>(Abs(point.x()) - trd.dx1(), pzPlusDz, trd.x2minusx1(), trd.dztimes2(), cross);
    Bool_t inx = cross >= 0;

    Bool_t iny;
    // inside Y?
    if(HasVaryingY<trdTypeT>::value != TrdTypes::kNo) {
      // If Trd type is unknown don't bother with a runtime check, assume
      // the general case
      PointLineOrientation<Backend>(Abs(point.y()) - trd.dy1(), pzPlusDz, trd.y2minusy1(), trd.dztimes2(), cross);
      iny = cross >= 0;
    }
    else {
      iny = Abs(point.y()) <= trd.dy1();
    }

    inside = inz & inx & iny;
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Contains(
      UnplacedTrd const &trd,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside) {

    localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
    UnplacedContains<Backend>(trd, localPoint, inside);
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Inside(UnplacedTrd const &trd,
                     Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point,
                     typename Backend::inside_v &inside) {
    typedef typename Backend::bool_v Bool_t;
    Vector3D<typename Backend::precision_v> localpoint;
    localpoint = transformation.Transform<transCodeT, rotCodeT>(point);
    inside = EInside::kOutside;

    Bool_t isInside;
    UnplacedContains<Backend>(trd, localpoint, isInside);
    MaskedAssign(isInside, EInside::kInside, &inside);
  }

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    static void DistanceToIn(
        UnplacedTrd const &trd,
        Transformation3D const &transformation,
        Vector3D<typename Backend::precision_v> const &point,
        Vector3D<typename Backend::precision_v> const &direction,
        typename Backend::precision_v const &stepMax,
        typename Backend::precision_v &distance) {
	
	}

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedTrd const &trd,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &dir,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance) {

  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(UnplacedTrd const &trd,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety) {

  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedTrd const &trd,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety) {

  }

};

}

#endif // VECGEOM_VOLUMES_KERNEL_TUBEIMPLEMENTATION_H_