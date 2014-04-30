/// @file ShapeImplementationHelper.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_SHAPEIMPLEMENTATIONHELPER_H_
#define VECGEOM_VOLUMES_SHAPEIMPLEMENTATIONHELPER_H_

#include "base/global.h"

#include "backend/scalar/backend.h"
#include "backend/backend.h"

namespace VECGEOM_NAMESPACE {

template <class Shape, class Specialization>
class ShapeImplementationHelper : public Shape {

public:

#ifndef VECGEOM_NVCC

  ShapeImplementationHelper(char const *const label,
                            LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : VPlacedVolume(label, logical_volume, transformation, this) {}

  ShapeImplementationHelper(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : ShapeImplementationHelper("", logical_volume, transformation) {}

#else // Compiling for CUDA

  __device__
  ShapeImplementationHelper(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            const int id)
      : VPlacedVolume(logical_volume, transformation, this, id) {}

#endif

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool Inside(Vector3D<Precision> const &point) const {
    EInside output;
    Specialization::template Inside<kScalar>(
      *this->GetUnplacedVolume(),
      *this->transformation(),
      point,
      output
    );
    return (output == kInside) ? true : false;
  }

#ifdef VECGEOM_VC

  template <class Container_t>
  void InsideTemplate(Container_t const &points, bool *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += VcPrecision::Size) {
      Vector3D<VcPrecision> point(
        VcPrecision(&points.x(i)),
        VcPrecision(&points.y(i)),
        VcPrecision(&points.z(i))
      );
      VcInt result;
      Specialization::template Inside<kVc>(
        *this->GetUnplacedVolume(),
        *this->transformation(),
        point,
        result
      );
      for (int j = 0; j < VcPrecision::Size; ++j) {
        output[j+i] = (result[j] == kInside) ? true : false;
      }
    }
  }

#else // Scalar default

  template <class Container_t>
  void InsideTemplate(Container_t const &points, bool *const output) const {
    for (int i = 0, i_max = points.size(); ++i) {
      EInside result;
      Specialization::template Inside<kScalar>(
        *this->GetUnplacedVolume(),
        *this->transformation(),
        point,
        result
      );
      output[i] = (result[j] == kInside) ? true : false;
    }
  }

#endif

  virtual void Inside(AOS3D<Precision> const &points,
                      bool *const output) const {
    InsideTemplate(points, output);
  }

  virtual void Inside(SOA3D<Precision> const &points,
                      bool *const output) const {
    InsideTemplate(points, output);
  }

}; // End class ShapeImplementationHelper

} // End global namespace

#endif // VECGEOM_VOLUMES_SHAPEIMPLEMENTATIONHELPER_H_