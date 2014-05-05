/// @file ShapeImplementationHelper.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_SHAPEIMPLEMENTATIONHELPER_H_
#define VECGEOM_VOLUMES_SHAPEIMPLEMENTATIONHELPER_H_

#include "base/global.h"

#include "backend/scalar/backend.h"
#include "backend/backend.h"
#include "base/soa3d.h"
#include "base/aos3d.h"

namespace VECGEOM_NAMESPACE {

template <class Shape, class Specialization>
class ShapeImplementationHelper : public Shape {

public:

#ifndef VECGEOM_NVCC

  ShapeImplementationHelper(char const *const label,
                            LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox)
      : Shape(label, logical_volume, transformation, boundingBox) {}

  ShapeImplementationHelper(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox)
      : ShapeImplementationHelper("", logical_volume, transformation,
                                  boundingBox) {}

#else // Compiling for CUDA

  __device__
  ShapeImplementationHelper(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            const int id)
      : VPlacedVolume(logical_volume, transformation, this, id) {}

#endif

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool Inside(Vector3D<Precision> const &point,
                      Vector3D<Precision> &localPoint) const {
    int output;
    Specialization::template Inside<kScalar>(
      *this->GetUnplacedVolume(),
      *this->transformation(),
      point,
      localPoint,
      output
    );
    return (output == EInside::kInside) ? true : false;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool UnplacedInside(Vector3D<Precision> const &point) const {
    int output;
    Specialization::template UnplacedInside<kScalar>(
      *this->GetUnplacedVolume(),
      point,
      output
    );
    return (output == EInside::kInside) ? true : false;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToIn(Vector3D<Precision> const &point,
                                 Vector3D<Precision> const &direction,
                                 const Precision stepMax = kInfinity) const {
    Precision output = kInfinity;
    Specialization::template DistanceToIn<kScalar>(
      *this->GetUnplacedVolume(),
      *this->transformation(),
      point,
      direction,
      stepMax,
      output
    );
    return output;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToOut(Vector3D<Precision> const &point,
                                  Vector3D<Precision> const &direction,
                                  const Precision stepMax = kInfinity) const {
    Precision output = kInfinity;
    Specialization::template DistanceToOut<kScalar>(
      *this->GetUnplacedVolume(),
      point,
      direction,
      stepMax,
      output
    );
    return output;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision SafetyToIn(Vector3D<Precision> const &point) const {
    Precision output = kInfinity;
    Specialization::template SafetyToIn<kScalar>(
      *this->GetUnplacedVolume(),
      *this->transformation(),
      point,
      output
    );
    return output;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision SafetyToOut(Vector3D<Precision> const &point) const {
    Precision output = kInfinity;
    Specialization::template SafetyToOut<kScalar>(
      *this->GetUnplacedVolume(),
      point,
      output
    );
    return output;
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
      Vector3D<VcPrecision> localPoint;
      VcInt result;
      Specialization::template Inside<kVc>(
        *this->GetUnplacedVolume(),
        *this->transformation(),
        point,
        localPoint,
        result
      );
      for (unsigned j = 0; j < VcInt::Size; ++j) {
        output[j+i] = (result[j] == EInside::kInside) ? true : false;
      }
    }
  }

  template <class Container_t>
  void DistanceToInTemplate(Container_t const &points,
                            Container_t const &directions,
                            Precision const *const stepMax,
                            Precision *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += VcPrecision::Size) {
      Vector3D<VcPrecision> point(
        VcPrecision(&points.x(i)),
        VcPrecision(&points.y(i)),
        VcPrecision(&points.z(i))
      );
      Vector3D<VcPrecision> direction(
        VcPrecision(&directions.x(i)),
        VcPrecision(&directions.y(i)),
        VcPrecision(&directions.z(i))
      );
      VcPrecision stepMaxVc = VcPrecision(&stepMax[i]);
      VcPrecision result;
      Specialization::template DistanceToIn<kVc>(
        *this->GetUnplacedVolume(),
        *this->transformation(),
        point,
        direction,
        stepMaxVc,
        result
      );
      result.store(&output[i]);
    }
  }

  template <class Container_t>
  void DistanceToOutTemplate(Container_t const &points,
                             Container_t const &directions,
                             Precision const *const stepMax,
                             Precision *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += VcPrecision::Size) {
      Vector3D<VcPrecision> point(
        VcPrecision(&points.x(i)),
        VcPrecision(&points.y(i)),
        VcPrecision(&points.z(i))
      );
      Vector3D<VcPrecision> direction(
        VcPrecision(&directions.x(i)),
        VcPrecision(&directions.y(i)),
        VcPrecision(&directions.z(i))
      );
      VcPrecision stepMaxVc = VcPrecision(&stepMax[i]);
      VcPrecision result;
      Specialization::template DistanceToOut<kVc>(
        *this->GetUnplacedVolume(),
        point,
        direction,
        stepMaxVc,
        result
      );
      result.store(&output[i]);
    }
  }

  template <class Container_t>
  void SafetyToInTemplate(Container_t const &points,
                          Precision *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += VcPrecision::Size) {
      Vector3D<VcPrecision> point(
        VcPrecision(&points.x(i)),
        VcPrecision(&points.y(i)),
        VcPrecision(&points.z(i))
      );
      VcPrecision result;
      Specialization::template SafetyToIn<kVc>(
        *this->GetUnplacedVolume(),
        *this->transformation(),
        point,
        result
      );
      result.store(&output[i]);
    }
  }

  template <class Container_t>
  void SafetyToOutTemplate(Container_t const &points,
                           Precision *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += VcPrecision::Size) {
      Vector3D<VcPrecision> point(
        VcPrecision(&points.x(i)),
        VcPrecision(&points.y(i)),
        VcPrecision(&points.z(i))
      );
      VcPrecision result;
      Specialization::template SafetyToOut<kVc>(
        *this->GetUnplacedVolume(),
        point,
        result
      );
      result.store(&output[i]);
    }
  }

#else // Scalar default

  template <class Container_t>
  void InsideTemplate(Container_t const &points, bool *const output) const {
    for (int i = 0, i_max = points.size(); ++i) {
      int result;
      Specialization::template Inside<kScalar>(
        *this->GetUnplacedVolume(),
        *this->transformation(),
        point,
        result
      );
      output[i] = (result[j] == kInside) ? true : false;
    }
  }

  template <class Container_t>
  void DistanceToInTemplate(Container_t const &points,
                            Container_t const &directions,
                            Precision const *const stepMax,
                            Precision *const output) const {
    for (int i = 0, i_max = points.size(); ++i) {
      Specialization::template DistanceToIn<kScalar>(
        *this->GetUnplacedVolume(),
        *this->transformation(),
        points[i],
        directions[i],
        output[i]
      );
    }
  }

  template <class Container_t>
  void DistanceToOutTemplate(Container_t const &points,
                             Container_t const &directions,
                             Precision const *const stepMax,
                             Precision *const output) const {
    for (int i = 0, i_max = points.size(); ++i) {
      Specialization::template DistanceToOut<kScalar>(
        *this->GetUnplacedVolume(),
        points[i],
        directions[i],
        output[i]
      );
    }
  }

  template <class Container_t>
  void SafetyToInTemplate(Container_t const &points,
                          Precision *const output) const {
    for (int i = 0, i_max = points.size(); ++i) {
      Specialization::template SafetyToIn<kScalar>(
        *this->GetUnplacedVolume(),
        *this->transformation(),
        point,
        output[i]
      );
    }
  }

  template <class Container_t>
  void SafetyToOutTemplate(Container_t const &points,
                           Precision *const output) const {
    for (int i = 0, i_max = points.size(); ++i) {
      Specialization::template SafetyToOut<kScalar>(
        *this->GetUnplacedVolume(),
        point,
        output[i]
      );
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

  virtual void DistanceToIn(AOS3D<Precision> const &points,
                            AOS3D<Precision> const &directions,
                            Precision const *const stepMax,
                            Precision *const output) const {
    DistanceToInTemplate(points, directions, stepMax, output);
  }

  virtual void DistanceToIn(SOA3D<Precision> const &points,
                            SOA3D<Precision> const &directions,
                            Precision const *const stepMax,
                            Precision *const output) const {
    DistanceToInTemplate(points, directions, stepMax, output);
  }

  virtual void DistanceToOut(AOS3D<Precision> const &points,
                             AOS3D<Precision> const &directions,
                             Precision const *const stepMax,
                             Precision *const output) const {
    DistanceToOutTemplate(points, directions, stepMax, output);
  }

  virtual void DistanceToOut(SOA3D<Precision> const &points,
                             SOA3D<Precision> const &directions,
                             Precision const *const stepMax,
                             Precision *const output) const {
    DistanceToOutTemplate(points, directions, stepMax, output);
  }

  virtual void SafetyToIn(AOS3D<Precision> const &points,
                          Precision *const output) const {
    SafetyToInTemplate(points, output);
  }

  virtual void SafetyToIn(SOA3D<Precision> const &points,
                          Precision *const output) const {
    SafetyToInTemplate(points, output);
  }

  virtual void SafetyToOut(AOS3D<Precision> const &points,
                          Precision *const output) const {
    SafetyToOutTemplate(points, output);
  }

  virtual void SafetyToOut(SOA3D<Precision> const &points,
                          Precision *const output) const {
    SafetyToOutTemplate(points, output);
  }

}; // End class ShapeImplementationHelper

} // End global namespace

#endif // VECGEOM_VOLUMES_SHAPEIMPLEMENTATIONHELPER_H_