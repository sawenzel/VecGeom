/// \file ShapeImplementationHelper.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_SHAPEIMPLEMENTATIONHELPER_H_
#define VECGEOM_VOLUMES_SHAPEIMPLEMENTATIONHELPER_H_

#include "base/Global.h"

#include "backend/scalar/Backend.h"
#include "backend/Backend.h"
#include "base/SOA3D.h"
#include "base/AOS3D.h"
#include "volumes/PlacedBox.h"

#include <algorithm>

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
                            PlacedBox const *const boundingBox,
                            const int id)
      : Shape(logical_volume, transformation, boundingBox, id) {}

#endif

  VECGEOM_CUDA_HEADER_BOTH
  virtual Inside_t Inside(Vector3D<Precision> const &point) const {
    Inside_t output = EInside::kOutside;
    Specialization::template Inside<kScalar>(
      *this->GetUnplacedVolume(),
      *this->transformation(),
      point,
      output
    );
    return output;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool Contains(Vector3D<Precision> const &point) const {
    bool output = false;
    Vector3D<Precision> localPoint;
    Specialization::template Contains<kScalar>(
      *this->GetUnplacedVolume(),
      *this->transformation(),
      point,
      localPoint,
      output
    );
    return output;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool Contains(Vector3D<Precision> const &point,
                        Vector3D<Precision> &localPoint) const {
    bool output = false;
    Specialization::template Contains<kScalar>(
      *this->GetUnplacedVolume(),
      *this->transformation(),
      point,
      localPoint,
      output
    );
    return output;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool UnplacedContains(Vector3D<Precision> const &point) const {
    bool output = false;
    Specialization::template UnplacedContains<kScalar>(
      *this->GetUnplacedVolume(),
      point,
      output
    );
    return output;
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
  void ContainsTemplate(Container_t const &points, bool *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += VcPrecision::Size) {
      Vector3D<VcPrecision> point(
        VcPrecision(&points.x(i)),
        VcPrecision(&points.y(i)),
        VcPrecision(&points.z(i))
      );
      Vector3D<VcPrecision> localPoint;
      VcBool result(false);
      Specialization::template Contains<kVc>(
        *this->GetUnplacedVolume(),
        *this->transformation(),
        point,
        localPoint,
        result
      );
      for (unsigned j = 0; j < VcPrecision::Size; ++j) {
        output[j+i] = result[j];
      }
    }
  }

  template <class Container_t>
  void InsideTemplate(Container_t const &points, Inside_t *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += VcPrecision::Size) {
      Vector3D<VcPrecision> point(
        VcPrecision(&points.x(i)),
        VcPrecision(&points.y(i)),
        VcPrecision(&points.z(i))
      );
      VcInside result = EInside::kOutside;
      Specialization::template Inside<kVc>(
        *this->GetUnplacedVolume(),
        *this->transformation(),
        point,
        result
      );
      for (unsigned j = 0; j < VcPrecision::Size; ++j) output[j+i] = result[j];
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
      VcPrecision result = kInfinity;
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
      VcPrecision result = kInfinity;
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
      VcPrecision result = kInfinity;
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
      VcPrecision result = kInfinity;
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
  void ContainsTemplate(Container_t const &points, bool *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; ++i) {
      Vector3D<Precision> localPoint;
      Specialization::template Contains<kScalar>(
        *this->GetUnplacedVolume(),
        *this->transformation(),
        points[i],
        localPoint,
        output[i]
      );
    }
  }

  template <class Container_t>
  void InsideTemplate(Container_t const &points,
                      Inside_t *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; ++i) {
      Inside_t result = EInside::kOutside;
      Specialization::template Inside<kScalar>(
        *this->GetUnplacedVolume(),
        *this->transformation(),
        points[i],
        result
      );
      output[i] = result;
    }
  }

  template <class Container_t>
  void DistanceToInTemplate(Container_t const &points,
                            Container_t const &directions,
                            Precision const *const stepMax,
                            Precision *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; ++i) {
      Specialization::template DistanceToIn<kScalar>(
        *this->GetUnplacedVolume(),
        *this->transformation(),
        points[i],
        directions[i],
        stepMax[i],
        output[i]
      );
    }
  }

  template <class Container_t>
  void DistanceToOutTemplate(Container_t const &points,
                             Container_t const &directions,
                             Precision const *const stepMax,
                             Precision *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; ++i) {
      Specialization::template DistanceToOut<kScalar>(
        *this->GetUnplacedVolume(),
        points[i],
        directions[i],
        stepMax[i],
        output[i]
      );
    }
  }

  template <class Container_t>
  void SafetyToInTemplate(Container_t const &points,
                          Precision *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; ++i) {
      Specialization::template SafetyToIn<kScalar>(
        *this->GetUnplacedVolume(),
        *this->transformation(),
        points[i],
        output[i]
      );
    }
  }

  template <class Container_t>
  void SafetyToOutTemplate(Container_t const &points,
                           Precision *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; ++i) {
      Specialization::template SafetyToOut<kScalar>(
        *this->GetUnplacedVolume(),
        points[i],
        output[i]
      );
    }
  }

#endif

  virtual void Contains(AOS3D<Precision> const &points,
                        bool *const output) const {
    ContainsTemplate(points, output);
  }

  virtual void Contains(SOA3D<Precision> const &points,
                        bool *const output) const {
    ContainsTemplate(points, output);
  }

  virtual void Inside(AOS3D<Precision> const &points,
                      Inside_t *const output) const {
    InsideTemplate(points, output);
  }

  virtual void Inside(SOA3D<Precision> const &points,
                      Inside_t *const output) const {
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